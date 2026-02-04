#!/usr/bin/env bash
# ===========================================
# LawSelect RL 训练脚本 (3B 版本，2×60G 优化)
# 
# 硬件要求: 2 × 60G GPU
# 用法: CUDA_VISIBLE_DEVICES=0,1 bash bash/agent/train_rl_lawselect.sh
#
# 训练目标：
# - 保持高 MRR（3B 基座已有 0.93）
# - 提高 Recall@10（让模型多选一些）
# ===========================================

set -euo pipefail

ROOT="/data-share/chenxuanyi/internship/JuDGE_RL"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_VERBOSITY=error
export PYTORCH_ALLOC_CONF=expandable_segments:True

# vLLM 环境变量（避免 V1 引擎冲突）
export VLLM_USE_V1=1
export VLLM_USE_FLASHINFER_SAMPLER=0

# ============== 配置 ==============
# 使用 3B 模型（效果接近 7B，2×60G 可充分利用）
BASE_MODEL="/data-share/LLM/models--Qwen--Qwen2.5-3B-Instruct/snapshots/aa8e72537993ba99e69dfaafa59ed015b17504d1"
MODEL_TYPE="qwen2"
MASTER_PORT="${MASTER_PORT:-27107}"

# GPU 配置（2×60G）
gpu_ids="${CUDA_VISIBLE_DEVICES:-4,7}"
IFS=',' read -ra parts <<< "$gpu_ids"
gpu_num=${#parts[@]}

cd "${ROOT}"

# ============== 任务配置 ==============
DATA_DIR="${ROOT}/data/agent_rl/law_select_train.jsonl"
REWARD_FUNC="law_select_reward"
OUT_DIR="${ROOT}/output/rl_lawselect_3b_lora"

# 训练参数（2×60G 充分利用，大幅增加探索）
MAX_INPUT_LENGTH=10000
MAX_OUTPUT_LENGTH=1024
NUM_GENERATIONS=${NUM_GENERATIONS:-16}  # 2×60G 可支持大探索
BATCH_SIZE=2                            # 增大 batch
GRAD_ACCUM=16                           # 有效 batch = 2*2*16 = 64
LEARNING_RATE="2e-5"
BETA="0.04"               # 略降 KL 惩罚，鼓励更多探索
TEMPERATURE="0.9"         # 略升温度，增加多样性
NUM_EPOCHS="3"

# LoRA 配置（增大 rank 提升表达能力）
LORA_RANK="128"
LORA_ALPHA="256"
LORA_DROPOUT="0.05"

# ============== 检查 ==============
if [[ ! -f "${DATA_DIR}" ]]; then
    echo "[ERROR] 训练数据不存在: ${DATA_DIR}"
    echo "请先运行: bash bash/agent/prepare_agent_rl_data.sh"
    exit 1
fi

export HF_DATASETS_CACHE="${ROOT}/.cache"

echo "=========================================="
echo "  LawSelect RL (3B LoRA) - vLLM 加速版"
echo "=========================================="
echo "  模型: Qwen2.5-3B-Instruct"
echo "  GPU: ${gpu_num} 卡"
echo "  vLLM: 启用（生成加速 5-10x）"
echo "  num_generations: ${NUM_GENERATIONS}"
echo "  batch_size: ${BATCH_SIZE}, grad_accum: ${GRAD_ACCUM}"
echo "  LoRA rank: ${LORA_RANK}"
echo "  beta: ${BETA}, lr: ${LEARNING_RATE}, temp: ${TEMPERATURE}"
echo "  输出: ${OUT_DIR}"
echo "=========================================="

# ============== 运行训练 ==============
CUDA_VISIBLE_DEVICES=$gpu_ids \
NPROC_PER_NODE=$gpu_num \
MASTER_PORT=$MASTER_PORT \
swift rlhf \
  --rlhf_type grpo \
  --model "${BASE_MODEL}" \
  --model_type "${MODEL_TYPE}" \
  --external_plugins "${ROOT}/train/src/agent_rl_plugin.py" \
  --reward_funcs "${REWARD_FUNC}" \
  --train_type lora \
  --lora_rank ${LORA_RANK} \
  --lora_alpha ${LORA_ALPHA} \
  --lora_dropout ${LORA_DROPOUT} \
  --torch_dtype bfloat16 \
  --dataset "${DATA_DIR}" \
  --max_length ${MAX_INPUT_LENGTH} \
  --max_completion_length ${MAX_OUTPUT_LENGTH} \
  --num_train_epochs ${NUM_EPOCHS} \
  --per_device_train_batch_size ${BATCH_SIZE} \
  --gradient_accumulation_steps ${GRAD_ACCUM} \
  --learning_rate "${LEARNING_RATE}" \
  --num_generations ${NUM_GENERATIONS} \
  --temperature ${TEMPERATURE} \
  --beta ${BETA} \
  --save_steps 100 \
  --save_only_model true \
  --save_total_limit 3 \
  --logging_steps 1 \
  --output_dir "${OUT_DIR}" \
  --warmup_ratio 0.1 \
  --gradient_checkpointing true \
  --use_vllm true \
  --vllm_gpu_memory_utilization 0.3 \
  --vllm_max_model_len 12000 \
  "$@"

echo "✅ 训练完成！模型: ${OUT_DIR}"
echo "下一步: bash bash/agent/merge_agent_lora.sh lawselect  # 合并 LoRA"
