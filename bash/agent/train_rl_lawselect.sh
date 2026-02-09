#!/usr/bin/env bash
# ===========================================
# LawSelect RL 训练脚本
# 用法: CUDA_VISIBLE_DEVICES=0,1,2,3 bash bash/agent/train_rl_lawselect_4gpu.sh
#
# 训练目标：
# - 优化 Recall@5 和 Precision@5
# - 让前 5 条法条包含尽可能多的相关法条
# - 使用 7B 模型获得更强的语义理解能力
#
# 奖励函数 (law_select_reward_v2):
# - Recall@5:    0.45（最重要）
# - Precision@5: 0.35
# - Recall@10:   0.15
# - Quantity:    0.05
# ===========================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../paths.sh"
ROOT="${PROJECT_ROOT}"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_VERBOSITY=error
export PYTORCH_ALLOC_CONF=expandable_segments:True

# vLLM 环境变量
export VLLM_USE_V1=1
export VLLM_USE_FLASHINFER_SAMPLER=0

# ============== 配置 ==============
# 使用 7B 模型（更强的语义理解能力）
BASE_MODEL="${QWEN25_7B_MODEL_PATH}"
MODEL_TYPE="qwen2"
MASTER_PORT="${MASTER_PORT:-27101}"

# GPU 配置（4卡：3×80G + 1×60G）
gpu_ids="${CUDA_VISIBLE_DEVICES:-0,1,3,7}"
IFS=',' read -ra parts <<< "$gpu_ids"
gpu_num=${#parts[@]}

cd "${ROOT}"

# ============== 任务配置 ==============
DATA_DIR="${ROOT}/data/agent_rl/law_select_train.jsonl"
REWARD_FUNC="law_select_reward_v2"  # 使用优化后的奖励函数
OUT_DIR="${ROOT}/output/rl_lawselect_7b_lora"

# 训练参数（4卡优化，充分探索）
MAX_INPUT_LENGTH=10000
MAX_OUTPUT_LENGTH=1024
NUM_GENERATIONS=${NUM_GENERATIONS:-16}  # 增加探索（4卡可支持）
BATCH_SIZE=4                            # 4卡可增大
GRAD_ACCUM=8                            # 有效 batch = 4*4*8 = 128
LEARNING_RATE="1e-5"                    # 略降学习率，更稳定
BETA="0.02"                             # 降低 KL 惩罚，鼓励更多探索
TEMPERATURE="0.8"                       # 略降温度，更聚焦
NUM_EPOCHS="5"                          # 增加训练轮数

# LoRA 配置
LORA_RANK="128"
LORA_ALPHA="256"
LORA_DROPOUT="0.05"

# vLLM 配置（4卡需要调整）
VLLM_GPU_MEMORY_UTIL="0.3"              # 80G 卡可以用更多

# ============== 检查 ==============
if [[ ! -f "${DATA_DIR}" ]]; then
    echo "[ERROR] 训练数据不存在: ${DATA_DIR}"
    echo "请先运行: bash bash/agent/prepare_agent_rl_data.sh"
    exit 1
fi

if [[ ! -d "${BASE_MODEL}" ]]; then
    echo "[ERROR] 基座模型不存在: ${BASE_MODEL}"
    exit 1
fi

export HF_DATASETS_CACHE="${ROOT}/.cache"

echo "=========================================="
echo "  LawSelect RL (7B LoRA) - 4卡优化版"
echo "=========================================="
echo "  目标: 前 5 条包含尽可能多的相关法条"
echo "  奖励: Recall@5(0.45) + Precision@5(0.35) + Recall@10(0.15)"
echo "  模型: Qwen2.5-7B-Instruct"
echo "  GPU: ${gpu_num} 卡"
echo "  vLLM: 启用"
echo "  num_generations: ${NUM_GENERATIONS}"
echo "  batch_size: ${BATCH_SIZE}, grad_accum: ${GRAD_ACCUM}"
echo "  有效 batch: $((BATCH_SIZE * gpu_num * GRAD_ACCUM))"
echo "  LoRA rank: ${LORA_RANK}"
echo "  beta: ${BETA}, lr: ${LEARNING_RATE}, temp: ${TEMPERATURE}"
echo "  epochs: ${NUM_EPOCHS}"
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
  --save_steps 25 \
  --save_only_model true \
  --save_total_limit 20 \
  --logging_steps 1 \
  --output_dir "${OUT_DIR}" \
  --warmup_ratio 0.1 \
  --gradient_checkpointing true \
  --use_vllm true \
  --vllm_gpu_memory_utilization ${VLLM_GPU_MEMORY_UTIL} \
  --vllm_max_model_len 12000 \
  "$@"

echo "✅ 训练完成！模型: ${OUT_DIR}"
echo "下一步: bash bash/agent/merge_agent_lora.sh   # 合并 LoRA"
