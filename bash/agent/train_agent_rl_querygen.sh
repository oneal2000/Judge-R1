#!/usr/bin/env bash
# ===========================================
# Agent RL 训练脚本 - QueryGen 任务
# 
# 使用 GRPO 强化学习提升 Qwen2.5-3B-Instruct 
# 根据案件事实生成检索查询的能力
#
# 用法:
#   bash bash/agent/train_agent_rl_querygen.sh
# ===========================================

set -euo pipefail

ROOT="/data-share/chenxuanyi/internship/JuDGE_RL"

# 添加项目根目录到 PYTHONPATH
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"

# 抑制 tokenizer 警告
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_VERBOSITY=error
export PYTHONWARNINGS="ignore::UserWarning"

# ============== 配置 ==============
# 基础模型: Qwen2.5-3B-Instruct
BASE_MODEL="${BASE_MODEL:-/data-share/LLM/models--Qwen--Qwen2.5-3B-Instruct/snapshots/aa8e72537993ba99e69dfaafa59ed015b17504d1}"
MODEL_TYPE="${MODEL_TYPE:-qwen2}"

# 训练类型: full 或 lora（3B 模型推荐 full，效果更好）
TRAIN_TYPE="${TRAIN_TYPE:-full}"

# 端口配置
MASTER_PORT="${MASTER_PORT:-27100}"

# GPU 配置
gpu_ids="${CUDA_VISIBLE_DEVICES:-0,1,4,6}"
IFS=',' read -ra parts <<< "$gpu_ids"
gpu_num=${#parts[@]}

cd "${ROOT}"

# ============== QueryGen 任务配置 ==============
echo "[CONFIG] 任务: QueryGen (生成检索查询)"
DATA_DIR="${ROOT}/data/agent_rl/query_gen_train.jsonl"
REWARD_FUNC="query_gen_reward"
OUT_DIR="${ROOT}/output/agent_rl_querygen_3b_v3"
MAX_INPUT_LENGTH=3000
MAX_OUTPUT_LENGTH=512
NUM_GENERATIONS=16      # 增加采样数量，提高多样性（更多样本 = 更好的梯度估计）
BATCH_SIZE=1            # 全参数训练显存大，保持 batch=1
GRAD_ACCUM=32           # 增加梯度累积，更稳定的更新

# 根据训练类型调整输出目录和学习率
if [[ "${TRAIN_TYPE}" == "lora" ]]; then
    OUT_DIR="${OUT_DIR}_lora"
    LR_DEFAULT="2e-5"  # LoRA 可以用更高的学习率
else
    LR_DEFAULT="5e-6"  # 全参数需要更小的学习率
fi

LEARNING_RATE="${LEARNING_RATE:-${LR_DEFAULT}}"

# ============== 检查数据文件 ==============
if [[ ! -f "${DATA_DIR}" ]]; then
    echo "[ERROR] 训练数据不存在: ${DATA_DIR}"
    echo ""
    echo "请先生成训练数据:"
    echo "  bash bash/agent/prepare_agent_rl_data.sh"
    exit 1
fi

export HF_DATASETS_CACHE="${ROOT}/.cache"

echo "=========================================="
echo "  Agent RL 训练 (GRPO) - QueryGen"
echo "=========================================="
echo "  任务: QueryGen (生成检索查询)"
echo "  模型: ${BASE_MODEL}"
echo "  数据: ${DATA_DIR}"
echo "  奖励函数: ${REWARD_FUNC}"
echo "  训练类型: ${TRAIN_TYPE}"
echo "  输入长度: ${MAX_INPUT_LENGTH}"
echo "  输出长度: ${MAX_OUTPUT_LENGTH}"
echo "  学习率: ${LEARNING_RATE}"
echo "  输出: ${OUT_DIR}"
echo "  GPU数量: ${gpu_num}"
echo "=========================================="

# ============== 训练参数 ==============
# 关键改进 v3：
# - beta = 0.15，更强的 KL 惩罚防止模型偏离基座太远
# - num_generations = 16，更多采样 = 更好的梯度估计
# - num_train_epochs = 3，更充分的训练
# - temperature = 1.0，最大化采样多样性
# - warmup_ratio = 0.1，更长的预热
# - 新增 gradient_checkpointing 节省显存
TRAIN_ARGS=(
  --rlhf_type grpo
  --model "${BASE_MODEL}"
  --model_type "${MODEL_TYPE}"
  --external_plugins "${ROOT}/train/src/agent_rl_plugin.py"
  --reward_funcs "${REWARD_FUNC}"
  --train_type "${TRAIN_TYPE}"
  --torch_dtype bfloat16
  --dataset "${DATA_DIR}"
  --max_length ${MAX_INPUT_LENGTH}
  --max_completion_length ${MAX_OUTPUT_LENGTH}
  --num_train_epochs 3
  --per_device_train_batch_size ${BATCH_SIZE}
  --gradient_accumulation_steps ${GRAD_ACCUM}
  --learning_rate "${LEARNING_RATE}"
  --num_generations ${NUM_GENERATIONS}
  --temperature 1.0
  --save_steps 200
  --save_only_model true
  --save_total_limit 5
  --logging_steps 1
  --output_dir "${OUT_DIR}"
  --warmup_ratio 0.10
  --dataloader_num_workers 2
  --beta 0.15
  --repetition_penalty 1.0
  --report_to tensorboard
  --ddp_backend nccl
  --ddp_find_unused_parameters false
  --gradient_checkpointing true
  --use_vllm false
)

# LoRA 配置
if [[ "${TRAIN_TYPE}" == "lora" ]]; then
  echo "[CONFIG] 使用 LoRA 训练（rank=64, alpha=128）"
  TRAIN_ARGS+=(
    --lora_rank 64
    --lora_alpha 128
    --lora_dropout 0.05
  )
fi

echo "[RUN] CUDA_VISIBLE_DEVICES=${gpu_ids}, NPROC_PER_NODE=${gpu_num}"
echo "[RUN] MASTER_PORT=${MASTER_PORT}"

CUDA_VISIBLE_DEVICES=$gpu_ids \
NPROC_PER_NODE=$gpu_num \
MASTER_PORT=$MASTER_PORT \
swift rlhf \
  "${TRAIN_ARGS[@]}" \
  "$@"

echo ""
echo "=========================================="
echo "✅ QueryGen RL 训练完成！"
echo "=========================================="
echo "模型保存到: ${OUT_DIR}"
echo ""
echo "下一步:"
if [[ "${TRAIN_TYPE}" == "lora" ]]; then
    echo "  1. 合并 LoRA: bash bash/loramerge.sh ${OUT_DIR}"
    echo "  2. 测试效果: bash bash/agent/run_law_agent_pipeline.sh"
else
    echo "  1. 测试效果: bash bash/agent/run_law_agent_pipeline.sh"
fi
echo ""
echo "提示: run_law_agent_pipeline.sh 会自动查找最新的 RL checkpoint"
