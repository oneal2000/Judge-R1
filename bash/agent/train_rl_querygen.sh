#!/usr/bin/env bash
# ===========================================
# QueryGen RL 训练脚本
# 用法: CUDA_VISIBLE_DEVICES=0,1,2,3 bash bash/agent/train_rl_querygen.sh
# ===========================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../paths.sh"
ROOT="${PROJECT_ROOT}"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_VERBOSITY=error
export PYTORCH_ALLOC_CONF=expandable_segments:True

# ============== 配置 ==============
BASE_MODEL="${QWEN25_7B_MODEL_PATH}"
MODEL_TYPE="qwen2"
MASTER_PORT="${MASTER_PORT:-27106}"

# GPU 配置（需要 4 卡复现 v3）
gpu_ids="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
IFS=',' read -ra parts <<< "$gpu_ids"
gpu_num=${#parts[@]}

cd "${ROOT}"

# ============== 任务配置 ==============
DATA_DIR="${ROOT}/data/agent_rl/query_gen_train.jsonl"
REWARD_FUNC="query_gen_reward"
OUT_DIR="${ROOT}/output/rl_querygen_7b_lora"

# 训练参数（与 v3 一致，需要 4 卡）
MAX_INPUT_LENGTH=3000
MAX_OUTPUT_LENGTH=1024
NUM_GENERATIONS=${NUM_GENERATIONS:-16}
BATCH_SIZE=1
GRAD_ACCUM=32
LEARNING_RATE="2e-5"
BETA="0.1"
TEMPERATURE="1.0"
NUM_EPOCHS="5"

# LoRA 配置（与 v3 一致）
LORA_RANK="64"
LORA_ALPHA="128"
LORA_DROPOUT="0.05"

# ============== 检查 ==============
if [[ ! -f "${DATA_DIR}" ]]; then
    echo "[ERROR] 训练数据不存在: ${DATA_DIR}"
    echo "请先运行: bash bash/agent/prepare_agent_rl_data.sh"
    exit 1
fi

export HF_DATASETS_CACHE="${ROOT}/.cache"

echo "=========================================="
echo "  QueryGen RL (7B LoRA)"
echo "=========================================="
echo "  模型: ${BASE_MODEL}"
echo "  GPU: ${gpu_num} 卡"
echo "  num_generations: ${NUM_GENERATIONS}"
echo "  beta: ${BETA}, lr: ${LEARNING_RATE}"
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
  --save_steps 200 \
  --save_only_model true \
  --save_total_limit 5 \
  --logging_steps 1 \
  --output_dir "${OUT_DIR}" \
  --warmup_ratio 0.1 \
  --gradient_checkpointing true \
  --use_vllm false \
  "$@"

echo "✅ 训练完成！模型: ${OUT_DIR}"
echo "下一步: bash bash/agent/merge_agent_lora.sh  # 合并 LoRA"
