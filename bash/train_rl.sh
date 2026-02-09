#!/usr/bin/env bash
set -euo pipefail

# Load centralized path configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/paths.sh"
ROOT="${PROJECT_ROOT}"

# ============== 参数化配置 ==============
# 环境变量:
#   MODEL_NAME  - 选择基座模型 (默认: qwen2)
#                 qwen3  -> Qwen3-4B-Thinking
#                 qwen2  -> Qwen2.5-3B-Instruct
#   EXPERIMENT  - 训练模式 (默认: sft_full)
#                 sft_full  -> SFT 模型上做全参 GRPO
#                 base_full -> 基座模型上做全参 GRPO
#   USE_MRAG    - 是否使用 MRAG 数据 (默认: false)
#                 sft_full  + USE_MRAG=false -> SFT模型 → RL（无MRAG）
#                 sft_full  + USE_MRAG=true  -> SFT+MRAG模型 → RL（有MRAG）
#                 base_full + USE_MRAG=false -> 基座模型 → RL（无MRAG）
#                 base_full + USE_MRAG=true  -> 基座模型 → RL（有MRAG）
#   USE_VLLM    - 是否使用外部 vLLM (默认: false)
#   MODEL       - 直接指定训练起点模型路径（覆盖自动推导）
#   OUT_DIR     - 直接指定输出目录（覆盖自动生成）
#   SFT_MODEL_DEFAULT - 直接指定 SFT 模型路径（覆盖自动推导）
#
# 每个模型有 4 种 RL 实验:
#   EXPERIMENT=sft_full                       -> SFT → RL       (sft_rl)
#   EXPERIMENT=sft_full  USE_MRAG=true        -> SFT+MRAG → RL  (sft_mrag_rl)
#   EXPERIMENT=base_full                      -> Base → RL      (rl)
#   EXPERIMENT=base_full USE_MRAG=true        -> Base → RL+MRAG (mrag_rl)
#
# 用法:
#   bash bash/train_rl.sh                                        # SFT → RL（无MRAG）
#   USE_MRAG=true bash bash/train_rl.sh                          # SFT+MRAG → RL
#   EXPERIMENT=base_full bash bash/train_rl.sh                   # Base → RL（无MRAG）
#   EXPERIMENT=base_full USE_MRAG=true bash bash/train_rl.sh     # Base → RL（有MRAG）
#   MODEL_NAME=qwen3 bash bash/train_rl.sh                       # Qwen3 SFT → RL
#   MODEL_NAME=qwen3 EXPERIMENT=base_full USE_MRAG=true bash bash/train_rl.sh
# ========================================

MODEL_NAME="${MODEL_NAME:-qwen3}"
EXPERIMENT="${EXPERIMENT:-sft_full}"
USE_MRAG=${USE_MRAG:-false}

# 根据 MODEL_NAME 设置模型路径和类型
case "${MODEL_NAME}" in
  qwen3)
    BASE_MODEL_DEFAULT="${QWEN3_MODEL_PATH}"
    MODEL_TYPE="${MODEL_TYPE:-qwen3_thinking}"
    MODEL_LABEL="qwen3-4b"
    ;;
  qwen2|qwen2.5)
    BASE_MODEL_DEFAULT="${QWEN25_MODEL_PATH}"
    MODEL_TYPE="${MODEL_TYPE:-qwen2}"
    MODEL_LABEL="qwen2.5-3b"
    ;;
  *)
    echo "[ERROR] Unknown MODEL_NAME=${MODEL_NAME}, options: qwen3 | qwen2"
    exit 1
    ;;
esac

# 根据MRAG模式选择数据目录
if [[ "${USE_MRAG}" == "true" ]]; then
  DATA_DIR="${ROOT}/data/rl_train_mrag"
  MAX_INPUT_LENGTH=6000  # MRAG需要更长输入
  MRAG_SUFFIX="_mrag"
  echo "[CONFIG] MRAG模式: 输入长度=${MAX_INPUT_LENGTH}"
else
  DATA_DIR="${ROOT}/data/rl_train"
  MAX_INPUT_LENGTH=3000  # 标准模式较短即可
  MRAG_SUFFIX=""
  echo "[CONFIG] 标准模式: 输入长度=${MAX_INPUT_LENGTH}"
fi

# SFT模型路径（自动根据 MODEL_NAME + MRAG 推导，可通过环境变量覆盖）
SFT_MODEL_DEFAULT="${SFT_MODEL_DEFAULT:-${ROOT}/output/sft_${MODEL_LABEL}_lora${MRAG_SUFFIX}/merge}"

MAX_OUTPUT_LENGTH=4096

MASTER_PORT=${MASTER_PORT:-27001}
VLLM_HOST=${VLLM_HOST:-127.0.0.1}
VLLM_PORT=${VLLM_PORT:-20420}
USE_VLLM=${USE_VLLM:-false}

cd "${ROOT}"

# 输出目录命名规则:
#   sft_full  + 无MRAG -> rl_{MODEL_LABEL}_grpo_sft_full
#   sft_full  + MRAG   -> rl_{MODEL_LABEL}_grpo_sft_mrag_full
#   base_full + 无MRAG -> rl_{MODEL_LABEL}_grpo_base_full
#   base_full + MRAG   -> rl_{MODEL_LABEL}_grpo_base_mrag_full
case "${EXPERIMENT}" in
  sft_full)
    echo "[CONFIG] SFT${MRAG_SUFFIX} 模型上做全参 GRPO"
    MODEL="${MODEL:-${SFT_MODEL_DEFAULT}}"
    TRAIN_TYPE="full"
    OUT_DIR="${OUT_DIR:-${ROOT}/output/rl_${MODEL_LABEL}_grpo_sft${MRAG_SUFFIX}_full}"
    LR_DEFAULT="5e-6"
    ;;
  base_full)
    echo "[CONFIG] 基座模型上做全参 GRPO（MRAG=${USE_MRAG}）"
    MODEL="${MODEL:-${BASE_MODEL_DEFAULT}}"
    TRAIN_TYPE="full"
    OUT_DIR="${OUT_DIR:-${ROOT}/output/rl_${MODEL_LABEL}_grpo_base${MRAG_SUFFIX}_full}"
    LR_DEFAULT="5e-6"
    ;;
  *)
    echo "[ERROR] 未知 EXPERIMENT=${EXPERIMENT}，请使用 sft_full / base_full"
    exit 1
    ;;
esac

LEARNING_RATE="${LEARNING_RATE:-${LR_DEFAULT}}"

gpu_ids=${CUDA_VISIBLE_DEVICES:-0,3,4,7}
IFS=',' read -ra parts <<< "$gpu_ids"
gpu_num=${#parts[@]}

export HF_DATASETS_CACHE="${ROOT}/.cache"

echo "=========================================="
echo "  RL 训练 (GRPO)"
echo "  模型: ${MODEL} (${MODEL_NAME})"
echo "  实验: ${EXPERIMENT}"
echo "  MRAG: ${USE_MRAG}"
echo "  数据: ${DATA_DIR}"
echo "  输入长度: ${MAX_INPUT_LENGTH}"
echo "  输出长度: ${MAX_OUTPUT_LENGTH}"
echo "  输出: ${OUT_DIR}"
echo "  GPU数量: ${gpu_num}"
echo "=========================================="

# 公共训练参数
COMMON_ARGS=(
  --rlhf_type grpo
  --model "${MODEL}"
  --model_type "${MODEL_TYPE}"
  --external_plugins "${ROOT}/train/src/rl_plugin1.py"
  --reward_funcs legal_doc_reward
  --train_type "${TRAIN_TYPE}"
  --torch_dtype bfloat16
  --dataset "${DATA_DIR}"
  --max_length ${MAX_INPUT_LENGTH}  
  --max_completion_length ${MAX_OUTPUT_LENGTH}
  --num_train_epochs 1
  --per_device_train_batch_size 1
  --gradient_accumulation_steps 16
  --learning_rate "${LEARNING_RATE}"
  --num_generations 16
  --temperature 0.8
  --save_steps 250
  --save_only_model true
  --save_total_limit 5
  --logging_steps 5
  --output_dir "${OUT_DIR}"
  --warmup_ratio 0.03
  --dataloader_num_workers 4
  --beta 0.05
  --repetition_penalty 1.1
  --report_to tensorboard
  --ddp_backend nccl
  --ddp_find_unused_parameters false
)

if [[ "${USE_VLLM}" == "true" ]]; then
  echo "[INFO] USE_VLLM=true, 使用外部 vLLM server: ${VLLM_HOST}:${VLLM_PORT}"
  COMMON_ARGS+=(
    --use_vllm true
    --vllm_mode server
    --vllm_server_host "${VLLM_HOST}"
    --vllm_server_port "${VLLM_PORT}"
  )
else
  echo "[INFO] USE_VLLM=false, 不使用外部 vLLM（进程内推理）"
  COMMON_ARGS+=(
    --use_vllm false
  )
fi

echo "[RUN] CUDA_VISIBLE_DEVICES=${gpu_ids}, NPROC_PER_NODE=${gpu_num}"
echo "[RUN] Using MASTER_PORT=${MASTER_PORT}"

CUDA_VISIBLE_DEVICES=$gpu_ids \
NPROC_PER_NODE=$gpu_num \
MASTER_PORT=$MASTER_PORT \
swift rlhf \
  "${COMMON_ARGS[@]}" \
  "$@"

echo "✅ RL 训练完成"
