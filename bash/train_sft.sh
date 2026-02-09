#!/usr/bin/env bash
set -euo pipefail

# ============== 参数化配置 ==============
# 环境变量:
#   MODEL_NAME  - 选择基座模型 (默认: qwen2)
#                 qwen3  -> Qwen3-4B-Thinking
#                 qwen2  -> Qwen2.5-3B-Instruct
#   USE_MRAG    - 是否使用 MRAG 数据 (默认: false)
#   MODEL       - 直接指定模型路径（覆盖 MODEL_NAME 的默认路径）
#   OUT_DIR     - 直接指定输出目录（覆盖自动生成的路径）
#   MASTER_PORT - DeepSpeed master port (默认: 29650)
#
# 用法:
#   bash bash/train_sft.sh                                  # Qwen2.5, 无MRAG
#   MODEL_NAME=qwen3 bash bash/train_sft.sh                 # Qwen3, 无MRAG
#   MODEL_NAME=qwen3 USE_MRAG=true bash bash/train_sft.sh   # Qwen3, MRAG
#   MODEL_NAME=qwen2 USE_MRAG=true bash bash/train_sft.sh   # Qwen2.5, MRAG
# ========================================

MODEL_NAME="${MODEL_NAME:-qwen2}"
USE_MRAG=${USE_MRAG:-false}
MASTER_PORT=${MASTER_PORT:-29650}
VISIBLE_GPUS="${CUDA_VISIBLE_DEVICES:-0,3,4,7}"
INCLUDE="${INCLUDE:-localhost:${VISIBLE_GPUS}}"

# Load centralized path configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/paths.sh"
cd "${PROJECT_ROOT}"

# 根据 MODEL_NAME 设置模型路径和标签
case "${MODEL_NAME}" in
  qwen3)
    MODEL="${MODEL:-${QWEN3_MODEL_PATH}}"
    MODEL_LABEL="qwen3-4b"
    ;;
  qwen2|qwen2.5)
    MODEL="${MODEL:-${QWEN25_MODEL_PATH}}"
    MODEL_LABEL="qwen2.5-3b"
    ;;
  *)
    echo "[ERROR] Unknown MODEL_NAME=${MODEL_NAME}, options: qwen3 | qwen2"
    exit 1
    ;;
esac

validate_path "MODEL" "${MODEL}"

# 根据 MRAG 模式选择训练数据
if [[ "${USE_MRAG}" == "true" ]]; then
  TRAIN_DATA="data/train_sft_mrag.json"
  MAX_SRC=6000  # MRAG需要更长输入
  MRAG_SUFFIX="_mrag"
  echo "[CONFIG] MRAG模式: 输入长度=${MAX_SRC}"
else
  TRAIN_DATA="data/train_sft.json"
  MAX_SRC=3000  # 标准模式较短即可
  MRAG_SUFFIX=""
  echo "[CONFIG] 标准模式: 输入长度=${MAX_SRC}"
fi

# 自动生成输出目录（可通过 OUT_DIR 环境变量覆盖）
OUT_DIR="${OUT_DIR:-output/sft_${MODEL_LABEL}_lora${MRAG_SUFFIX}}"

# 输出长度统一
MAX_TAR=4096

echo "=========================================="
echo "  SFT 训练"
echo "  模型: ${MODEL} (${MODEL_NAME})"
echo "  MRAG: ${USE_MRAG}"
echo "  数据: ${TRAIN_DATA}"
echo "  输入长度: ${MAX_SRC}"
echo "  输出长度: ${MAX_TAR}"
echo "  输出: ${OUT_DIR}"
echo "=========================================="

export NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

CUDA_VISIBLE_DEVICES=${VISIBLE_GPUS} \
deepspeed --include=${INCLUDE} --master_port ${MASTER_PORT} train/src/train.py \
  --train_path "${TRAIN_DATA}" \
  --max_src ${MAX_SRC} \
  --max_tar ${MAX_TAR} \
  --workers 4 \
  --model_name_or_path "${MODEL}" \
  --tokenizer_name_or_path "${MODEL}" \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 16 \
  --num_train_epochs 5 \
  --learning_rate 2e-5 \
  --output_dir "${OUT_DIR}" \
  --deepspeed_config train/src/ds_config.json

sleep 3

if [ -f "${OUT_DIR}/zero_to_fp32.py" ]; then
  python "${OUT_DIR}/zero_to_fp32.py" "${OUT_DIR}" "${OUT_DIR}/pytorch_model.bin"
fi

echo "✅ SFT 训练完成"
echo "输出目录: ${OUT_DIR}"
ls -lh "${OUT_DIR}"
