#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# LegalOne SFT 训练脚本（支持 4B / 1.7B）
#
# 用法:
#   bash bash/legalone/train_sft.sh
#   USE_MRAG=true bash bash/legalone/train_sft.sh
#   LEGALONE_MODEL_SET=1.7b bash bash/legalone/train_sft.sh
#   LEGALONE_MODEL_SET=4b USE_MRAG=true bash bash/legalone/train_sft.sh
# ============================================================

LEGALONE_MODEL_SET=${LEGALONE_MODEL_SET:-4b}  # 4b | 1.7b
USE_MRAG=${USE_MRAG:-false}
MASTER_PORT=${MASTER_PORT:-29651}
INCLUDE=${INCLUDE:-"localhost:2"}

cd /data-share/chenxuanyi/internship/JuDGE_RL

# 选择模型
case "${LEGALONE_MODEL_SET}" in
    4b)
        MODEL="${LEGALONE_4B_MODEL_PATH:-/data-share/chenxuanyi/LLM/LegalOne-4B}"
        MODEL_TAG="legalone-4b"
        MODEL_NAME_SHOW="LegalOne-4B"
        ;;
    1.7b|1_7b|17b)
        MODEL="${LEGALONE_17B_MODEL_PATH:-/data-share/chenxuanyi/LLM/LegalOne-1.7B}"
        MODEL_TAG="legalone-1.7b"
        MODEL_NAME_SHOW="LegalOne-1.7B"
        ;;
    *)
        echo "❌ 错误: 不支持的 LEGALONE_MODEL_SET=${LEGALONE_MODEL_SET} (可选: 4b | 1.7b)"
        exit 1
        ;;
esac

# 根据 MRAG 模式选择输出目录和训练数据
if [[ "${USE_MRAG}" == "true" ]]; then
    OUT_DIR="output/sft_${MODEL_TAG}_lora_mrag"
    TRAIN_DATA="data/train_sft_mrag.json"
    MAX_SRC=6000  # MRAG 需要更长输入
    echo "[CONFIG] MRAG模式: 输入长度=${MAX_SRC}"
else
    OUT_DIR="output/sft_${MODEL_TAG}_lora"
    TRAIN_DATA="data/train_sft.json"
    MAX_SRC=3000  # 标准模式
    echo "[CONFIG] 标准模式: 输入长度=${MAX_SRC}"
fi

# 检查训练数据是否存在
if [[ ! -f "${TRAIN_DATA}" ]]; then
    echo "❌ 错误: 训练数据不存在: ${TRAIN_DATA}"
    echo "   请先运行数据生成脚本:"
    if [[ "${USE_MRAG}" == "true" ]]; then
        echo "   USE_MRAG=true bash bash/data_train.sh"
    else
        echo "   bash bash/data_train.sh"
    fi
    exit 1
fi

# 检查模型是否存在
if [[ ! -d "${MODEL}" ]]; then
    echo "❌ 错误: 模型不存在: ${MODEL}"
    echo "   请先运行: LEGALONE_MODEL_SET=${LEGALONE_MODEL_SET} bash bash/legalone/download_model.sh"
    exit 1
fi

# 输出长度统一
MAX_TAR=4096

echo "=========================================="
echo "  LegalOne SFT 训练"
echo "  模型: ${MODEL_NAME_SHOW}"
echo "  模型路径: ${MODEL}"
echo "  MRAG: ${USE_MRAG}"
echo "  数据: ${TRAIN_DATA}"
echo "  输入长度: ${MAX_SRC}"
echo "  输出长度: ${MAX_TAR}"
echo "  输出: ${OUT_DIR}"
echo "=========================================="

export NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3} \
deepspeed --include="${INCLUDE}" --master_port "${MASTER_PORT}" train/src/train.py \
    --train_path "${TRAIN_DATA}" \
    --max_src "${MAX_SRC}" \
    --max_tar "${MAX_TAR}" \
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

if [[ -f "${OUT_DIR}/zero_to_fp32.py" ]]; then
    python "${OUT_DIR}/zero_to_fp32.py" "${OUT_DIR}" "${OUT_DIR}/pytorch_model.bin"
fi

echo "✅ ${MODEL_NAME_SHOW} SFT 训练完成"
echo "输出目录: ${OUT_DIR}"
ls -lh "${OUT_DIR}"
