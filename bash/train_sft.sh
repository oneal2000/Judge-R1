#!/usr/bin/env bash
set -euo pipefail

# MODEL="/data-share/chenxuanyi/LLM/Qwen3-4B-Thinking-2507"
# OUT_DIR="output/sft_qwen3-4b_lora"
MODEL="/data-share/LLM/models--Qwen--Qwen2.5-3B-Instruct/snapshots/aa8e72537993ba99e69dfaafa59ed015b17504d1"
OUT_DIR="output/sft_qwen2.5-3b_lora_mrag"

USE_MRAG=${USE_MRAG:-false}
MASTER_PORT=${MASTER_PORT:-29650}
INCLUDE="localhost:0,1,2,3"
cd /data-share/chenxuanyi/internship/JuDGE_RL

# 根据模型和MRAG模式选择训练数据

if [[ "${USE_MRAG}" == "true" ]]; then
  TRAIN_DATA="data/train_sft_mrag.json"
  MAX_SRC=6000  # MRAG需要更长输入
  echo "[CONFIG] MRAG模式: 输入长度=${MAX_SRC}"
else
  TRAIN_DATA="data/train_sft.json"
  MAX_SRC=3000  # 标准模式较短即可
  echo "[CONFIG] 标准模式: 输入长度=${MAX_SRC}"
fi

# 输出长度统一
MAX_TAR=4096

echo "=========================================="
echo "  SFT 训练"
echo "  模型: ${MODEL}"
echo "  MRAG: ${USE_MRAG}"
echo "  数据: ${TRAIN_DATA}"
echo "  输入长度: ${MAX_SRC}"
echo "  输出长度: ${MAX_TAR}"
echo "  输出: ${OUT_DIR}"
echo "=========================================="

export NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3} \
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