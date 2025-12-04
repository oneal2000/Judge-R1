#!/usr/bin/env bash
set -euo pipefail

# MODEL="/data-share/chenxuanyi/LLM/Qwen3-4B-Thinking-2507"
# OUT_DIR="output/sft_qwen3-4b_full"
MODEL="/data-share/LLM/models--Qwen--Qwen2.5-3B-Instruct/snapshots/aa8e72537993ba99e69dfaafa59ed015b17504d1"
OUT_DIR="output/sft_qwen2.5-3b_full"
MASTER_PORT=${MASTER_PORT:-29630}
INCLUDE="localhost:4"

cd /data-share/chenxuanyi/internship/JuDGE_RL

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,7} \
deepspeed --include=${INCLUDE} --master_port ${MASTER_PORT} train/src/train.py \
  --train_path data/train_sft.json \
  --max_src 2048 \
  --max_tar 2048 \
  --workers 4 \
  --model_name_or_path "${MODEL}" \
  --tokenizer_name_or_path "${MODEL}" \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 16 \
  --num_train_epochs 3 \
  --learning_rate 5e-5 \
  --output_dir "${OUT_DIR}" \
  --deepspeed_config train/src/ds_config.json \
  --disable_lora

# 合并 ZeRO 权重
if [ -f "${OUT_DIR}/zero_to_fp32.py" ]; then
  python "${OUT_DIR}/zero_to_fp32.py" "${OUT_DIR}" "${OUT_DIR}/pytorch_model.bin"
fi
