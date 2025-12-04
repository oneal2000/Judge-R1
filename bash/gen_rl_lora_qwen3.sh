#!/usr/bin/env bash
set -euo pipefail

# 推理 LoRA RL（不合并），base + adapter
BASE="${BASE:-/data-share/chenxuanyi/internship/JuDGE_RL/output/sft_qwen3-4b_full}"
ADAPTER="${ADAPTER:-/data-share/chenxuanyi/internship/JuDGE_RL/output/rl_qwen3-4b_grpo_sft_lora/v5-20251203-031643/checkpoint-250}"
DATA="${DATA:-/data-share/chenxuanyi/internship/JuDGE_RL/data/test.json}"
OUT="${OUT:-/data-share/chenxuanyi/internship/JuDGE_RL/outputs/qwen3_rl_lora_raw.json}"
cd /data-share/chenxuanyi/internship/JuDGE_RL
mkdir -p outputs

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} \
python train/deploy/inf_rl_lora.py \
  --base_model "${BASE}" \
  --adapter "${ADAPTER}" \
  --dataset_path "${DATA}" \
  --output_path "${OUT}"

echo "gen_rl_lora done, output -> ${OUT}"
