#!/usr/bin/env bash
set -euo pipefail

SFT_DIR="/data-share/chenxuanyi/internship/JuDGE_RL/output/sft_qwen3-4b_full"
OUT="/data-share/chenxuanyi/internship/JuDGE_RL/outputs/qwen3_sft_raw.json"

cd /data-share/chenxuanyi/internship/JuDGE_RL
mkdir -p outputs

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-3} \
python train/deploy/inf_ft.py \
  --suffix "${SFT_DIR}" \
  --dataset_path "data/test.json" \
  --output_path "${OUT}"
