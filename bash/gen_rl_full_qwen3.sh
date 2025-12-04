#!/usr/bin/env bash
set -euo pipefail

# 推理全参 RL 模型（已合并权重）
MODEL="${MODEL:-/data-share/chenxuanyi/internship/JuDGE_RL/output/rl_qwen3-4b_grpo/merged}"
OUT="${OUT:-/data-share/chenxuanyi/internship/JuDGE_RL/outputs/qwen3_rl_full_raw.json}"
DATA="${DATA:-/data-share/chenxuanyi/internship/JuDGE_RL/data/test.json}"

cd /data-share/chenxuanyi/internship/JuDGE_RL
mkdir -p outputs

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} \
python train/deploy/inf_direct.py \
  --suffix "${MODEL}" \
  --dataset_path "${DATA}" \
  --output_path "${OUT}"

echo "gen_rl_full done, output -> ${OUT}"
