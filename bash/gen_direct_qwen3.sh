#!/usr/bin/env bash
set -euo pipefail

MODEL="/data-share/chenxuanyi/LLM/Qwen3-4B-Thinking-2507"
OUT="/data-share/chenxuanyi/internship/JuDGE_RL/outputs/qwen3_direct_raw.json"

cd /data-share/chenxuanyi/internship/JuDGE_RL
mkdir -p outputs

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} \
python train/deploy/inf_direct.py \
  --suffix "${MODEL}" \
  --dataset_path "data/test.json" \
  --output_path "${OUT}"
