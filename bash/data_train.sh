#!/usr/bin/env bash
set -euo pipefail

cd /data-share/chenxuanyi/internship/JuDGE_RL

# 准备 SFT 数据
python script/sft_data.py \
  --src data/train.json \
  --dst data/train_sft.json \
  --seed 42

cd /data-share/chenxuanyi/internship/JuDGE_RL
# 准备 RL 数据（可通过 MAX_SAMPLES 控制子集大小）
python script/rl_data.py \
  --src /data-share/chenxuanyi/internship/JuDGE_RL/data/train.json \
  --dst_dir /data-share/chenxuanyi/internship/JuDGE_RL/data/rl_train \
  ${MAX_SAMPLES:+--max_samples "${MAX_SAMPLES}"} \
  --seed 42

