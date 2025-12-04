#!/usr/bin/env bash
set -euo pipefail

cd /data-share/chenxuanyi/internship/JuDGE_RL/evaluation

# 这里列出你要评估的 5 个 gen_file
GEN_FILES=(
  "../outputs/qwen3_direct.jsonl"
  "../outputs/qwen3_sft.jsonl"
  "../outputs/qwen3_rl_full.jsonl"
  "../outputs/qwen3_rl_lora.jsonl"
  "../outputs/qwen3_icl.jsonl"
)

EXP_FILE="../data/expected.jsonl"
TMPDIR_BASE="/data-share/chenxuanyi/tmp"

for gen in "${GEN_FILES[@]}"; do
  # 打个标题，方便在日志里看是哪一个
  echo "=================================================="
  echo "Evaluating: ${gen}"
  echo "=================================================="

  # 第一个 eval：calc.py
  PYTHONPATH=../evaluation \
  python calc.py \
    --gen_file "${gen}" \
    --exp_file "${EXP_FILE}"

  # 第二个 eval：calc_rel.py
  TMPDIR="${TMPDIR_BASE}" PYTHONPATH=../evaluation \
  python calc_rel.py \
    --gen_file "${gen}" \
    --exp_file "${EXP_FILE}"

  echo
done
