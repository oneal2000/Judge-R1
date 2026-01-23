#!/usr/bin/env bash
set -euo pipefail

cd /data-share/chenxuanyi/internship/JuDGE_RL/evaluation

# MRAG模式配置
USE_MRAG=${USE_MRAG:-false}
if [[ "${USE_MRAG}" == "true" ]]; then
  SUFFIX="_mrag"
else
  SUFFIX=""
fi

EXP_FILE="/data-share/chenxuanyi/internship/JuDGE_RL/data/expected.jsonl"

# 所有待评测的文件
GEN_FILES=(
  "../outputs/qwen25_direct${SUFFIX}.jsonl"
  "../outputs/qwen25_icl${SUFFIX}.jsonl"
  "../outputs/qwen25_sft${SUFFIX}.jsonl"
  "../outputs/qwen25_rl${SUFFIX}.jsonl"
  "../outputs/qwen3_direct${SUFFIX}.jsonl"
  "../outputs/qwen3_icl${SUFFIX}.jsonl"
  "../outputs/qwen3_sft${SUFFIX}.jsonl"
  "../outputs/qwen3_rl${SUFFIX}.jsonl"
)

echo "=========================================="
echo "  评测开始"
echo "  MRAG模式: ${USE_MRAG}"
echo "  后缀: ${SUFFIX}"
echo "=========================================="

# 创建结果汇总文件
RESULT_SUMMARY="../result/eval_summary${SUFFIX}.txt"
mkdir -p ../result
echo "评测结果汇总 - $(date)" > "${RESULT_SUMMARY}"
echo "==========================================" >> "${RESULT_SUMMARY}"

for gen_file in "${GEN_FILES[@]}"; do
  if [ ! -f "$gen_file" ]; then
    echo "[SKIP] $gen_file not found"
    continue
  fi
  
  # 提取文件名用于显示
  filename=$(basename "$gen_file" .jsonl)
  
  echo ""
  echo "=========================================="
  echo "评测: $filename"
  echo "=========================================="
  
  # 写入汇总文件
  echo "" >> "${RESULT_SUMMARY}"
  echo ">>> $filename" >> "${RESULT_SUMMARY}"
  echo "-------------------------------------------" >> "${RESULT_SUMMARY}"
  
  # 1. 法律准确性评测 (calc.py)
  echo ">>> [1/2] 法律准确性评测 (Crime, Penalty, Law Article)..."
  echo "[法律准确性]" >> "${RESULT_SUMMARY}"
  python calc.py --gen_file "$gen_file" --exp_file "$EXP_FILE" 2>&1 | tee -a "${RESULT_SUMMARY}"
  
  # 2. 文本质量评测 (calc_rel.py)
  echo ">>> [2/2] 文本质量评测 (METEOR, BERTScore)..."
  echo "[文本质量]" >> "${RESULT_SUMMARY}"
  python calc_rel.py --gen_file "$gen_file" --exp_file "$EXP_FILE" 2>&1 | tee -a "${RESULT_SUMMARY}"
  
  echo "-------------------------------------------" >> "${RESULT_SUMMARY}"
done

echo ""
echo "=========================================="
echo "✅ 评测完成！"
echo "结果汇总保存至: ${RESULT_SUMMARY}"
echo "=========================================="