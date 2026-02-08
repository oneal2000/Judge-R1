#!/usr/bin/env bash
set -euo pipefail

cd /data-share/chenxuanyi/internship/JuDGE_RL/evaluation

# 所有模型前缀和实验模式
PREFIXES="qwen25 qwen3"
ALL_MODES="direct icl sft mrag rl sft_mrag sft_rl mrag_rl sft_mrag_rl"

EXP_FILE="/data-share/chenxuanyi/internship/JuDGE_RL/data/expected.jsonl"

# 动态收集所有存在的 .jsonl 文件
GEN_FILES=()
for prefix in $PREFIXES; do
    for mode in $ALL_MODES; do
        f="../outputs/${prefix}_${mode}.jsonl"
        if [ -f "$f" ]; then
            GEN_FILES+=("$f")
        fi
    done
done

if [ ${#GEN_FILES[@]} -eq 0 ]; then
    echo "⚠️  没有找到任何 .jsonl 文件，请先运行 bash bash/gen.sh 和 bash bash/convert.sh"
    exit 0
fi

echo "=========================================="
echo "  评测开始"
echo "  找到 ${#GEN_FILES[@]} 个文件待评测"
echo "=========================================="

# 创建结果汇总文件
RESULT_SUMMARY="../result/eval_summary.txt"
mkdir -p ../result
echo "评测结果汇总 - $(date)" > "${RESULT_SUMMARY}"
echo "==========================================" >> "${RESULT_SUMMARY}"

for gen_file in "${GEN_FILES[@]}"; do
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
echo "共评测 ${#GEN_FILES[@]} 个文件"
echo "结果汇总保存至: ${RESULT_SUMMARY}"
echo "=========================================="
