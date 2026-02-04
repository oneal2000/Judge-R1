#!/usr/bin/env bash
set -euo pipefail

cd /data-share/chenxuanyi/internship/JuDGE_RL/evaluation

# ============================================================
# LegalOne-4B 评估脚本
# 
# 用法:
#   bash bash/legalone/eval.sh               # 标准模式
#   USE_MRAG=true bash bash/legalone/eval.sh # MRAG 模式
# ============================================================

USE_MRAG=${USE_MRAG:-false}
if [[ "${USE_MRAG}" == "true" ]]; then
    SUFFIX="_mrag"
else
    SUFFIX=""
fi

EXP_FILE="/data-share/chenxuanyi/internship/JuDGE_RL/data/expected.jsonl"

# LegalOne-4B 待评测的文件
GEN_FILES=(
    "../outputs/legalone_direct${SUFFIX}.jsonl"
    "../outputs/legalone_icl${SUFFIX}.jsonl"
    "../outputs/legalone_sft${SUFFIX}.jsonl"
)

echo "=========================================="
echo "  LegalOne-4B 评测"
echo "  MRAG模式: ${USE_MRAG}"
echo "  后缀: ${SUFFIX}"
echo "=========================================="

# 创建结果汇总文件
RESULT_SUMMARY="../result/eval_legalone_summary${SUFFIX}.txt"
mkdir -p ../result
echo "LegalOne-4B 评测结果汇总 - $(date)" > "${RESULT_SUMMARY}"
echo "==========================================" >> "${RESULT_SUMMARY}"

for gen_file in "${GEN_FILES[@]}"; do
    if [ ! -f "$gen_file" ]; then
        echo "[SKIP] $gen_file not found"
        continue
    fi
    
    filename=$(basename "$gen_file" .jsonl)
    
    echo ""
    echo "=========================================="
    echo "评测: $filename"
    echo "=========================================="
    
    echo "" >> "${RESULT_SUMMARY}"
    echo ">>> $filename" >> "${RESULT_SUMMARY}"
    echo "-------------------------------------------" >> "${RESULT_SUMMARY}"
    
    # 1. 法律准确性评测
    echo ">>> [1/2] 法律准确性评测 (Crime, Penalty, Law Article)..."
    echo "[法律准确性]" >> "${RESULT_SUMMARY}"
    python calc.py --gen_file "$gen_file" --exp_file "$EXP_FILE" 2>&1 | tee -a "${RESULT_SUMMARY}"
    
    # 2. 文本质量评测
    echo ">>> [2/2] 文本质量评测 (METEOR, BERTScore)..."
    echo "[文本质量]" >> "${RESULT_SUMMARY}"
    python calc_rel.py --gen_file "$gen_file" --exp_file "$EXP_FILE" 2>&1 | tee -a "${RESULT_SUMMARY}"
    
    echo "-------------------------------------------" >> "${RESULT_SUMMARY}"
done

echo ""
echo "=========================================="
echo "✅ LegalOne-4B 评测完成！"
echo "结果汇总保存至: ${RESULT_SUMMARY}"
echo "=========================================="
