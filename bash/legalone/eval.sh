#!/usr/bin/env bash
set -euo pipefail

cd /data-share/chenxuanyi/internship/JuDGE_RL/evaluation

# ============================================================
# LegalOne 评估脚本（4B + 1.7B）
# ============================================================

USE_MRAG=${USE_MRAG:-false}
LEGALONE_MODEL_SET=${LEGALONE_MODEL_SET:-all}   # all | 4b | 1.7b

if [[ "${USE_MRAG}" == "true" ]]; then
    SUFFIX="_mrag"
else
    SUFFIX=""
fi

EXP_FILE="/data-share/chenxuanyi/internship/JuDGE_RL/data/expected.jsonl"

MODEL_PREFIXES=()
case "${LEGALONE_MODEL_SET}" in
    all)
        MODEL_PREFIXES=("legalone" "legalone17b")
        ;;
    4b)
        MODEL_PREFIXES=("legalone")
        ;;
    1.7b|1_7b|17b)
        MODEL_PREFIXES=("legalone17b")
        ;;
    *)
        echo "❌ 不支持的 LEGALONE_MODEL_SET=${LEGALONE_MODEL_SET} (可选: all|4b|1.7b)"
        exit 1
        ;;
esac

GEN_FILES=()
for prefix in "${MODEL_PREFIXES[@]}"; do
    GEN_FILES+=("../outputs/${prefix}_direct${SUFFIX}.jsonl")
    GEN_FILES+=("../outputs/${prefix}_icl${SUFFIX}.jsonl")
    GEN_FILES+=("../outputs/${prefix}_sft${SUFFIX}.jsonl")
done

echo "=========================================="
echo "  LegalOne 评测"
echo "  模型集合: ${LEGALONE_MODEL_SET}"
echo "  MRAG模式: ${USE_MRAG}"
echo "  后缀: ${SUFFIX}"
echo "=========================================="

RESULT_SUMMARY="../result/eval_legalone_summary${SUFFIX}.txt"
mkdir -p ../result
echo "LegalOne 评测结果汇总 - $(date)" > "${RESULT_SUMMARY}"
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

    echo ">>> [1/2] 法律准确性评测 (Crime, Penalty, Law Article)..."
    echo "[法律准确性]" >> "${RESULT_SUMMARY}"
    python calc.py --gen_file "$gen_file" --exp_file "$EXP_FILE" 2>&1 | tee -a "${RESULT_SUMMARY}"

    echo ">>> [2/2] 文本质量评测 (METEOR, BERTScore)..."
    echo "[文本质量]" >> "${RESULT_SUMMARY}"
    python calc_rel.py --gen_file "$gen_file" --exp_file "$EXP_FILE" 2>&1 | tee -a "${RESULT_SUMMARY}"

    echo "-------------------------------------------" >> "${RESULT_SUMMARY}"
done

echo ""
echo "=========================================="
echo "✅ LegalOne 评测完成！"
echo "结果汇总保存至: ${RESULT_SUMMARY}"
echo "=========================================="
