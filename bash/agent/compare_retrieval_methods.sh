#!/bin/bash
# ===========================================
# 对比多种法条检索方法的效果
#
# 包括：
# 1. MRAG (Dense only): fact → Dense
# 2. MRAG (Dense + Reranker): fact → Dense → Reranker
# 3. Agent Baseline: fact → QueryGen → Dense → Reranker → LLM-Select
# 4. Agent (Both RL): 使用 RL 训练后的 QueryGen 和 LawSelect
# 5. Hybrid Fusion: MRAG + Agent → RRF 融合 → Reranker → LLM-Select
#
# 用法:
#   bash bash/agent/compare_retrieval_methods.sh
# ===========================================

set -e

PROJECT_ROOT="/data-share/chenxuanyi/internship/JuDGE_RL"
OUTPUT_DIR="${PROJECT_ROOT}/mrag/retriever_output"
QRELS_FILE="${PROJECT_ROOT}/mrag/retriever_data/qrels_test.tsv"

echo "==========================================="
echo "  法条检索方法对比"
echo "==========================================="

# 检查各种方法的结果文件是否存在并评估
check_and_eval() {
    local name=$1
    local runfile=$2
    local output=$3
    
    if [[ -f "${runfile}" ]]; then
        echo ""
        echo ">>> 评估 ${name}..."
        python "${PROJECT_ROOT}/mrag/eval_retriever.py" \
            --runfile "${runfile}" \
            --qrels "${QRELS_FILE}" \
            --output "${output}"
    else
        echo ""
        echo ">>> [跳过] ${name}: 结果文件不存在"
        echo "   ${runfile}"
    fi
}

# 评估各种方法
check_and_eval "MRAG (Dense only)" \
    "${OUTPUT_DIR}/law_runfile_test.tsv" \
    "${OUTPUT_DIR}/eval_mrag_dense.txt"

check_and_eval "MRAG (Dense + Reranker)" \
    "${OUTPUT_DIR}/law_runfile_reranked_test.tsv" \
    "${OUTPUT_DIR}/eval_mrag_reranked.txt"

# Agent Baseline: 检查新旧两种命名
if [[ -f "${OUTPUT_DIR}/law_runfile_agent_baseline_test.tsv" ]]; then
    check_and_eval "Agent Baseline (无 RL)" \
        "${OUTPUT_DIR}/law_runfile_agent_baseline_test.tsv" \
        "${OUTPUT_DIR}/eval_agent_baseline.txt"
elif [[ -f "${OUTPUT_DIR}/law_runfile_agent_test.tsv" ]]; then
    check_and_eval "Agent Baseline (无 RL) [旧命名]" \
        "${OUTPUT_DIR}/law_runfile_agent_test.tsv" \
        "${OUTPUT_DIR}/eval_agent_baseline.txt"
fi

check_and_eval "Agent (QueryGen RL)" \
    "${OUTPUT_DIR}/law_runfile_agent_qg_rl_test.tsv" \
    "${OUTPUT_DIR}/eval_agent_qg_rl.txt"

check_and_eval "Agent (LawSelect RL)" \
    "${OUTPUT_DIR}/law_runfile_agent_ls_rl_test.tsv" \
    "${OUTPUT_DIR}/eval_agent_ls_rl.txt"

check_and_eval "Agent (Both RL)" \
    "${OUTPUT_DIR}/law_runfile_agent_both_rl_test.tsv" \
    "${OUTPUT_DIR}/eval_agent_both_rl.txt"

check_and_eval "Hybrid Fusion" \
    "${OUTPUT_DIR}/law_runfile_hybrid_test.tsv" \
    "${OUTPUT_DIR}/eval_hybrid.txt"

# 生成汇总报告
echo ""
echo "==========================================="
echo "  生成汇总报告"
echo "==========================================="

SUMMARY_FILE="${OUTPUT_DIR}/comparison_summary.txt"
echo "法条检索方法对比报告 - $(date)" > "${SUMMARY_FILE}"
echo "==========================================" >> "${SUMMARY_FILE}"
echo "" >> "${SUMMARY_FILE}"
echo "方法说明:" >> "${SUMMARY_FILE}"
echo "  1. MRAG (Dense only): fact → Dense" >> "${SUMMARY_FILE}"
echo "  2. MRAG (Dense + Reranker): fact → Dense → Reranker" >> "${SUMMARY_FILE}"
echo "  3. Agent Baseline: fact → QueryGen → Dense → Reranker → LLM-Select" >> "${SUMMARY_FILE}"
echo "  4. Agent (Both RL): QueryGen RL + LawSelect RL" >> "${SUMMARY_FILE}"
echo "  5. Hybrid Fusion: MRAG + Agent → RRF 融合 → Reranker → LLM-Select" >> "${SUMMARY_FILE}"
echo "" >> "${SUMMARY_FILE}"

for eval_file in \
    "${OUTPUT_DIR}/eval_mrag_dense.txt" \
    "${OUTPUT_DIR}/eval_mrag_reranked.txt" \
    "${OUTPUT_DIR}/eval_agent_baseline.txt" \
    "${OUTPUT_DIR}/eval_agent_qg_rl.txt" \
    "${OUTPUT_DIR}/eval_agent_ls_rl.txt" \
    "${OUTPUT_DIR}/eval_agent_both_rl.txt" \
    "${OUTPUT_DIR}/eval_hybrid.txt"
do
    if [[ -f "${eval_file}" ]]; then
        filename=$(basename "${eval_file}" .txt)
        echo "-------------------------------------------" >> "${SUMMARY_FILE}"
        echo ">>> ${filename}" >> "${SUMMARY_FILE}"
        cat "${eval_file}" >> "${SUMMARY_FILE}"
    fi
done

echo "" >> "${SUMMARY_FILE}"
echo "==========================================" >> "${SUMMARY_FILE}"

echo ""
echo ">>> 汇总报告已保存到: ${SUMMARY_FILE}"
echo ""
cat "${SUMMARY_FILE}"

echo ""
echo "==========================================="
echo "✅ 对比完成！"
echo "==========================================="
