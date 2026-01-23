#!/bin/bash
set -e

PROJECT_ROOT="/data-share/chenxuanyi/internship/JuDGE_RL"

# 路径配置
DENSE_RUNFILE="${PROJECT_ROOT}/mrag/retriever_output/law_runfile_test.tsv"
RERANKED_RUNFILE="${PROJECT_ROOT}/mrag/retriever_output/law_runfile_reranked_test.tsv"
QRELS_FILE="${PROJECT_ROOT}/mrag/retriever_data/qrels_test.tsv"

# 输出文件
DENSE_OUTPUT="${PROJECT_ROOT}/mrag/retriever_output/eval_dense_results.txt"
RERANKED_OUTPUT="${PROJECT_ROOT}/mrag/retriever_output/eval_reranked_results.txt"

echo "=========================================="
echo "  评测检索性能 (DenseRetriever vs Reranker)"
echo "=========================================="
echo "DenseRetriever 结果: ${DENSE_RUNFILE}"
echo "Reranker 结果: ${RERANKED_RUNFILE}"
echo "标准答案: ${QRELS_FILE}"
echo "=========================================="

# 检查文件是否存在
if [ ! -f "${DENSE_RUNFILE}" ]; then
    echo "错误: DenseRetriever 结果文件不存在: ${DENSE_RUNFILE}"
    exit 1
fi

if [ ! -f "${QRELS_FILE}" ]; then
    echo "错误: 标准答案文件不存在: ${QRELS_FILE}"
    exit 1
fi

# 1. 评测 DenseRetriever
echo ""
echo ">>> [1/2] 评测 DenseRetriever..."
python "${PROJECT_ROOT}/mrag/eval_retriever.py" \
    --runfile "${DENSE_RUNFILE}" \
    --qrels "${QRELS_FILE}" \
    --output "${DENSE_OUTPUT}"

# 2. 评测 Reranker（如果文件存在）
if [ -f "${RERANKED_RUNFILE}" ]; then
    echo ""
    echo ">>> [2/2] 评测 Reranker..."
    python "${PROJECT_ROOT}/mrag/eval_retriever.py" \
        --runfile "${RERANKED_RUNFILE}" \
        --qrels "${QRELS_FILE}" \
        --output "${RERANKED_OUTPUT}"
else
    echo ""
    echo ">>> [2/2] 警告: Reranker 结果文件不存在，跳过: ${RERANKED_RUNFILE}"
    echo "       提示: 如果已训练 Reranker，请先运行 bash/reranker/run_reranker.sh"
fi

echo ""
echo "=========================================="
echo "✅ 评测完成！"
echo "=========================================="
echo "DenseRetriever 结果: ${DENSE_OUTPUT}"
if [ -f "${RERANKED_RUNFILE}" ]; then
    echo "Reranker 结果: ${RERANKED_OUTPUT}"
fi
echo "=========================================="