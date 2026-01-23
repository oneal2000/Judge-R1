#!/bin/bash
# 步骤4: 对测试集执行法条检索和案例检索

set -e

PROJECT_ROOT="/data-share/chenxuanyi/internship/JuDGE_RL"

# 路径配置
MODEL_PATH="${PROJECT_ROOT}/output/law_retriever"
EMBEDDINGS_DIR="${PROJECT_ROOT}/mrag/retriever_output"
QUERIES_FILE="${PROJECT_ROOT}/mrag/retriever_data/queries_test.jsonl"
LAW_OUTPUT_FILE="${PROJECT_ROOT}/mrag/retriever_output/law_runfile_test.tsv"
CASE_OUTPUT_FILE="${PROJECT_ROOT}/mrag/retriever_output/case_runfile_test.tsv"

# 检索参数
TOP_K=${TOP_K:-50}

# GPU 设置
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

echo "=========================================="
echo "执行法条检索和案例检索"
echo "=========================================="
echo "查询文件: ${QUERIES_FILE}"
echo "Top-K: ${TOP_K}"
echo "=========================================="

# 检索法条
echo ""
echo "[1/2] 检索法条..."
python "${PROJECT_ROOT}/mrag/encode_and_retrieve.py" retrieve \
    --model_path "${MODEL_PATH}" \
    --embeddings_dir "${EMBEDDINGS_DIR}" \
    --queries "${QUERIES_FILE}" \
    --output_file "${LAW_OUTPUT_FILE}" \
    --top_k ${TOP_K} \
    --batch_size 32

# 检索案例
echo ""
echo "[2/2] 检索案例..."
python "${PROJECT_ROOT}/mrag/encode_and_retrieve.py" retrieve_case \
    --model_path "${MODEL_PATH}" \
    --embeddings_dir "${EMBEDDINGS_DIR}" \
    --queries "${QUERIES_FILE}" \
    --output_file "${CASE_OUTPUT_FILE}" \
    --top_k 5 \
    --batch_size 32

echo ""
echo "完成！检索结果："
echo "  法条检索(Top-50):${LAW_OUTPUT_FILE}"
echo "  案例检索(Top-5):${CASE_OUTPUT_FILE}"