#!/bin/bash
# 步骤3: 使用训练好的模型编码法条库和案例库

set -e

PROJECT_ROOT="/data-share/chenxuanyi/internship/JuDGE_RL"

# 模型路径（训练好的检索模型）
MODEL_PATH="${PROJECT_ROOT}/output/law_retriever"
# 法条库路径
LAW_CORPUS="${PROJECT_ROOT}/data/law_corpus.jsonl"
# 案例库路径
CASE_CORPUS="${PROJECT_ROOT}/data/case_corpus.jsonl"
# 输出目录
OUTPUT_DIR="${PROJECT_ROOT}/mrag/retriever_output"

# GPU 设置
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

echo "=========================================="
echo "  编码法条库和案例库"
echo "=========================================="
echo "模型: ${MODEL_PATH}"
echo "法条库: ${LAW_CORPUS}"
echo "案例库: ${CASE_CORPUS}"
echo "=========================================="

# 编码法条库
echo ""
echo "[1/2] 编码法条库..."
python "${PROJECT_ROOT}/mrag/encode_and_retrieve.py" encode \
    --model_path "${MODEL_PATH}" \
    --law_corpus "${LAW_CORPUS}" \
    --output_dir "${OUTPUT_DIR}" \
    --batch_size 64 \
    --max_length 512

# 编码案例库
echo ""
echo "[2/2] 编码案例库..."
python "${PROJECT_ROOT}/mrag/encode_and_retrieve.py" encode_case \
    --model_path "${MODEL_PATH}" \
    --case_corpus "${CASE_CORPUS}" \
    --output_dir "${OUTPUT_DIR}" \
    --batch_size 64 \
    --max_length 512

echo ""
echo "完成！输出文件："
echo "  法条库："
echo "    - ${OUTPUT_DIR}/law_embeddings.npy"
echo "    - ${OUTPUT_DIR}/law_ids.json"
echo "  案例库："
echo "    - ${OUTPUT_DIR}/case_embeddings.npy"
echo "    - ${OUTPUT_DIR}/case_ids.json"
echo "    - ${OUTPUT_DIR}/case_fds.json"