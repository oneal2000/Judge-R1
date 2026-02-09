#!/bin/bash
# 对比评测：基座模型 vs 预训练模型
# 只输出最终对比报告，自动清理所有中间文件

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../paths.sh"

# 模型路径
BASE_MODEL="${ROBERTA_MODEL_PATH}"
TRAINED_MODEL="${PROJECT_ROOT}/output/law_retriever"

# 使用临时目录存储中间文件
TEMP_DIR=$(mktemp -d)
BASE_OUTPUT_DIR="${TEMP_DIR}/base"
TRAINED_OUTPUT_DIR="${TEMP_DIR}/trained"
mkdir -p "${BASE_OUTPUT_DIR}" "${TRAINED_OUTPUT_DIR}"

# 设置清理函数（即使脚本出错也会清理）
cleanup() {
    echo "清理临时文件..."
    rm -rf "${TEMP_DIR}"
}
trap cleanup EXIT

# 数据路径
LAW_CORPUS="${PROJECT_ROOT}/data/law_corpus.jsonl"
CASE_CORPUS="${PROJECT_ROOT}/data/case_corpus.jsonl"
QUERIES_FILE="${PROJECT_ROOT}/mrag/retriever_data/queries_test.jsonl"
QRELS_FILE="${PROJECT_ROOT}/mrag/retriever_data/qrels_test.tsv"

# 检索参数
TOP_K=${TOP_K:-50}

# GPU 设置
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

# 最终输出文件
COMPARISON_FILE="${PROJECT_ROOT}/mrag/retriever_output/comparison_report.txt"
mkdir -p "$(dirname "${COMPARISON_FILE}")"

echo "=========================================="
echo "  对比评测：基座模型 vs 预训练模型"
echo "=========================================="
echo "基座模型: ${BASE_MODEL}"
echo "预训练模型: ${TRAINED_MODEL}"
echo "Top-K: ${TOP_K}"
echo "=========================================="

# ============================================
# 测试基座模型
# ============================================
echo ""
echo ">>> [1/2] 测试基座模型..."

echo "[1.1] 编码法条库..."
python "${PROJECT_ROOT}/mrag/encode_and_retrieve.py" encode \
    --model_path "${BASE_MODEL}" \
    --law_corpus "${LAW_CORPUS}" \
    --output_dir "${BASE_OUTPUT_DIR}" \
    --batch_size 64 \
    --max_length 256

echo "[1.2] 检索法条..."
python "${PROJECT_ROOT}/mrag/encode_and_retrieve.py" retrieve \
    --model_path "${BASE_MODEL}" \
    --embeddings_dir "${BASE_OUTPUT_DIR}" \
    --queries "${QUERIES_FILE}" \
    --output_file "${BASE_OUTPUT_DIR}/law_runfile_test.tsv" \
    --top_k ${TOP_K} \
    --batch_size 32

echo "[1.3] 评测检索性能..."
python "${PROJECT_ROOT}/mrag/eval_retriever.py" \
    --runfile "${BASE_OUTPUT_DIR}/law_runfile_test.tsv" \
    --qrels "${QRELS_FILE}" \
    --output "${BASE_OUTPUT_DIR}/eval_results.txt"

# ============================================
# 测试预训练模型
# ============================================
echo ""
echo ">>> [2/2] 测试预训练模型..."

echo "[2.1] 编码法条库..."
python "${PROJECT_ROOT}/mrag/encode_and_retrieve.py" encode \
    --model_path "${TRAINED_MODEL}" \
    --law_corpus "${LAW_CORPUS}" \
    --output_dir "${TRAINED_OUTPUT_DIR}" \
    --batch_size 64 \
    --max_length 256

echo "[2.2] 检索法条..."
python "${PROJECT_ROOT}/mrag/encode_and_retrieve.py" retrieve \
    --model_path "${TRAINED_MODEL}" \
    --embeddings_dir "${TRAINED_OUTPUT_DIR}" \
    --queries "${QUERIES_FILE}" \
    --output_file "${TRAINED_OUTPUT_DIR}/law_runfile_test.tsv" \
    --top_k ${TOP_K} \
    --batch_size 32

echo "[2.3] 评测检索性能..."
python "${PROJECT_ROOT}/mrag/eval_retriever.py" \
    --runfile "${TRAINED_OUTPUT_DIR}/law_runfile_test.tsv" \
    --qrels "${QRELS_FILE}" \
    --output "${TRAINED_OUTPUT_DIR}/eval_results.txt"

# ============================================
# 生成对比报告
# ============================================
echo ""
echo ">>> [3/3] 生成对比报告..."

cat > "${COMPARISON_FILE}" << EOF
==========================================
  检索模型对比报告
==========================================

测试配置:
  - Top-K: ${TOP_K}
  - 测试集: ${QUERIES_FILE}
  - 标准答案: ${QRELS_FILE}

------------------------------------------
基座模型结果 (${BASE_MODEL})
------------------------------------------
EOF

cat "${BASE_OUTPUT_DIR}/eval_results.txt" >> "${COMPARISON_FILE}"

cat >> "${COMPARISON_FILE}" << EOF

------------------------------------------
预训练模型结果 (${TRAINED_MODEL})
------------------------------------------
EOF

cat "${TRAINED_OUTPUT_DIR}/eval_results.txt" >> "${COMPARISON_FILE}"

echo ""
echo "=========================================="
echo "  对比评测完成！"
echo "=========================================="
echo ""
echo "输出文件: ${COMPARISON_FILE}"
echo ""
cat "${COMPARISON_FILE}"