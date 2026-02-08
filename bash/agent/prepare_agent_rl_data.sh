#!/bin/bash
# ===========================================
# 准备 Agent RL 训练数据
#
# 生成两个任务的训练数据：
# 1. QueryGen: 根据案件事实生成检索查询
# 2. LawSelect: 从候选法条中筛选相关法条

set -e

ROOT="/data-share/chenxuanyi/internship/JuDGE_RL"

# 添加项目根目录到 PYTHONPATH，确保模块可以正确导入
export PYTHONPATH="${ROOT}:${PYTHONPATH}"

# 数据路径
TRAIN_DATA="${ROOT}/data/train.json"
LAW_CORPUS="${ROOT}/data/law_corpus.jsonl"
OUTPUT_DIR="${ROOT}/data/agent_rl"

# 检索模型路径（用于生成与推理一致的候选）
DENSE_MODEL="${ROOT}/output/law_retriever"
RERANKER_MODEL="${ROOT}/reranker/train"

echo "=========================================="
echo "  准备 Agent RL 训练数据"
echo "=========================================="
echo "  训练数据: ${TRAIN_DATA}"
echo "  法条库: ${LAW_CORPUS}"
echo "  输出目录: ${OUTPUT_DIR}"
echo "  Dense Model: ${DENSE_MODEL}"
echo "  Reranker Model: ${RERANKER_MODEL}"
echo "=========================================="

# 检查输入文件
if [[ ! -f "${TRAIN_DATA}" ]]; then
    echo "[ERROR] 训练数据不存在: ${TRAIN_DATA}"
    exit 1
fi

if [[ ! -f "${LAW_CORPUS}" ]]; then
    echo "[ERROR] 法条库不存在: ${LAW_CORPUS}"
    exit 1
fi

# 生成数据
# LawSelect: 使用 Dense + Reranker 生成候选（与推理时完全一致）
# - Dense top-50 → Reranker top-20 → 强制加入遗漏正例 → 打乱
# - 候选分布与推理时 LawSelect 看到的完全一致
#
# 如果 Dense/Reranker 模型不存在，自动回退到 n-gram 困难负例
EXTRA_ARGS=""
if [[ -d "${DENSE_MODEL}" && -d "${RERANKER_MODEL}" ]]; then
    echo "[INFO] 使用 Dense + Reranker 生成候选（推荐，与推理一致）"
    EXTRA_ARGS="--dense_model ${DENSE_MODEL} --reranker_model ${RERANKER_MODEL}"
else
    echo "[WARN] Dense/Reranker 模型不存在，回退到 n-gram 困难负例"
fi

python "${ROOT}/mrag/agent/gen_agent_rl_data.py" \
    --train_data "${TRAIN_DATA}" \
    --law_corpus "${LAW_CORPUS}" \
    --output_dir "${OUTPUT_DIR}" \
    --task both \
    --num_candidates 20 \
    --seed 42 \
    ${EXTRA_ARGS}

echo ""
echo "=========================================="
echo "✅ 数据准备完成！"
echo "=========================================="
echo ""
echo "生成的文件:"
echo "  - ${OUTPUT_DIR}/query_gen_train.jsonl"
echo "  - ${OUTPUT_DIR}/law_select_train.jsonl"
echo ""
