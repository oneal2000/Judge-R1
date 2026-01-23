#!/bin/bash
# ===========================================
# 准备 Agent RL 训练数据
#
# 生成两个任务的训练数据：
# 1. QueryGen: 根据案件事实生成检索查询
# 2. LawSelect: 从候选法条中筛选相关法条
# ===========================================

set -e

ROOT="/data-share/chenxuanyi/internship/JuDGE_RL"

# 添加项目根目录到 PYTHONPATH，确保模块可以正确导入
export PYTHONPATH="${ROOT}:${PYTHONPATH}"

# 数据路径
TRAIN_DATA="${ROOT}/data/train.json"
LAW_CORPUS="${ROOT}/data/law_corpus.jsonl"
OUTPUT_DIR="${ROOT}/data/agent_rl"

echo "=========================================="
echo "  准备 Agent RL 训练数据"
echo "=========================================="
echo "  训练数据: ${TRAIN_DATA}"
echo "  法条库: ${LAW_CORPUS}"
echo "  输出目录: ${OUTPUT_DIR}"
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

# 生成数据（启用困难负例，提高 LawSelect 训练效果）
python "${ROOT}/mrag/agent/gen_agent_rl_data.py" \
    --train_data "${TRAIN_DATA}" \
    --law_corpus "${LAW_CORPUS}" \
    --output_dir "${OUTPUT_DIR}" \
    --task both \
    --num_candidates 20 \
    --num_hard_negatives 15 \
    --seed 42

echo ""
echo "=========================================="
echo "✅ 数据准备完成！"
echo "=========================================="
echo ""
echo "生成的文件:"
echo "  - ${OUTPUT_DIR}/query_gen_train.jsonl"
echo "  - ${OUTPUT_DIR}/law_select_train.jsonl"
echo ""
echo "下一步:"
echo "  1. 训练 QueryGen: bash bash/agent/train_agent_rl.sh query_gen"
echo "  2. 训练 LawSelect: bash bash/agent/train_agent_rl.sh law_select"
