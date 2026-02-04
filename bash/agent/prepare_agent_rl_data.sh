#!/bin/bash
# ===========================================
# 准备 Agent RL 训练数据
#
# 生成两个任务的训练数据：
# 1. QueryGen: 根据案件事实生成检索查询
# 2. LawSelect: 从候选法条中筛选相关法条
#
# 参数说明（v7: 只需 top-10，节省显存）：
#   - num_candidates=20: 总候选法条数（正例 ~5 + 困难负例 15）
#   - num_hard_negatives=15: 困难负例数量
#
# v7 改动：下游任务只需 10 条法条，减少候选数量以节省显存
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

# 生成数据（v7: 候选数量减少到 20，节省显存，目标输出 10 条）
# 正例约 5 个 + 困难负例 15 个 = 20 候选
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
