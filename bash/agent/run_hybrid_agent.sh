#!/bin/bash
# ===========================================
# Hybrid 融合脚本（输出级融合）
#
# 融合已有的 Agent 和 MRAG 输出文件，无需重新加载模型。
# 这是 run_law_agent_pipeline.sh FUSE_WITH_MRAG=true 的便捷入口。
#
# 用法:
#   # 使用默认路径（自动查找最佳 Agent + MRAG 文件）
#   bash bash/agent/run_hybrid_agent.sh
#
#   # 指定文件
#   AGENT_FILE=path/to/agent.tsv MRAG_FILE=path/to/mrag.tsv \
#     bash bash/agent/run_hybrid_agent.sh
#
#   # 尝试所有融合策略
#   FUSION_STRATEGY=all bash bash/agent/run_hybrid_agent.sh
# ===========================================

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../paths.sh"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

OUTPUT_DIR="${PROJECT_ROOT}/mrag/retriever_output"
QRELS_FILE="${PROJECT_ROOT}/mrag/retriever_data/qrels_test.tsv"

# 融合策略: agent_first / rrf / score_merge / all
FUSION_STRATEGY=${FUSION_STRATEGY:-rrf}
MAX_LAWS=${MAX_LAWS:-20}

# ============== 自动查找文件 ==============
# Agent 文件（优先级：Both RL > LS RL > QG RL > Baseline）
if [[ -z "${AGENT_FILE:-}" ]]; then
    for f in \
        "${OUTPUT_DIR}/ablation_both_rl.tsv" \
        "${OUTPUT_DIR}/law_runfile_agent_both_rl_test.tsv" \
        "${OUTPUT_DIR}/ablation_ls_rl.tsv" \
        "${OUTPUT_DIR}/law_runfile_agent_ls_rl_test.tsv" \
        "${OUTPUT_DIR}/ablation_baseline.tsv" \
        "${OUTPUT_DIR}/law_runfile_agent_baseline_test.tsv"; do
        if [[ -f "$f" ]]; then
            AGENT_FILE="$f"
            break
        fi
    done
fi

# MRAG 文件
if [[ -z "${MRAG_FILE:-}" ]]; then
    for f in \
        "${OUTPUT_DIR}/ablation_mrag.tsv" \
        "${OUTPUT_DIR}/law_runfile_reranked_test.tsv"; do
        if [[ -f "$f" ]]; then
            MRAG_FILE="$f"
            break
        fi
    done
fi

# 检查文件
if [[ -z "${AGENT_FILE:-}" || ! -f "${AGENT_FILE}" ]]; then
    echo "❌ 未找到 Agent 输出文件"
    echo "   请先运行 Agent 管线: bash bash/agent/run_law_agent_pipeline.sh"
    echo "   或指定 AGENT_FILE 环境变量"
    exit 1
fi

if [[ -z "${MRAG_FILE:-}" || ! -f "${MRAG_FILE}" ]]; then
    echo "❌ 未找到 MRAG 输出文件"
    echo "   请先运行 eval_ablation.sh Phase 1 生成 MRAG 结果"
    echo "   或指定 MRAG_FILE 环境变量"
    exit 1
fi

echo "==========================================="
echo "  Hybrid 融合（输出级）"
echo "==========================================="
echo "  Agent 文件: ${AGENT_FILE}"
echo "  MRAG  文件: ${MRAG_FILE}"
echo "  融合策略:   ${FUSION_STRATEGY}"
echo "  最大法条数: ${MAX_LAWS}"
echo "==========================================="

# 确定输出文件名
if [[ "${FUSION_STRATEGY}" == "all" ]]; then
    OUTPUT_FILE="${OUTPUT_DIR}/fused_hybrid.tsv"
else
    OUTPUT_FILE="${OUTPUT_DIR}/fused_hybrid_${FUSION_STRATEGY}.tsv"
fi

python "${PROJECT_ROOT}/mrag/agent/fuse_results.py" \
    --agent_file "${AGENT_FILE}" \
    --mrag_file "${MRAG_FILE}" \
    --output_file "${OUTPUT_FILE}" \
    --strategy "${FUSION_STRATEGY}" \
    --max_laws ${MAX_LAWS} \
    --qrels "${QRELS_FILE}"

echo ""
echo "==========================================="
echo "✅ 完成！"
echo "  输出: ${OUTPUT_FILE}"
echo "==========================================="
