#!/bin/bash
# ===========================================
# Hybrid Fusion Agent Pipeline
# 
# 融合 MRAG 和 Agent 两条检索路线：
# 1. MRAG 路线: fact → Dense → top-50
# 2. Agent 路线: fact → QueryGen → Dense → top-50
# 3. RRF 融合（来源信息仅用于统计，不传递给 LLM）
# 4. Reranker 统一重排 → top-25
# 5. LLM Select（使用与训练一致的统一提示词）
#
# 优势：
# - Reranker 看到更全面的候选集
# - 双重命中提供可靠信号（内部统计）
# - 训练推理一致的提示词格式
# - 不受单一路线上限限制
#
# 用法:
#   CUDA_VISIBLE_DEVICES=0 bash bash/agent/run_hybrid_agent.sh
# ===========================================

set -e

PROJECT_ROOT="/data-share/chenxuanyi/internship/JuDGE_RL"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"

# ============== 配置 ==============
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-7}
export CUDA_VISIBLE_DEVICES
GPU_MEMORY_UTIL=${GPU_MEMORY_UTIL:-0.45}

# LLM 模型 (QueryGen + LawSelect)
# 基座模型
BASE_MODEL="/data-share/LLM/models--Qwen--Qwen2.5-3B-Instruct/snapshots/aa8e72537993ba99e69dfaafa59ed015b17504d1"

# RL 模型路径（可选）
QUERYGEN_RL_DIR="${PROJECT_ROOT}/output/agent_rl_querygen_3b_v3"
LAWSELECT_RL_DIR="${PROJECT_ROOT}/output/agent_rl_lawselect_3b_v3"

# 自动查找最新的 checkpoint
find_latest_checkpoint() {
    local dir=$1
    if [[ ! -d "$dir" ]]; then
        echo ""
        return
    fi
    local latest_version=$(ls -d "$dir"/v* 2>/dev/null | sort -V | tail -1)
    if [[ -z "$latest_version" ]]; then
        echo ""
        return
    fi
    local latest_ckpt=$(ls -d "$latest_version"/checkpoint-* 2>/dev/null | sort -t- -k2 -n | tail -1)
    echo "$latest_ckpt"
}

QUERYGEN_MODEL=$(find_latest_checkpoint "$QUERYGEN_RL_DIR")
LAWSELECT_MODEL=$(find_latest_checkpoint "$LAWSELECT_RL_DIR")

# 如果没有 RL 模型，使用基座模型
if [[ -z "$QUERYGEN_MODEL" ]]; then
    QUERYGEN_MODEL="$BASE_MODEL"
    echo "使用基座模型作为 QueryGen"
fi
if [[ -z "$LAWSELECT_MODEL" ]]; then
    LAWSELECT_MODEL="$BASE_MODEL"
    echo "使用基座模型作为 LawSelect"
fi

# 检索模型
DENSE_MODEL="${PROJECT_ROOT}/output/law_retriever"
RERANKER_MODEL="${PROJECT_ROOT}/reranker/train"

# 数据路径
LAW_CORPUS="${PROJECT_ROOT}/data/law_corpus.jsonl"
TEST_DATA="${PROJECT_ROOT}/data/test.json"
OUTPUT_DIR="${PROJECT_ROOT}/mrag/retriever_output"

# 检索参数
DENSE_TOP_K=50    # 每条路线 Dense 检索 top-K
FUSION_TOP_K=80   # RRF 融合后取 top-K
RERANK_TOP_K=25   # Reranker 重排后取 top-K
MIN_SELECTED=5    # LLM 最少选择的法条数
RRF_K=60          # RRF 参数
BATCH_SIZE=4
TENSOR_PARALLEL_SIZE=1

echo "==========================================="
echo "  Hybrid Fusion Agent Pipeline"
echo "==========================================="
echo "  QueryGen 模型: ${QUERYGEN_MODEL}"
echo "  LawSelect 模型: ${LAWSELECT_MODEL}"
echo "  Dense Model: ${DENSE_MODEL}"
echo "  Reranker Model: ${RERANKER_MODEL}"
echo "==========================================="
echo "  Dense top-k: ${DENSE_TOP_K} (per path)"
echo "  Fusion top-k: ${FUSION_TOP_K}"
echo "  Rerank top-k: ${RERANK_TOP_K}"
echo "  RRF k: ${RRF_K}"
echo "==========================================="

mkdir -p "${OUTPUT_DIR}"

# 检查模型
if [ ! -d "${DENSE_MODEL}" ]; then
    echo "错误: Dense Model 路径不存在: ${DENSE_MODEL}"
    exit 1
fi

if [ ! -d "${RERANKER_MODEL}" ]; then
    echo "错误: Reranker Model 路径不存在: ${RERANKER_MODEL}"
    exit 1
fi

# 运行 Hybrid Agent
python "${PROJECT_ROOT}/mrag/agent/hybrid_agent.py" \
    --llm_model "${BASE_MODEL}" \
    --law_corpus "${LAW_CORPUS}" \
    --dense_model "${DENSE_MODEL}" \
    --reranker_model "${RERANKER_MODEL}" \
    --input_file "${TEST_DATA}" \
    --output_file "${OUTPUT_DIR}/law_runfile_hybrid_test.tsv" \
    --querygen_model "${QUERYGEN_MODEL}" \
    --lawselect_model "${LAWSELECT_MODEL}" \
    --dense_top_k ${DENSE_TOP_K} \
    --fusion_top_k ${FUSION_TOP_K} \
    --rerank_top_k ${RERANK_TOP_K} \
    --min_selected ${MIN_SELECTED} \
    --rrf_k ${RRF_K} \
    --batch_size ${BATCH_SIZE} \
    --device cuda \
    --use_vllm \
    --tensor_parallel_size ${TENSOR_PARALLEL_SIZE} \
    --gpu_memory_util ${GPU_MEMORY_UTIL} \
    --save_details

echo ""
echo "==========================================="
echo "  Hybrid Agent 完成！"
echo "==========================================="

# 评估
echo ""
echo ">>> 评估检索效果..."
if [ -f "${PROJECT_ROOT}/mrag/eval_retriever.py" ]; then
    python "${PROJECT_ROOT}/mrag/eval_retriever.py" \
        --runfile "${OUTPUT_DIR}/law_runfile_hybrid_test.tsv" \
        --qrels "${PROJECT_ROOT}/mrag/retriever_data/qrels_test.tsv" \
        --output "${OUTPUT_DIR}/eval_hybrid_results.txt"
    
    echo ""
    echo ">>> Hybrid Agent 评估结果:"
    cat "${OUTPUT_DIR}/eval_hybrid_results.txt"
fi

echo ""
echo "==========================================="
echo "输出文件:"
echo "  - 检索结果: ${OUTPUT_DIR}/law_runfile_hybrid_test.tsv"
echo "  - 详细输出: ${OUTPUT_DIR}/law_runfile_hybrid_test_details.json"
echo "  - 评估结果: ${OUTPUT_DIR}/eval_hybrid_results.txt"
echo "==========================================="
