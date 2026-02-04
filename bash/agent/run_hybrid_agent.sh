#!/bin/bash
# ===========================================
# Hybrid Fusion Agent Pipeline
# 
# 融合 MRAG 和 Agent 两条检索路线：
#   1. MRAG 路线: fact → Dense → top-K
#   2. Agent 路线: fact → QueryGen(RL) → Dense → top-K
#   3. RRF 融合两路结果
#   4. Reranker 统一重排
#   5. LLM Select (基座) / 直接输出
#
# 用法:
#   CUDA_VISIBLE_DEVICES=0 bash bash/agent/run_hybrid_agent.sh
#   SKIP_LLM_SELECT=false bash bash/agent/run_hybrid_agent.sh  # 启用 LLM Select
# ===========================================

set -e

PROJECT_ROOT="/data-share/chenxuanyi/internship/JuDGE_RL"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"

# ============== 配置 ==============
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export CUDA_VISIBLE_DEVICES
GPU_MEMORY_UTIL=${GPU_MEMORY_UTIL:-0.40}

# 基座模型 (用于 LawSelect)
BASE_MODEL="/data-share/chenxuanyi/LLM/Qwen2.5-7B-Instruct"

# QueryGen RL 模型 (合并后)
QUERYGEN_RL_MERGED="${PROJECT_ROOT}/output/rl_querygen_7b_lora/merge"
if [[ -d "$QUERYGEN_RL_MERGED" && -f "$QUERYGEN_RL_MERGED/config.json" ]]; then
    QUERYGEN_MODEL="$QUERYGEN_RL_MERGED"
    echo "✅ 使用 QueryGen RL: ${QUERYGEN_MODEL}"
else
    QUERYGEN_MODEL="$BASE_MODEL"
    echo "⚠️ QueryGen RL 未找到，使用基座模型"
fi

# LawSelect 固定使用基座模型
LAWSELECT_MODEL="$BASE_MODEL"

# 检索模型
DENSE_MODEL="${PROJECT_ROOT}/output/law_retriever"
RERANKER_MODEL="${PROJECT_ROOT}/reranker/train"

# 数据路径
LAW_CORPUS="${PROJECT_ROOT}/data/law_corpus.jsonl"
TEST_DATA="${PROJECT_ROOT}/data/test.json"
OUTPUT_DIR="${PROJECT_ROOT}/mrag/retriever_output"

# 检索参数
DENSE_TOP_K=${DENSE_TOP_K:-100}   # 每条路线检索数
FUSION_TOP_K=${FUSION_TOP_K:-150} # RRF 融合后保留数
RERANK_TOP_K=${RERANK_TOP_K:-50}  # Reranker 输出数
MIN_SELECTED=${MIN_SELECTED:-20}  # LLM 最少选择数
RRF_K=60
BATCH_SIZE=4
TENSOR_PARALLEL_SIZE=1

# 融合模式
# mrag_first: MRAG 优先（保留 MRAG 排序，Agent 补充）
# weighted_rrf: 加权 RRF
# rrf: 标准 RRF
FUSION_MODE=${FUSION_MODE:-mrag_first}

# 跳过 LLM Select（直接使用 Reranker 结果）
SKIP_LLM_SELECT=${SKIP_LLM_SELECT:-true}

echo "==========================================="
echo "  Hybrid Fusion Agent"
echo "==========================================="
echo "  QueryGen: ${QUERYGEN_MODEL}"
echo "  LawSelect: ${LAWSELECT_MODEL} (基座)"
echo "  融合模式: ${FUSION_MODE}"
echo "  跳过 LLM Select: ${SKIP_LLM_SELECT}"
echo "==========================================="

mkdir -p "${OUTPUT_DIR}"

# 构建可选参数
OPTIONAL_ARGS=""
[[ "${SKIP_LLM_SELECT}" == "true" ]] && OPTIONAL_ARGS="${OPTIONAL_ARGS} --skip_llm_select"

# 运行 Hybrid Agent
python "${PROJECT_ROOT}/mrag/agent/hybrid_agent.py" \
    --llm_model "${BASE_MODEL}" \
    --law_corpus "${LAW_CORPUS}" \
    --dense_model "${DENSE_MODEL}" \
    --reranker_model "${RERANKER_MODEL}" \
    --input_file "${TEST_DATA}" \
    --output_file "${OUTPUT_DIR}/law_runfile_hybrid.tsv" \
    --querygen_model "${QUERYGEN_MODEL}" \
    --lawselect_model "${LAWSELECT_MODEL}" \
    --dense_top_k ${DENSE_TOP_K} \
    --fusion_top_k ${FUSION_TOP_K} \
    --rerank_top_k ${RERANK_TOP_K} \
    --min_selected ${MIN_SELECTED} \
    --rrf_k ${RRF_K} \
    --fusion_mode ${FUSION_MODE} \
    --batch_size ${BATCH_SIZE} \
    --device cuda \
    --use_vllm \
    --tensor_parallel_size ${TENSOR_PARALLEL_SIZE} \
    --gpu_memory_util ${GPU_MEMORY_UTIL} \
    --save_details \
    ${OPTIONAL_ARGS}

# 评估
echo ""
echo ">>> 评估检索效果..."
python "${PROJECT_ROOT}/mrag/eval_retriever.py" \
    --runfile "${OUTPUT_DIR}/law_runfile_hybrid.tsv" \
    --qrels "${PROJECT_ROOT}/mrag/retriever_data/qrels_test.tsv" \
    --output "${OUTPUT_DIR}/eval_hybrid.txt"

echo ""
echo ">>> Hybrid Agent 结果:"
cat "${OUTPUT_DIR}/eval_hybrid.txt"

echo ""
echo "==========================================="
echo "✅ 完成！"
echo "  - 检索结果: ${OUTPUT_DIR}/law_runfile_hybrid.tsv"
echo "  - 评估结果: ${OUTPUT_DIR}/eval_hybrid.txt"
echo "==========================================="
