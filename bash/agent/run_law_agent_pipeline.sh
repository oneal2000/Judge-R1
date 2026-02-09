#!/bin/bash
# ===========================================
# LLM-based 法条检索 Agent 完整 Pipeline
#
#   1. 运行 Agent 检索（QueryGen → Dense → Reranker → LawSelect）
#   2. 可选：输出级融合 Agent + MRAG 结果（比 Dense 层融合效果更好）
#   3. 生成 MRAG 数据
#   4. 评估检索效果
#
# 用法:
#   # 仅运行 Agent
#   CUDA_VISIBLE_DEVICES=0 bash bash/agent/run_law_agent_pipeline.sh
#
#   # Agent + MRAG 融合（需要先有 MRAG 结果）
#   FUSE_WITH_MRAG=true MRAG_FILE=mrag/retriever_output/ablation_mrag.tsv \
#     bash bash/agent/run_law_agent_pipeline.sh
#
#   # 指定 RL 模型路径
#   QUERYGEN_MODEL=/path/to/qg LAWSELECT_MODEL=/path/to/ls \
#     bash bash/agent/run_law_agent_pipeline.sh
# ===========================================

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../paths.sh"

# 添加项目根目录到 PYTHONPATH
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

# ============== CUDA 环境 ==============
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
export VLLM_USE_V1=1
export VLLM_USE_FLASHINFER_SAMPLER=0

# ============== 配置 ==============
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-7}
export CUDA_VISIBLE_DEVICES
GPU_MEMORY_UTIL=${GPU_MEMORY_UTIL:-0.85}

# 基座 LLM 模型
BASE_MODEL_7B="${QWEN25_7B_MODEL_PATH}"
BASE_MODEL_3B="${QWEN25_MODEL_PATH}"
if [[ -d "$BASE_MODEL_7B" ]]; then
    BASE_MODEL="$BASE_MODEL_7B"
else
    BASE_MODEL="$BASE_MODEL_3B"
fi
LLM_MODEL="${LLM_MODEL:-${BASE_MODEL}}"

# ============== RL 模型自动查找 ==============
find_latest_checkpoint() {
    local dir=$1
    if [[ ! -d "$dir" ]]; then echo ""; return; fi
    local latest_version=$(ls -d "$dir"/v* 2>/dev/null | sort -V | tail -1)
    if [[ -z "$latest_version" ]]; then echo ""; return; fi
    local latest_ckpt=$(ls -d "$latest_version"/checkpoint-* 2>/dev/null | sort -t- -k2 -n | tail -1)
    echo "$latest_ckpt"
}

# QueryGen RL 模型
QUERYGEN_MERGED_7B="${PROJECT_ROOT}/output/rl_querygen_7b_lora/merge"
QUERYGEN_RL_DIR_3B="${PROJECT_ROOT}/output/agent_rl_querygen_3b_v3"
QUERYGEN_MODEL_3B=$(find_latest_checkpoint "$QUERYGEN_RL_DIR_3B")
if [[ -z "${QUERYGEN_MODEL:-}" ]]; then
    if [[ -d "$QUERYGEN_MERGED_7B" && -f "$QUERYGEN_MERGED_7B/config.json" ]]; then
        QUERYGEN_MODEL="$QUERYGEN_MERGED_7B"
    elif [[ -n "$QUERYGEN_MODEL_3B" ]]; then
        QUERYGEN_MODEL="$QUERYGEN_MODEL_3B"
    else
        QUERYGEN_MODEL=""
    fi
fi

# LawSelect RL 模型
LAWSELECT_MERGED_7B="${PROJECT_ROOT}/output/rl_lawselect_7b_lora/merge"
LAWSELECT_RL_DIR_3B_V4="${PROJECT_ROOT}/output/agent_rl_lawselect_3b_v4"
LAWSELECT_RL_DIR_3B_V3="${PROJECT_ROOT}/output/agent_rl_lawselect_3b_v3"
LAWSELECT_MODEL_3B_V4=$(find_latest_checkpoint "$LAWSELECT_RL_DIR_3B_V4")
LAWSELECT_MODEL_3B_V3=$(find_latest_checkpoint "$LAWSELECT_RL_DIR_3B_V3")
if [[ -z "${LAWSELECT_MODEL:-}" ]]; then
    if [[ -d "$LAWSELECT_MERGED_7B" && -f "$LAWSELECT_MERGED_7B/config.json" ]]; then
        LAWSELECT_MODEL="$LAWSELECT_MERGED_7B"
    elif [[ -n "$LAWSELECT_MODEL_3B_V4" ]]; then
        LAWSELECT_MODEL="$LAWSELECT_MODEL_3B_V4"
    elif [[ -n "$LAWSELECT_MODEL_3B_V3" ]]; then
        LAWSELECT_MODEL="$LAWSELECT_MODEL_3B_V3"
    else
        LAWSELECT_MODEL=""
    fi
fi

# Dense Retriever / Reranker
DENSE_MODEL="${DENSE_MODEL:-${PROJECT_ROOT}/output/law_retriever}"
RERANKER_MODEL="${RERANKER_MODEL:-${PROJECT_ROOT}/reranker/train}"

# 数据路径
LAW_CORPUS="${PROJECT_ROOT}/data/law_corpus.jsonl"
TEST_DATA="${PROJECT_ROOT}/data/test.json"
QRELS_FILE="${PROJECT_ROOT}/mrag/retriever_data/qrels_test.tsv"
OUTPUT_DIR="${PROJECT_ROOT}/mrag/retriever_output"

# 检索参数
DENSE_TOP_K=${DENSE_TOP_K:-50}
RERANK_TOP_K=${RERANK_TOP_K:-20}
BATCH_SIZE=${BATCH_SIZE:-4}
TENSOR_PARALLEL_SIZE=${TENSOR_PARALLEL_SIZE:-1}
MIN_SELECTED=${MIN_SELECTED:-5}

# 融合参数
FUSE_WITH_MRAG=${FUSE_WITH_MRAG:-false}    # 是否融合 MRAG 结果
MRAG_FILE="${MRAG_FILE:-${OUTPUT_DIR}/ablation_mrag.tsv}"  # MRAG TSV 路径
FUSION_STRATEGY=${FUSION_STRATEGY:-rrf}     # agent_first / rrf / score_merge

# ============== Step 0: 检查模型 ==============
echo "==========================================="
echo "  LLM-based 法条检索 Agent（统一版）"
echo "==========================================="
echo ""
echo ">>> 检查模型路径..."

[[ ! -d "${LLM_MODEL}" ]] && echo "❌ 基座模型不存在: ${LLM_MODEL}" && exit 1
echo "✅ 基座模型: ${LLM_MODEL}"

[[ ! -d "${DENSE_MODEL}" ]] && echo "❌ Dense Model 不存在: ${DENSE_MODEL}" && exit 1
echo "✅ Dense Retriever: ${DENSE_MODEL}"

[[ ! -d "${RERANKER_MODEL}" ]] && echo "❌ Reranker 不存在: ${RERANKER_MODEL}" && exit 1
echo "✅ Reranker: ${RERANKER_MODEL}"

HAS_QUERYGEN_RL=false
if [[ -n "${QUERYGEN_MODEL}" && -d "${QUERYGEN_MODEL}" ]]; then
    echo "✅ QueryGen RL: ${QUERYGEN_MODEL}"
    HAS_QUERYGEN_RL=true
else
    echo "⚠️  QueryGen RL 不存在，使用基座模型"
    QUERYGEN_MODEL=""
fi

HAS_LAWSELECT_RL=false
if [[ -n "${LAWSELECT_MODEL}" && -d "${LAWSELECT_MODEL}" ]]; then
    echo "✅ LawSelect RL: ${LAWSELECT_MODEL}"
    HAS_LAWSELECT_RL=true
else
    echo "⚠️  LawSelect RL 不存在，使用基座模型"
    LAWSELECT_MODEL=""
fi

# 确定输出文件名
if [[ "${HAS_QUERYGEN_RL}" == "true" && "${HAS_LAWSELECT_RL}" == "true" ]]; then
    OUTPUT_SUFFIX="both_rl"
    MODE_DESC="Both RL (QueryGen + LawSelect)"
elif [[ "${HAS_QUERYGEN_RL}" == "true" ]]; then
    OUTPUT_SUFFIX="qg_rl"
    MODE_DESC="QueryGen RL only"
elif [[ "${HAS_LAWSELECT_RL}" == "true" ]]; then
    OUTPUT_SUFFIX="ls_rl"
    MODE_DESC="LawSelect RL only"
else
    OUTPUT_SUFFIX="baseline"
    MODE_DESC="Baseline (无 RL)"
fi

OUTPUT_FILE="${OUTPUT_DIR}/law_runfile_agent_${OUTPUT_SUFFIX}_test.tsv"
mkdir -p "${OUTPUT_DIR}"

# ============== Step 1: 运行 Agent ==============
echo ""
echo "==========================================="
echo "Step 1: 运行 LLM-based 法条检索 Agent"
echo "==========================================="
echo "  模式: ${MODE_DESC}"
echo "  Dense top-k: ${DENSE_TOP_K}, Rerank top-k: ${RERANK_TOP_K}"
echo "  输出: ${OUTPUT_FILE}"
echo "==========================================="

CMD_ARGS=(
    --llm_model "${LLM_MODEL}"
    --law_corpus "${LAW_CORPUS}"
    --dense_model "${DENSE_MODEL}"
    --reranker_model "${RERANKER_MODEL}"
    --input_file "${TEST_DATA}"
    --output_file "${OUTPUT_FILE}"
    --dense_top_k ${DENSE_TOP_K}
    --rerank_top_k ${RERANK_TOP_K}
    --min_selected ${MIN_SELECTED}
    --batch_size ${BATCH_SIZE}
    --device cuda
    --use_vllm
    --tensor_parallel_size ${TENSOR_PARALLEL_SIZE}
    --gpu_memory_util ${GPU_MEMORY_UTIL}
    --save_details
)

[[ -n "${QUERYGEN_MODEL}" ]] && CMD_ARGS+=(--querygen_model "${QUERYGEN_MODEL}")
[[ -n "${LAWSELECT_MODEL}" ]] && CMD_ARGS+=(--lawselect_model "${LAWSELECT_MODEL}")

python "${PROJECT_ROOT}/mrag/agent/law_agent.py" "${CMD_ARGS[@]}"
echo "Agent 检索完成"

# ============== Step 2: 评估 Agent ==============
echo ""
echo "==========================================="
echo "Step 2: 评估 Agent 检索效果"
echo "==========================================="

EVAL_OUTPUT="${OUTPUT_FILE%.tsv}_eval.txt"
python "${PROJECT_ROOT}/mrag/eval_retriever.py" \
    --runfile "${OUTPUT_FILE}" \
    --qrels "${QRELS_FILE}" \
    --output "${EVAL_OUTPUT}"

echo "Agent 评估结果:"
cat "${EVAL_OUTPUT}"

# ============== Step 3: 可选 — 融合 MRAG ==============
FINAL_OUTPUT="${OUTPUT_FILE}"  # 默认使用 Agent 输出

if [[ "${FUSE_WITH_MRAG}" == "true" ]]; then
    echo ""
    echo "==========================================="
    echo "Step 3: 输出级融合 Agent + MRAG"
    echo "==========================================="

    if [[ -f "${MRAG_FILE}" ]]; then
        FUSED_FILE="${OUTPUT_DIR}/law_runfile_fused_${OUTPUT_SUFFIX}_test.tsv"

        python "${PROJECT_ROOT}/mrag/agent/fuse_results.py" \
            --agent_file "${OUTPUT_FILE}" \
            --mrag_file "${MRAG_FILE}" \
            --output_file "${FUSED_FILE}" \
            --strategy "${FUSION_STRATEGY}" \
            --max_laws 20 \
            --qrels "${QRELS_FILE}"

        FINAL_OUTPUT="${FUSED_FILE}"
        echo ""
        echo "融合后使用: ${FUSED_FILE}"
    else
        echo "⚠️  MRAG 文件不存在: ${MRAG_FILE}"
        echo "   跳过融合，使用 Agent 结果"
        echo "   提示: 先运行 eval_ablation.sh 的 Phase 1 生成 MRAG 结果"
    fi
else
    echo ""
    echo "[提示] 设置 FUSE_WITH_MRAG=true 可自动融合 MRAG 结果提升 R@10"
fi

# ============== Step 4: 生成 MRAG 数据 ==============
echo ""
echo "==========================================="
echo "Step 4: 生成 MRAG 测试数据"
echo "==========================================="

MRAG_OUTPUT="${PROJECT_ROOT}/data/test_mrag_agent_${OUTPUT_SUFFIX}.json"

python "${PROJECT_ROOT}/script/sft_data.py" \
    --src "${TEST_DATA}" \
    --dst "${MRAG_OUTPUT}" \
    --split test \
    --use_mrag \
    --law_runfile "${FINAL_OUTPUT}" \
    --law_corpus "${LAW_CORPUS}"

echo "MRAG 测试数据已生成: ${MRAG_OUTPUT}"

# ============== 完成 ==============
echo ""
echo "==========================================="
echo "Pipeline 完成！"
echo "==========================================="
echo "  模式: ${MODE_DESC}"
echo "  Agent 结果: ${OUTPUT_FILE}"
[[ "${FUSE_WITH_MRAG}" == "true" && -f "${FUSED_FILE:-}" ]] && \
echo "  融合结果: ${FUSED_FILE}"
echo "  MRAG 数据: ${MRAG_OUTPUT}"
echo ""
echo "下一步: 使用 MRAG 数据推理"
echo "  USE_MRAG=true LAW_SOURCE=agent_${OUTPUT_SUFFIX} bash bash/gen.sh"
