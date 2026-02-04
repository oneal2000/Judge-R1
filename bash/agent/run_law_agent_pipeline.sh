#!/bin/bash
# ===========================================
# LLM-based 法条检索 Agent 完整 Pipeline
# 
# 使用 RL 训练后的 QueryGen 和 LawSelect 模型
# Pipeline: QueryGen(RL) → Dense Retriever (top-100) → Reranker (top-50) → LawSelect(RL)
#
# 参数与 MRAG 对齐（MRAG 输出 50 条法条）：
#   - DENSE_TOP_K=100: 扩大候选池
#   - RERANK_TOP_K=50: 与 MRAG 输出数量对齐
#   - MIN_SELECTED=20: 最终输出与 MRAG 对齐
#
# 包含: 
#   1. 运行 Agent 检索（使用 RL 模型）
#   2. 生成 MRAG 数据
#   3. 评估检索效果
# ===========================================

set -e

PROJECT_ROOT="/data-share/chenxuanyi/internship/JuDGE_RL"

# 添加项目根目录到 PYTHONPATH
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"

# ============== 配置 ==============
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-7}  
export CUDA_VISIBLE_DEVICES
GPU_MEMORY_UTIL=${GPU_MEMORY_UTIL:-0.5}

# 基座 LLM 模型（7B，用于 fallback）
BASE_MODEL_7B="/data-share/chenxuanyi/LLM/Qwen2.5-7B-Instruct"
BASE_MODEL_3B="/data-share/LLM/models--Qwen--Qwen2.5-3B-Instruct/snapshots/aa8e72537993ba99e69dfaafa59ed015b17504d1"
# 优先使用 7B 模型
if [[ -d "$BASE_MODEL_7B" ]]; then
    BASE_MODEL="$BASE_MODEL_7B"
else
    BASE_MODEL="$BASE_MODEL_3B"
fi
LLM_MODEL="${LLM_MODEL:-${BASE_MODEL}}"

# ============== RL 模型自动查找 ==============
# 自动查找最新的 checkpoint
find_latest_checkpoint() {
    local dir=$1
    if [[ ! -d "$dir" ]]; then
        echo ""
        return
    fi
    # 查找最新的 version 目录
    local latest_version=$(ls -d "$dir"/v* 2>/dev/null | sort -V | tail -1)
    if [[ -z "$latest_version" ]]; then
        echo ""
        return
    fi
    # 查找最新的 checkpoint
    local latest_ckpt=$(ls -d "$latest_version"/checkpoint-* 2>/dev/null | sort -t- -k2 -n | tail -1)
    echo "$latest_ckpt"
}

# QueryGen RL 模型（优先合并后的 7B，其次 3B full）
# 注意：LoRA checkpoint 无法直接被 vLLM 加载，需要先合并
# 运行 bash bash/agent/merge_agent_lora.sh 生成合并模型
QUERYGEN_MERGED_7B="${PROJECT_ROOT}/output/agent_rl_querygen_7b_v1_lora/merge"
QUERYGEN_RL_DIR_3B="${PROJECT_ROOT}/output/agent_rl_querygen_3b_v3"
QUERYGEN_MODEL_3B=$(find_latest_checkpoint "$QUERYGEN_RL_DIR_3B")
if [[ -d "$QUERYGEN_MERGED_7B" && -f "$QUERYGEN_MERGED_7B/config.json" ]]; then
    QUERYGEN_MODEL="${QUERYGEN_MODEL:-$QUERYGEN_MERGED_7B}"
elif [[ -n "$QUERYGEN_MODEL_3B" ]]; then
    QUERYGEN_MODEL="${QUERYGEN_MODEL:-$QUERYGEN_MODEL_3B}"
else
    QUERYGEN_MODEL=""
fi

# LawSelect RL 模型（优先合并后的 7B，其次 3B full）
LAWSELECT_MERGED_7B="${PROJECT_ROOT}/output/agent_rl_lawselect_7b_v1_lora/merge"
LAWSELECT_RL_DIR_3B_V4="${PROJECT_ROOT}/output/agent_rl_lawselect_3b_v4"
LAWSELECT_RL_DIR_3B_V3="${PROJECT_ROOT}/output/agent_rl_lawselect_3b_v3"
LAWSELECT_MODEL_3B_V4=$(find_latest_checkpoint "$LAWSELECT_RL_DIR_3B_V4")
LAWSELECT_MODEL_3B_V3=$(find_latest_checkpoint "$LAWSELECT_RL_DIR_3B_V3")
if [[ -d "$LAWSELECT_MERGED_7B" && -f "$LAWSELECT_MERGED_7B/config.json" ]]; then
    LAWSELECT_MODEL="${LAWSELECT_MODEL:-$LAWSELECT_MERGED_7B}"
elif [[ -n "$LAWSELECT_MODEL_3B_V4" ]]; then
    LAWSELECT_MODEL="${LAWSELECT_MODEL:-$LAWSELECT_MODEL_3B_V4}"
elif [[ -n "$LAWSELECT_MODEL_3B_V3" ]]; then
    LAWSELECT_MODEL="${LAWSELECT_MODEL:-$LAWSELECT_MODEL_3B_V3}"
else
    LAWSELECT_MODEL=""
fi

# Dense Retriever 模型
DENSE_MODEL="${DENSE_MODEL:-${PROJECT_ROOT}/output/law_retriever}"

# Reranker 模型
RERANKER_MODEL="${RERANKER_MODEL:-${PROJECT_ROOT}/reranker/train}"

# 数据路径
LAW_CORPUS="${PROJECT_ROOT}/data/law_corpus.jsonl"
TEST_DATA="${PROJECT_ROOT}/data/test.json"
OUTPUT_DIR="${PROJECT_ROOT}/mrag/retriever_output"

# 检索参数（与 MRAG 对齐：MRAG 输出 50 条法条）
DENSE_TOP_K=100  # Dense 检索取 top-100（扩大候选池）
RERANK_TOP_K=50  # Rerank 后取 top-50（与 MRAG 的 50 对齐）
BATCH_SIZE=4
TENSOR_PARALLEL_SIZE=1
MIN_SELECTED=20  # LLM 最少选择的法条数（与 MRAG 对齐）

# ============== Step 0: 检查模型 ==============
echo "==========================================="
echo "  LLM-based 法条检索 Agent (RL 模型)"
echo "==========================================="
echo ""
echo ">>> 检查模型路径..."

# 检查基座模型
if [ ! -d "${LLM_MODEL}" ]; then
    echo "❌ 错误: 基座模型不存在: ${LLM_MODEL}"
    exit 1
fi
echo "✅ 基座模型: ${LLM_MODEL}"

# 检查 Dense Retriever
if [ ! -d "${DENSE_MODEL}" ]; then
    echo "❌ 错误: Dense Model 路径不存在: ${DENSE_MODEL}"
    exit 1
fi
echo "✅ Dense Retriever: ${DENSE_MODEL}"

# 检查 Reranker
if [ ! -d "${RERANKER_MODEL}" ]; then
    echo "❌ 错误: Reranker Model 路径不存在: ${RERANKER_MODEL}"
    exit 1
fi
echo "✅ Reranker: ${RERANKER_MODEL}"

# 检查 QueryGen RL 模型
HAS_QUERYGEN_RL=false
if [[ -n "${QUERYGEN_MODEL}" && -d "${QUERYGEN_MODEL}" ]]; then
    echo "✅ QueryGen RL: ${QUERYGEN_MODEL}"
    HAS_QUERYGEN_RL=true
else
    echo "⚠️  QueryGen RL 不存在，将使用基座模型"
    QUERYGEN_MODEL=""
fi

# 检查 LawSelect RL 模型
HAS_LAWSELECT_RL=false
if [[ -n "${LAWSELECT_MODEL}" && -d "${LAWSELECT_MODEL}" ]]; then
    echo "✅ LawSelect RL: ${LAWSELECT_MODEL}"
    HAS_LAWSELECT_RL=true
else
    echo "⚠️  LawSelect RL 不存在，将使用基座模型"
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
echo "  基座模型: ${LLM_MODEL}"
[[ -n "${QUERYGEN_MODEL}" ]] && echo "  QueryGen RL: ${QUERYGEN_MODEL}"
[[ -n "${LAWSELECT_MODEL}" ]] && echo "  LawSelect RL: ${LAWSELECT_MODEL}"
echo "  Dense Model: ${DENSE_MODEL}"
echo "  Reranker Model: ${RERANKER_MODEL}"
echo "  Dense top-k: ${DENSE_TOP_K}"
echo "  Rerank top-k: ${RERANK_TOP_K}"
echo "  输出文件: ${OUTPUT_FILE}"
echo "==========================================="

# 构建命令参数
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

# 添加 RL 模型参数
[[ -n "${QUERYGEN_MODEL}" ]] && CMD_ARGS+=(--querygen_model "${QUERYGEN_MODEL}")
[[ -n "${LAWSELECT_MODEL}" ]] && CMD_ARGS+=(--lawselect_model "${LAWSELECT_MODEL}")

python "${PROJECT_ROOT}/mrag/agent/law_agent.py" "${CMD_ARGS[@]}"

echo "Agent 检索完成"

# ============== Step 2: 生成 MRAG 数据 ==============
echo ""
echo "==========================================="
echo "Step 2: 生成 MRAG 测试数据"
echo "==========================================="

MRAG_OUTPUT="${PROJECT_ROOT}/data/test_mrag_agent_${OUTPUT_SUFFIX}.json"

# 使用 sft_data.py 生成 MRAG 格式的测试数据
python "${PROJECT_ROOT}/script/sft_data.py" \
    --src "${TEST_DATA}" \
    --dst "${MRAG_OUTPUT}" \
    --split test \
    --use_mrag \
    --law_runfile "${OUTPUT_FILE}" \
    --law_corpus "${LAW_CORPUS}"

echo "MRAG 测试数据已生成: ${MRAG_OUTPUT}"

# ============== Step 3: 评估  ==============
echo ""
echo "==========================================="
echo "Step 3: 评估检索效果"
echo "==========================================="

EVAL_OUTPUT="${OUTPUT_FILE%.tsv}_eval.txt"

if [ -f "${PROJECT_ROOT}/mrag/eval_retriever.py" ]; then
    python "${PROJECT_ROOT}/mrag/eval_retriever.py" \
        --runfile "${OUTPUT_FILE}" \
        --qrels "${PROJECT_ROOT}/mrag/retriever_data/qrels_test.tsv" \
        --output "${EVAL_OUTPUT}"
    
    echo ""
    echo "评估结果:"
    cat "${EVAL_OUTPUT}"
else
    echo "跳过评估（未找到 eval_retriever.py）"
fi

# ============== 完成 ==============
echo ""
echo "==========================================="
echo "Pipeline 完成！"
echo "==========================================="
echo ""
echo "模式: ${MODE_DESC}"
echo ""
echo "输出文件:"
echo "  - Agent 检索结果: ${OUTPUT_FILE}"
echo "  - 详细输出: ${OUTPUT_FILE%.tsv}_details.json"
echo "  - MRAG 测试数据: ${MRAG_OUTPUT}"
echo "  - 评估结果: ${EVAL_OUTPUT}"
echo ""
echo "下一步: 使用生成的 MRAG 数据进行推理"
echo "  USE_MRAG=true LAW_SOURCE=agent_${OUTPUT_SUFFIX} bash bash/gen.sh"
