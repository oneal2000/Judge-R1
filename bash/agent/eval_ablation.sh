#!/bin/bash

# 用法: CUDA_VISIBLE_DEVICES=0 bash bash/agent/eval_ablation.sh

set -e

PROJECT_ROOT="/data-share/chenxuanyi/internship/JuDGE_RL"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"

# CUDA 环境
export CUDA_HOME="/usr/local/cuda-12.4"
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
export VLLM_USE_V1=1
export VLLM_USE_FLASHINFER_SAMPLER=0

# ============== 模型配置 ==============
BASE_MODEL="/data-share/chenxuanyi/LLM/Qwen2.5-7B-Instruct"
QUERYGEN_RL_MERGED="${PROJECT_ROOT}/output/rl_querygen_7b_lora/merge"
LAWSELECT_RL_MERGED="${PROJECT_ROOT}/output/rl_lawselect_7b_lora/merge"
DENSE_MODEL="${PROJECT_ROOT}/output/law_retriever"
RERANKER_MODEL="${PROJECT_ROOT}/reranker/train"

# 数据路径
LAW_CORPUS="${PROJECT_ROOT}/data/law_corpus.jsonl"
TEST_DATA="${PROJECT_ROOT}/data/test.json"
QRELS_FILE="${PROJECT_ROOT}/mrag/retriever_data/qrels_test.tsv"
OUTPUT_DIR="${PROJECT_ROOT}/mrag/retriever_output"

# 检索参数（与训练保持一致）
DENSE_TOP_K=50
RERANK_TOP_K=20
MIN_SELECTED=5
BATCH_SIZE=4
GPU_MEMORY_UTIL=0.85

# 融合参数（输出级融合，不再在 Dense 层融合）
FUSION_STRATEGY="rrf"  # agent_first / rrf / score_merge / all

# GPU
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export CUDA_VISIBLE_DEVICES

echo "=========================================="
echo "  检索消融实验（公平对比版 · 9 种配置）"
echo "=========================================="
echo "  Dense 编码: 法条 max_length=256, 查询 max_length=512"
echo "  基座模型: ${BASE_MODEL}"
echo "=========================================="

# ============== 工具函数 ==============
check_model() {
    local path=$1 name=$2
    if [[ -d "$path" ]]; then
        echo "✅ ${name}: ${path}"
        return 0
    else
        echo "❌ ${name} 不存在: ${path}"
        return 1
    fi
}

check_model "${BASE_MODEL}" "基座模型" || exit 1
check_model "${DENSE_MODEL}" "Dense Retriever" || exit 1
check_model "${RERANKER_MODEL}" "Reranker" || exit 1

HAS_QUERYGEN_RL=false
if check_model "${QUERYGEN_RL_MERGED}" "QueryGen RL"; then
    HAS_QUERYGEN_RL=true
else
    echo "⚠️  QueryGen RL 未合并，#5 #7 将跳过"
fi

HAS_LAWSELECT_RL=false
if check_model "${LAWSELECT_RL_MERGED}" "LawSelect RL"; then
    HAS_LAWSELECT_RL=true
else
    echo "⚠️  LawSelect RL 未合并，#6 #7 将跳过"
fi

mkdir -p "${OUTPUT_DIR}"

# ============== 公共参数 ==============
COMMON_ARGS=(
    --law_corpus "${LAW_CORPUS}"
    --dense_model "${DENSE_MODEL}"
    --reranker_model "${RERANKER_MODEL}"
    --input_file "${TEST_DATA}"
    --dense_top_k ${DENSE_TOP_K}
    --rerank_top_k ${RERANK_TOP_K}
    --min_selected ${MIN_SELECTED}
    --batch_size ${BATCH_SIZE}
    --device cuda
    --use_vllm
    --tensor_parallel_size 1
    --gpu_memory_util ${GPU_MEMORY_UTIL}
    --save_details
)

# ============== Agent 实验函数 ==============
run_agent_experiment() {
    local exp_name=$1
    local output_file=$2
    local querygen_model=$3
    local skip_qg=$4
    local skip_reranker=$5
    local skip_ls=$6
    local lawselect_model=${7:-}

    echo ""
    echo "==========================================="
    echo "  实验: ${exp_name}"
    echo "==========================================="
    echo "  QueryGen:  ${querygen_model:-跳过}"
    echo "  Reranker:  $( [[ "${skip_reranker}" == "true" ]] && echo "跳过" || echo "启用" )"
    echo "  LawSelect: ${lawselect_model:-$( [[ "${skip_ls}" == "true" ]] && echo "跳过" || echo "${BASE_MODEL} (基座)" )}"
    echo "  输出: ${output_file}"
    echo "==========================================="

    local CMD_ARGS=(
        --llm_model "${BASE_MODEL}"
        --output_file "${output_file}"
        "${COMMON_ARGS[@]}"
    )

    [[ "${skip_qg}" == "true" ]] && CMD_ARGS+=(--skip_querygen)
    [[ "${skip_reranker}" == "true" ]] && CMD_ARGS+=(--skip_reranker)
    [[ "${skip_ls}" == "true" ]] && CMD_ARGS+=(--skip_lawselect)

    if [[ "${skip_qg}" != "true" && -n "${querygen_model}" ]]; then
        CMD_ARGS+=(--querygen_model "${querygen_model}")
    fi
    if [[ "${skip_ls}" != "true" && -n "${lawselect_model}" ]]; then
        CMD_ARGS+=(--lawselect_model "${lawselect_model}")
    fi

    python "${PROJECT_ROOT}/mrag/agent/law_agent.py" "${CMD_ARGS[@]}"

    # 评估
    local eval_output="${output_file%.tsv}_eval.txt"
    python "${PROJECT_ROOT}/mrag/eval_retriever.py" \
        --runfile "${output_file}" \
        --qrels "${QRELS_FILE}" \
        --output "${eval_output}"

    echo ">>> ${exp_name} 结果:"
    cat "${eval_output}"
}

# ============== 输出级融合实验函数 ==============
run_fusion_experiment() {
    local exp_name=$1
    local agent_file=$2
    local mrag_file=$3
    local output_file=$4
    local strategy=${5:-rrf}

    echo ""
    echo "==========================================="
    echo "  实验: ${exp_name}"
    echo "==========================================="
    echo "  Agent 文件: ${agent_file}"
    echo "  MRAG  文件: ${mrag_file}"
    echo "  融合策略:   ${strategy}"
    echo "  输出: ${output_file}"
    echo "==========================================="

    python "${PROJECT_ROOT}/mrag/agent/fuse_results.py" \
        --agent_file "${agent_file}" \
        --mrag_file "${mrag_file}" \
        --output_file "${output_file}" \
        --strategy "${strategy}" \
        --max_laws 20 \
        --qrels "${QRELS_FILE}"


    local eval_output="${output_file%.tsv}_eval.txt"
    python "${PROJECT_ROOT}/mrag/eval_retriever.py" \
        --runfile "${output_file}" \
        --qrels "${QRELS_FILE}" \
        --output "${eval_output}"

    echo ">>> ${exp_name} 结果:"
    cat "${eval_output}"
}

# ==============================================================
#  Phase 1: 检索基线
# ==============================================================
echo ""
echo "============================================"
echo "  Phase 1: 检索基线"
echo "============================================"

# #0 Dense Only (skip QG + skip Reranker + skip LS)
run_agent_experiment \
    "#0 Dense Only" \
    "${OUTPUT_DIR}/ablation_dense_only.tsv" \
    "" "true" "true" "true"

# #1 Dense + Reranker = MRAG (skip QG + skip LS)
run_agent_experiment \
    "#1 Dense + Reranker (MRAG)" \
    "${OUTPUT_DIR}/ablation_mrag.tsv" \
    "" "true" "false" "true"

# ==============================================================
#  Phase 2: Agent 组件消融
# ==============================================================
echo ""
echo "============================================"
echo "  Phase 2: Agent 组件消融"
echo "============================================"

# #2 No QueryGen (skip QG + LS基座)
run_agent_experiment \
    "#2 No QueryGen (LS基座)" \
    "${OUTPUT_DIR}/ablation_no_querygen.tsv" \
    "" "true" "false" "false"

# #3 No LawSelect (QG基座 + skip LS)
run_agent_experiment \
    "#3 No LawSelect (QG基座)" \
    "${OUTPUT_DIR}/ablation_no_lawselect.tsv" \
    "${BASE_MODEL}" "false" "false" "true"

# #4 Baseline (QG基座 + LS基座)
run_agent_experiment \
    "#4 Baseline (QG基座 + LS基座)" \
    "${OUTPUT_DIR}/ablation_baseline.tsv" \
    "${BASE_MODEL}" "false" "false" "false"

# ==============================================================
#  Phase 3: Agent RL 消融
# ==============================================================
echo ""
echo "============================================"
echo "  Phase 3: Agent RL 消融 "
echo "============================================"

# #5 QueryGen RL only (QG=RL + LS=基座)
if [[ "${HAS_QUERYGEN_RL}" == "true" ]]; then
    run_agent_experiment \
        "#5 QueryGen RL only (QG=RL, LS=基座)" \
        "${OUTPUT_DIR}/ablation_qg_rl.tsv" \
        "${QUERYGEN_RL_MERGED}" "false" "false" "false"
fi

# #6 LawSelect RL only (QG=基座 + LS=RL)
if [[ "${HAS_LAWSELECT_RL}" == "true" ]]; then
    run_agent_experiment \
        "#6 LawSelect RL only (QG=基座, LS=RL)" \
        "${OUTPUT_DIR}/ablation_ls_rl.tsv" \
        "${BASE_MODEL}" "false" "false" "false" \
        "${LAWSELECT_RL_MERGED}"
fi

# #7 Both RL (QG=RL + LS=RL)
if [[ "${HAS_QUERYGEN_RL}" == "true" && "${HAS_LAWSELECT_RL}" == "true" ]]; then
    run_agent_experiment \
        "#7 Both RL (QG=RL, LS=RL)" \
        "${OUTPUT_DIR}/ablation_both_rl.tsv" \
        "${QUERYGEN_RL_MERGED}" "false" "false" "false" \
        "${LAWSELECT_RL_MERGED}"
fi

# ==============================================================
#  Phase 4: 输出级融合（复用已有的 Agent 和 MRAG 输出）
# ==============================================================
echo ""
echo "============================================"
echo "  Phase 4: 输出级融合"
echo "============================================"
echo "  直接融合 Agent 和 MRAG 的 TSV 输出文件"
echo "============================================"

# 确定最佳 Agent 文件：优先 Both RL > LS RL > Baseline
if [[ -f "${OUTPUT_DIR}/ablation_both_rl.tsv" ]]; then
    BEST_AGENT_FILE="${OUTPUT_DIR}/ablation_both_rl.tsv"
    BEST_AGENT_NAME="Both RL (#7)"
elif [[ -f "${OUTPUT_DIR}/ablation_ls_rl.tsv" ]]; then
    BEST_AGENT_FILE="${OUTPUT_DIR}/ablation_ls_rl.tsv"
    BEST_AGENT_NAME="LS RL (#6)"
elif [[ -f "${OUTPUT_DIR}/ablation_baseline.tsv" ]]; then
    BEST_AGENT_FILE="${OUTPUT_DIR}/ablation_baseline.tsv"
    BEST_AGENT_NAME="Baseline (#4)"
else
    echo "⚠️  未找到 Agent 输出文件，跳过 Phase 4"
    BEST_AGENT_FILE=""
fi

MRAG_FILE="${OUTPUT_DIR}/ablation_mrag.tsv"

if [[ -n "${BEST_AGENT_FILE}" && -f "${MRAG_FILE}" ]]; then
    echo "  Agent 文件: ${BEST_AGENT_FILE} (${BEST_AGENT_NAME})"
    echo "  MRAG  文件: ${MRAG_FILE}"

    # #8 输出级融合
    run_fusion_experiment \
        "#8 Hybrid: ${BEST_AGENT_NAME} + MRAG (输出级 ${FUSION_STRATEGY})" \
        "${BEST_AGENT_FILE}" \
        "${MRAG_FILE}" \
        "${OUTPUT_DIR}/ablation_hybrid.tsv" \
        "${FUSION_STRATEGY}"
else
    echo "⚠️  缺少必要文件，跳过 Phase 4"
fi

# ==============================================================
#  汇总
# ==============================================================
echo ""
echo "==========================================="
echo "  检索消融实验结果汇总"
echo "==========================================="

SUMMARY_FILE="${OUTPUT_DIR}/ablation_summary.txt"
cat > "${SUMMARY_FILE}" << EOF
检索消融实验- $(date)
==========================================
Dense Retriever: 法条 max_length=256, 查询 max_length=512
配置: dense_top_k=${DENSE_TOP_K}, rerank_top_k=${RERANK_TOP_K}
Hybrid: 输出级融合, fusion_strategy=${FUSION_STRATEGY}
所有实验共享同一 Dense Retriever 和 Reranker 实例

实验配置:
  --- Phase 1: 检索基线 ---
  #0  Dense Only:         skip QG + skip Reranker + skip LS
  #1  Dense + Reranker:   skip QG + skip LS (= MRAG)

  --- Phase 2: Agent 组件消融 ---
  #2  No QueryGen:        skip QG + LS(基座)
  #3  No LawSelect:       QG(基座) + skip LS
  #4  Baseline:           QG(基座) + LS(基座)

  --- Phase 3: Agent RL ---
  #5  QG RL only:         QG(RL) + LS(基座)
  #6  LS RL only:         QG(基座) + LS(RL)
  #7  Both RL:            QG(RL) + LS(RL)

  --- Phase 4: 输出级融合 ---
  #8  Hybrid:             Agent TSV + MRAG TSV (输出级 ${FUSION_STRATEGY} 融合)

  2×2 RL 消融矩阵:
                  LawSelect基座    LawSelect RL
    QG 基座          #4               #6
    QG RL            #5               #7

==========================================
EOF

# 收集结果
EVAL_FILES=(
    "${OUTPUT_DIR}/ablation_dense_only_eval.txt"
    "${OUTPUT_DIR}/ablation_mrag_eval.txt"
    "${OUTPUT_DIR}/ablation_no_querygen_eval.txt"
    "${OUTPUT_DIR}/ablation_no_lawselect_eval.txt"
    "${OUTPUT_DIR}/ablation_baseline_eval.txt"
    "${OUTPUT_DIR}/ablation_qg_rl_eval.txt"
    "${OUTPUT_DIR}/ablation_ls_rl_eval.txt"
    "${OUTPUT_DIR}/ablation_both_rl_eval.txt"
    "${OUTPUT_DIR}/ablation_hybrid_eval.txt"
)

EVAL_NAMES=(
    "#0 Dense Only"
    "#1 Dense + Reranker (MRAG)"
    "#2 No QueryGen"
    "#3 No LawSelect"
    "#4 Baseline (QG基座+LS基座)"
    "#5 QueryGen RL only"
    "#6 LawSelect RL only"
    "#7 Both RL (QG=RL+LS=RL)"
    "#8 Hybrid MRAG+Agent"
)

for i in "${!EVAL_FILES[@]}"; do
    f="${EVAL_FILES[$i]}"
    name="${EVAL_NAMES[$i]}"
    echo "" >> "${SUMMARY_FILE}"
    if [[ -f "$f" ]]; then
        echo "--- ${name} ---" >> "${SUMMARY_FILE}"
        cat "$f" >> "${SUMMARY_FILE}"
    else
        echo "--- ${name} --- [未运行]" >> "${SUMMARY_FILE}"
    fi
done

cat "${SUMMARY_FILE}"

echo ""
echo "✅ 检索消融实验完成！结果: ${SUMMARY_FILE}"
