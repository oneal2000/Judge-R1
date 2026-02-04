#!/bin/bash
# ===========================================
# Agent 消融实验脚本
# 
# 对比配置（6 种）:
#   1. MRAG: 无 QueryGen + 无 LawSelect（纯 Dense + Rerank）
#   2. No QueryGen: 无 QueryGen + 有 LawSelect
#   3. No LawSelect: 有 QueryGen(基座) + 无 LawSelect
#   4. Baseline: QueryGen(基座) + LawSelect(基座)
#   5. QueryGen RL: QueryGen(RL) + LawSelect(基座)
#   6. Full Agent: QueryGen(RL) + LawSelect(基座) [同5，为后续扩展保留]
#
# 用法: CUDA_VISIBLE_DEVICES=0 bash bash/agent/eval_ablation.sh
# ===========================================

set -e

PROJECT_ROOT="/data-share/chenxuanyi/internship/JuDGE_RL"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"

# 设置 CUDA 环境（flashinfer JIT 编译需要）
export CUDA_HOME="/usr/local/cuda-12.4"
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
export VLLM_USE_V1=1
export VLLM_USE_FLASHINFER_SAMPLER=0  # 禁用 FlashInfer 采样，使用默认 PyTorch 采样

# ============== 模型配置 ==============
BASE_MODEL="/data-share/chenxuanyi/LLM/Qwen2.5-7B-Instruct"

# QueryGen RL 模型 (合并后的 LoRA)
QUERYGEN_RL_MERGED="${PROJECT_ROOT}/output/rl_querygen_7b_lora/merge"

# 检索模型
DENSE_MODEL="${PROJECT_ROOT}/output/law_retriever"
RERANKER_MODEL="${PROJECT_ROOT}/reranker/train"

# 数据路径
LAW_CORPUS="${PROJECT_ROOT}/data/law_corpus.jsonl"
TEST_DATA="${PROJECT_ROOT}/data/test.json"
QRELS_FILE="${PROJECT_ROOT}/mrag/retriever_data/qrels_test.tsv"
OUTPUT_DIR="${PROJECT_ROOT}/mrag/retriever_output"

# 检索参数（v7: 只需要 top-10，节省显存）
DENSE_TOP_K=50
RERANK_TOP_K=10
MIN_SELECTED=5
BATCH_SIZE=4
GPU_MEMORY_UTIL=0.5 

# GPU 设置
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export CUDA_VISIBLE_DEVICES

echo "=========================================="
echo "  Agent 消融实验（完整版）"
echo "=========================================="
echo "基座模型: ${BASE_MODEL}"
echo "QueryGen RL: ${QUERYGEN_RL_MERGED}"
echo "=========================================="

# 检查模型
check_model() {
    local path=$1
    local name=$2
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
    echo "⚠️ QueryGen RL 未合并，请先运行: bash bash/agent/merge_agent_lora.sh"
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

run_experiment() {
    local exp_name=$1
    local output_file=$2
    local querygen_model=$3
    local skip_qg=$4
    local skip_ls=$5
    
    echo ""
    echo "==========================================="
    echo "  实验: ${exp_name}"
    echo "==========================================="
    echo "  QueryGen: ${querygen_model:-跳过}"
    echo "  LawSelect: ${BASE_MODEL} (基座)"
    echo "  skip_querygen: ${skip_qg}"
    echo "  skip_lawselect: ${skip_ls}"
    echo "  输出: ${output_file}"
    echo "==========================================="
    
    # 构建命令参数
    local CMD_ARGS=(
        --llm_model "${BASE_MODEL}"
        --output_file "${output_file}"
        "${COMMON_ARGS[@]}"
    )
    
    # 添加 querygen_model（如果不跳过）
    if [[ "${skip_qg}" != "true" && -n "${querygen_model}" ]]; then
        CMD_ARGS+=(--querygen_model "${querygen_model}")
    fi
    
    # 添加跳过选项
    [[ "${skip_qg}" == "true" ]] && CMD_ARGS+=(--skip_querygen)
    [[ "${skip_ls}" == "true" ]] && CMD_ARGS+=(--skip_lawselect)
    
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

# ============== 实验 1: MRAG (无 QueryGen + 无 LawSelect) ==============
run_experiment \
    "MRAG (无QueryGen+无LawSelect)" \
    "${OUTPUT_DIR}/law_runfile_mrag_only.tsv" \
    "" \
    "true" \
    "true"

# ============== 实验 2: No QueryGen (无 QueryGen + 有 LawSelect) ==============
run_experiment \
    "No QueryGen (有LawSelect)" \
    "${OUTPUT_DIR}/law_runfile_no_querygen.tsv" \
    "" \
    "true" \
    "false"

# ============== 实验 3: No LawSelect (有 QueryGen基座 + 无 LawSelect) ==============
run_experiment \
    "No LawSelect (有QueryGen基座)" \
    "${OUTPUT_DIR}/law_runfile_no_lawselect.tsv" \
    "${BASE_MODEL}" \
    "false" \
    "true"

# ============== 实验 4: Baseline (QueryGen基座 + LawSelect基座) ==============
run_experiment \
    "Baseline (基座模型)" \
    "${OUTPUT_DIR}/law_runfile_baseline.tsv" \
    "${BASE_MODEL}" \
    "false" \
    "false"

# ============== 实验 5: QueryGen RL ==============
if [[ "${HAS_QUERYGEN_RL}" == "true" ]]; then
    run_experiment \
        "QueryGen RL" \
        "${OUTPUT_DIR}/law_runfile_querygen_rl.tsv" \
        "${QUERYGEN_RL_MERGED}" \
        "false" \
        "false"
fi

# ============== 汇总 ==============
echo ""
echo "==========================================="
echo "  消融实验结果汇总"
echo "==========================================="

SUMMARY_FILE="${OUTPUT_DIR}/ablation_summary.txt"
cat > "${SUMMARY_FILE}" << EOF
Agent 消融实验 - $(date)
==========================================
配置: Dense top-k=${DENSE_TOP_K}, Rerank top-k=${RERANK_TOP_K}

实验配置说明:
  1. MRAG: 无QueryGen + 无LawSelect（纯 Dense→Rerank）
  2. No QueryGen: 无QueryGen + 有LawSelect（fact直接检索）
  3. No LawSelect: 有QueryGen基座 + 无LawSelect（跳过LLM筛选）
  4. Baseline: QueryGen基座 + LawSelect基座
  5. QueryGen RL: QueryGen(RL) + LawSelect基座

==========================================
EOF

for f in "${OUTPUT_DIR}/law_runfile_mrag_only_eval.txt" \
         "${OUTPUT_DIR}/law_runfile_no_querygen_eval.txt" \
         "${OUTPUT_DIR}/law_runfile_no_lawselect_eval.txt" \
         "${OUTPUT_DIR}/law_runfile_baseline_eval.txt" \
         "${OUTPUT_DIR}/law_runfile_querygen_rl_eval.txt"; do
    if [[ -f "$f" ]]; then
        echo "" >> "${SUMMARY_FILE}"
        echo "--- $(basename "$f" _eval.txt) ---" >> "${SUMMARY_FILE}"
        cat "$f" >> "${SUMMARY_FILE}"
    fi
done

cat "${SUMMARY_FILE}"

echo ""
echo "✅ 消融实验完成！结果: ${SUMMARY_FILE}"
