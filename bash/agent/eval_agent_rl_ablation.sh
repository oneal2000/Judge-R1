#!/bin/bash
# ===========================================
# Agent RL 消融实验对比脚本
# 
# 运行四种配置来验证 QueryGen RL 和 LawSelect RL 的效果:
#   1. Baseline: 两个都不用 RL（基座模型）
#   2. QueryGen RL only: 只用 QueryGen RL
#   3. LawSelect RL only: 只用 LawSelect RL  
#   4. Both RL: 两个都用 RL
#
# 用法: CUDA_VISIBLE_DEVICES=7 bash bash/agent/eval_agent_rl_ablation.sh
# ===========================================

set -e

PROJECT_ROOT="/data-share/chenxuanyi/internship/JuDGE_RL"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"

# ============== 模型配置 ==============
# 基座模型
BASE_MODEL="/data-share/LLM/models--Qwen--Qwen2.5-3B-Instruct/snapshots/aa8e72537993ba99e69dfaafa59ed015b17504d1"

# RL 训练后的模型 (v3 版本 - 改进的奖励函数)
# v3 改进：
# - QueryGen: 直接评估 Dense 阶段的 Recall@50 + MRR + HitRatio
# - LawSelect: 全量 Recall + MRR + NDCG + Precision
# - beta = 0.15 防止偏离基座
# - num_generations = 16/12 增加采样多样性
# 会自动查找最新的 checkpoint
QUERYGEN_RL_DIR="${PROJECT_ROOT}/output/agent_rl_querygen_3b_v3"
LAWSELECT_RL_DIR="${PROJECT_ROOT}/output/agent_rl_lawselect_3b_v3"

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

QUERYGEN_RL_MODEL=$(find_latest_checkpoint "$QUERYGEN_RL_DIR")
LAWSELECT_RL_MODEL=$(find_latest_checkpoint "$LAWSELECT_RL_DIR")

# 如果没找到 v2，尝试 v1 版本
if [[ -z "$QUERYGEN_RL_MODEL" ]]; then
    QUERYGEN_RL_MODEL=$(find_latest_checkpoint "${PROJECT_ROOT}/output/agent_rl_querygen_3b")
fi
if [[ -z "$LAWSELECT_RL_MODEL" ]]; then
    LAWSELECT_RL_MODEL=$(find_latest_checkpoint "${PROJECT_ROOT}/output/agent_rl_lawselect_3b")
fi

# 检索模型
DENSE_MODEL="${PROJECT_ROOT}/output/law_retriever"
RERANKER_MODEL="${PROJECT_ROOT}/reranker/train"

# 数据路径
LAW_CORPUS="${PROJECT_ROOT}/data/law_corpus.jsonl"
TEST_DATA="${PROJECT_ROOT}/data/test.json"
QRELS_FILE="${PROJECT_ROOT}/mrag/retriever_data/qrels_test.tsv"
OUTPUT_DIR="${PROJECT_ROOT}/mrag/retriever_output"

# 检索参数
DENSE_TOP_K=50
RERANK_TOP_K=20
MIN_SELECTED=5
BATCH_SIZE=4
GPU_MEMORY_UTIL=0.40

# GPU 设置
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-7}
export CUDA_VISIBLE_DEVICES

echo "=========================================="
echo "  Agent RL 消融实验"
echo "=========================================="
echo "基座模型: ${BASE_MODEL}"
echo "QueryGen RL: ${QUERYGEN_RL_MODEL}"
echo "LawSelect RL: ${LAWSELECT_RL_MODEL}"
echo "=========================================="

# 检查模型是否存在
check_model() {
    local model_path=$1
    local model_name=$2
    if [[ ! -d "${model_path}" ]]; then
        echo "❌ 错误: ${model_name} 不存在: ${model_path}"
        exit 1
    fi
    echo "✅ ${model_name}: ${model_path}"
}

echo ""
echo ">>> 检查模型路径..."
check_model "${BASE_MODEL}" "基座模型"
check_model "${DENSE_MODEL}" "Dense Retriever"
check_model "${RERANKER_MODEL}" "Reranker"

# QueryGen RL 和 LawSelect RL 可能不存在（正在训练中）
if [[ -n "${QUERYGEN_RL_MODEL}" && -d "${QUERYGEN_RL_MODEL}" ]]; then
    echo "✅ QueryGen RL: ${QUERYGEN_RL_MODEL}"
    HAS_QUERYGEN_RL=true
else
    echo "⚠️ QueryGen RL 不存在，将跳过相关实验"
    echo "   请先运行: bash bash/agent/train_agent_rl_querygen.sh"
    HAS_QUERYGEN_RL=false
fi

if [[ -n "${LAWSELECT_RL_MODEL}" && -d "${LAWSELECT_RL_MODEL}" ]]; then
    echo "✅ LawSelect RL: ${LAWSELECT_RL_MODEL}"
    HAS_LAWSELECT_RL=true
else
    echo "⚠️ LawSelect RL 不存在，将跳过相关实验"
    echo "   请先运行: bash bash/agent/train_agent_rl_lawselect.sh"
    HAS_LAWSELECT_RL=false
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
    local llm_model=$3
    local querygen_model=$4
    local lawselect_model=$5
    
    echo ""
    echo "==========================================="
    echo "  实验: ${exp_name}"
    echo "==========================================="
    echo "  LLM Model: ${llm_model}"
    [[ -n "${querygen_model}" ]] && echo "  QueryGen Model: ${querygen_model}"
    [[ -n "${lawselect_model}" ]] && echo "  LawSelect Model: ${lawselect_model}"
    echo "  输出: ${output_file}"
    echo "==========================================="
    
    # 构建命令
    CMD_ARGS=(
        --llm_model "${llm_model}"
        --output_file "${output_file}"
        "${COMMON_ARGS[@]}"
    )
    
    [[ -n "${querygen_model}" ]] && CMD_ARGS+=(--querygen_model "${querygen_model}")
    [[ -n "${lawselect_model}" ]] && CMD_ARGS+=(--lawselect_model "${lawselect_model}")
    
    # 运行
    python "${PROJECT_ROOT}/mrag/agent/law_agent.py" "${CMD_ARGS[@]}"
    
    # 评估
    echo ""
    echo ">>> 评估 ${exp_name}..."
    local eval_output="${output_file%.tsv}_eval.txt"
    python "${PROJECT_ROOT}/mrag/eval_retriever.py" \
        --runfile "${output_file}" \
        --qrels "${QRELS_FILE}" \
        --output "${eval_output}"
    
    echo ""
    echo ">>> ${exp_name} 评估结果:"
    cat "${eval_output}"
}

# ============== 实验 1: Baseline (无 RL) ==============
run_experiment \
    "Baseline (无 RL)" \
    "${OUTPUT_DIR}/law_runfile_agent_baseline_test.tsv" \
    "${BASE_MODEL}" \
    "" ""

# ============== 实验 2: QueryGen RL only ==============
if [[ "${HAS_QUERYGEN_RL}" == "true" ]]; then
    run_experiment \
        "QueryGen RL only" \
        "${OUTPUT_DIR}/law_runfile_agent_qg_rl_test.tsv" \
        "${BASE_MODEL}" \
        "${QUERYGEN_RL_MODEL}" ""
fi

# ============== 实验 3: LawSelect RL only ==============
if [[ "${HAS_LAWSELECT_RL}" == "true" ]]; then
    run_experiment \
        "LawSelect RL only" \
        "${OUTPUT_DIR}/law_runfile_agent_ls_rl_test.tsv" \
        "${BASE_MODEL}" \
        "" "${LAWSELECT_RL_MODEL}"
fi

# ============== 实验 4: Both RL ==============
if [[ "${HAS_QUERYGEN_RL}" == "true" && "${HAS_LAWSELECT_RL}" == "true" ]]; then
    run_experiment \
        "Both RL (QueryGen + LawSelect)" \
        "${OUTPUT_DIR}/law_runfile_agent_both_rl_test.tsv" \
        "${BASE_MODEL}" \
        "${QUERYGEN_RL_MODEL}" "${LAWSELECT_RL_MODEL}"
fi

# ============== 汇总结果 ==============
echo ""
echo "==========================================="
echo "  消融实验结果汇总"
echo "==========================================="

SUMMARY_FILE="${OUTPUT_DIR}/ablation_summary.txt"
echo "Agent RL 消融实验结果 - $(date)" > "${SUMMARY_FILE}"
echo "==========================================" >> "${SUMMARY_FILE}"

echo "" >> "${SUMMARY_FILE}"
echo "实验配置:" >> "${SUMMARY_FILE}"
echo "  Dense top-k: ${DENSE_TOP_K}" >> "${SUMMARY_FILE}"
echo "  Rerank top-k: ${RERANK_TOP_K}" >> "${SUMMARY_FILE}"
echo "  Min selected: ${MIN_SELECTED}" >> "${SUMMARY_FILE}"
echo "" >> "${SUMMARY_FILE}"

# 汇总各实验结果（只包含本次消融实验生成的文件）
for eval_file in \
    "${OUTPUT_DIR}/law_runfile_agent_baseline_test_eval.txt" \
    "${OUTPUT_DIR}/law_runfile_agent_qg_rl_test_eval.txt" \
    "${OUTPUT_DIR}/law_runfile_agent_ls_rl_test_eval.txt" \
    "${OUTPUT_DIR}/law_runfile_agent_both_rl_test_eval.txt"
do
    if [[ -f "${eval_file}" ]]; then
        filename=$(basename "${eval_file}" .txt)
        echo "-------------------------------------------" >> "${SUMMARY_FILE}"
        echo ">>> ${filename}" >> "${SUMMARY_FILE}"
        cat "${eval_file}" >> "${SUMMARY_FILE}"
    fi
done

echo "" >> "${SUMMARY_FILE}"
echo "==========================================" >> "${SUMMARY_FILE}"

echo ""
echo ">>> 汇总结果已保存到: ${SUMMARY_FILE}"
echo ""
cat "${SUMMARY_FILE}"

echo ""
echo "==========================================="
echo "✅ 消融实验完成！"
echo "==========================================="
echo ""
echo "输出文件:"
echo "  - Baseline: ${OUTPUT_DIR}/law_runfile_agent_test.tsv"
[[ "${HAS_QUERYGEN_RL}" == "true" ]] && echo "  - QueryGen RL: ${OUTPUT_DIR}/law_runfile_agent_qg_rl_test.tsv"
[[ "${HAS_LAWSELECT_RL}" == "true" ]] && echo "  - LawSelect RL: ${OUTPUT_DIR}/law_runfile_agent_ls_rl_test.tsv"
[[ "${HAS_QUERYGEN_RL}" == "true" && "${HAS_LAWSELECT_RL}" == "true" ]] && echo "  - Both RL: ${OUTPUT_DIR}/law_runfile_agent_both_rl_test.tsv"
echo "  - 汇总: ${SUMMARY_FILE}"
