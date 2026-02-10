#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/paths.sh"
cd "${PROJECT_ROOT}"
mkdir -p outputs

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-5}"
TP_SIZE="${TP_SIZE:-1}"

# ===========================================
# 参数化配置
# ===========================================
# 9 种推理模式 (每个模型都适用):
#   direct      -> 基座模型直接推理
#   icl         -> 基座模型 Few-shot 推理
#   sft         -> SFT 模型推理（无MRAG）
#   mrag        -> 基座模型 + MRAG数据推理
#   rl          -> Base→RL 模型推理（无MRAG）
#   sft_mrag    -> SFT+MRAG 模型推理
#   sft_rl      -> SFT→RL 模型推理（无MRAG）
#   mrag_rl     -> Base→RL+MRAG 模型推理
#   sft_mrag_rl -> SFT+MRAG→RL 模型推理
#
# 环境变量:
#   MODEL_NAME - 运行哪些模型, 逗号分隔 (默认: qwen3,qwen2)
#   MODES      - 运行哪些推理模式, 逗号分隔 (默认: all)
#                all -> 以上全部 9 种
#
# RL 模型路径（按需通过环境变量覆盖，默认指向 output 目录下的标准命名）:
#   RL_BASE_QWEN3_PATH / RL_BASE_QWEN25_PATH       - Base→RL
#   RL_SFT_QWEN3_PATH / RL_SFT_QWEN25_PATH         - SFT→RL
#   RL_BASE_MRAG_QWEN3_PATH / RL_BASE_MRAG_QWEN25_PATH - Base→RL+MRAG
#   RL_SFT_MRAG_QWEN3_PATH / RL_SFT_MRAG_QWEN25_PATH  - SFT+MRAG→RL
#
# 用法:
#   bash bash/gen.sh                                              # 所有模型所有模式
#   MODES=direct,icl bash bash/gen.sh                             # 只跑 direct 和 icl
#   MODEL_NAME=qwen3 MODES=sft_rl,sft_mrag_rl bash bash/gen.sh   # Qwen3 的 SFT→RL 系列
#   MODEL_NAME=qwen2 MODES=mrag,mrag_rl bash bash/gen.sh         # Qwen2.5 的 MRAG 系列
# ===========================================

MODEL_NAME="${MODEL_NAME:-qwen3,qwen2}"
MODES="${MODES:-all}"
SCRIPT="train/deploy/inf.py"

# 展开 "all" 关键字
[[ "${MODEL_NAME}" == "all" ]] && MODEL_NAME="qwen3,qwen2"
[[ "${MODES}" == "all" ]] && MODES="direct,icl,sft,mrag,rl,sft_mrag,sft_rl,mrag_rl,sft_mrag_rl"

# ========== 基座模型路径 ==========
QWEN3_BASE="${QWEN3_MODEL_PATH}"
QWEN25_BASE="${QWEN25_MODEL_PATH}"

# ========== SFT 模型路径 ==========
SFT_QWEN3="${SFT_QWEN3:-output/sft_qwen3-4b_lora/merge}"
SFT_MRAG_QWEN3="${SFT_MRAG_QWEN3:-output/sft_qwen3-4b_lora_mrag/merge}"
SFT_QWEN25="${SFT_QWEN25:-output/sft_qwen2.5-3b_lora/merge}"
SFT_MRAG_QWEN25="${SFT_MRAG_QWEN25:-output/sft_qwen2.5-3b_lora_mrag/merge}"

# ========== RL 模型路径（默认指向 output 目录，按需通过环境变量覆盖）==========
# Base → RL（无MRAG）
RL_BASE_QWEN3_PATH="${RL_BASE_QWEN3_PATH:-output/rl_qwen3-4b_grpo_base_full}"
RL_BASE_QWEN25_PATH="${RL_BASE_QWEN25_PATH:-output/rl_qwen2.5-3b_grpo_base_full}"
# SFT → RL（无MRAG）
RL_SFT_QWEN3_PATH="${RL_SFT_QWEN3_PATH:-output/rl_qwen3-4b_grpo_sft_full}"
RL_SFT_QWEN25_PATH="${RL_SFT_QWEN25_PATH:-output/rl_qwen2.5-3b_grpo_sft_full}"
# Base → RL（有MRAG）
RL_BASE_MRAG_QWEN3_PATH="${RL_BASE_MRAG_QWEN3_PATH:-output/rl_qwen3-4b_grpo_base_mrag_full}"
RL_BASE_MRAG_QWEN25_PATH="${RL_BASE_MRAG_QWEN25_PATH:-output/rl_qwen2.5-3b_grpo_base_mrag_full}"
# SFT+MRAG → RL
RL_SFT_MRAG_QWEN3_PATH="${RL_SFT_MRAG_QWEN3_PATH:-output/rl_qwen3-4b_grpo_sft_mrag_full}"
RL_SFT_MRAG_QWEN25_PATH="${RL_SFT_MRAG_QWEN25_PATH:-output/rl_qwen2.5-3b_grpo_sft_mrag_full}"

# ========== 数据集路径 ==========
DATASET_RAW="data/test.json"
DATASET_SFT="data/test_sft.json"
DATASET_MRAG="data/test_sft_mrag.json"

echo "=========================================="
echo "  推理配置"
echo "  模型: ${MODEL_NAME}"
echo "  模式: ${MODES}"
echo "=========================================="

# 通用推理函数
run_inference() {
    local model_path="$1"
    local label="$2"
    local inf_mode="$3"
    local dataset="$4"
    local output="$5"

    if [ ! -d "${model_path}" ]; then
        echo "⚠️  [SKIP] ${label}: 模型不存在 ${model_path}"
        return 0
    fi
    if [ ! -f "${dataset}" ]; then
        echo "⚠️  [SKIP] ${label}: 数据集不存在 ${dataset}"
        return 0
    fi

    echo "--------------------------------------------"
    echo "[RUN] ${label}"
    echo "  Model:   ${model_path}"
    echo "  Dataset: ${dataset}"
    echo "  Mode:    ${inf_mode}"
    echo "  Output:  ${output}"
    python $SCRIPT \
        --model_path "${model_path}" \
        --dataset_path "${dataset}" \
        --output_path "${output}" \
        --mode "${inf_mode}" \
        --tensor_parallel_size $TP_SIZE \
        --gpu_memory_utilization 0.3
}

# 根据 model_family 和 exp_mode 分发推理任务
dispatch_inference() {
    local model_family="$1"
    local exp_mode="$2"
    local prefix base sft sft_mrag rl_base rl_sft rl_base_mrag rl_sft_mrag

    case "${model_family}" in
        qwen3)
            prefix="qwen3"
            base="${QWEN3_BASE}"
            sft="${SFT_QWEN3}"
            sft_mrag="${SFT_MRAG_QWEN3}"
            rl_base="${RL_BASE_QWEN3_PATH}"
            rl_sft="${RL_SFT_QWEN3_PATH}"
            rl_base_mrag="${RL_BASE_MRAG_QWEN3_PATH}"
            rl_sft_mrag="${RL_SFT_MRAG_QWEN3_PATH}"
            ;;
        qwen2|qwen25|qwen2.5)
            prefix="qwen25"
            base="${QWEN25_BASE}"
            sft="${SFT_QWEN25}"
            sft_mrag="${SFT_MRAG_QWEN25}"
            rl_base="${RL_BASE_QWEN25_PATH}"
            rl_sft="${RL_SFT_QWEN25_PATH}"
            rl_base_mrag="${RL_BASE_MRAG_QWEN25_PATH}"
            rl_sft_mrag="${RL_SFT_MRAG_QWEN25_PATH}"
            ;;
        *)
            echo "[WARN] 未知模型: ${model_family}，跳过"
            return 0
            ;;
    esac

    # ============================================================
    # 9 种实验模式 → (模型路径, inf.py模式, 数据集, 输出文件)
    # ============================================================
    case "${exp_mode}" in
        direct)
            run_inference "${base}" "${prefix} direct" "direct" \
                "${DATASET_RAW}" "outputs/${prefix}_direct_raw.json"
            ;;
        icl)
            run_inference "${base}" "${prefix} icl" "icl" \
                "${DATASET_RAW}" "outputs/${prefix}_icl_raw.json"
            ;;
        sft)
            run_inference "${sft}" "${prefix} sft" "sft" \
                "${DATASET_SFT}" "outputs/${prefix}_sft_raw.json"
            ;;
        mrag)
            # 基座模型 + MRAG 数据（测试检索增强对基座模型的效果）
            run_inference "${base}" "${prefix} mrag" "sft" \
                "${DATASET_MRAG}" "outputs/${prefix}_mrag_raw.json"
            ;;
        rl)
            # Base → RL（无SFT、无MRAG）
            run_inference "${rl_base}" "${prefix} rl (base→RL)" "rl" \
                "${DATASET_SFT}" "outputs/${prefix}_rl_raw.json"
            ;;
        sft_mrag)
            run_inference "${sft_mrag}" "${prefix} sft_mrag" "sft" \
                "${DATASET_MRAG}" "outputs/${prefix}_sft_mrag_raw.json"
            ;;
        sft_rl)
            # SFT → RL（无MRAG）
            run_inference "${rl_sft}" "${prefix} sft_rl (SFT→RL)" "rl" \
                "${DATASET_SFT}" "outputs/${prefix}_sft_rl_raw.json"
            ;;
        mrag_rl)
            # Base → RL + MRAG 数据
            run_inference "${rl_base_mrag}" "${prefix} mrag_rl (base→RL+MRAG)" "rl" \
                "${DATASET_MRAG}" "outputs/${prefix}_mrag_rl_raw.json"
            ;;
        sft_mrag_rl)
            # SFT+MRAG → RL
            run_inference "${rl_sft_mrag}" "${prefix} sft_mrag_rl (SFT+MRAG→RL)" "rl" \
                "${DATASET_MRAG}" "outputs/${prefix}_sft_mrag_rl_raw.json"
            ;;
        *)
            echo "[WARN] 未知模式: ${exp_mode}，跳过"
            ;;
    esac
}

echo ">>> Starting Inference Tasks..."

IFS=',' read -ra MODEL_LIST <<< "${MODEL_NAME}"
IFS=',' read -ra MODE_LIST <<< "${MODES}"

for model in "${MODEL_LIST[@]}"; do
    model=$(echo "${model}" | xargs)
    for mode in "${MODE_LIST[@]}"; do
        mode=$(echo "${mode}" | xargs)
        dispatch_inference "${model}" "${mode}"
    done
done

echo "✅ All inference tasks completed!"
