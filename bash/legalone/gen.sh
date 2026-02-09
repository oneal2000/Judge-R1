#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../paths.sh"
cd "${PROJECT_ROOT}"
mkdir -p outputs

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

# ============================================================
# LegalOne 推理脚本（4B + 1.7B）
#
# 用法:
#   bash bash/legalone/gen.sh
#   USE_MRAG=true bash bash/legalone/gen.sh
#   LEGALONE_MODEL_SET=1.7b bash bash/legalone/gen.sh
#   LEGALONE_MODEL_SET=4b MODES=direct,icl bash bash/legalone/gen.sh
# ============================================================

USE_MRAG=${USE_MRAG:-false}
LEGALONE_MODEL_SET=${LEGALONE_MODEL_SET:-4b}   # all | 4b | 1.7b
MODES=${MODES:-direct,icl,sft}                  # direct,icl,sft
SCRIPT="train/deploy/inf.py"

TP_SIZE=${TP_SIZE:-1}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.40}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-64}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-0}  # 0 表示按 inf.py 自动推断

# 模型路径（可覆盖）
BASE_MODEL_4B="${LEGALONE_4B_MODEL_PATH}"
BASE_MODEL_17B="${LEGALONE_17B_MODEL_PATH}"

# 根据 MRAG 模式选择数据集和 SFT 模型
if [[ "${USE_MRAG}" == "true" ]]; then
    SFT_DATASET="data/test_sft_mrag.json"
    SUFFIX="_mrag"
    SFT_MODEL_4B=${SFT_MODEL_4B:-output/sft_legalone-4b_lora_mrag/merge}
    SFT_MODEL_17B=${SFT_MODEL_17B:-output/sft_legalone-1.7b_lora_mrag/merge}
else
    SFT_DATASET="data/test_sft.json"
    SUFFIX=""
    SFT_MODEL_4B=${SFT_MODEL_4B:-output/sft_legalone-4b_lora/merge}
    SFT_MODEL_17B=${SFT_MODEL_17B:-output/sft_legalone-1.7b_lora/merge}
fi

if [[ ! -f "${SFT_DATASET}" ]]; then
    echo "❌ 错误: 测试集不存在: ${SFT_DATASET}"
    if [[ "${USE_MRAG}" == "true" ]]; then
        echo "   请先运行: USE_MRAG=true bash bash/data_train.sh"
    else
        echo "   请先运行: bash bash/data_train.sh"
    fi
    exit 1
fi

if [[ ! -f "data/test.json" ]]; then
    echo "❌ 错误: data/test.json 不存在"
    exit 1
fi

IFS=',' read -ra MODE_LIST <<< "${MODES}"

run_one_mode() {
    local display_name="$1"
    local model_path="$2"
    local output_prefix="$3"
    local mode="$4"
    local dataset_path="$5"
    local output_path="$6"

    if [[ ! -d "${model_path}" ]]; then
        echo "[SKIP] ${display_name} ${mode}: 模型不存在 ${model_path}"
        return
    fi

    echo "[RUN] ${display_name} ${mode}"
    echo "      model=${model_path}"
    echo "      dataset=${dataset_path}"
    echo "      output=${output_path}"

    args=(
        --model_path "${model_path}"
        --dataset_path "${dataset_path}"
        --output_path "${output_path}"
        --mode "${mode}"
        --tensor_parallel_size "${TP_SIZE}"
        --gpu_memory_utilization "${GPU_MEMORY_UTILIZATION}"
        --max_num_seqs "${MAX_NUM_SEQS}"
    )

    if [[ "${MAX_MODEL_LEN}" -gt 0 ]]; then
        args+=(--max_model_len "${MAX_MODEL_LEN}")
    fi

    python "${SCRIPT}" "${args[@]}"
}

run_one_model() {
    local display_name="$1"
    local base_model="$2"
    local sft_model="$3"
    local output_prefix="$4"

    echo ""
    echo "=========================================="
    echo "  ${display_name} 推理"
    echo "  MRAG: ${USE_MRAG}"
    echo "  Base: ${base_model}"
    echo "  SFT : ${sft_model}"
    echo "=========================================="

    local idx=0
    for mode in "${MODE_LIST[@]}"; do
        idx=$((idx + 1))
        case "${mode}" in
            direct)
                run_one_mode "${display_name}" "${base_model}" "${output_prefix}" "direct" \
                    "data/test.json" "outputs/${output_prefix}_direct${SUFFIX}_raw.json"
                ;;
            icl)
                run_one_mode "${display_name}" "${base_model}" "${output_prefix}" "icl" \
                    "data/test.json" "outputs/${output_prefix}_icl${SUFFIX}_raw.json"
                ;;
            sft)
                run_one_mode "${display_name}" "${sft_model}" "${output_prefix}" "sft" \
                    "${SFT_DATASET}" "outputs/${output_prefix}_sft${SUFFIX}_raw.json"
                ;;
            *)
                echo "❌ 不支持的 MODE=${mode} (可选: direct,icl,sft)"
                exit 1
                ;;
        esac
    done
}

echo "=========================================="
echo "  LegalOne 推理总配置"
echo "  模型集合: ${LEGALONE_MODEL_SET}"
echo "  模式: ${MODES}"
echo "  MRAG: ${USE_MRAG}"
echo "  TP: ${TP_SIZE}"
echo "  GPU_MEM_UTIL: ${GPU_MEMORY_UTILIZATION}"
echo "  MAX_NUM_SEQS: ${MAX_NUM_SEQS}"
echo "=========================================="

case "${LEGALONE_MODEL_SET}" in
    all)
        run_one_model "LegalOne-4B" "${BASE_MODEL_4B}" "${SFT_MODEL_4B}" "legalone"
        run_one_model "LegalOne-1.7B" "${BASE_MODEL_17B}" "${SFT_MODEL_17B}" "legalone17b"
        ;;
    4b)
        run_one_model "LegalOne-4B" "${BASE_MODEL_4B}" "${SFT_MODEL_4B}" "legalone"
        ;;
    1.7b|1_7b|17b)
        run_one_model "LegalOne-1.7B" "${BASE_MODEL_17B}" "${SFT_MODEL_17B}" "legalone17b"
        ;;
    *)
        echo "❌ 不支持的 LEGALONE_MODEL_SET=${LEGALONE_MODEL_SET} (可选: all|4b|1.7b)"
        exit 1
        ;;
esac

echo ""
echo "✅ LegalOne 推理完成"
