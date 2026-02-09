#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# 下载 LegalOne 系列模型（4B / 1.7B）
#
# 用法:
#   bash bash/legalone/download_model.sh
#   LEGALONE_MODEL_SET=1.7b bash bash/legalone/download_model.sh
#   LEGALONE_MODEL_SET=4b bash bash/legalone/download_model.sh
# ============================================================

LEGALONE_ROOT_DIR="${LEGALONE_ROOT_DIR:-/data-share/chenxuanyi/LLM}"
LEGALONE_MODEL_SET="${LEGALONE_MODEL_SET:-all}"   # all | 4b | 1.7b

# 向后兼容: 若手动指定 MODEL_ID，则按旧逻辑下载单模型
if [[ -n "${LEGALONE_MODEL_ID:-}" ]]; then
    MODEL_ID="${LEGALONE_MODEL_ID}"
    TARGET_DIR="${LEGALONE_TARGET_DIR:-${TARGET_DIR:-${HOME}/LLM/LegalOne-4B}}"
    echo "=========================================="
    echo "  下载自定义 LegalOne 模型"
    echo "  模型: ${MODEL_ID}"
    echo "  目标: ${TARGET_DIR}"
    echo "=========================================="
    if [[ -f "${TARGET_DIR}/config.json" ]]; then
        echo "✅ 模型已存在: ${TARGET_DIR}"
        exit 0
    fi
    mkdir -p "${TARGET_DIR}"
    huggingface-cli download \
        "${MODEL_ID}" \
        --local-dir "${TARGET_DIR}" \
        --local-dir-use-symlinks False
    echo "✅ 下载完成: ${TARGET_DIR}"
    exit 0
fi

download_one() {
    local model_id="$1"
    local target_dir="$2"

    echo "------------------------------------------"
    echo "模型: ${model_id}"
    echo "目标: ${target_dir}"

    if [[ -f "${target_dir}/config.json" ]]; then
        echo "✅ 已存在，跳过"
        return 0
    fi

    mkdir -p "${target_dir}"
    huggingface-cli download \
        "${model_id}" \
        --local-dir "${target_dir}" \
        --local-dir-use-symlinks False
    echo "✅ 下载完成: ${target_dir}"
}

echo "=========================================="
echo "  下载 LegalOne 模型"
echo "  模型集合: ${LEGALONE_MODEL_SET}"
echo "  根目录: ${LEGALONE_ROOT_DIR}"
echo "=========================================="

case "${LEGALONE_MODEL_SET}" in
    all)
        download_one "CSHaitao/LegalOne-4B" "${LEGALONE_4B_TARGET_DIR:-${LEGALONE_ROOT_DIR}/LegalOne-4B}"
        download_one "CSHaitao/LegalOne-1.7B" "${LEGALONE_17B_TARGET_DIR:-${LEGALONE_ROOT_DIR}/LegalOne-1.7B}"
        ;;
    4b)
        download_one "CSHaitao/LegalOne-4B" "${LEGALONE_4B_TARGET_DIR:-${LEGALONE_ROOT_DIR}/LegalOne-4B}"
        ;;
    1.7b|1_7b|17b)
        download_one "CSHaitao/LegalOne-1.7B" "${LEGALONE_17B_TARGET_DIR:-${LEGALONE_ROOT_DIR}/LegalOne-1.7B}"
        ;;
    *)
        echo "❌ 不支持的 LEGALONE_MODEL_SET=${LEGALONE_MODEL_SET} (可选: all|4b|1.7b)"
        exit 1
        ;;
esac

echo ""
echo "✅ 全部下载任务完成"
