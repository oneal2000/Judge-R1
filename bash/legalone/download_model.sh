#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# 下载 LegalOne-4B 模型
# ============================================================

MODEL_ID="CSHaitao/LegalOne-4B"
TARGET_DIR="/data-share/chenxuanyi/LLM/LegalOne-4B"

echo "=========================================="
echo "  下载 LegalOne-4B 模型"
echo "  模型: ${MODEL_ID}"
echo "  目标: ${TARGET_DIR}"
echo "=========================================="

# 检查目标目录是否已存在
if [ -d "${TARGET_DIR}" ] && [ -f "${TARGET_DIR}/config.json" ]; then
    echo "✅ 模型已存在: ${TARGET_DIR}"
    echo "如需重新下载，请先删除目录"
    exit 0
fi

# 创建目标目录
mkdir -p "${TARGET_DIR}"

# 使用 huggingface-cli 下载
echo "[INFO] 开始下载模型..."
huggingface-cli download \
    "${MODEL_ID}" \
    --local-dir "${TARGET_DIR}" \
    --local-dir-use-symlinks False

echo ""
echo "✅ 下载完成!"
echo "模型路径: ${TARGET_DIR}"
ls -lh "${TARGET_DIR}"
