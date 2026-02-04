#!/usr/bin/env bash
set -euo pipefail
cd /data-share/chenxuanyi/internship/JuDGE_RL

# ============================================================
# LegalOne-4B LoRA 合并脚本
# 
# 用法:
#   bash bash/legalone/loramerge.sh               # 标准模式
#   USE_MRAG=true bash bash/legalone/loramerge.sh # MRAG 模式
# ============================================================

# LegalOne-4B 基座模型
BASEMODEL="/data-share/chenxuanyi/LLM/LegalOne-4B"

USE_MRAG=${USE_MRAG:-false}

# 根据 MRAG 模式选择 SFT 输出目录
if [[ "${USE_MRAG}" == "true" ]]; then
    DIR="/data-share/chenxuanyi/internship/JuDGE_RL/output/sft_legalone-4b_lora_mrag"
else
    DIR="/data-share/chenxuanyi/internship/JuDGE_RL/output/sft_legalone-4b_lora"
fi

# LoRA 配置 (与 Qwen3-4B-Thinking 保持一致)
LORA_RANK=128
LORA_ALPHA=256
TARGET_MODULES="q_proj,k_proj,v_proj,o_proj"

echo "=========================================="
echo "  LegalOne-4B LoRA 合并"
echo "  基座模型: ${BASEMODEL}"
echo "  Adapter: ${DIR}"
echo "  MRAG: ${USE_MRAG}"
echo "  LoRA: rank=${LORA_RANK}, alpha=${LORA_ALPHA}"
echo "=========================================="

# 检查是否需要提取 adapter
if [ ! -f "${DIR}/adapter_config.json" ]; then
    echo "[STEP 1] Extracting LoRA Adapter..."
    
    # 检查 pytorch_model.bin 是否存在
    if [ ! -f "${DIR}/pytorch_model.bin" ] && [ ! -d "${DIR}/pytorch_model.bin" ]; then
        echo "ERROR: ${DIR}/pytorch_model.bin not found!"
        echo "Please run: python ${DIR}/zero_to_fp32.py ${DIR} ${DIR}/pytorch_model.bin"
        exit 1
    fi
    
    python train/deploy/extract_lora.py \
        --base_model "${BASEMODEL}" \
        --mixed_dir "${DIR}" \
        --output_dir "${DIR}/extracted_adapter" \
        --lora_rank ${LORA_RANK} \
        --lora_alpha ${LORA_ALPHA} \
        --target_modules "${TARGET_MODULES}"
    
    ADAPTER_PATH="${DIR}/extracted_adapter"
else
    echo "[STEP 1] Found standard adapter files, skipping extraction."
    ADAPTER_PATH="${DIR}"
fi

# 合并权重
echo "[STEP 2] Merging Base + LoRA..."
mkdir -p "${DIR}/merge"

python train/deploy/merge_lora.py \
    --base_model "${BASEMODEL}" \
    --adapter_dir "${ADAPTER_PATH}" \
    --output_dir "${DIR}/merge"

echo "[STEP 3] Copying Tokenizer files..."
for file in tokenizer.json tokenizer_config.json vocab.json merges.txt \
            special_tokens_map.json added_tokens.json generation_config.json \
            chat_template.jinja; do
    if [ -f "${BASEMODEL}/${file}" ]; then
        cp "${BASEMODEL}/${file}" "${DIR}/merge/" && echo "  Copied: ${file}"
    fi
done

cp "${BASEMODEL}"/tokenizer* "${DIR}/merge/" 2>/dev/null || true

echo "✅ LegalOne-4B LoRA 合并完成!"
echo "合并后模型: ${DIR}/merge"
ls -lh "${DIR}/merge"
