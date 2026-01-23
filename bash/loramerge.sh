#!/usr/bin/env bash
set -euo pipefail
cd /data-share/chenxuanyi/internship/JuDGE_RL

# ============== 配置区 ==============
BASEMODEL="/data-share/chenxuanyi/LLM/Qwen3-4B-Thinking-2507"
DIR="/data-share/chenxuanyi/internship/JuDGE_RL/output/sft_qwen3-4b_lora_mrag"
# BASEMODEL="/data-share/LLM/models--Qwen--Qwen2.5-3B-Instruct/snapshots/aa8e72537993ba99e69dfaafa59ed015b17504d1"
# DIR="/data-share/chenxuanyi/internship/JuDGE_RL/output/sft_qwen2.5-3b_lora"
# LoRA 配置
LORA_RANK=128
LORA_ALPHA=256
# 注意：这必须与 train.py 中的 target_modules 完全一致
TARGET_MODULES="q_proj,k_proj,v_proj,o_proj"

# =====================================
echo "Base: ${BASEMODEL}"
echo "Adapter Source: ${DIR}"
echo "LoRA Config: rank=${LORA_RANK}, alpha=${LORA_ALPHA}"
echo "Target Modules: ${TARGET_MODULES}"

# 检查是否需要提取 adapter
if [ ! -f "${DIR}/adapter_config.json" ]; then
    echo "[STEP 1] Extracting LoRA Adapter..."
    
    # 检查 pytorch_model.bin 是否存在（可能是文件或目录）
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
            special_tokens_map.json added_tokens.json generation_config.json; do
    if [ -f "${BASEMODEL}/${file}" ]; then
        cp "${BASEMODEL}/${file}" "${DIR}/merge/" && echo "  Copied: ${file}"
    fi
done

cp "${BASEMODEL}"/tokenizer* "${DIR}/merge/" 2>/dev/null || true

echo "✅ All Done! Merged model is at: ${DIR}/merge"
ls -lh "${DIR}/merge"