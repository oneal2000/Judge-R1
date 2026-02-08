#!/usr/bin/env bash
set -euo pipefail
cd /data-share/chenxuanyi/internship/JuDGE_RL

# ============== 参数化配置 ==============
# 合并 SFT LoRA 模型到基座模型
#
# 注意: Agent RL 模型 (querygen/lawselect) 请使用专用脚本:
#   bash bash/agent/merge_agent_lora.sh querygen
#   bash bash/agent/merge_agent_lora.sh lawselect
#
# 通过 MERGE_CONFIG 环境变量选择要合并的 SFT 模型配置:
#   sft_qwen3_mrag   - Qwen3-4B SFT MRAG 模型 (默认)
#   sft_qwen3        - Qwen3-4B SFT 模型 (无MRAG)
#   sft_qwen2_mrag   - Qwen2.5-3B SFT MRAG 模型
#   sft_qwen2        - Qwen2.5-3B SFT 模型 (无MRAG)
#
# 也可直接通过 BASEMODEL, DIR, LORA_RANK, LORA_ALPHA 环境变量覆盖预设值
#
# 用法:
#   bash bash/loramerge.sh                                  # 默认: sft_qwen3_mrag
#   MERGE_CONFIG=sft_qwen3 bash bash/loramerge.sh           # Qwen3 SFT (无MRAG)
#   MERGE_CONFIG=sft_qwen2 bash bash/loramerge.sh           # Qwen2.5 SFT (无MRAG)
#   MERGE_CONFIG=sft_qwen2_mrag bash bash/loramerge.sh      # Qwen2.5 SFT MRAG
# ========================================

MERGE_CONFIG="${MERGE_CONFIG:-sft_qwen3_mrag}"

case "${MERGE_CONFIG}" in
  sft_qwen3_mrag)
    BASEMODEL="${BASEMODEL:-/data-share/chenxuanyi/LLM/Qwen3-4B-Thinking-2507}"
    DIR="${DIR:-/data-share/chenxuanyi/internship/JuDGE_RL/output/sft_qwen3-4b_lora_mrag}"
    LORA_RANK="${LORA_RANK:-128}"
    LORA_ALPHA="${LORA_ALPHA:-256}"
    ;;
  sft_qwen3)
    BASEMODEL="${BASEMODEL:-/data-share/chenxuanyi/LLM/Qwen3-4B-Thinking-2507}"
    DIR="${DIR:-/data-share/chenxuanyi/internship/JuDGE_RL/output/sft_qwen3-4b_lora}"
    LORA_RANK="${LORA_RANK:-128}"
    LORA_ALPHA="${LORA_ALPHA:-256}"
    ;;
  sft_qwen2_mrag|sft_qwen25_mrag)
    BASEMODEL="${BASEMODEL:-/data-share/LLM/models--Qwen--Qwen2.5-3B-Instruct/snapshots/aa8e72537993ba99e69dfaafa59ed015b17504d1}"
    DIR="${DIR:-/data-share/chenxuanyi/internship/JuDGE_RL/output/sft_qwen2.5-3b_lora_mrag}"
    LORA_RANK="${LORA_RANK:-128}"
    LORA_ALPHA="${LORA_ALPHA:-256}"
    ;;
  sft_qwen2|sft_qwen25)
    BASEMODEL="${BASEMODEL:-/data-share/LLM/models--Qwen--Qwen2.5-3B-Instruct/snapshots/aa8e72537993ba99e69dfaafa59ed015b17504d1}"
    DIR="${DIR:-/data-share/chenxuanyi/internship/JuDGE_RL/output/sft_qwen2.5-3b_lora}"
    LORA_RANK="${LORA_RANK:-128}"
    LORA_ALPHA="${LORA_ALPHA:-256}"
    ;;
  *)
    echo "[ERROR] 未知 MERGE_CONFIG=${MERGE_CONFIG}"
    echo "可选 SFT 配置: sft_qwen3_mrag | sft_qwen3 | sft_qwen2_mrag | sft_qwen2"
    echo ""
    echo "Agent RL 模型请使用: bash bash/agent/merge_agent_lora.sh [querygen|lawselect]"
    exit 1
    ;;
esac

# LoRA Target Modules（通用，无需修改）
# 注意：这必须与训练时的 target_modules 完全一致
TARGET_MODULES="q_proj,k_proj,v_proj,o_proj"

# =====================================
echo "=========================================="
echo "  SFT LoRA 合并"
echo "  配置: ${MERGE_CONFIG}"
echo "  Base: ${BASEMODEL}"
echo "  Adapter Source: ${DIR}"
echo "  LoRA Config: rank=${LORA_RANK}, alpha=${LORA_ALPHA}"
echo "  Target Modules: ${TARGET_MODULES}"
echo "=========================================="

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
