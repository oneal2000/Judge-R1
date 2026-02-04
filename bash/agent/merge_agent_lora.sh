#!/usr/bin/env bash
# ===========================================
# 合并 QueryGen RL LoRA 模型到基座模型
# 
# 用法: bash bash/agent/merge_agent_lora.sh
# ===========================================

set -euo pipefail
cd /data-share/chenxuanyi/internship/JuDGE_RL

# 基座模型
BASEMODEL="/data-share/chenxuanyi/LLM/Qwen2.5-7B-Instruct"

# LoRA 目录
LORA_DIR="output/rl_querygen_7b_lora"
OUTPUT_DIR="${LORA_DIR}/merge"

# 自动查找最新的 checkpoint
find_latest_checkpoint() {
    local dir=$1
    if [[ ! -d "$dir" ]]; then
        echo ""
        return
    fi
    local latest_version=$(ls -d "$dir"/v* 2>/dev/null | sort -V | tail -1)
    if [[ -z "$latest_version" ]]; then
        echo ""
        return
    fi
    local latest_ckpt=$(ls -d "$latest_version"/checkpoint-* 2>/dev/null | \
        sed 's/.*checkpoint-//' | sort -n | tail -1)
    if [[ -n "$latest_ckpt" ]]; then
        echo "$latest_version/checkpoint-$latest_ckpt"
    else
        echo ""
    fi
}

echo "==========================================="
echo "  合并 QueryGen RL LoRA"
echo "==========================================="

# 查找最新 checkpoint
CHECKPOINT=$(find_latest_checkpoint "$LORA_DIR")

if [[ -z "$CHECKPOINT" || ! -d "$CHECKPOINT" ]]; then
    echo "❌ 找不到 checkpoint in ${LORA_DIR}"
    echo "请先运行: bash bash/agent/train_rl_querygen.sh"
    exit 1
fi

echo "  基座模型: ${BASEMODEL}"
echo "  LoRA Checkpoint: ${CHECKPOINT}"
echo "  输出目录: ${OUTPUT_DIR}"
echo "==========================================="

# 检查 adapter_config.json
if [[ ! -f "${CHECKPOINT}/adapter_config.json" ]]; then
    echo "❌ ${CHECKPOINT}/adapter_config.json 不存在"
    exit 1
fi

mkdir -p "${OUTPUT_DIR}"

# 合并权重
echo "[STEP 1] Merging Base + LoRA..."
python train/deploy/merge_lora.py \
    --base_model "${BASEMODEL}" \
    --adapter_dir "${CHECKPOINT}" \
    --output_dir "${OUTPUT_DIR}"

# 复制 tokenizer 文件
echo "[STEP 2] Copying Tokenizer files..."
for file in tokenizer.json tokenizer_config.json vocab.json merges.txt \
            special_tokens_map.json added_tokens.json generation_config.json \
            chat_template.jinja; do
    if [[ -f "${BASEMODEL}/${file}" ]]; then
        cp "${BASEMODEL}/${file}" "${OUTPUT_DIR}/" && echo "  Copied: ${file}"
    fi
done
cp "${BASEMODEL}"/tokenizer* "${OUTPUT_DIR}/" 2>/dev/null || true

echo ""
echo "==========================================="
echo "✅ 合并完成: ${OUTPUT_DIR}"
echo "==========================================="
ls -lh "${OUTPUT_DIR}"/*.safetensors 2>/dev/null | head -5 || ls -lh "${OUTPUT_DIR}"/*.bin 2>/dev/null | head -5

echo ""
echo "下一步:"
echo "  bash bash/agent/eval_ablation.sh    # 消融实验"
echo "  bash bash/agent/run_hybrid_agent.sh # Hybrid 融合"
