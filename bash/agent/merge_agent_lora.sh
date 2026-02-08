#!/usr/bin/env bash
# ===========================================
# 合并 LoRA 模型到基座模型
# 
# 用法:
#   bash bash/agent/merge_agent_lora.sh querygen                    # 自动找最新 checkpoint
#   bash bash/agent/merge_agent_lora.sh lawselect                   # 自动找最新 checkpoint
#   bash bash/agent/merge_agent_lora.sh querygen checkpoint-1000    # 指定 checkpoint
#   bash bash/agent/merge_agent_lora.sh lawselect v3-20260205-205057/checkpoint-1250  # 指定版本+checkpoint
# ===========================================

set -euo pipefail
cd /data-share/chenxuanyi/internship/JuDGE_RL

# 基座模型
BASEMODEL="/data-share/chenxuanyi/LLM/Qwen2.5-7B-Instruct"

# LoRA 目录映射
declare -A LORA_DIRS=(
    ["querygen"]="output/rl_querygen_7b_lora"
    ["lawselect"]="output/rl_lawselect_7b_lora"
)

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

# 合并函数
do_merge() {
    local name=$1
    local checkpoint=$2
    local lora_dir=${LORA_DIRS[$name]}
    local output_dir="${lora_dir}/merge"

    echo "==========================================="
    echo "  合并 ${name} LoRA"
    echo "==========================================="
    echo "  基座模型: ${BASEMODEL}"
    echo "  LoRA Checkpoint: ${checkpoint}"
    echo "  输出目录: ${output_dir}"
    echo "==========================================="

    if [[ ! -f "${checkpoint}/adapter_config.json" ]]; then
        echo "❌ ${checkpoint}/adapter_config.json 不存在"
        exit 1
    fi

    mkdir -p "${output_dir}"

    echo "[STEP 1] Merging Base + LoRA..."
    python train/deploy/merge_lora.py \
        --base_model "${BASEMODEL}" \
        --adapter_dir "${checkpoint}" \
        --output_dir "${output_dir}"

    echo "[STEP 2] Copying Tokenizer files..."
    for file in tokenizer.json tokenizer_config.json vocab.json merges.txt \
                special_tokens_map.json added_tokens.json generation_config.json \
                chat_template.jinja; do
        if [[ -f "${BASEMODEL}/${file}" ]]; then
            cp "${BASEMODEL}/${file}" "${output_dir}/" && echo "  Copied: ${file}"
        fi
    done
    cp "${BASEMODEL}"/tokenizer* "${output_dir}/" 2>/dev/null || true

    echo ""
    echo "✅ ${name} 合并完成: ${output_dir}"
    ls -lh "${output_dir}"/*.safetensors 2>/dev/null | head -5 || \
        ls -lh "${output_dir}"/*.bin 2>/dev/null | head -5
}

# ============== 主逻辑 ==============
if [[ $# -lt 1 ]]; then
    echo "可用的 LoRA 目录:"
    for key in "${!LORA_DIRS[@]}"; do
        echo "  ${key}: ${LORA_DIRS[$key]}"
        if [[ -d "${LORA_DIRS[$key]}" ]]; then
            ls -d "${LORA_DIRS[$key]}"/v*/checkpoint-* 2>/dev/null | sed 's/^/    /'
        fi
    done
    exit 1
fi

NAME=$1
if [[ -z "${LORA_DIRS[$NAME]+x}" ]]; then
    echo "❌ 未知任务: ${NAME}，可选: querygen, lawselect"
    exit 1
fi

LORA_DIR="${LORA_DIRS[$NAME]}"

if [[ $# -ge 2 ]]; then
    # 用户指定了 checkpoint
    USER_CKPT=$2
    if [[ -d "${LORA_DIR}/${USER_CKPT}" ]]; then
        CHECKPOINT="${LORA_DIR}/${USER_CKPT}"
    elif [[ -d "${USER_CKPT}" ]]; then
        CHECKPOINT="${USER_CKPT}"
    else
        echo "❌ 找不到 checkpoint: ${USER_CKPT}"
        echo "尝试过的路径:"
        echo "  - ${LORA_DIR}/${USER_CKPT}"
        echo "  - ${USER_CKPT}"
        echo ""
        echo "可用的 checkpoints:"
        ls -d "${LORA_DIR}"/v*/checkpoint-* 2>/dev/null | sed 's/^/  /'
        exit 1
    fi
else
    # 自动查找最新
    CHECKPOINT=$(find_latest_checkpoint "$LORA_DIR")
    if [[ -z "$CHECKPOINT" || ! -d "$CHECKPOINT" ]]; then
        echo "❌ 找不到 checkpoint in ${LORA_DIR}"
        exit 1
    fi
fi

do_merge "$NAME" "$CHECKPOINT"
