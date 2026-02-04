#!/usr/bin/env bash
set -euo pipefail
cd /data-share/chenxuanyi/internship/JuDGE_RL
mkdir -p outputs

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1,3,5,6}
TP_SIZE=4

# ============================================================
# LegalOne-4B 推理脚本
# 
# 用法:
#   bash bash/legalone/gen.sh                      # 标准模式
#   USE_MRAG=true bash bash/legalone/gen.sh        # MRAG 模式
# ============================================================

USE_MRAG=${USE_MRAG:-false}
SCRIPT="train/deploy/inf.py"  # 复用统一的推理脚本

# LegalOne-4B 模型路径
BASE_MODEL="/data-share/chenxuanyi/LLM/LegalOne-4B"

# 根据 MRAG 模式选择数据集和 SFT 模型
if [[ "${USE_MRAG}" == "true" ]]; then
    DATASET="data/test_sft_mrag.json"
    SUFFIX="_mrag"
    SFT_MODEL="output/sft_legalone-4b_lora_mrag/merge"
    
    if [[ ! -f "${DATASET}" ]]; then
        echo "❌ 错误: MRAG 测试集不存在: ${DATASET}"
        echo "   请先运行: USE_MRAG=true bash bash/data_train.sh"
        exit 1
    fi
    
    echo "=========================================="
    echo "  LegalOne-4B 推理: MRAG 增强模式"
    echo "  测试集: ${DATASET}"
    echo "  输出后缀: ${SUFFIX}"
    echo "=========================================="
else
    DATASET="data/test_sft.json"
    SUFFIX=""
    SFT_MODEL="output/sft_legalone-4b_lora/merge"
    
    if [[ ! -f "${DATASET}" ]]; then
        echo "❌ 错误: 测试集不存在: ${DATASET}"
        echo "   请先运行: bash bash/data_train.sh"
        exit 1
    fi
    
    echo "=========================================="
    echo "  LegalOne-4B 推理: 标准模式"
    echo "  测试集: ${DATASET}"
    echo "=========================================="
fi

echo ""
echo ">>> 模型配置："
echo "  基座模型: ${BASE_MODEL}"
echo "  SFT 模型: ${SFT_MODEL}"
echo ""

echo ">>> Starting LegalOne-4B Inference Tasks..."

# ========================================================
# 1. Direct (基座模型) - 使用原始测试数据
# ========================================================
echo "[1/3] Running LegalOne-4B Direct..."
if [ ! -d "${BASE_MODEL}" ]; then
    echo "Warning: ${BASE_MODEL} not found, skipping..."
    echo "   请先运行: bash bash/legalone/download_model.sh"
else
    python $SCRIPT \
        --model_path "${BASE_MODEL}" \
        --dataset_path "data/test.json" \
        --output_path "outputs/legalone_direct${SUFFIX}_raw.json" \
        --mode direct \
        --tensor_parallel_size $TP_SIZE
fi

# ========================================================
# 2. ICL (Few-shot) - 使用原始测试数据
# ========================================================
echo "[2/3] Running LegalOne-4B ICL..."
if [ ! -d "${BASE_MODEL}" ]; then
    echo "Warning: ${BASE_MODEL} not found, skipping..."
else
    python $SCRIPT \
        --model_path "${BASE_MODEL}" \
        --dataset_path "data/test.json" \
        --output_path "outputs/legalone_icl${SUFFIX}_raw.json" \
        --mode icl \
        --tensor_parallel_size $TP_SIZE
fi

# ========================================================
# 3. SFT (LoRA Merge) - 使用预先格式化的测试集
# ========================================================
echo "[3/3] Running LegalOne-4B SFT..."
if [ ! -d "${SFT_MODEL}" ]; then
    echo "Warning: ${SFT_MODEL} not found, skipping..."
    echo "   请先运行 SFT 训练和 LoRA 合并"
else
    python $SCRIPT \
        --model_path "${SFT_MODEL}" \
        --dataset_path "${DATASET}" \
        --output_path "outputs/legalone_sft${SUFFIX}_raw.json" \
        --mode sft \
        --tensor_parallel_size $TP_SIZE
fi

echo ""
echo "✅ LegalOne-4B inference tasks completed!"
