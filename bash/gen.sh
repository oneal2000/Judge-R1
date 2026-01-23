#!/usr/bin/env bash
set -euo pipefail
cd /data-share/chenxuanyi/internship/JuDGE_RL
mkdir -p outputs

export CUDA_VISIBLE_DEVICES=4,5
TP_SIZE=2

# ===========================================
# 配置参数
# ===========================================
# USE_MRAG: 是否使用检索增强 (true/false)
#
# 使用方法:
#   bash bash/gen.sh                      # 标准模式
#   USE_MRAG=true bash bash/gen.sh        # MRAG 模式
#
# 注意: 运行前请确保已生成对应的测试集数据
#   bash bash/data_train.sh               # 生成标准模式测试集
#   USE_MRAG=true bash bash/data_train.sh # 生成 MRAG 模式测试集
# ===========================================

USE_MRAG=${USE_MRAG:-false}
SCRIPT="train/deploy/inf.py"

# Qwen3-4B RL模型路径
RL_QWEN3_PATH=${RL_QWEN3_PATH:-"output/rl_qwen3-4b_grpo_full/v1-20260118-210344/checkpoint-501"}

# Qwen2.5-3B RL模型路径
RL_QWEN25_PATH=${RL_QWEN25_PATH:-"output/rl_qwen2.5-3b_grpo_sft_full/v17-20260117-091241/checkpoint-501"}

# 根据 MRAG 模式选择数据集和模型
if [[ "${USE_MRAG}" == "true" ]]; then
    # MRAG 模式：使用预先格式化的测试集
    DATASET="data/test_sft_mrag.json"
    SUFFIX="_mrag"
    
    SFT_MODEL_QWEN3="output/sft_qwen3-4b_lora_mrag/merge"
    SFT_MODEL_QWEN25="output/sft_qwen2.5-3b_lora_mrag/merge"
    
    # 检查测试集是否存在
    if [[ ! -f "${DATASET}" ]]; then
        echo "❌ 错误: MRAG 测试集不存在: ${DATASET}"
        echo "   请先运行: USE_MRAG=true bash bash/data_train.sh"
        exit 1
    fi
    
    echo "=========================================="
    echo "  推理模式: MRAG 增强"
    echo "  测试集: ${DATASET}"
    echo "  输出后缀: ${SUFFIX}"
    echo "=========================================="
else
    # 标准模式：使用预先格式化的测试集
    DATASET="data/test_sft.json"
    SUFFIX=""
    
    SFT_MODEL_QWEN3="output/sft_qwen3-4b_lora/merge"
    SFT_MODEL_QWEN25="output/sft_qwen2.5-3b_lora/merge"
    
    # 检查测试集是否存在
    if [[ ! -f "${DATASET}" ]]; then
        echo "❌ 错误: 测试集不存在: ${DATASET}"
        echo "   请先运行: bash bash/data_train.sh"
        exit 1
    fi
    
    echo "=========================================="
    echo "  推理模式: 标准（无检索增强）"
    echo "  测试集: ${DATASET}"
    echo "=========================================="
fi

echo ""
echo ">>> 模型配置："
echo "  Qwen3 RL:   ${RL_QWEN3_PATH}"
echo "  Qwen2.5 RL: ${RL_QWEN25_PATH}"
echo "  Qwen3 SFT:  ${SFT_MODEL_QWEN3}"
echo "  Qwen2.5 SFT: ${SFT_MODEL_QWEN25}"
echo ""

echo ">>> Starting Inference Tasks..."

# ========================================================
# 1. Qwen3-4B-Thinking (推理模型系列)
# ========================================================

# 1.1 Direct (基座) - 使用原始测试数据
echo "[1/8] Running Qwen3 Direct..."
python $SCRIPT \
  --model_path "/data-share/chenxuanyi/LLM/Qwen3-4B-Thinking-2507" \
  --dataset_path "data/test.json" \
  --output_path "outputs/qwen3_direct${SUFFIX}_raw.json" \
  --mode direct \
  --tensor_parallel_size $TP_SIZE

# 1.2 ICL (Few-shot) - 使用原始测试数据
echo "[2/8] Running Qwen3 ICL..."
python $SCRIPT \
  --model_path "/data-share/chenxuanyi/LLM/Qwen3-4B-Thinking-2507" \
  --dataset_path "data/test.json" \
  --output_path "outputs/qwen3_icl${SUFFIX}_raw.json" \
  --mode icl \
  --tensor_parallel_size $TP_SIZE

# 1.3 SFT (LoRA Merge) - 使用预先格式化的测试集
echo "[3/8] Running Qwen3 SFT..."
if [ ! -d "${SFT_MODEL_QWEN3}" ]; then
  echo "Warning: ${SFT_MODEL_QWEN3} not found, skipping..."
else
  python $SCRIPT \
    --model_path "${SFT_MODEL_QWEN3}" \
    --dataset_path "${DATASET}" \
    --output_path "outputs/qwen3_sft${SUFFIX}_raw.json" \
    --mode sft \
    --tensor_parallel_size $TP_SIZE
fi

# 1.4 RL (最新 checkpoint) - 使用预先格式化的测试集
echo "[4/8] Running Qwen3 RL..."
if [ ! -d "${RL_QWEN3_PATH}" ]; then
  echo "Warning: ${RL_QWEN3_PATH} not found, skipping..."
else
  echo "Using model: ${RL_QWEN3_PATH}"
  python $SCRIPT \
    --model_path "${RL_QWEN3_PATH}" \
    --dataset_path "${DATASET}" \
    --output_path "outputs/qwen3_rl${SUFFIX}_raw.json" \
    --mode rl \
    --tensor_parallel_size $TP_SIZE
fi

# ========================================================
# 2. Qwen2.5-3B-Instruct (普通模型系列)
# ========================================================

# 2.1 Direct (基座) - 使用原始测试数据
echo "[5/8] Running Qwen2.5 Direct..."
python $SCRIPT \
  --model_path "/data-share/LLM/models--Qwen--Qwen2.5-3B-Instruct/snapshots/aa8e72537993ba99e69dfaafa59ed015b17504d1" \
  --dataset_path "data/test.json" \
  --output_path "outputs/qwen25_direct${SUFFIX}_raw.json" \
  --mode direct \
  --tensor_parallel_size $TP_SIZE

# 2.2 ICL (Few-shot) - 使用原始测试数据
echo "[6/8] Running Qwen2.5 ICL..."
python $SCRIPT \
  --model_path "/data-share/LLM/models--Qwen--Qwen2.5-3B-Instruct/snapshots/aa8e72537993ba99e69dfaafa59ed015b17504d1" \
  --dataset_path "data/test.json" \
  --output_path "outputs/qwen25_icl${SUFFIX}_raw.json" \
  --mode icl \
  --tensor_parallel_size $TP_SIZE

# 2.3 SFT (LoRA Merge) - 使用预先格式化的测试集
echo "[7/8] Running Qwen2.5 SFT..."
if [ ! -d "${SFT_MODEL_QWEN25}" ]; then
  echo "Warning: ${SFT_MODEL_QWEN25} not found, skipping..."
else
  python $SCRIPT \
    --model_path "${SFT_MODEL_QWEN25}" \
    --dataset_path "${DATASET}" \
    --output_path "outputs/qwen25_sft${SUFFIX}_raw.json" \
    --mode sft \
    --tensor_parallel_size $TP_SIZE
fi

# 2.4 RL (最新 checkpoint) - 使用预先格式化的测试集
echo "[8/8] Running Qwen2.5 RL..."
if [ ! -d "${RL_QWEN25_PATH}" ]; then
  echo "Warning: ${RL_QWEN25_PATH} not found, skipping..."
else
  echo "Using model: ${RL_QWEN25_PATH}"
  python $SCRIPT \
    --model_path "${RL_QWEN25_PATH}" \
    --dataset_path "${DATASET}" \
    --output_path "outputs/qwen25_rl${SUFFIX}_raw.json" \
    --mode rl \
    --tensor_parallel_size $TP_SIZE
fi

echo "✅ All inference tasks completed!"
