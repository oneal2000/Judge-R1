#!/usr/bin/env bash
set -euo pipefail

ROOT="/data-share/chenxuanyi/internship/JuDGE_RL"
# BASE_MODEL_DEFAULT="/data-share/chenxuanyi/LLM/Qwen3-4B-Thinking-2507"
SFT_MODEL_DEFAULT="/data-share/chenxuanyi/internship/JuDGE_RL/output/sft_qwen3-4b_lora_mrag/merge"
MODEL_TYPE="${MODEL_TYPE:-qwen3_thinking}"
# BASE_MODEL_DEFAULT="/data-share/LLM/models--Qwen--Qwen2.5-3B-Instruct/snapshots/aa8e72537993ba99e69dfaafa59ed015b17504d1"
# # SFT_MODEL_DEFAULT="/data-share/chenxuanyi/internship/JuDGE_RL/output/sft_qwen2.5-3b_lora/merge"
# MODEL_TYPE="${MODEL_TYPE:-qwen2}"
EXPERIMENT="${EXPERIMENT:-sft_full}"
USE_MRAG=${USE_MRAG:-false}

# 根据MRAG模式选择数据目录
if [[ "${USE_MRAG}" == "true" ]]; then
  DATA_DIR="${ROOT}/data/rl_train_mrag"
  MAX_INPUT_LENGTH=6000  # MRAG需要更长输入
  echo "[CONFIG] MRAG模式: 输入长度=${MAX_INPUT_LENGTH}"
else
  DATA_DIR="${ROOT}/data/rl_train"
  MAX_INPUT_LENGTH=3000  # 标准模式较短即可
  echo "[CONFIG] 标准模式: 输入长度=${MAX_INPUT_LENGTH}"
fi
MAX_OUTPUT_LENGTH=4096

MASTER_PORT=${MASTER_PORT:-27000}
VLLM_HOST=${VLLM_HOST:-127.0.0.1}
VLLM_PORT=${VLLM_PORT:-20420}
USE_VLLM=${USE_VLLM:-false}

cd "${ROOT}"

case "${EXPERIMENT}" in
  sft_full)
    echo "[CONFIG] SFT 模型上做全参 GRPO"
    MODEL="${MODEL:-${SFT_MODEL_DEFAULT}}"
    # echo "[CONFIG] 基座模型上做全参 GRPO"
    # MODEL="${MODEL:-${BASE_MODEL_DEFAULT}}"
    TRAIN_TYPE="full"
    # OUT_DIR="${ROOT}/output/rl_qwen3-4b_grpo_full"
    OUT_DIR="${ROOT}/output/rl_qwen3-4b_grpo_sft_full"
    # OUT_DIR="${ROOT}/output/rl_qwen2.5-3b_grpo_sft_full"
    # OUT_DIR="${ROOT}/output/rl_qwen2.5-3b_grpo_full"
    LR_DEFAULT="5e-6"       
    ;;
  sft_lora)
    echo "[CONFIG] EXPERIMENT=sft_lora -> SFT 模型上做 LoRA GRPO"
    MODEL="${MODEL:-${SFT_MODEL_DEFAULT}}"
    TRAIN_TYPE="lora"
    OUT_DIR="${ROOT}/output/rl_qwen3-4b_grpo_sft_lora"
    # OUT_DIR="${ROOT}/output/rl_qwen2.5-3b_grpo_sft_lora"
    LR_DEFAULT="5e-5"      
    ;;
  *)
    echo "[ERROR] 未知 EXPERIMENT=${EXPERIMENT}，请使用 sft_full 或 sft_lora"
    exit 1
    ;;
esac

LEARNING_RATE="${LEARNING_RATE:-${LR_DEFAULT}}"

gpu_ids=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
IFS=',' read -ra parts <<< "$gpu_ids"
gpu_num=${#parts[@]}

export HF_DATASETS_CACHE="${ROOT}/.cache"

echo "=========================================="
echo "  RL 训练 (GRPO)"
echo "  模型: ${MODEL}"
echo "  数据: ${DATA_DIR}"
echo "  MRAG: ${USE_MRAG}"
echo "  输入长度: ${MAX_INPUT_LENGTH}"
echo "  输出长度: ${MAX_OUTPUT_LENGTH}"
echo "  输出: ${OUT_DIR}"
echo "  GPU数量: ${gpu_num}"
echo "=========================================="

# 公共训练参数
COMMON_ARGS=(
  --rlhf_type grpo
  --model "${MODEL}"
  --model_type "${MODEL_TYPE}"
  --external_plugins "${ROOT}/train/src/rl_plugin1.py"
  --reward_funcs legal_doc_reward
  --train_type "${TRAIN_TYPE}"
  --torch_dtype bfloat16
  --dataset "${DATA_DIR}"
  --max_length ${MAX_INPUT_LENGTH}  
  --max_completion_length ${MAX_OUTPUT_LENGTH}
  --num_train_epochs 1
  --per_device_train_batch_size 1
  --gradient_accumulation_steps 16
  --learning_rate "${LEARNING_RATE}"
  --num_generations 16
  --temperature 0.8
  --save_steps 300
  --save_only_model true
  --save_total_limit 5
  --logging_steps 1
  --output_dir "${OUT_DIR}"
  --warmup_ratio 0.03
  --dataloader_num_workers 4
  --beta 0.05
  --repetition_penalty 1.1
  --report_to tensorboard
  --ddp_backend nccl
  --ddp_find_unused_parameters false
)

if [[ "${TRAIN_TYPE}" == "lora" ]]; then
  echo "[CONFIG] 使用 LoRA 训练参数（rank=64, alpha=128, dropout=0.05）"
  COMMON_ARGS+=(
    --lora_rank 64
    --lora_alpha 128
    --lora_dropout 0.05
  )
fi

if [[ "${USE_VLLM}" == "true" ]]; then
  echo "[INFO] USE_VLLM=true, 使用外部 vLLM server: ${VLLM_HOST}:${VLLM_PORT}"
  COMMON_ARGS+=(
    --use_vllm true
    --vllm_mode server
    --vllm_server_host "${VLLM_HOST}"
    --vllm_server_port "${VLLM_PORT}"
  )
else
  echo "[INFO] USE_VLLM=false, 不使用外部 vLLM（进程内推理）"
  COMMON_ARGS+=(
    --use_vllm false
  )
fi

echo "[RUN] CUDA_VISIBLE_DEVICES=${gpu_ids}, NPROC_PER_NODE=${gpu_num}"
echo "[RUN] Using MASTER_PORT=${MASTER_PORT}"

CUDA_VISIBLE_DEVICES=$gpu_ids \
NPROC_PER_NODE=$gpu_num \
MASTER_PORT=$MASTER_PORT \
swift rlhf \
  "${COMMON_ARGS[@]}" \
  "$@"

echo "✅ RL 训练完成"