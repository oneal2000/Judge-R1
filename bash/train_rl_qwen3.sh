#!/usr/bin/env bash
set -euo pipefail


ROOT="/data-share/chenxuanyi/internship/JuDGE_RL"
BASE_MODEL_DEFAULT="/data-share/chenxuanyi/LLM/Qwen3-4B-Thinking-2507"
# SFT_MODEL_DEFAULT="/data-share/chenxuanyi/internship/JuDGE_RL/output/sft_qwen3-4b_full"
SFT_MODEL_DEFAULT="/data-share/chenxuanyi/internship/JuDGE_RL/output/sft_qwen2.5-3b_full"
MODEL_TYPE="${MODEL_TYPE:-qwen3_thinking}"
EXPERIMENT="${EXPERIMENT:-sft_full}"

DATA_SRC="${ROOT}/data/train.json"
DATA_DIR="${ROOT}/data/rl_train"

MASTER_PORT=${MASTER_PORT:-29640}
VLLM_HOST=${VLLM_HOST:-127.0.0.1}
VLLM_PORT=${VLLM_PORT:-20420}
USE_VLLM=${USE_VLLM:-true}

cd "${ROOT}"

case "${EXPERIMENT}" in
  sft_full)
    echo "[CONFIG] EXPERIMENT=sft_full -> SFT 模型上做全参 GRPO"

    MODEL="${MODEL:-${SFT_MODEL_DEFAULT}}"
    TRAIN_TYPE="full"
    # OUT_DIR="${ROOT}/output/rl_qwen3-4b_grpo_sft_full"
    OUT_DIR="${ROOT}/output/rl_qwen2.5-3b_grpo_sft_full"
    LR_DEFAULT="5e-6"       # 全参 RL 学习率略小一点
    ;;

  sft_lora)
    echo "[CONFIG] EXPERIMENT=sft_lora -> SFT 模型上做 LoRA GRPO"

    MODEL="${MODEL:-${SFT_MODEL_DEFAULT}}"
    TRAIN_TYPE="lora"
    OUT_DIR="${ROOT}/output/rl_qwen3-4b_grpo_sft_lora"
    LR_DEFAULT="5e-5"       # LoRA 一般可以用更大的 lr
    ;;

  *)
    echo "[ERROR] 未知 EXPERIMENT=${EXPERIMENT}，请使用 sft_full 或 sft_lora"
    exit 1
    ;;
esac

LEARNING_RATE="${LEARNING_RATE:-${LR_DEFAULT}}"

gpu_ids=${CUDA_VISIBLE_DEVICES:-5,6}
IFS=',' read -ra parts <<< "$gpu_ids"
gpu_num=${#parts[@]}

export HF_DATASETS_CACHE="${ROOT}/.cache"

# 公共训练参数
COMMON_ARGS=(
  --rlhf_type grpo
  --model "${MODEL}"
  --model_type "${MODEL_TYPE:-qwen3_thinking}"
  --external_plugins "${ROOT}/train/src/rl_plugin.py"
  --reward_funcs legal_doc_reward
  --train_type "${TRAIN_TYPE}"
  --torch_dtype bfloat16
  --dataset "${DATA_DIR}"
  --max_length 4096
  --max_completion_length 2048
  --num_train_epochs 1
  --per_device_train_batch_size 1
  --gradient_accumulation_steps 16
  --learning_rate "${LEARNING_RATE}"
  --num_generations 8
  --temperature 0.8
  --save_steps 250
  --save_only_model true
  --save_total_limit 5
  --logging_steps 1
  --output_dir "${OUT_DIR}"
  --warmup_ratio 0.03
  --dataloader_num_workers 4
  --beta 0.001
  --report_to tensorboard
  --logging_dir "${OUT_DIR}/tb"
)

if [[ "${TRAIN_TYPE}" == "lora" ]]; then
  echo "[CONFIG] 使用 LoRA 训练参数（rank=64, alpha=128, dropout=0.05）"
  COMMON_ARGS+=(
    --lora_rank 64
    --lora_alpha 128
    --lora_dropout 0.05

  )
fi

# 根据 USE_VLLM 决定是否追加 vLLM 参数
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
echo "[RUN] MODEL=${MODEL}"
echo "[RUN] TRAIN_TYPE=${TRAIN_TYPE}, LR=${LEARNING_RATE}"
echo "[RUN] OUTPUT_DIR=${OUT_DIR}"

CUDA_VISIBLE_DEVICES=$gpu_ids \
NPROC_PER_NODE=$gpu_num \
swift rlhf \
  "${COMMON_ARGS[@]}" \
  "$@"
