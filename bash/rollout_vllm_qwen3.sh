#!/usr/bin/env bash
set -euo pipefail

# 启动 vLLM rollout 服务，为 RL 训练提供生成
# 环境：/data-share/yeesuanAI08/tuyiteng/anaconda3/envs/swift
# 端口、显存占用、使用 GPU 可按需修改

MODEL="/data-share/chenxuanyi/LLM/Qwen3-4B-Thinking-2507"
PORT=${PORT:-20420}
GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.5}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-4096}

# 默认使用 1 张卡；如需指定卡号，可在调用时设置 CUDA_VISIBLE_DEVICES，例如：
# CUDA_VISIBLE_DEVICES=0 bash rollout_vllm_qwen3.sh

swift rollout \
  --model "${MODEL}" \
  --gpu_memory_utilization "${GPU_MEM_UTIL}" \
  --max_model_len "${MAX_MODEL_LEN}" \
  --port "${PORT}"

echo "vLLM rollout started on port ${PORT} with model ${MODEL}, gpu_memory_utilization=${GPU_MEM_UTIL}, max_model_len=${MAX_MODEL_LEN}"
