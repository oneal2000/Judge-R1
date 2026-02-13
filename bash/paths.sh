#!/usr/bin/env bash
# ================================================================
#  paths.sh — Centralized path configuration
#
#  ALL model paths and the project root are defined here.
#  Every other script sources this file instead of hardcoding paths.
#
#  To reproduce: copy this file, edit the paths below, done.
#  You can also override any variable via environment variables:
#    QWEN3_MODEL_PATH=/my/path bash bash/train_sft.sh
# ================================================================

# -------- Project root (auto-detected from this file's location) --------
export PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

# -------- Generation base models --------
export QWEN3_MODEL_PATH="${QWEN3_MODEL_PATH:-/path/to/Qwen3-4B}"
export QWEN25_MODEL_PATH="${QWEN25_MODEL_PATH:-/path/to/Qwen2.5-3B-Instruct}"

# -------- Agent model (7B, for QueryGen & LawSelect) --------
export QWEN25_7B_MODEL_PATH="${QWEN25_7B_MODEL_PATH:-/path/to/Qwen2.5-7B-Instruct}"

# -------- Retriever / Reranker base model --------
export ROBERTA_MODEL_PATH="${ROBERTA_MODEL_PATH:-/path/to/chinese-roberta-wwm-ext}"

# -------- BERT model (for BERTScore in RL reward) --------
export BERT_MODEL_PATH="${BERT_MODEL_PATH:-/path/to/bert-base-chinese}"

# -------- Validation helper --------
validate_path() {
    local name="$1" path="$2"
    if [ ! -d "$path" ] && [ ! -f "$path" ]; then
        echo "========================================================"
        echo "  ERROR: $name not found"
        echo "  Expected: $path"
        echo ""
        echo "  Please edit bash/paths.sh or set the environment variable:"
        echo "    export $name=\"/your/actual/path\""
        echo "========================================================"
        exit 1
    fi
}
