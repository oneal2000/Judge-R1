#!/usr/bin/env bash
set -euo pipefail

cd /data-share/chenxuanyi/internship/JuDGE_RL

# ============================================================
# 数据生成脚本
# 
# 用法:
#   bash bash/data_train.sh              # 标准模式
#   USE_MRAG=true bash bash/data_train.sh  # MRAG 模式
#
# 输出:
#   标准模式:
#     - data/train_sft.json       (SFT 训练集)
#     - data/test_sft.json        (SFT 测试集)
#     - data/rl_train/train.jsonl (RL 训练集)
#
#   MRAG 模式:
#     - data/train_sft_mrag.json       (SFT 训练集)
#     - data/test_sft_mrag.json        (SFT 测试集)
#     - data/rl_train_mrag/train.jsonl (RL 训练集)
# ============================================================

USE_MRAG=${USE_MRAG:-false}

# 语料库路径
LAW_CORPUS="data/law_corpus.jsonl"
CASE_CORPUS="data/case_corpus.jsonl"

# 检索结果路径
if [[ "${USE_MRAG}" == "true" ]]; then
    # 训练集检索结果
    LAW_RUNFILE_TRAIN="data/retrieval/law_runfile_train"
    CASE_RUNFILE_TRAIN="data/retrieval/case_runfile_train"
    
    # 测试集检索结果（优先使用 Reranker 重排序后的结果）
    if [ -f "mrag/retriever_output/law_runfile_reranked_test.tsv" ]; then
        LAW_RUNFILE_TEST="mrag/retriever_output/law_runfile_reranked_test.tsv"
    else
        LAW_RUNFILE_TEST="mrag/retriever_output/law_runfile_test.tsv"
    fi
    CASE_RUNFILE_TEST="mrag/retriever_output/case_runfile_test.tsv"
    
    # 输出路径
    TRAIN_SFT="data/train_sft_mrag.json"
    TEST_SFT="data/test_sft_mrag.json"
    RL_TRAIN_DIR="data/rl_train_mrag"
else
    # 输出路径
    TRAIN_SFT="data/train_sft.json"
    TEST_SFT="data/test_sft.json"
    RL_TRAIN_DIR="data/rl_train"
fi

echo "=========================================="
echo "  数据生成: SFT + RL (训练集 + 测试集)"
echo "  MRAG 模式: ${USE_MRAG}"
echo "=========================================="

# ============================================================
# 1. 生成 SFT 训练集
# ============================================================
echo ""
echo ">>> [1/4] 生成 SFT 训练集..."
if [[ "${USE_MRAG}" == "true" ]]; then
    echo "    使用 MRAG 增强"
    python script/sft_data.py \
        --src data/train.json \
        --dst "${TRAIN_SFT}" \
        --split train \
        --use_mrag \
        --law_runfile "${LAW_RUNFILE_TRAIN}" \
        --case_runfile "${CASE_RUNFILE_TRAIN}" \
        --law_corpus "${LAW_CORPUS}" \
        --case_corpus "${CASE_CORPUS}"
else
    echo "    标准模式"
    python script/sft_data.py \
        --src data/train.json \
        --dst "${TRAIN_SFT}" \
        --split train
fi

# ============================================================
# 2. 生成 SFT 测试集
# ============================================================
echo ""
echo ">>> [2/4] 生成 SFT 测试集..."
if [[ "${USE_MRAG}" == "true" ]]; then
    echo "    使用 MRAG 增强"
    
    # 检查测试集检索结果是否存在
    if [[ ! -f "${LAW_RUNFILE_TEST}" ]]; then
        echo "    ⚠️  警告: 法条检索结果不存在: ${LAW_RUNFILE_TEST}"
        echo "    请先运行检索脚本生成测试集的检索结果"
        echo "    跳过 MRAG 测试集生成..."
    else
        python script/sft_data.py \
            --src data/test.json \
            --dst "${TEST_SFT}" \
            --split test \
            --use_mrag \
            --law_runfile "${LAW_RUNFILE_TEST}" \
            --case_runfile "${CASE_RUNFILE_TEST}" \
            --law_corpus "${LAW_CORPUS}" \
            --case_corpus "${CASE_CORPUS}"
    fi
else
    echo "    标准模式"
    python script/sft_data.py \
        --src data/test.json \
        --dst "${TEST_SFT}" \
        --split test
fi

# ============================================================
# 3. 生成 RL 训练集
# ============================================================
echo ""
echo ">>> [3/4] 生成 RL 训练集..."
if [[ "${USE_MRAG}" == "true" ]]; then
    echo "    使用 MRAG 增强"
    python script/rl_data.py \
        --src data/train.json \
        --dst_dir "${RL_TRAIN_DIR}" \
        --use_mrag \
        --law_runfile "${LAW_RUNFILE_TRAIN}" \
        --case_runfile "${CASE_RUNFILE_TRAIN}" \
        --law_corpus "${LAW_CORPUS}" \
        --case_corpus "${CASE_CORPUS}" \
        ${MAX_SAMPLES:+--max_samples "${MAX_SAMPLES}"} \
        --seed 42
else
    echo "    标准模式"
    python script/rl_data.py \
        --src data/train.json \
        --dst_dir "${RL_TRAIN_DIR}" \
        ${MAX_SAMPLES:+--max_samples "${MAX_SAMPLES}"} \
        --seed 42
fi

# ============================================================
# 4. 汇总
# ============================================================
echo ""
echo "=========================================="
echo "✅ 数据生成完成"
echo ""
echo "输出文件:"
echo "  SFT 训练集: ${TRAIN_SFT}"
echo "  SFT 测试集: ${TEST_SFT}"
echo "  RL 训练集:  ${RL_TRAIN_DIR}/train.jsonl"
echo "=========================================="
