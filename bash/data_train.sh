#!/usr/bin/env bash
set -euo pipefail

cd /data-share/chenxuanyi/internship/JuDGE_RL

# ============================================================
# 数据生成脚本
#
# 生成 SFT 训练/测试数据 和 RL 训练数据。
# MRAG 模式下，prompt 中包含检索到的相关法条和相似案例。
#
# 数据来源（MRAG=true 时）:
#   法条:
#     训练集 → Dense Retriever K-fold 结果 (避免数据泄露)
#     测试集 → Agent+MRAG 输出级融合结果 (最优检索)
#             fallback: Reranker 结果 → Dense 结果
#   案例:
#     训练集 → 无（Dense Retriever 仅对测试集做了案例检索）
#     测试集 → Dense Retriever 案例检索 (top-5)
#
# 前置步骤（生成检索结果）:
#   1. bash bash/retriever/retrieve.sh          # Dense 检索 + 案例检索
#   2. bash bash/reranker/run_reranker.sh       # Reranker 重排
#   3. bash bash/agent/eval_ablation.sh         # Agent 消融 (生成 #1 MRAG + #7 Agent)
#   4. bash bash/agent/run_hybrid_agent.sh      # 输出级融合 (生成 fused_rrf.tsv)
#
# 用法:
#   bash bash/data_train.sh                # 标准模式
#   USE_MRAG=true bash bash/data_train.sh  # MRAG 模式（使用融合法条 + 案例）
#
# 输出:
#   标准模式:
#     - data/train_sft.json       (SFT 训练集)
#     - data/test_sft.json        (SFT 测试集)
#     - data/rl_train/train.jsonl (RL 训练集)
#
#   MRAG 模式:
#     - data/train_sft_mrag.json       (SFT 训练集, 含法条)
#     - data/test_sft_mrag.json        (SFT 测试集, 含法条+案例)
#     - data/rl_train_mrag/train.jsonl (RL 训练集, 含法条)
# ============================================================

USE_MRAG=${USE_MRAG:-false}

# 语料库路径
LAW_CORPUS="data/law_corpus.jsonl"
CASE_CORPUS="data/case_corpus.jsonl"

# 输出路径
if [[ "${USE_MRAG}" == "true" ]]; then
    TRAIN_SFT="data/train_sft_mrag.json"
    TEST_SFT="data/test_sft_mrag.json"
    RL_TRAIN_DIR="data/rl_train_mrag"
else
    TRAIN_SFT="data/train_sft.json"
    TEST_SFT="data/test_sft.json"
    RL_TRAIN_DIR="data/rl_train"
fi

# ============================================================
# MRAG 检索结果路径
# ============================================================
if [[ "${USE_MRAG}" == "true" ]]; then

    # --- 训练集法条 ---
    # 使用 K-fold Dense Retriever 结果（避免训练集数据泄露）
    # 由 bash/retriever/kfold_train_retriever.sh 生成
    if [[ -f "mrag/retriever_output/law_runfile_train_kfold.tsv" ]]; then
        LAW_RUNFILE_TRAIN="mrag/retriever_output/law_runfile_train_kfold.tsv"
    elif [[ -f "mrag/retriever_output/law_runfile_train.tsv" ]]; then
        LAW_RUNFILE_TRAIN="mrag/retriever_output/law_runfile_train.tsv"
    else
        echo "❌ 错误: 训练集法条检索结果不存在"
        echo "   请先运行: bash bash/retriever/retrieve.sh"
        exit 1
    fi

    # --- 训练集案例 ---
    # 训练集暂无案例检索结果（Dense Retriever 仅对测试集做了案例检索）
    CASE_RUNFILE_TRAIN=""

    # --- 测试集法条 ---
    # 优先级: 融合结果 > Reranker > Dense
    # 融合结果由 bash/agent/run_hybrid_agent.sh 生成
    if [[ -f "mrag/retriever_output/fused_hybrid_rrf.tsv" ]]; then
        LAW_RUNFILE_TEST="mrag/retriever_output/fused_hybrid_rrf.tsv"
    elif [[ -f "mrag/retriever_output/fused_rrf.tsv" ]]; then
        LAW_RUNFILE_TEST="mrag/retriever_output/fused_rrf.tsv"
    elif [[ -f "mrag/retriever_output/ablation_both_rl.tsv" ]]; then
        # Agent Both RL（次优）
        LAW_RUNFILE_TEST="mrag/retriever_output/ablation_both_rl.tsv"
    elif [[ -f "mrag/retriever_output/law_runfile_reranked_test.tsv" ]]; then
        LAW_RUNFILE_TEST="mrag/retriever_output/law_runfile_reranked_test.tsv"
    elif [[ -f "mrag/retriever_output/law_runfile_test.tsv" ]]; then
        LAW_RUNFILE_TEST="mrag/retriever_output/law_runfile_test.tsv"
    else
        echo "❌ 错误: 测试集法条检索结果不存在"
        echo "   请先运行: bash bash/agent/eval_ablation.sh"
        exit 1
    fi

    # --- 测试集案例 ---
    # 由 bash/retriever/retrieve.sh 的 retrieve_case 步骤生成
    CASE_RUNFILE_TEST="mrag/retriever_output/case_runfile_test.tsv"

    echo "=========================================="
    echo "  数据生成: SFT + RL (MRAG 模式)"
    echo "=========================================="
    echo "  训练集法条: ${LAW_RUNFILE_TRAIN}"
    echo "  训练集案例: ${CASE_RUNFILE_TRAIN:-无}"
    echo "  测试集法条: ${LAW_RUNFILE_TEST}"
    echo "  测试集案例: ${CASE_RUNFILE_TEST}"
    echo "=========================================="
else
    echo "=========================================="
    echo "  数据生成: SFT + RL (标准模式)"
    echo "=========================================="
fi

# ============================================================
# 1. 生成 SFT 训练集
# ============================================================
echo ""
echo ">>> [1/4] 生成 SFT 训练集..."
if [[ "${USE_MRAG}" == "true" ]]; then
    echo "    使用 MRAG 增强"
    TRAIN_CMD=(
        python script/sft_data.py
        --src data/train.json
        --dst "${TRAIN_SFT}"
        --split train
        --use_mrag
        --law_runfile "${LAW_RUNFILE_TRAIN}"
        --law_corpus "${LAW_CORPUS}"
    )
    # 训练集有案例检索结果时才加入
    if [[ -n "${CASE_RUNFILE_TRAIN}" && -f "${CASE_RUNFILE_TRAIN}" ]]; then
        TRAIN_CMD+=(--case_runfile "${CASE_RUNFILE_TRAIN}" --case_corpus "${CASE_CORPUS}")
    else
        echo "    (训练集无案例检索，仅使用法条)"
    fi
    "${TRAIN_CMD[@]}"
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

    if [[ ! -f "${LAW_RUNFILE_TEST}" ]]; then
        echo "    ⚠️  警告: 法条检索结果不存在: ${LAW_RUNFILE_TEST}"
        echo "    跳过 MRAG 测试集生成..."
    else
        TEST_CMD=(
            python script/sft_data.py
            --src data/test.json
            --dst "${TEST_SFT}"
            --split test
            --use_mrag
            --law_runfile "${LAW_RUNFILE_TEST}"
            --law_corpus "${LAW_CORPUS}"
        )
        # 测试集有案例检索结果时加入
        if [[ -f "${CASE_RUNFILE_TEST}" ]]; then
            TEST_CMD+=(--case_runfile "${CASE_RUNFILE_TEST}" --case_corpus "${CASE_CORPUS}")
        else
            echo "    (测试集无案例检索，仅使用法条)"
        fi
        "${TEST_CMD[@]}"
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
    RL_CMD=(
        python script/rl_data.py
        --src data/train.json
        --dst_dir "${RL_TRAIN_DIR}"
        --use_mrag
        --law_runfile "${LAW_RUNFILE_TRAIN}"
        --law_corpus "${LAW_CORPUS}"
        --seed 42
    )
    # 训练集有案例检索结果时加入
    if [[ -n "${CASE_RUNFILE_TRAIN}" && -f "${CASE_RUNFILE_TRAIN}" ]]; then
        RL_CMD+=(--case_runfile "${CASE_RUNFILE_TRAIN}" --case_corpus "${CASE_CORPUS}")
    fi
    if [[ -n "${MAX_SAMPLES:-}" ]]; then
        RL_CMD+=(--max_samples "${MAX_SAMPLES}")
    fi
    "${RL_CMD[@]}"
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
if [[ "${USE_MRAG}" == "true" ]]; then
    echo ""
    echo "检索来源:"
    echo "  训练集法条: ${LAW_RUNFILE_TRAIN}"
    echo "  测试集法条: ${LAW_RUNFILE_TEST}"
    [[ -f "${CASE_RUNFILE_TEST}" ]] && echo "  测试集案例: ${CASE_RUNFILE_TEST}"
fi
echo "=========================================="
