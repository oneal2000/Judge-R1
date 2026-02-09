# #!/bin/bash
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../paths.sh"
RERANKER_DIR="${PROJECT_ROOT}/reranker"

PRETRAINED_MODEL="${ROBERTA_MODEL_PATH}"

# K-Fold 合并后的数据
RUNFILE="${PROJECT_ROOT}/mrag/retriever_output/law_runfile_train_kfold.tsv"
RERANKER_SOURCE_DATA="${PROJECT_ROOT}/mrag/retriever_data/reranker_source_data_kfold.json"

# GPU 设置
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

cd "${RERANKER_DIR}"

echo "=========================================="
echo "  训练 Reranker（K-Fold 数据）"
echo "=========================================="
echo "基础模型: ${PRETRAINED_MODEL}"
echo "数据策略: K-Fold 交叉挖掘（完整 2004 条数据）"
echo "检索结果: ${RUNFILE}"
echo "源数据: ${RERANKER_SOURCE_DATA}"
echo "=========================================="

# 检查必要文件是否存在
if [ ! -f "${RUNFILE}" ]; then
    echo "错误: 检索结果文件不存在: ${RUNFILE}"
    echo "请先运行: bash bash/retriever/kfold_hybrid_train.sh"
    exit 1
fi

if [ ! -f "${RERANKER_SOURCE_DATA}" ]; then
    echo "错误: 源数据文件不存在: ${RERANKER_SOURCE_DATA}"
    echo "请先运行: bash bash/retriever/kfold_hybrid_train.sh"
    exit 1
fi

# 统计数据量
echo ""
echo "数据统计:"
echo "  检索结果行数: $(wc -l < "${RUNFILE}")"
echo "  源数据行数: $(wc -l < "${RERANKER_SOURCE_DATA}")"
echo ""

# 1. 构建训练数据（增加负例数量）
echo "[1/2] 构建训练数据..."
python helpers/build_train_from_ranking.py \
    --tokenizer_name "${PRETRAINED_MODEL}" \
    --random \
    --run_file_train "${RUNFILE}" \
    --output_train_file reranker_train_kfold.json \
    --qry_train_file "${RERANKER_SOURCE_DATA}" \
    --law_data_file "${PROJECT_ROOT}/data/law_corpus.jsonl" \
    --truncate 200 \
    --q_truncate 512 \
    --sample_from_top 50 \
    --n_sample 40

echo "训练数据构建完成"

# 2. 训练模型（增强训练）
echo "[2/2] 训练 Reranker 模型..."
python run_reranker.py \
    --output_dir train \
    --model_name_or_path "${PRETRAINED_MODEL}" \
    --do_train \
    --train_path reranker_train_kfold.json \
    --max_len 512 \
    --fp16 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --train_group_size 20 \
    --num_train_epochs 8 \
    --learning_rate 2e-5 \
    --warmup_ratio 0.1 \
    --weight_decay 0.01 \
    --overwrite_output_dir \
    --dataloader_num_workers 8 \
    --logging_steps 10 \
    --save_strategy "no"

echo ""
echo "=========================================="
echo "  Reranker 训练完成！"
echo "=========================================="
echo "模型保存到: ${RERANKER_DIR}/train"
echo ""
echo "下一步:"
echo "  1. 对测试集重排序: bash bash/reranker/run_reranker.sh"
echo "  2. 评测检索性能: bash bash/retriever/eval_retriever.sh"