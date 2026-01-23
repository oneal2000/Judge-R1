#!/bin/bash
# 使用 Reranker 对测试集进行重排序

set -e

PROJECT_ROOT="/data-share/chenxuanyi/internship/JuDGE_RL"
RERANKER_DIR="${PROJECT_ROOT}/reranker"

# 预训练模型路径
PRETRAINED_MODEL="/data-share/chenxuanyi/LLM/chinese-roberta-wwm"

# 训练好的 Reranker 模型
RERANKER_MODEL="${RERANKER_DIR}/train"

# GPU 设置
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

cd "${RERANKER_DIR}"

echo "=========================================="
echo "  使用 Reranker 重排序测试集"
echo "=========================================="

# 对测试集进行重排序
PART="test"
RUNFILE="${PROJECT_ROOT}/mrag/retriever_output/law_runfile_${PART}.tsv"

echo "[1/3] 构建推理数据..."
mkdir -p result/${PART}
python helpers/topk_text_2_json.py \
    --tokenizer "${PRETRAINED_MODEL}" \
    --save_to result/${PART}/all.json \
    --generate_id_to result/${PART}/ids.tsv \
    --truncate 200 \
    --q_truncate 512 \
    --qry_file "${PROJECT_ROOT}/data/${PART}.json" \
    --law_data_file "${PROJECT_ROOT}/data/law_corpus.jsonl" \
    --run_file_train "${RUNFILE}"

echo "推理数据构建完成"

echo "[2/3] Reranker 推理..."
mkdir -p score/${PART}
python run_reranker.py \
    --output_dir score/${PART} \
    --model_name_or_path "${RERANKER_MODEL}" \
    --tokenizer_name "${PRETRAINED_MODEL}" \
    --do_predict \
    --max_len 512 \
    --fp16 \
    --per_device_eval_batch_size 128 \
    --dataloader_num_workers 32 \
    --pred_path result/${PART}/all.json \
    --pred_id_file result/${PART}/ids.tsv \
    --rank_score_path score/${PART}/score.txt

echo "推理完成"

echo "[3/3] 转换为 TREC 格式..."
python helpers/score_to_tein.py \
    --score_file score/${PART}/score.txt \
    --reranker_run_file score/${PART}/reranker_run_file_${PART} \
    --part ${PART}

# 复制重排序结果到 retriever_output 目录
cp score/${PART}/reranker_run_file_${PART} "${PROJECT_ROOT}/mrag/retriever_output/law_runfile_reranked_${PART}.tsv"

echo ""
echo "重排序完成！结果保存到:"
echo "  - ${RERANKER_DIR}/score/${PART}/reranker_run_file_${PART}"
echo "  - ${PROJECT_ROOT}/mrag/retriever_output/law_runfile_reranked_${PART}.tsv"