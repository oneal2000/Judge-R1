# #!/bin/bash
# # K-Fold 混合策略训练流程
# #
# # 混合策略的核心思想：
# # 1. 用全部 2004 条数据训练一个高质量的 Dense Retriever（用于最终检索）
# # 2. 用 K-Fold 方式训练临时模型，只为 Reranker 挖掘无泄露的困难负例
# # 3. 合并所有 Fold 的检索结果，得到完整的 Reranker 训练数据
# #
# # 优点：
# # - Dense Retriever 保持最高质量（全量数据训练）
# # - Reranker 的困难负例无数据泄露


# set -e
# export TOKENIZERS_PARALLELISM=false
# PROJECT_ROOT="(set via paths.sh)"
# KFOLD_DATA_DIR="${PROJECT_ROOT}/mrag/kfold_data"
# KFOLD_OUTPUT_DIR="${PROJECT_ROOT}/mrag/kfold_output"

# # 模型配置
# BASE_MODEL="${ROBERTA_MODEL_PATH}"
# LAW_CORPUS="${PROJECT_ROOT}/data/law_corpus.jsonl"

# # K-Fold 配置
# NUM_FOLDS=4
# TOP_K=50  # 检索 Top-K 用于 Reranker 训练

# # GPU 设置
# export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-5}

# echo "=========================================="
# echo "  K-Fold 混合策略训练流程"
# echo "=========================================="
# echo "策略说明:"
# echo "  1. 用全部数据训练最终 Dense Retriever"
# echo "  2. 用 K-Fold 临时模型挖掘 Reranker 困难负例"
# echo "=========================================="
# echo "折数: ${NUM_FOLDS}"
# echo "基础模型: ${BASE_MODEL}"
# echo "检索 Top-K: ${TOP_K}"
# echo "=========================================="

# # ============================================
# # 步骤 1: 准备 K-Fold 数据
# # ============================================
# echo ""
# echo ">>> [步骤 1] 准备 K-Fold 数据..."
# python "${PROJECT_ROOT}/mrag/gen_kfold_data.py"

# mkdir -p "${KFOLD_OUTPUT_DIR}"

# # ============================================
# # 步骤 2: 用全部数据训练最终的 Dense Retriever
# # ============================================
# echo ""
# echo "=========================================="
# echo "  步骤 2: 训练最终 Dense Retriever（全量数据）"
# echo "=========================================="

# FINAL_MODEL="${PROJECT_ROOT}/output/law_retriever"
# FINAL_EMBEDDINGS="${PROJECT_ROOT}/mrag/retriever_output"

# # 生成全量训练数据
# echo "[2.1] 生成全量 Dense Retriever 训练数据..."
# python -c "
# import json
# import random
# from pathlib import Path

# base_dir = Path('${PROJECT_ROOT}')
# train_path = base_dir / 'data' / 'train.json'
# law_corpus_path = base_dir / 'data' / 'law_corpus.jsonl'
# output_path = base_dir / 'mrag' / 'retriever_data' / 'dense_train_full.jsonl'

# # 加载法条库
# law_dict = {}
# with open(law_corpus_path, 'r', encoding='utf-8') as f:
#     for line in f:
#         item = json.loads(line)
#         law_id = int(item['text_id'])
#         law_text = f\"{item['name']}：{item['text']}\"
#         law_dict[law_id] = law_text

# # 加载训练数据
# with open(train_path, 'r', encoding='utf-8') as f:
#     train_data = [json.loads(line) for line in f]

# # 生成训练样本
# all_law_ids = list(law_dict.keys())
# examples = []

# for entry in train_data:
#     fact = entry['text']
#     law_ids = entry['la']
    
#     positive_texts = [law_dict[lid] for lid in law_ids if lid in law_dict]
#     if not positive_texts:
#         continue
    
#     negative_ids = [lid for lid in all_law_ids if lid not in law_ids]
#     sampled_neg_ids = random.sample(negative_ids, min(7, len(negative_ids)))
#     negative_texts = [law_dict[lid] for lid in sampled_neg_ids]
    
#     for pos_text in positive_texts:
#         examples.append({
#             'query': fact,
#             'positives': [pos_text],
#             'negatives': negative_texts
#         })

# output_path.parent.mkdir(parents=True, exist_ok=True)
# with open(output_path, 'w', encoding='utf-8') as f:
#     for ex in examples:
#         f.write(json.dumps(ex, ensure_ascii=False) + '\n')

# print(f'生成 {len(examples)} 条全量训练样本')
# "

# echo "[2.2] 训练最终 Dense Retriever..."
# python "${PROJECT_ROOT}/mrag/train_retriever.py" \
#     --model_name_or_path "${BASE_MODEL}" \
#     --train_data "${PROJECT_ROOT}/mrag/retriever_data/dense_train_full.jsonl" \
#     --law_corpus "${LAW_CORPUS}" \
#     --output_dir "${FINAL_MODEL}" \
#     --num_epochs 5 \
#     --batch_size 16 \
#     --learning_rate 2e-5 \
#     --temperature 0.02 \
#     --num_hard_negs 7 \
#     --max_query_len 512 \
#     --max_passage_len 512

# echo "[2.3] 编码法条库..."
# python "${PROJECT_ROOT}/mrag/encode_and_retrieve.py" encode \
#     --model_path "${FINAL_MODEL}" \
#     --law_corpus "${LAW_CORPUS}" \
#     --output_dir "${FINAL_EMBEDDINGS}" \
#     --batch_size 64 \
#     --max_length 512

# echo "最终 Dense Retriever 训练完成！"

# # ============================================
# # 步骤 3: K-Fold 临时模型挖掘困难负例
# # ============================================
# echo ""
# echo "=========================================="
# echo "  步骤 3: K-Fold 挖掘 Reranker 困难负例"
# echo "=========================================="

# for fold_idx in $(seq 1 ${NUM_FOLDS}); do
#     echo ""
#     echo "--- Fold ${fold_idx}/${NUM_FOLDS} ---"
    
#     FOLD_DATA_DIR="${KFOLD_DATA_DIR}/fold_${fold_idx}"
#     FOLD_OUTPUT_DIR="${KFOLD_OUTPUT_DIR}/fold_${fold_idx}"
#     TEMP_MODEL="${FOLD_OUTPUT_DIR}/temp_model"
    
#     mkdir -p "${FOLD_OUTPUT_DIR}"
    
#     # 3.1 训练临时 Dense Retriever（只用于挖掘困难负例）
#     echo "[Fold ${fold_idx}] 训练临时 Dense Retriever..."
#     python "${PROJECT_ROOT}/mrag/train_retriever.py" \
#         --model_name_or_path "${BASE_MODEL}" \
#         --train_data "${FOLD_DATA_DIR}/dense_train.jsonl" \
#         --law_corpus "${LAW_CORPUS}" \
#         --output_dir "${TEMP_MODEL}" \
#         --num_epochs 3 \
#         --batch_size 16 \
#         --learning_rate 2e-5 \
#         --temperature 0.02 \
#         --num_hard_negs 7 \
#         --max_query_len 512 \
#         --max_passage_len 512
    
#     # 3.2 编码法条库
#     echo "[Fold ${fold_idx}] 编码法条库..."
#     python "${PROJECT_ROOT}/mrag/encode_and_retrieve.py" encode \
#         --model_path "${TEMP_MODEL}" \
#         --law_corpus "${LAW_CORPUS}" \
#         --output_dir "${FOLD_OUTPUT_DIR}" \
#         --batch_size 64 \
#         --max_length 512
    
#     # 3.3 检索当前 Fold 的数据（挖掘困难负例）
#     echo "[Fold ${fold_idx}] 检索困难负例..."
#     python "${PROJECT_ROOT}/mrag/encode_and_retrieve.py" retrieve \
#         --model_path "${TEMP_MODEL}" \
#         --embeddings_dir "${FOLD_OUTPUT_DIR}" \
#         --queries "${FOLD_DATA_DIR}/queries.jsonl" \
#         --output_file "${FOLD_OUTPUT_DIR}/runfile.tsv" \
#         --top_k ${TOP_K} \
#         --batch_size 32
    
#     # 删除临时模型节省空间
#     echo "[Fold ${fold_idx}] 清理临时模型..."
#     rm -rf "${TEMP_MODEL}"
    
#     echo "[Fold ${fold_idx}] 完成！"
# done

# # ============================================
# # 步骤 4: 合并所有 Fold 的检索结果
# # ============================================
# echo ""
# echo "=========================================="
# echo "  步骤 4: 合并检索结果"
# echo "=========================================="

# MERGED_RUNFILE="${PROJECT_ROOT}/mrag/retriever_output/law_runfile_train_kfold.tsv"
# MERGED_SOURCE_DATA="${PROJECT_ROOT}/mrag/retriever_data/reranker_source_data_kfold.json"

# # 合并检索结果
# echo "合并 runfile..."
# cat /dev/null > "${MERGED_RUNFILE}"
# for fold_idx in $(seq 1 ${NUM_FOLDS}); do
#     cat "${KFOLD_OUTPUT_DIR}/fold_${fold_idx}/runfile.tsv" >> "${MERGED_RUNFILE}"
# done
# echo "  合并后的 runfile: ${MERGED_RUNFILE}"
# echo "  总行数: $(wc -l < "${MERGED_RUNFILE}")"

# # 合并源数据
# echo "合并源数据..."
# cat /dev/null > "${MERGED_SOURCE_DATA}"
# for fold_idx in $(seq 1 ${NUM_FOLDS}); do
#     cat "${KFOLD_DATA_DIR}/fold_${fold_idx}/source_data.json" >> "${MERGED_SOURCE_DATA}"
# done
# echo "  合并后的源数据: ${MERGED_SOURCE_DATA}"
# echo "  总行数: $(wc -l < "${MERGED_SOURCE_DATA}")"

# # ============================================
# # 完成
# # ============================================
# echo ""
# echo "=========================================="
# echo "  K-Fold 混合策略训练完成！"
# echo "=========================================="
# echo ""
# echo "输出文件:"
# echo "  - 最终 Dense Retriever: ${FINAL_MODEL}"
# echo "  - 困难负例 runfile: ${MERGED_RUNFILE}"
# echo "  - 源数据: ${MERGED_SOURCE_DATA}"
# echo ""
# echo "优势说明:"
# echo "  ✅ Dense Retriever 用全量数据训练，质量最高"
# echo "  ✅ Reranker 困难负例无数据泄露，质量更高"
# echo ""
# echo "下一步: bash bash/reranker/train_reranker_kfold.sh"


#!/bin/bash
# K-Fold 混合策略训练流程
#
# 混合策略的核心思想：
# 1. 用全部 2004 条数据训练一个高质量的 Dense Retriever（用于最终检索）
# 2. 用 K-Fold 方式训练临时模型，只为 Reranker 挖掘无泄露的困难负例
# 3. 合并所有 Fold 的检索结果，得到完整的 Reranker 训练数据
#
# 优点：
# - Dense Retriever 保持最高质量（全量数据训练）
# - Reranker 的困难负例无数据泄露


set -e
export TOKENIZERS_PARALLELISM=false
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../paths.sh"
KFOLD_DATA_DIR="${PROJECT_ROOT}/mrag/kfold_data"
KFOLD_OUTPUT_DIR="${PROJECT_ROOT}/mrag/kfold_output"

# 模型配置
BASE_MODEL="${ROBERTA_MODEL_PATH}"
LAW_CORPUS="${PROJECT_ROOT}/data/law_corpus.jsonl"

# K-Fold 配置
NUM_FOLDS=4
TOP_K=50  # 检索 Top-K 用于 Reranker 训练

# GPU 设置
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-7}

echo "=========================================="
echo "  K-Fold 混合策略训练流程"
echo "=========================================="
echo "策略说明:"
echo "  1. 用全部数据训练最终 Dense Retriever"
echo "  2. 用 K-Fold 临时模型挖掘 Reranker 困难负例"
echo "=========================================="
echo "折数: ${NUM_FOLDS}"
echo "基础模型: ${BASE_MODEL}"
echo "检索 Top-K: ${TOP_K}"
echo "=========================================="

# ============================================
# 步骤 1: 准备 K-Fold 数据
# ============================================
echo ""
echo ">>> [步骤 1] 准备 K-Fold 数据..."
python "${PROJECT_ROOT}/mrag/gen_kfold_data.py"

mkdir -p "${KFOLD_OUTPUT_DIR}"

# ============================================
# 步骤 2: 用全部数据训练最终的 Dense Retriever
# ============================================
echo ""
echo "=========================================="
echo "  步骤 2: 训练最终 Dense Retriever（全量数据）"
echo "=========================================="

FINAL_MODEL="${PROJECT_ROOT}/output/law_retriever"
FINAL_EMBEDDINGS="${PROJECT_ROOT}/mrag/retriever_output"

# 生成全量训练数据
echo "[2.1] 生成全量 Dense Retriever 训练数据..."
python -c "
import json
import random
from pathlib import Path

base_dir = Path('${PROJECT_ROOT}')
train_path = base_dir / 'data' / 'train.json'
law_corpus_path = base_dir / 'data' / 'law_corpus.jsonl'
output_path = base_dir / 'mrag' / 'retriever_data' / 'dense_train_full.jsonl'

# 加载法条库
law_dict = {}
with open(law_corpus_path, 'r', encoding='utf-8') as f:
    for line in f:
        item = json.loads(line)
        law_id = int(item['text_id'])
        law_text = f\"{item['name']}：{item['text']}\"
        law_dict[law_id] = law_text

# 加载训练数据
with open(train_path, 'r', encoding='utf-8') as f:
    train_data = [json.loads(line) for line in f]

# 生成训练样本
all_law_ids = list(law_dict.keys())
examples = []

for entry in train_data:
    fact = entry['text']
    law_ids = entry['la']
    
    positive_texts = [law_dict[lid] for lid in law_ids if lid in law_dict]
    if not positive_texts:
        continue
    
    negative_ids = [lid for lid in all_law_ids if lid not in law_ids]
    sampled_neg_ids = random.sample(negative_ids, min(7, len(negative_ids)))
    negative_texts = [law_dict[lid] for lid in sampled_neg_ids]
    
    for pos_text in positive_texts:
        examples.append({
            'query': fact,
            'positives': [pos_text],
            'negatives': negative_texts
        })

output_path.parent.mkdir(parents=True, exist_ok=True)
with open(output_path, 'w', encoding='utf-8') as f:
    for ex in examples:
        f.write(json.dumps(ex, ensure_ascii=False) + '\n')

print(f'生成 {len(examples)} 条全量训练样本')
"

echo "[2.2] 训练最终 Dense Retriever..."
python "${PROJECT_ROOT}/mrag/train_retriever.py" \
    --model_name_or_path "${BASE_MODEL}" \
    --train_data "${PROJECT_ROOT}/mrag/retriever_data/dense_train_full.jsonl" \
    --law_corpus "${LAW_CORPUS}" \
    --output_dir "${FINAL_MODEL}" \
    --num_epochs 5 \
    --batch_size 16 \
    --learning_rate 2e-5 \
    --temperature 0.02 \
    --num_hard_negs 7 \
    --max_query_len 512 \
    --max_passage_len 512

echo "[2.3] 编码法条库..."
python "${PROJECT_ROOT}/mrag/encode_and_retrieve.py" encode \
    --model_path "${FINAL_MODEL}" \
    --law_corpus "${LAW_CORPUS}" \
    --output_dir "${FINAL_EMBEDDINGS}" \
    --batch_size 64 \
    --max_length 512

echo "最终 Dense Retriever 训练完成！"

# ============================================
# 步骤 3: K-Fold 临时模型挖掘困难负例
# ============================================
echo ""
echo "=========================================="
echo "  步骤 3: K-Fold 挖掘 Reranker 困难负例"
echo "=========================================="
echo "【优化】临时模型训练 5 epochs，与最终模型一致"
echo "=========================================="

for fold_idx in $(seq 1 ${NUM_FOLDS}); do
    echo ""
    echo "--- Fold ${fold_idx}/${NUM_FOLDS} ---"
    
    FOLD_DATA_DIR="${KFOLD_DATA_DIR}/fold_${fold_idx}"
    FOLD_OUTPUT_DIR="${KFOLD_OUTPUT_DIR}/fold_${fold_idx}"
    TEMP_MODEL="${FOLD_OUTPUT_DIR}/temp_model"
    
    mkdir -p "${FOLD_OUTPUT_DIR}"
    
    # 3.1 训练临时 Dense Retriever（提高训练质量）
    echo "[Fold ${fold_idx}] 训练临时 Dense Retriever (5 epochs)..."
    python "${PROJECT_ROOT}/mrag/train_retriever.py" \
        --model_name_or_path "${BASE_MODEL}" \
        --train_data "${FOLD_DATA_DIR}/dense_train.jsonl" \
        --law_corpus "${LAW_CORPUS}" \
        --output_dir "${TEMP_MODEL}" \
        --num_epochs 5 \
        --batch_size 16 \
        --learning_rate 2e-5 \
        --temperature 0.02 \
        --num_hard_negs 10 \
        --max_query_len 512 \
        --max_passage_len 512
    
    # 3.2 编码法条库
    echo "[Fold ${fold_idx}] 编码法条库..."
    python "${PROJECT_ROOT}/mrag/encode_and_retrieve.py" encode \
        --model_path "${TEMP_MODEL}" \
        --law_corpus "${LAW_CORPUS}" \
        --output_dir "${FOLD_OUTPUT_DIR}" \
        --batch_size 64 \
        --max_length 512
    
    # 3.3 检索当前 Fold 的数据（挖掘困难负例）
    echo "[Fold ${fold_idx}] 检索困难负例..."
    python "${PROJECT_ROOT}/mrag/encode_and_retrieve.py" retrieve \
        --model_path "${TEMP_MODEL}" \
        --embeddings_dir "${FOLD_OUTPUT_DIR}" \
        --queries "${FOLD_DATA_DIR}/queries.jsonl" \
        --output_file "${FOLD_OUTPUT_DIR}/runfile.tsv" \
        --top_k ${TOP_K} \
        --batch_size 32
    
    # 删除临时模型节省空间
    echo "[Fold ${fold_idx}] 清理临时模型..."
    rm -rf "${TEMP_MODEL}"
    
    echo "[Fold ${fold_idx}] 完成！"
done

# ============================================
# 步骤 4: 合并所有 Fold 的检索结果
# ============================================
echo ""
echo "=========================================="
echo "  步骤 4: 合并检索结果"
echo "=========================================="

MERGED_RUNFILE="${PROJECT_ROOT}/mrag/retriever_output/law_runfile_train_kfold.tsv"
MERGED_SOURCE_DATA="${PROJECT_ROOT}/mrag/retriever_data/reranker_source_data_kfold.json"

# 合并检索结果
echo "合并 runfile..."
cat /dev/null > "${MERGED_RUNFILE}"
for fold_idx in $(seq 1 ${NUM_FOLDS}); do
    cat "${KFOLD_OUTPUT_DIR}/fold_${fold_idx}/runfile.tsv" >> "${MERGED_RUNFILE}"
done
echo "  合并后的 runfile: ${MERGED_RUNFILE}"
echo "  总行数: $(wc -l < "${MERGED_RUNFILE}")"

# 合并源数据
echo "合并源数据..."
cat /dev/null > "${MERGED_SOURCE_DATA}"
for fold_idx in $(seq 1 ${NUM_FOLDS}); do
    cat "${KFOLD_DATA_DIR}/fold_${fold_idx}/source_data.json" >> "${MERGED_SOURCE_DATA}"
done
echo "  合并后的源数据: ${MERGED_SOURCE_DATA}"
echo "  总行数: $(wc -l < "${MERGED_SOURCE_DATA}")"

# ============================================
# 步骤 5: 清理 K-Fold 临时文件（保留 runfile）
# ============================================
echo ""
echo "=========================================="
echo "  步骤 5: 清理 K-Fold 临时文件"
echo "=========================================="

for fold_idx in $(seq 1 ${NUM_FOLDS}); do
    FOLD_OUTPUT_DIR="${KFOLD_OUTPUT_DIR}/fold_${fold_idx}"
    
    # 删除临时模型目录（可能是 temp_model 或 model）
    if [[ -d "${FOLD_OUTPUT_DIR}/temp_model" ]]; then
        echo "[Fold ${fold_idx}] 删除临时模型 temp_model..."
        rm -rf "${FOLD_OUTPUT_DIR}/temp_model"
    fi
    if [[ -d "${FOLD_OUTPUT_DIR}/model" ]]; then
        echo "[Fold ${fold_idx}] 删除临时模型 model..."
        rm -rf "${FOLD_OUTPUT_DIR}/model"
    fi
    
    # 删除 embeddings（已用于检索，不再需要）
    if [[ -f "${FOLD_OUTPUT_DIR}/law_embeddings.npy" ]]; then
        echo "[Fold ${fold_idx}] 删除临时 embeddings..."
        rm -f "${FOLD_OUTPUT_DIR}/law_embeddings.npy"
        rm -f "${FOLD_OUTPUT_DIR}/law_ids.json"
    fi
done

echo "清理完成！只保留 runfile.tsv 用于后续训练"

# ============================================
# 完成
# ============================================
echo ""
echo "=========================================="
echo "  K-Fold 混合策略训练完成！"
echo "=========================================="
echo ""
echo "输出文件:"
echo "  - 最终 Dense Retriever: ${FINAL_MODEL}"
echo "  - 困难负例 runfile: ${MERGED_RUNFILE}"
echo "  - 源数据: ${MERGED_SOURCE_DATA}"
echo ""
echo "优势说明:"
echo "  ✅ Dense Retriever 用全量数据训练，质量最高"
echo "  ✅ Reranker 困难负例无数据泄露，质量更高"
echo "  ✅ 临时模型训练 5 epochs，与最终模型分布更一致"
echo ""
echo "下一步: bash bash/reranker/train_reranker_kfold.sh"