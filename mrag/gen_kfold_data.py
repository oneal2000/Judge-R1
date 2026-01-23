"""
K-Fold 交叉挖掘数据准备

策略：
- 将训练集分成 K 份（默认 4 份）
- 每个 Fold：用其他 K-1 份训练 Dense Retriever，检索剩余 1 份
- 合并所有 Fold 的检索结果，得到完整的 Reranker 训练数据

优点：
- Reranker 仍有完整的训练数据（所有样本）
- 每条数据的困难负例都来自"未见过它的" Dense Retriever
- 避免数据泄露
"""
import json
import random
import os
from pathlib import Path
from typing import List, Dict, Tuple

# K-Fold 配置
NUM_FOLDS = 4  # 默认 4 折

def load_train_data(train_path: str) -> List[Dict]:
    """加载训练数据"""
    with open(train_path, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]
    return data

def load_law_corpus(law_corpus_path: str) -> Dict[int, str]:
    """加载法条库"""
    law_dict = {}
    with open(law_corpus_path, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            law_id = int(item['text_id'])
            law_text = f"{item['name']}：{item['text']}"
            law_dict[law_id] = law_text
    return law_dict

def split_into_folds(data: List[Dict], num_folds: int, seed: int = 42) -> List[List[Dict]]:
    """
    将数据分成 K 份
    
    Args:
        data: 完整训练数据
        num_folds: 折数
        seed: 随机种子（保证可复现）
    
    Returns:
        folds: K 个数据列表
    """
    random.seed(seed)
    indices = list(range(len(data)))
    random.shuffle(indices)
    
    folds = []
    fold_size = len(data) // num_folds
    
    for i in range(num_folds):
        start = i * fold_size
        # 最后一折包含所有剩余数据
        end = start + fold_size if i < num_folds - 1 else len(data)
        fold_indices = indices[start:end]
        folds.append([data[idx] for idx in fold_indices])
    
    return folds

def generate_dense_train_data(train_data: List[Dict], law_dict: Dict[int, str], num_negatives: int = 7) -> List[Dict]:
    """
    生成 Dense Retriever 训练数据
    格式: {"query": "案件事实", "positives": ["法条1"], "negatives": ["法条2", ...]}
    """
    all_law_ids = list(law_dict.keys())
    examples = []
    
    for entry in train_data:
        fact = entry['text']
        law_ids = entry['la']
        
        positive_texts = []
        for law_id in law_ids:
            if law_id in law_dict:
                positive_texts.append(law_dict[law_id])
        
        if not positive_texts:
            continue
        
        negative_ids = [lid for lid in all_law_ids if lid not in law_ids]
        sampled_neg_ids = random.sample(negative_ids, min(num_negatives, len(negative_ids)))
        negative_texts = [law_dict[lid] for lid in sampled_neg_ids]
        
        for pos_text in positive_texts:
            examples.append({
                'query': fact,
                'positives': [pos_text],
                'negatives': negative_texts
            })
    
    return examples

def generate_queries(data: List[Dict], output_queries: str, output_qrels: str):
    """生成查询文件和标准答案"""
    with open(output_queries, 'w', encoding='utf-8') as fq, \
         open(output_qrels, 'w', encoding='utf-8') as fr:
        for entry in data:
            # 查询文件
            query_item = {
                'query_id': entry['text_id'],
                'query': entry['text']
            }
            fq.write(json.dumps(query_item, ensure_ascii=False) + '\n')
            
            # qrels 文件
            query_id = entry['text_id']
            for law_id in entry['la']:
                fr.write(f"{query_id}\t0\t{law_id}\t1\n")

def main():
    base_dir = Path(__file__).parent.parent
    train_path = base_dir / 'data' / 'train.json'
    law_corpus_path = base_dir / 'data' / 'law_corpus.jsonl'
    output_dir = base_dir / 'mrag' / 'kfold_data'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print(f"  K-Fold 交叉挖掘数据准备 (K={NUM_FOLDS})")
    print("=" * 60)
    
    # 加载数据
    print(f"\n[1/4] 加载数据...")
    train_data = load_train_data(train_path)
    law_dict = load_law_corpus(law_corpus_path)
    print(f"  训练数据: {len(train_data)} 条")
    print(f"  法条库: {len(law_dict)} 条")
    
    # 分割成 K 份
    print(f"\n[2/4] 分割数据为 {NUM_FOLDS} 份...")
    folds = split_into_folds(train_data, NUM_FOLDS)
    for i, fold in enumerate(folds):
        print(f"  Fold {i+1}: {len(fold)} 条")
    
    # 为每个 Fold 生成数据
    print(f"\n[3/4] 为每个 Fold 生成训练和检索数据...")
    
    for fold_idx in range(NUM_FOLDS):
        fold_dir = output_dir / f'fold_{fold_idx+1}'
        fold_dir.mkdir(parents=True, exist_ok=True)
        
        # 训练数据 = 除当前 Fold 外的所有数据
        train_folds = [folds[i] for i in range(NUM_FOLDS) if i != fold_idx]
        train_data_for_dense = [item for fold in train_folds for item in fold]
        
        # 检索数据 = 当前 Fold
        retrieval_data = folds[fold_idx]
        
        print(f"\n  --- Fold {fold_idx+1} ---")
        print(f"  Dense Retriever 训练数据: {len(train_data_for_dense)} 条")
        print(f"  检索数据（生成困难负例）: {len(retrieval_data)} 条")
        
        # 生成 Dense Retriever 训练数据
        dense_train = generate_dense_train_data(train_data_for_dense, law_dict)
        dense_train_path = fold_dir / 'dense_train.jsonl'
        with open(dense_train_path, 'w', encoding='utf-8') as f:
            for ex in dense_train:
                f.write(json.dumps(ex, ensure_ascii=False) + '\n')
        print(f"  Dense 训练样本: {len(dense_train)} 条 -> {dense_train_path}")
        
        # 生成检索查询文件（用于挖掘困难负例）
        queries_path = fold_dir / 'queries.jsonl'
        qrels_path = fold_dir / 'qrels.tsv'
        generate_queries(retrieval_data, queries_path, qrels_path)
        print(f"  检索查询: {len(retrieval_data)} 条 -> {queries_path}")
        
        # 保存原始数据（用于后续合并）
        source_data_path = fold_dir / 'source_data.json'
        with open(source_data_path, 'w', encoding='utf-8') as f:
            for entry in retrieval_data:
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    
    # 生成测试集查询（用于最终评测）
    print(f"\n[4/4] 生成测试集查询文件...")
    test_path = base_dir / 'data' / 'test.json'
    test_data = load_train_data(test_path)
    test_output_dir = base_dir / 'mrag' / 'retriever_data'
    test_output_dir.mkdir(parents=True, exist_ok=True)
    generate_queries(test_data, test_output_dir / 'queries_test.jsonl', test_output_dir / 'qrels_test.tsv')
    print(f"  测试集查询: {len(test_data)} 条")
    
    print("\n" + "=" * 60)
    print("  数据准备完成！")
    print("=" * 60)
    print(f"\n输出目录: {output_dir}")
    print(f"\n每个 Fold 包含:")
    print(f"  - dense_train.jsonl: Dense Retriever 训练数据")
    print(f"  - queries.jsonl: 检索查询（用于挖掘困难负例）")
    print(f"  - qrels.tsv: 标准答案")
    print(f"  - source_data.json: 原始数据")
    print(f"\n下一步: bash bash/retriever/kfold_train.sh")

if __name__ == '__main__':
    main()
