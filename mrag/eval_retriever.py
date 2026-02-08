"""
评测检索模型性能

核心指标（针对法律文书生成任务）:
- Recall@K: 最重要，确保关键法条被召回
- Precision@K: 次重要，控制噪声

注：MRR 适用于单答案检索任务，对于本任务中的多法条召回场景意义有限，故不作为主要评估指标。
"""
import json
import argparse
from collections import defaultdict

def load_qrels(qrels_path):
    """
    加载标准答案
    支持两种格式:
    1. TREC qrels 格式: query_id\t0\tdoc_id\trelevance
    2. JSON Lines 格式: {"text_id": "xxx", "la": [法条ID列表], ...}
    
    返回: {query_id: set(relevant_doc_ids)}
    """
    qrels = defaultdict(set)
    with open(qrels_path, 'r', encoding='utf-8') as f:
        first_line = f.readline().strip()

    is_json = first_line.startswith('{')

    if is_json:
        # JSON Lines 格式 (test.json)
        print(f"检测到 JSON Lines 格式的 qrels 文件")
        with open(qrels_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                    query_id = item.get('text_id', '')
                    law_ids = item.get('la', [])
                    
                    if query_id and law_ids:
                        # la 中的法条ID是数字，需要转为字符串以匹配 law_corpus.jsonl 的 text_id
                        for law_id in law_ids:
                            qrels[query_id].add(str(law_id))
                except json.JSONDecodeError:
                    continue
    else:
        # TREC qrels 格式
        print(f"检测到 TREC qrels 格式")
        with open(qrels_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 4:
                    query_id = parts[0]
                    doc_id = parts[2]
                    relevance = int(parts[3])
                    if relevance > 0:
                        qrels[query_id].add(str(doc_id))
    
    return qrels

def load_runfile(runfile_path):
    """
    加载检索结果 (TREC 格式)
    格式: query_id\tQ0\tdoc_id\trank\tscore\trun_name
    
    返回: {query_id: [(doc_id, rank, score), ...]}
    """
    results = defaultdict(list)
    with open(runfile_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 5:
                query_id = parts[0]
                doc_id = str(parts[2])
                rank = int(parts[3])
                score = float(parts[4])
                results[query_id].append((doc_id, rank, score))
    
    # 按 rank 排序
    for query_id in results:
        results[query_id].sort(key=lambda x: x[1])
    
    return results

def compute_recall_at_k(qrels, results, k):
    """计算 Recall@K"""
    recall_scores = []
    for query_id, relevant_docs in qrels.items():
        if query_id not in results:
            recall_scores.append(0.0)
            continue
        
        retrieved_docs = [doc_id for doc_id, _, _ in results[query_id][:k]]
        hits = len(set(retrieved_docs) & relevant_docs)
        recall = hits / len(relevant_docs) if relevant_docs else 0.0
        recall_scores.append(recall)
    
    return sum(recall_scores) / len(recall_scores) if recall_scores else 0.0

def compute_precision_at_k(qrels, results, k):
    """计算 Precision@K"""
    precision_scores = []
    for query_id, relevant_docs in qrels.items():
        if query_id not in results:
            precision_scores.append(0.0)
            continue
        
        retrieved_docs = [doc_id for doc_id, _, _ in results[query_id][:k]]
        hits = len(set(retrieved_docs) & relevant_docs)
        precision = hits / k if k > 0 else 0.0
        precision_scores.append(precision)
    
    return sum(precision_scores) / len(precision_scores) if precision_scores else 0.0

def compute_mrr(qrels, results):
    """计算 MRR (Mean Reciprocal Rank)"""
    rr_scores = []
    for query_id, relevant_docs in qrels.items():
        if query_id not in results:
            rr_scores.append(0.0)
            continue
        
        rr = 0.0
        for doc_id, rank, _ in results[query_id]:
            if doc_id in relevant_docs:
                rr = 1.0 / rank
                break
        rr_scores.append(rr)
    
    return sum(rr_scores) / len(rr_scores) if rr_scores else 0.0


def main():
    parser = argparse.ArgumentParser(description='评测检索模型性能')
    parser.add_argument('--runfile', type=str, required=True, help='检索结果文件 (TREC 格式)')
    parser.add_argument('--qrels', type=str, required=True, help='标准答案文件 (TREC qrels 或 JSON Lines)')
    parser.add_argument('--output', type=str, default=None, help='评测结果输出文件')
    args = parser.parse_args()
    
    print("加载标准答案...")
    qrels = load_qrels(args.qrels)
    print(f"共 {len(qrels)} 个查询")
    
    if len(qrels) == 0:
        print("警告: 未能加载任何 qrels，请检查文件格式!")
        return
    
    print("加载检索结果...")
    results = load_runfile(args.runfile)
    print(f"共 {len(results)} 个查询的检索结果")
    
    # 检查匹配情况
    matched_queries = set(qrels.keys()) & set(results.keys())
    print(f"匹配到 {len(matched_queries)} 个查询")
    
    if len(matched_queries) == 0:
        print("警告: qrels 和 runfile 中没有匹配的查询ID!")
        print(f"qrels 示例 ID: {list(qrels.keys())[:3]}")
        print(f"runfile 示例 ID: {list(results.keys())[:3]}")
        return
    
    # 检查检索结果的数量
    max_retrieved = max(len(docs) for docs in results.values()) if results else 0
    print(f"每个查询最多检索到 {max_retrieved} 个结果")
    
    if max_retrieved < 10:
        print(f"\n⚠️  警告: 检索结果数量 ({max_retrieved}) 小于评测的最大 K 值 (10)")
        print(f"   建议在检索时设置更大的 TOP_K 值")
        print(f"   否则 Recall@{max_retrieved+1} 及以上的指标将不准确！\n")
    
    print("\n" + "=" * 50)
    print("检索性能评测结果")
    print("=" * 50)
    
    # 计算各项指标（只评估 @5 和 @10，下游任务只需 10 条法条）
    metrics = {}
    
    for k in [5, 10]:
        recall = compute_recall_at_k(qrels, results, k)
        precision = compute_precision_at_k(qrels, results, k)
        
        metrics[f'Recall@{k}'] = recall
        metrics[f'Precision@{k}'] = precision
        
        # 如果 K 超过检索结果数量，添加警告
        warning = " ⚠️" if k > max_retrieved else ""
        print(f"Recall@{k}: {recall:.4f}{warning}")
        print(f"Precision@{k}: {precision:.4f}{warning}")
        print("-" * 30)
    
    print("=" * 50)
    
    # 保存结果
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            f.write("=" * 50 + "\n")
            f.write("法条检索模型评测结果\n")
            f.write("=" * 50 + "\n\n")
            
            for metric, value in metrics.items():
                f.write(f"{metric}: {value:.4f}\n")
        
        print(f"\n结果已保存到: {args.output}")


if __name__ == '__main__':
    main()