"""
输出级融合脚本 — 融合 Agent 和 MRAG 的 TSV 检索结果

核心思想：
  Agent 管线 (QueryGen → Dense → Reranker → LawSelect) 输出精筛后的法条（~7 条）
  MRAG 管线 (fact → Dense → Reranker) 输出 top-K 重排法条（~20 条）
  二者输出的 TSV 文件可直接在输出级融合，不需要重新加载模型。

为什么不在 Dense 检索层融合（旧 hybrid_agent.py 的做法）：
  1. 跳过了 Agent 最强组件 LawSelect — 丢掉了 RL 训练的判别能力
  2. Reranker 对融合池重排引入噪声 — 无关法条可能排到首位
  3. Agent 内部已经包含 Dense+Reranker+LawSelect 全链路，无需再外部 rerank

融合策略：
  agent_first（推荐）：保留 Agent 全部输出 → 追加 MRAG 中 Agent 未覆盖的法条
  rrf             ：基于 rank 的 Reciprocal Rank Fusion
  score_merge     ：取两路中较高的 score，按 score 排序

用法：
  python mrag/agent/fuse_results.py \
    --agent_file  mrag/retriever_output/ablation_both_rl.tsv \
    --mrag_file   mrag/retriever_output/ablation_mrag.tsv \
    --output_file mrag/retriever_output/fused_agent_first.tsv \
    --strategy agent_first \
    --max_laws 20 \
    --qrels mrag/retriever_data/qrels_test.tsv
"""

import argparse
import sys
import os
from collections import defaultdict, OrderedDict
from typing import Dict, List, Tuple, Optional


def load_runfile(path: str) -> Dict[str, List[Tuple[str, int, float]]]:
    """加载 TREC 格式 TSV: qid \\t Q0 \\t docid \\t rank \\t score \\t tag"""
    results: Dict[str, List[Tuple[str, int, float]]] = defaultdict(list)
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 5:
                continue
            qid, _, docid, rank, score = parts[0], parts[1], parts[2], int(parts[3]), float(parts[4])
            results[qid].append((docid, rank, score))
    # 按 rank 排序
    for qid in results:
        results[qid].sort(key=lambda x: x[1])
    return results


def save_runfile(
    results: Dict[str, List[Tuple[str, float]]],
    path: str,
    tag: str = "fused",
) -> None:
    """保存为 TREC 格式"""
    with open(path, "w", encoding="utf-8") as f:
        for qid in sorted(results.keys()):
            for rank, (docid, score) in enumerate(results[qid], 1):
                f.write(f"{qid}\tQ0\t{docid}\t{rank}\t{score:.6f}\t{tag}\n")


# ============================================================
#  融合策略
# ============================================================

def fuse_agent_first(
    agent: Dict[str, List[Tuple[str, int, float]]],
    mrag: Dict[str, List[Tuple[str, int, float]]],
    max_laws: int = 20,
) -> Dict[str, List[Tuple[str, float]]]:
    """
    Agent 优先融合：
    1. 保留 Agent 全部输出（高精度, 经 LawSelect 筛选）
    2. 追加 MRAG 中 Agent 未覆盖的法条（按 MRAG 原始排名）
    3. 截断到 max_laws

    预期效果：
      R@5  ≈ Agent（Agent 的 top-5 通常就是最准的）
      R@10 > Agent（MRAG 补充了 Agent 遗漏的法条）
    """
    all_qids = set(agent.keys()) | set(mrag.keys())
    fused: Dict[str, List[Tuple[str, float]]] = {}

    for qid in all_qids:
        merged: List[Tuple[str, float]] = []
        seen = set()

        # Step 1: Agent 全部保留
        for docid, _rank, score in agent.get(qid, []):
            if docid not in seen:
                seen.add(docid)
                merged.append((docid, score))

        # Step 2: MRAG 补充（Agent 中没有的）
        for docid, _rank, score in mrag.get(qid, []):
            if docid not in seen:
                seen.add(docid)
                # 给 MRAG 补充的法条一个较低的分数基线，保证排在 Agent 之后
                adjusted_score = score * 0.5 if score > 0 else 0.3
                merged.append((docid, adjusted_score))

        fused[qid] = merged[:max_laws]

    return fused


def fuse_rrf(
    agent: Dict[str, List[Tuple[str, int, float]]],
    mrag: Dict[str, List[Tuple[str, int, float]]],
    max_laws: int = 20,
    k: int = 60,
    agent_weight: float = 2.0,
    mrag_weight: float = 1.0,
) -> Dict[str, List[Tuple[str, float]]]:
    """
    加权 RRF 融合（基于输出排名）
    Agent 权重更高因为经过了 LawSelect 精筛
    """
    all_qids = set(agent.keys()) | set(mrag.keys())
    fused: Dict[str, List[Tuple[str, float]]] = {}

    for qid in all_qids:
        scores: Dict[str, float] = defaultdict(float)

        for docid, rank, _ in agent.get(qid, []):
            scores[docid] += agent_weight / (k + rank)

        for docid, rank, _ in mrag.get(qid, []):
            scores[docid] += mrag_weight / (k + rank)

        # 按 RRF 分数排序
        sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        fused[qid] = sorted_docs[:max_laws]

    return fused


def fuse_score_merge(
    agent: Dict[str, List[Tuple[str, int, float]]],
    mrag: Dict[str, List[Tuple[str, int, float]]],
    max_laws: int = 20,
) -> Dict[str, List[Tuple[str, float]]]:
    """
    分数合并：取两路中较高的 score，按 score 排序
    """
    all_qids = set(agent.keys()) | set(mrag.keys())
    fused: Dict[str, List[Tuple[str, float]]] = {}

    for qid in all_qids:
        scores: Dict[str, float] = {}

        for docid, _rank, score in agent.get(qid, []):
            scores[docid] = max(scores.get(docid, float("-inf")), score)

        for docid, _rank, score in mrag.get(qid, []):
            scores[docid] = max(scores.get(docid, float("-inf")), score)

        sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        fused[qid] = sorted_docs[:max_laws]

    return fused


# ============================================================
#  评测（可选，直接调用 eval_retriever 的逻辑）
# ============================================================

def evaluate(runfile_path: str, qrels_path: str) -> Dict[str, float]:
    """直接调用 eval_retriever.py 的逻辑"""
    # 复用 eval_retriever 的加载和评测函数
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    sys.path.insert(0, project_root)
    from mrag.eval_retriever import load_qrels, load_runfile as eval_load_runfile
    from mrag.eval_retriever import compute_recall_at_k, compute_precision_at_k

    qrels = load_qrels(qrels_path)
    results = eval_load_runfile(runfile_path)

    metrics = {}
    for k_val in [5, 10]:
        metrics[f"Recall@{k_val}"] = compute_recall_at_k(qrels, results, k_val)
        metrics[f"Precision@{k_val}"] = compute_precision_at_k(qrels, results, k_val)

    return metrics


def print_comparison(
    agent_metrics: Dict[str, float],
    mrag_metrics: Dict[str, float],
    fused_metrics: Dict[str, float],
    strategy: str,
) -> None:
    """打印对比表格"""
    print("\n" + "=" * 70)
    print(f"  融合效果对比 (策略: {strategy})")
    print("=" * 70)
    print(f"{'指标':<16} {'Agent':>10} {'MRAG':>10} {'Fused':>10} {'vs Agent':>10} {'vs MRAG':>10}")
    print("-" * 70)

    for metric in ["Recall@5", "Precision@5", "Recall@10", "Precision@10"]:
        a = agent_metrics.get(metric, 0)
        m = mrag_metrics.get(metric, 0)
        f = fused_metrics.get(metric, 0)
        da = f - a
        dm = f - m
        da_str = f"{da:+.4f}" if da != 0 else "  --"
        dm_str = f"{dm:+.4f}" if dm != 0 else "  --"
        print(f"{metric:<16} {a:>10.4f} {m:>10.4f} {f:>10.4f} {da_str:>10} {dm_str:>10}")

    print("=" * 70)


# ============================================================
#  统计分析
# ============================================================

def print_stats(
    agent: Dict[str, List[Tuple[str, int, float]]],
    mrag: Dict[str, List[Tuple[str, int, float]]],
    fused: Dict[str, List[Tuple[str, float]]],
) -> None:
    """打印融合统计"""
    all_qids = set(agent.keys()) | set(mrag.keys())

    total_agent_only = 0
    total_mrag_only = 0
    total_both = 0
    total_agent_count = 0
    total_mrag_count = 0
    total_fused_count = 0

    for qid in all_qids:
        agent_ids = {d[0] for d in agent.get(qid, [])}
        mrag_ids = {d[0] for d in mrag.get(qid, [])}
        fused_ids = {d[0] for d in fused.get(qid, [])}

        both = agent_ids & mrag_ids
        agent_only = agent_ids - mrag_ids
        mrag_only = mrag_ids - agent_ids

        total_both += len(both)
        total_agent_only += len(agent_only)
        total_mrag_only += len(mrag_only)
        total_agent_count += len(agent_ids)
        total_mrag_count += len(mrag_ids)
        total_fused_count += len(fused_ids)

    n = len(all_qids)
    print(f"\n--- 融合统计 ({n} 个查询) ---")
    print(f"  Agent 平均输出: {total_agent_count/n:.1f} 条/查询")
    print(f"  MRAG  平均输出: {total_mrag_count/n:.1f} 条/查询")
    print(f"  融合  平均输出: {total_fused_count/n:.1f} 条/查询")
    print(f"  双重命中: {total_both/n:.1f} 条/查询")
    print(f"  Agent 独有: {total_agent_only/n:.1f} 条/查询")
    print(f"  MRAG  独有: {total_mrag_only/n:.1f} 条/查询")


def main():
    parser = argparse.ArgumentParser(
        description="输出级融合 Agent 和 MRAG 的 TSV 检索结果",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # Agent 优先融合（推荐）
  python mrag/agent/fuse_results.py \\
    --agent_file mrag/retriever_output/ablation_both_rl.tsv \\
    --mrag_file  mrag/retriever_output/ablation_mrag.tsv \\
    --output_file mrag/retriever_output/fused_agent_first.tsv \\
    --strategy agent_first \\
    --qrels mrag/retriever_data/qrels_test.tsv

  # 尝试所有策略并对比
  python mrag/agent/fuse_results.py \\
    --agent_file mrag/retriever_output/ablation_both_rl.tsv \\
    --mrag_file  mrag/retriever_output/ablation_mrag.tsv \\
    --output_file mrag/retriever_output/fused.tsv \\
    --strategy all \\
    --qrels mrag/retriever_data/qrels_test.tsv
""",
    )
    parser.add_argument("--agent_file", type=str, required=True,
                        help="Agent 管线的 TSV 输出文件（经 LawSelect 筛选）")
    parser.add_argument("--mrag_file", type=str, required=True,
                        help="MRAG 管线的 TSV 输出文件（Dense + Reranker）")
    parser.add_argument("--output_file", type=str, required=True,
                        help="融合后的 TSV 输出文件")
    parser.add_argument("--strategy", type=str, default="agent_first",
                        choices=["agent_first", "rrf", "score_merge", "all"],
                        help="融合策略 (默认 agent_first)")
    parser.add_argument("--max_laws", type=int, default=20,
                        help="每个查询最多输出的法条数 (默认 20)")
    parser.add_argument("--rrf_k", type=int, default=60,
                        help="RRF 参数 k (默认 60)")
    parser.add_argument("--agent_weight", type=float, default=2.0,
                        help="RRF 策略中 Agent 的权重 (默认 2.0)")
    parser.add_argument("--mrag_weight", type=float, default=1.0,
                        help="RRF 策略中 MRAG 的权重 (默认 1.0)")
    parser.add_argument("--qrels", type=str, default=None,
                        help="标准答案文件，提供则自动评测并对比")

    args = parser.parse_args()

    print(f"[FuseResults] Agent 文件: {args.agent_file}")
    print(f"[FuseResults] MRAG  文件: {args.mrag_file}")
    print(f"[FuseResults] 策略: {args.strategy}")
    print(f"[FuseResults] 最大法条数: {args.max_laws}")

    # 加载
    agent = load_runfile(args.agent_file)
    mrag = load_runfile(args.mrag_file)
    print(f"[FuseResults] Agent: {len(agent)} 个查询")
    print(f"[FuseResults] MRAG:  {len(mrag)} 个查询")

    # 如果选择 "all"，尝试所有策略并对比
    strategies = ["agent_first", "rrf", "score_merge"] if args.strategy == "all" else [args.strategy]

    # 预计算 agent 和 mrag 的基线指标
    agent_metrics = None
    mrag_metrics = None
    if args.qrels:
        print("\n[FuseResults] 评测基线...")
        agent_metrics = evaluate(args.agent_file, args.qrels)
        mrag_metrics = evaluate(args.mrag_file, args.qrels)

        print(f"\n  Agent:  R@5={agent_metrics['Recall@5']:.4f}  P@5={agent_metrics['Precision@5']:.4f}  "
              f"R@10={agent_metrics['Recall@10']:.4f}  P@10={agent_metrics['Precision@10']:.4f}")
        print(f"  MRAG:   R@5={mrag_metrics['Recall@5']:.4f}  P@5={mrag_metrics['Precision@5']:.4f}  "
              f"R@10={mrag_metrics['Recall@10']:.4f}  P@10={mrag_metrics['Precision@10']:.4f}")

    for strategy in strategies:
        print(f"\n{'='*50}")
        print(f"  策略: {strategy}")
        print(f"{'='*50}")

        # 融合
        if strategy == "agent_first":
            fused = fuse_agent_first(agent, mrag, max_laws=args.max_laws)
        elif strategy == "rrf":
            fused = fuse_rrf(agent, mrag, max_laws=args.max_laws, k=args.rrf_k,
                             agent_weight=args.agent_weight, mrag_weight=args.mrag_weight)
        elif strategy == "score_merge":
            fused = fuse_score_merge(agent, mrag, max_laws=args.max_laws)
        else:
            raise ValueError(f"未知策略: {strategy}")

        # 统计
        print_stats(agent, mrag, fused)

        # 确定输出路径
        if len(strategies) > 1:
            base, ext = os.path.splitext(args.output_file)
            out_path = f"{base}_{strategy}{ext}"
        else:
            out_path = args.output_file

        # 保存
        save_runfile(fused, out_path, tag=f"fused_{strategy}")
        print(f"\n[FuseResults] 已保存: {out_path}")

        # 评测
        if args.qrels and agent_metrics and mrag_metrics:
            fused_metrics = evaluate(out_path, args.qrels)
            print_comparison(agent_metrics, mrag_metrics, fused_metrics, strategy)

    print("\n[FuseResults] 完成！")


if __name__ == "__main__":
    main()
