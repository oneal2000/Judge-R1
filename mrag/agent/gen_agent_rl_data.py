"""
生成 Agent RL 训练数据

包含两个任务：
1. QueryGen: 根据案件事实生成检索查询
2. LawSelect: 从候选法条中筛选相关法条

数据格式符合 swift rlhf 的要求

改进点：
- LawSelect 使用 Dense Retriever + Reranker 生成候选（与推理时完全一致）
- 训练和推理使用完全一致的提示词模板和格式
- 统一的文本截断长度
- Dense 编码参数对齐：法条 max_length=256, 查询 max_length=512
"""
import json
import argparse
import random
import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Set, Optional, Tuple
from tqdm import tqdm

import numpy as np

# 添加项目根目录到路径
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

# 导入共享提示词和配置（确保训练和推理使用一致的提示词）
from mrag.agent.prompts import (
    QUERYGEN_SYSTEM_PROMPT,
    QUERYGEN_USER_TEMPLATE,
    LAWSELECT_SYSTEM_PROMPT,
    LAWSELECT_USER_TEMPLATE,
    MAX_LAW_TEXT_LENGTH,
    MAX_FACT_LENGTH,
    truncate_law_text,
    truncate_fact,
    format_candidate_law,
)


# ============== Dense + Reranker 候选生成器（与推理一致）==============
class DenseRerankerRetriever:
    """
    使用真正的 Dense Retriever + Reranker 生成候选法条
    
    与 law_agent.py 推理时完全一致的检索管线：
    - Dense 编码：法条 max_length=256, 查询 max_length=512
    - Reranker: Cross-Encoder 重排
    
    这样训练和推理时 LawSelect 看到的候选分布完全一致。
    """
    
    def __init__(
        self,
        law_corpus_path: str,
        dense_model_path: str,
        reranker_model_path: str,
        device: str = "cuda",
    ):
        from mrag.agent.law_agent import DenseRetriever, LawReranker
        
        print("[DenseRerankerRetriever] 加载 Dense Retriever + Reranker（与推理一致）...")
        self.dense_retriever = DenseRetriever(
            law_corpus_path=law_corpus_path,
            dense_model_path=dense_model_path,
            device=device,
        )
        self.reranker = LawReranker(
            model_path=reranker_model_path,
            device=device,
        )
        print("[DenseRerankerRetriever] 加载完成")
    
    def retrieve_candidates(
        self, 
        fact: str, 
        dense_top_k: int = 50, 
        rerank_top_k: int = 20,
    ) -> List[Tuple[str, float]]:
        """
        执行 Dense + Reranker 检索，返回 (law_id, score) 列表
        
        与 law_agent.py 推理时完全一致:
        1. fact → Dense Retriever → top-50
        2. top-50 → Reranker → top-20
        """
        # Step 1: Dense 检索
        query = fact[:1500] if len(fact) > 1500 else fact
        dense_results = self.dense_retriever.search([query], top_k=dense_top_k)
        
        # Step 2: Reranker 重排
        reranked = self.reranker.rerank(fact, dense_results, top_k=rerank_top_k)
        
        return [(r.law_id, r.score) for r in reranked]


# ============== 旧版 n-gram 检索器（保留作为 fallback）==============
class SimpleRetrieverForNegatives:
    """
    简化版检索器，使用字符 n-gram 生成困难负例（旧版，作为 fallback）
    """
    
    def __init__(self, law_dict: Dict[int, Dict]):
        self.law_dict = law_dict
        self.law_ids = list(law_dict.keys())
        
        self.inverted_index: Dict[str, Set[int]] = {}
        self.law_ngrams: Dict[int, Set[str]] = {}
        
        print("[SimpleRetriever] Building n-gram index...")
        for law_id, info in tqdm(law_dict.items(), desc="Indexing"):
            text = f"{info['name']}：{info['text']}"
            ngrams = self._get_ngrams(text, n=2)
            self.law_ngrams[law_id] = ngrams
            for ng in ngrams:
                if ng not in self.inverted_index:
                    self.inverted_index[ng] = set()
                self.inverted_index[ng].add(law_id)
        
        print(f"[SimpleRetriever] Indexed {len(law_dict)} laws")
    
    def _get_ngrams(self, text: str, n: int = 2) -> Set[str]:
        text = text.lower()
        return {text[i:i+n] for i in range(len(text) - n + 1)}
    
    def retrieve(self, query: str, top_k: int = 50, exclude: Set[int] = None) -> List[int]:
        if exclude is None:
            exclude = set()
        query_ngrams = self._get_ngrams(query, n=2)
        scores: Dict[int, float] = {}
        for ng in query_ngrams:
            if ng in self.inverted_index:
                for law_id in self.inverted_index[ng]:
                    if law_id not in exclude:
                        scores[law_id] = scores.get(law_id, 0) + 1
        for law_id in scores:
            union_size = len(query_ngrams | self.law_ngrams[law_id])
            if union_size > 0:
                scores[law_id] /= union_size
        sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
        return sorted_ids[:top_k]


def gen_querygen_data(train_data: List[Dict], output_path: Path):
    """生成 QueryGen 任务的 RL 训练数据"""
    print(f"[QueryGen] 生成训练数据...")
    
    examples = []
    for item in tqdm(train_data, desc="Processing"):
        fact = item.get("text", "")
        true_law_ids = item.get("la", [])  # 真实法条ID，用于计算检索F1
        
        if not fact or len(fact) < 50:
            continue
        
        # 截断过长的事实（使用统一的截断函数）
        fact = truncate_fact(fact)
        
        example = {
            "messages": [
                {"role": "system", "content": QUERYGEN_SYSTEM_PROMPT},
                {"role": "user", "content": QUERYGEN_USER_TEMPLATE.format(fact=fact)},
            ],
            # 额外字段用于奖励计算
            "case_fact": fact,
            "ground_truth_laws": json.dumps(true_law_ids),  # 真实法条ID，用于计算检索后F1
            "task_type": "query_gen",
        }
        examples.append(example)
    
    # 保存
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + '\n')
    
    print(f"[QueryGen] 保存 {len(examples)} 条数据到 {output_path}")
    return examples


def load_law_corpus(law_corpus_path: Path) -> Dict[int, Dict]:
    """加载法条库"""
    law_dict = {}
    with open(law_corpus_path, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            law_id = int(item['text_id'])
            law_dict[law_id] = {
                'name': item['name'],
                'text': item['text'],
            }
    return law_dict


def gen_lawselect_data(
    train_data: List[Dict], 
    law_dict: Dict[int, Dict],
    output_path: Path,
    num_candidates: int = 20,
    num_hard_negatives: int = 15,
    use_hard_negatives: bool = True,
    dense_model_path: str = None,
    reranker_model_path: str = None,
    law_corpus_path: str = None,
    device: str = "cuda",
):
    """
    生成 LawSelect 任务的 RL 训练数据
    
    候选法条生成方式（按优先级）：
    1. Dense + Reranker（推荐，与推理一致）：需要 --dense_model 和 --reranker_model
    2. n-gram 困难负例（fallback）：不需要 GPU
    3. 随机负例：--no_hard_negatives
    
    当使用 Dense + Reranker 时：
    - fact → Dense top-50 → Reranker top-20
    - 如果正例不在 Reranker top-20 中，强制加入（替换末尾负例）
    - 打乱顺序（消除位置偏差）
    - 候选分布与推理时 LawSelect 看到的完全一致
    
    Args:
        train_data: 训练数据
        law_dict: 法条字典
        output_path: 输出路径
        num_candidates: 候选法条数量（默认 20，与推理时 rerank_top_k 对齐）
        num_hard_negatives: n-gram 困难负例数量（仅 fallback 模式使用）
        use_hard_negatives: 是否使用困难负例
        dense_model_path: Dense Retriever 模型路径（优先使用）
        reranker_model_path: Reranker 模型路径（优先使用）
        law_corpus_path: 法条库路径（Dense+Reranker 模式需要）
        device: GPU 设备
    """
    print(f"[LawSelect] 生成训练数据...")
    
    all_law_ids = list(law_dict.keys())
    
    # 选择候选生成方式
    use_dense_reranker = (
        dense_model_path is not None 
        and reranker_model_path is not None 
        and law_corpus_path is not None
    )
    
    dense_reranker = None
    ngram_retriever = None
    
    if use_dense_reranker:
        print("[LawSelect] 使用 Dense + Reranker 生成候选（与推理一致）")
        dense_reranker = DenseRerankerRetriever(
            law_corpus_path=law_corpus_path,
            dense_model_path=dense_model_path,
            reranker_model_path=reranker_model_path,
            device=device,
        )
    elif use_hard_negatives:
        print("[LawSelect] 使用 n-gram 困难负例（fallback 模式）")
        ngram_retriever = SimpleRetrieverForNegatives(law_dict)
    else:
        print("[LawSelect] 使用随机负例")
    
    examples = []
    stats = {"total": 0, "gt_in_candidates": 0, "gt_forced": 0}
    
    for item in tqdm(train_data, desc="Processing"):
        fact = item.get("text", "")
        true_law_ids = item.get("la", [])
        
        if not fact or len(fact) < 50 or not true_law_ids:
            continue
        
        # 截断过长的事实（与推理时 LawSelector 一致）
        fact_truncated = truncate_fact(fact, max_length=1500)
        
        # 正例
        positive_ids = [lid for lid in true_law_ids if lid in law_dict]
        if not positive_ids:
            continue
        positive_set = set(positive_ids)
        # 将 positive_ids 转为字符串（与 Dense Retriever 输出格式一致）
        positive_str_set = {str(lid) for lid in positive_ids}
        
        stats["total"] += 1
        
        # ============ 生成候选法条 ============
        if use_dense_reranker and dense_reranker is not None:
            # 方式 1: Dense + Reranker（与推理完全一致）
            reranker_results = dense_reranker.retrieve_candidates(
                fact, dense_top_k=50, rerank_top_k=num_candidates
            )
            
            # Reranker top-20 的法条 ID 列表
            reranker_ids = [int(law_id) for law_id, _ in reranker_results]
            reranker_id_set = set(reranker_ids)
            
            # 统计正例覆盖
            gt_in_reranker = positive_set & reranker_id_set
            stats["gt_in_candidates"] += len(gt_in_reranker)
            
            # 强制加入遗漏的正例（替换末尾的负例）
            missing_positives = [lid for lid in positive_ids if lid not in reranker_id_set]
            if missing_positives:
                stats["gt_forced"] += len(missing_positives)
                # 从 Reranker 结果末尾移除相应数量的负例
                # 保留前面的候选（排名靠前的更有价值）
                keep_count = max(0, num_candidates - len(missing_positives))
                candidate_ids = reranker_ids[:keep_count] + missing_positives
            else:
                candidate_ids = reranker_ids[:num_candidates]
            
            # 打乱顺序（消除位置偏差，与旧版一致）
            random.shuffle(candidate_ids)
        
        elif ngram_retriever is not None:
            # 方式 2: n-gram 困难负例（fallback）
            num_negatives_needed = max(0, num_candidates - len(positive_ids))
            actual_hard_neg_count = min(num_hard_negatives, num_negatives_needed)
            
            hard_neg_candidates = ngram_retriever.retrieve(
                fact, top_k=actual_hard_neg_count * 2, exclude=positive_set
            )
            negative_ids = list(hard_neg_candidates[:actual_hard_neg_count])
            
            if len(negative_ids) < num_negatives_needed:
                remaining = num_negatives_needed - len(negative_ids)
                random_pool = [lid for lid in all_law_ids 
                              if lid not in positive_set and lid not in set(negative_ids)]
                if random_pool:
                    negative_ids.extend(random.sample(random_pool, min(remaining, len(random_pool))))
            
            candidate_ids = list(positive_ids) + negative_ids[:num_negatives_needed]
            random.shuffle(candidate_ids)
        
        else:
            # 方式 3: 纯随机负例
            num_negatives_needed = max(0, num_candidates - len(positive_ids))
            negative_pool = [lid for lid in all_law_ids if lid not in positive_set]
            negative_ids = random.sample(negative_pool, min(num_negatives_needed, len(negative_pool)))
            candidate_ids = list(positive_ids) + negative_ids
            random.shuffle(candidate_ids)
        
        # ============ 构建 prompt ============
        candidate_texts = []
        for i, lid in enumerate(candidate_ids):
            lid_int = int(lid) if not isinstance(lid, int) else lid
            if lid_int not in law_dict:
                continue
            law_info = law_dict[lid_int]
            formatted = format_candidate_law(
                idx=i,
                law_id=str(lid_int),
                law_name=law_info['name'],
                law_text=law_info['text'],
                max_text_length=MAX_LAW_TEXT_LENGTH
            )
            candidate_texts.append(formatted)
        
        candidate_laws_str = "\n".join(candidate_texts)
        
        example = {
            "messages": [
                {"role": "system", "content": LAWSELECT_SYSTEM_PROMPT},
                {"role": "user", "content": LAWSELECT_USER_TEMPLATE.format(
                    fact=fact_truncated,
                    num_candidates=len(candidate_texts),
                    candidate_laws=candidate_laws_str,
                )},
            ],
            "ground_truth_laws": json.dumps(positive_ids),
            "task_type": "law_select",
        }
        examples.append(example)
    
    # 保存
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + '\n')
    
    # 统计信息
    print(f"[LawSelect] 保存 {len(examples)} 条数据到 {output_path}")
    if use_dense_reranker:
        total_gt = stats["gt_in_candidates"] + stats["gt_forced"]
        avg_coverage = stats["gt_in_candidates"] / total_gt * 100 if total_gt > 0 else 0
        print(f"[LawSelect] Dense+Reranker 候选统计:")
        print(f"  正例自然覆盖: {stats['gt_in_candidates']}/{total_gt} ({avg_coverage:.1f}%)")
        print(f"  正例强制加入: {stats['gt_forced']}")
    
    return examples


def main():
    parser = argparse.ArgumentParser(description="生成 Agent RL 训练数据")
    parser.add_argument("--train_data", type=str, required=True,
                        help="训练数据路径 (train.json)")
    parser.add_argument("--law_corpus", type=str, required=True,
                        help="法条库路径 (law_corpus.jsonl)")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="输出目录")
    parser.add_argument("--task", type=str, choices=["query_gen", "law_select", "both"],
                        default="both", help="生成哪个任务的数据")
    parser.add_argument("--num_candidates", type=int, default=20,
                        help="LawSelect 任务的候选法条数量（与推理时 rerank_top_k 对齐）")
    parser.add_argument("--num_hard_negatives", type=int, default=15,
                        help="LawSelect n-gram 困难负例数量（仅 fallback 模式）")
    parser.add_argument("--no_hard_negatives", action="store_true",
                        help="不使用困难负例（使用随机负例）")
    
    # Dense + Reranker 模式（推荐，与推理一致）
    parser.add_argument("--dense_model", type=str, default=None,
                        help="Dense Retriever 模型路径（启用 Dense+Reranker 模式）")
    parser.add_argument("--reranker_model", type=str, default=None,
                        help="Reranker 模型路径（启用 Dense+Reranker 模式）")
    parser.add_argument("--device", type=str, default="cuda",
                        help="GPU 设备")
    
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    
    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    # 加载数据
    print(f"加载训练数据: {args.train_data}")
    with open(args.train_data, 'r', encoding='utf-8') as f:
        train_data = [json.loads(line) for line in f]
    print(f"共 {len(train_data)} 条训练样本")
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if args.task in ["query_gen", "both"]:
        gen_querygen_data(
            train_data, 
            output_dir / "query_gen_train.jsonl"
        )
    
    if args.task in ["law_select", "both"]:
        print(f"加载法条库: {args.law_corpus}")
        law_dict = load_law_corpus(Path(args.law_corpus))
        print(f"共 {len(law_dict)} 条法条")
        
        gen_lawselect_data(
            train_data,
            law_dict,
            output_dir / "law_select_train.jsonl",
            num_candidates=args.num_candidates,
            num_hard_negatives=args.num_hard_negatives,
            use_hard_negatives=not args.no_hard_negatives,
            dense_model_path=args.dense_model,
            reranker_model_path=args.reranker_model,
            law_corpus_path=args.law_corpus,
            device=args.device,
        )
    
    print("\n✅ 数据生成完成！")
    print(f"输出目录: {output_dir}")


if __name__ == "__main__":
    main()
