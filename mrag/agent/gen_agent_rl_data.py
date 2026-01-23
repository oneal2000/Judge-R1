"""
生成 Agent RL 训练数据

包含两个任务：
1. QueryGen: 根据案件事实生成检索查询
2. LawSelect: 从候选法条中筛选相关法条

数据格式符合 swift rlhf 的要求

改进点：
- LawSelect 使用困难负例（Dense Retriever 检索的非正例法条）
- 训练和推理使用完全一致的提示词模板和格式
- 统一的文本截断长度
"""
import json
import argparse
import random
import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Set, Optional
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


# ============== 简化版检索器（用于生成困难负例）==============
class SimpleRetrieverForNegatives:
    """
    简化版检索器，用于生成困难负例
    使用 BM25 或简单的词重叠度进行快速检索
    """
    
    def __init__(self, law_dict: Dict[int, Dict]):
        self.law_dict = law_dict
        self.law_ids = list(law_dict.keys())
        
        # 构建简单的倒排索引（基于字符 n-gram）
        self.inverted_index: Dict[str, Set[int]] = {}
        self.law_ngrams: Dict[int, Set[str]] = {}
        
        print("[SimpleRetriever] Building index...")
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
        """提取字符 n-gram"""
        text = text.lower()
        ngrams = set()
        for i in range(len(text) - n + 1):
            ngrams.add(text[i:i+n])
        return ngrams
    
    def retrieve(self, query: str, top_k: int = 50, exclude: Set[int] = None) -> List[int]:
        """
        检索与 query 相关的法条
        
        Args:
            query: 查询文本
            top_k: 返回数量
            exclude: 要排除的法条 ID 集合
        
        Returns:
            法条 ID 列表（按相关度排序）
        """
        if exclude is None:
            exclude = set()
        
        query_ngrams = self._get_ngrams(query, n=2)
        
        # 计算每个法条的匹配分数
        scores: Dict[int, float] = {}
        for ng in query_ngrams:
            if ng in self.inverted_index:
                for law_id in self.inverted_index[ng]:
                    if law_id not in exclude:
                        if law_id not in scores:
                            scores[law_id] = 0
                        scores[law_id] += 1
        
        # 归一化分数（Jaccard 相似度）
        for law_id in scores:
            law_ngrams = self.law_ngrams[law_id]
            union_size = len(query_ngrams | law_ngrams)
            if union_size > 0:
                scores[law_id] = scores[law_id] / union_size
        
        # 按分数排序
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
):
    """
    生成 LawSelect 任务的 RL 训练数据
    
    改进：使用困难负例替代随机负例
    
    候选法条 = 真实相关法条 + 困难负例（通过检索获得）
    
    Args:
        train_data: 训练数据
        law_dict: 法条字典
        output_path: 输出路径
        num_candidates: 候选法条数量
        num_hard_negatives: 困难负例数量
        use_hard_negatives: 是否使用困难负例（False 则使用随机负例）
    """
    print(f"[LawSelect] 生成训练数据...")
    print(f"[LawSelect] 使用困难负例: {use_hard_negatives}")
    
    all_law_ids = list(law_dict.keys())
    
    # 初始化检索器（用于生成困难负例）
    retriever = None
    if use_hard_negatives:
        retriever = SimpleRetrieverForNegatives(law_dict)
    
    examples = []
    hard_neg_stats = []  # 统计困难负例数量
    
    for item in tqdm(train_data, desc="Processing"):
        fact = item.get("text", "")
        true_law_ids = item.get("la", [])
        
        if not fact or len(fact) < 50 or not true_law_ids:
            continue
        
        # 截断过长的事实（使用统一的截断函数）
        fact_truncated = truncate_fact(fact, max_length=1500)  # LawSelect 用更短的 fact
        
        # 构建候选法条列表
        # 正例：真实相关的法条
        positive_ids = [lid for lid in true_law_ids if lid in law_dict]
        if not positive_ids:
            continue
        
        positive_set = set(positive_ids)
        
        # 负例：困难负例 + 随机负例
        negative_ids = []
        
        if use_hard_negatives and retriever is not None:
            # 使用检索获取困难负例
            hard_neg_candidates = retriever.retrieve(
                fact, 
                top_k=num_hard_negatives * 2,  # 多检索一些，后面筛选
                exclude=positive_set
            )
            
            # 选择前 num_hard_negatives 个困难负例
            hard_negatives = hard_neg_candidates[:num_hard_negatives]
            negative_ids.extend(hard_negatives)
            hard_neg_stats.append(len(hard_negatives))
            
            # 如果困难负例不够，用随机负例补充
            if len(negative_ids) < num_hard_negatives:
                remaining = num_hard_negatives - len(negative_ids)
                random_pool = [lid for lid in all_law_ids 
                              if lid not in positive_set and lid not in set(negative_ids)]
                if random_pool:
                    random_negatives = random.sample(random_pool, min(remaining, len(random_pool)))
                    negative_ids.extend(random_negatives)
        else:
            # 使用随机负例
            negative_pool = [lid for lid in all_law_ids if lid not in positive_set]
            num_neg = min(num_hard_negatives, len(negative_pool))
            negative_ids = random.sample(negative_pool, num_neg)
        
        # 合并正例和负例，然后打乱顺序
        candidate_ids = positive_ids + negative_ids
        random.shuffle(candidate_ids)
        
        # 构建候选法条文本（使用统一的格式化函数）
        candidate_texts = []
        for i, lid in enumerate(candidate_ids):
            law_info = law_dict[lid]
            formatted = format_candidate_law(
                idx=i,
                law_id=str(lid),
                law_name=law_info['name'],
                law_text=law_info['text'],
                max_text_length=MAX_LAW_TEXT_LENGTH  # 使用统一的截断长度
            )
            candidate_texts.append(formatted)
        
        candidate_laws_str = "\n".join(candidate_texts)
        
        example = {
            "messages": [
                {"role": "system", "content": LAWSELECT_SYSTEM_PROMPT},
                {"role": "user", "content": LAWSELECT_USER_TEMPLATE.format(
                    fact=fact_truncated,
                    num_candidates=len(candidate_ids),
                    candidate_laws=candidate_laws_str,
                )},
            ],
            # 额外字段用于奖励计算
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
    if hard_neg_stats:
        avg_hard_neg = sum(hard_neg_stats) / len(hard_neg_stats)
        print(f"[LawSelect] 平均困难负例数量: {avg_hard_neg:.1f}")
    
    print(f"[LawSelect] 保存 {len(examples)} 条数据到 {output_path}")
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
                        help="LawSelect 任务的候选法条数量")
    parser.add_argument("--num_hard_negatives", type=int, default=15,
                        help="LawSelect 任务的困难负例数量")
    parser.add_argument("--no_hard_negatives", action="store_true",
                        help="不使用困难负例（使用随机负例）")
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
        )
    
    print("\n✅ 数据生成完成！")
    print(f"输出目录: {output_dir}")


if __name__ == "__main__":
    main()
