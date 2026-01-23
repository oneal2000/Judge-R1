"""
Agent RL 奖励函数 (改进版 v3)

核心目标：提升法条检索的召回率和准确率

设计理念：
- 奖励函数必须与最终评估指标强相关
- QueryGen 的目标是让 Dense Retriever 找到更多候选
- LawSelect 的目标是从候选中选出正确的法条，并且排序正确

改进点：
1. QueryGen: 直接评估 Dense 阶段的 Recall，而不是经过 Reranker 的
2. LawSelect: 使用全量召回 + MRR + NDCG 综合评估
3. 大幅降低格式权重，提高效果权重
4. 加入更多排序相关的指标
"""
import json
import re
import os
import sys
import warnings
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Set, Tuple
import numpy as np

# 在导入 transformers 之前就设置警告过滤
warnings.filterwarnings("ignore", category=UserWarning, message=".*overflowing tokens.*")
warnings.filterwarnings("ignore", category=UserWarning, message=".*Be aware.*")
warnings.filterwarnings("ignore", category=FutureWarning)

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
import transformers
transformers.logging.set_verbosity_error()

# 设置 transformers 的警告级别
logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)
logging.getLogger("transformers.tokenization_utils").setLevel(logging.ERROR)

from swift.plugin import ORM, orms
from swift.utils import get_logger

logger = get_logger()

# 项目根目录
ROOT = Path(__file__).resolve().parents[2]


def strip_code_fences(text: str) -> str:
    """去掉 ```json ... ``` 代码块外壳"""
    if not text:
        return ""
    text = re.sub(r"```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = text.replace("```", "")
    return text.strip()


def json_raw_decode_from(text: str, start_char: str) -> Optional[Any]:
    """从 text 中扫描 start_char（'[' 或 '{'），尝试 JSON 解析"""
    if not text:
        return None
    decoder = json.JSONDecoder()
    for m in re.finditer(re.escape(start_char), text):
        try:
            obj, _ = decoder.raw_decode(text[m.start():])
            return obj
        except json.JSONDecodeError:
            continue
    return None


# ============== 评估指标函数 ==============

def compute_recall(retrieved: List[str], ground_truth: Set[str]) -> float:
    """计算全量召回率（不限制 K）"""
    if not ground_truth:
        return 1.0
    if not retrieved:
        return 0.0
    
    retrieved_set = set(retrieved)
    hits = len(retrieved_set & ground_truth)
    return hits / len(ground_truth)


def compute_recall_at_k(retrieved: List[str], ground_truth: Set[str], k: int) -> float:
    """计算 Recall@K"""
    if not ground_truth:
        return 1.0
    if not retrieved:
        return 0.0
    
    retrieved_top_k = retrieved[:k]
    hits = len(set(retrieved_top_k) & ground_truth)
    return hits / len(ground_truth)


def compute_precision(retrieved: List[str], ground_truth: Set[str]) -> float:
    """计算全量精确率"""
    if not retrieved:
        return 0.0
    
    retrieved_set = set(retrieved)
    hits = len(retrieved_set & ground_truth)
    return hits / len(retrieved_set)


def compute_mrr(retrieved: List[str], ground_truth: Set[str]) -> float:
    """计算 Mean Reciprocal Rank"""
    if not ground_truth or not retrieved:
        return 0.0
    
    for i, doc_id in enumerate(retrieved):
        if doc_id in ground_truth:
            return 1.0 / (i + 1)
    return 0.0


def compute_ndcg(retrieved: List[str], ground_truth: Set[str], k: int = None) -> float:
    """计算 NDCG (可选指定 K)"""
    if not ground_truth or not retrieved:
        return 0.0
    
    if k is not None:
        retrieved = retrieved[:k]
    
    # DCG
    dcg = 0.0
    for i, doc_id in enumerate(retrieved):
        if doc_id in ground_truth:
            dcg += 1.0 / np.log2(i + 2)  # i+2 because log2(1) = 0
    
    # IDCG: 理想情况下所有 ground truth 都排在最前面
    ideal_k = min(len(retrieved), len(ground_truth)) if k is None else min(k, len(ground_truth))
    idcg = sum(1.0 / np.log2(i + 2) for i in range(ideal_k))
    
    return dcg / idcg if idcg > 0 else 0.0


def compute_hit_ratio(retrieved: List[str], ground_truth: Set[str]) -> float:
    """计算是否命中（至少找到一个正确的）"""
    if not ground_truth:
        return 1.0
    if not retrieved:
        return 0.0
    
    for doc_id in retrieved:
        if doc_id in ground_truth:
            return 1.0
    return 0.0


# ============== Dense Retriever 封装（用于在线评估）==============
class SimpleRetriever:
    """
    简化版检索器，用于奖励函数中计算检索效果
    使用单例模式，避免重复加载模型
    """
    _instance = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if SimpleRetriever._initialized:
            return
        
        # 默认路径
        self.dense_model_path = str(ROOT / "output" / "law_retriever")
        self.law_corpus_path = str(ROOT / "data" / "law_corpus.jsonl")
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # 懒加载标志
        self._dense_loaded = False
        self._corpus_loaded = False
        
        SimpleRetriever._initialized = True
        logger.info("[SimpleRetriever] Initialized (lazy loading)")
    
    def _load_corpus(self):
        """加载法条库"""
        if self._corpus_loaded:
            return
        
        self.law_dict = {}
        self.law_ids = []
        self.law_texts = []
        
        if not os.path.exists(self.law_corpus_path):
            logger.warning(f"[SimpleRetriever] Law corpus not found: {self.law_corpus_path}")
            self._corpus_loaded = True
            return
        
        with open(self.law_corpus_path, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line)
                law_id = str(item['text_id'])
                law_text = f"{item['name']}：{item['text']}"
                self.law_dict[law_id] = law_text
                self.law_ids.append(law_id)
                self.law_texts.append(law_text)
        
        logger.info(f"[SimpleRetriever] Loaded {len(self.law_texts)} laws")
        self._corpus_loaded = True
    
    def _load_dense_model(self):
        """加载 Dense Retriever"""
        if self._dense_loaded:
            return
        
        self._load_corpus()
        
        if not os.path.exists(self.dense_model_path):
            logger.warning(f"[SimpleRetriever] Dense model not found: {self.dense_model_path}")
            self._dense_loaded = True
            return
        
        try:
            self.dense_tokenizer = AutoTokenizer.from_pretrained(self.dense_model_path)
            self.dense_model = AutoModel.from_pretrained(self.dense_model_path).to(self.device)
            self.dense_model.eval()
            
            # 预编码法条库
            logger.info("[SimpleRetriever] Encoding law corpus...")
            self.law_embeddings = self._encode_texts(self.law_texts, max_length=256)
            
            logger.info("[SimpleRetriever] Dense model loaded")
            self._dense_loaded = True
        except Exception as e:
            logger.warning(f"[SimpleRetriever] Failed to load dense model: {e}")
            self._dense_loaded = True
    
    def _encode_texts(self, texts: List[str], max_length: int = 512) -> torch.Tensor:
        """编码文本为向量"""
        embeddings = []
        batch_size = 32
        
        with torch.no_grad():
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                for i in range(0, len(texts), batch_size):
                    batch = texts[i:i+batch_size]
                    inputs = self.dense_tokenizer(
                        batch, padding=True, truncation=True, 
                        max_length=max_length, return_tensors='pt',
                        return_overflowing_tokens=False
                    ).to(self.device)
                    
                    outputs = self.dense_model(**inputs)
                    emb = outputs.last_hidden_state[:, 0, :]
                    emb = F.normalize(emb, p=2, dim=1)
                    embeddings.append(emb.cpu())
        
        return torch.cat(embeddings, dim=0)
    
    def retrieve_with_scores(self, queries: List[str], top_k: int = 50) -> List[List[Tuple[str, float]]]:
        """执行检索，返回每个 query 的 top-k (法条 ID, 分数) 列表"""
        self._load_dense_model()
        
        if not hasattr(self, 'law_embeddings') or self.law_embeddings is None:
            return [[] for _ in queries]
        
        # 编码查询
        query_embeddings = self._encode_texts(queries, max_length=512)
        
        # 计算相似度
        scores = torch.matmul(query_embeddings, self.law_embeddings.T)
        
        results = []
        for i in range(len(queries)):
            top_scores, top_indices = torch.topk(scores[i], k=min(top_k, len(self.law_ids)))
            result = [(self.law_ids[idx], score) for idx, score in zip(top_indices.tolist(), top_scores.tolist())]
            results.append(result)
        
        return results
    
    def retrieve(self, queries: List[str], top_k: int = 50) -> List[List[str]]:
        """执行检索，返回每个 query 的 top-k 法条 ID"""
        results_with_scores = self.retrieve_with_scores(queries, top_k)
        return [[law_id for law_id, _ in result] for result in results_with_scores]
    
    def merge_results(self, queries: List[str], top_k: int = 50) -> Tuple[List[str], Dict[str, float]]:
        """
        合并多个查询的检索结果
        返回：(排序后的法条 ID 列表, 法条 ID -> 最高分数 的映射)
        """
        results = self.retrieve_with_scores(queries, top_k)
        
        all_scores = {}
        for result in results:
            for law_id, score in result:
                if law_id not in all_scores:
                    all_scores[law_id] = score
                else:
                    all_scores[law_id] = max(all_scores[law_id], score)
        
        # 按分数排序
        sorted_items = sorted(all_scores.items(), key=lambda x: x[1], reverse=True)
        sorted_ids = [law_id for law_id, _ in sorted_items[:top_k]]
        
        return sorted_ids, all_scores


# ============== QueryGen 奖励函数（改进版 v3）==============
class QueryGenReward(ORM):
    """
    QueryGen 任务的奖励函数
    
    核心思想：
    - QueryGen 的作用是生成多样化的查询，帮助 Dense Retriever 找到更多候选法条
    - 因此应该直接评估 Dense 阶段的召回效果，而不是经过 Reranker 的
    - 多样性很重要，因为不同的查询可以覆盖案件的不同方面
    
    奖励公式：
        Reward = 0.05 * Format + 0.20 * Diversity + 0.75 * DenseScore
        
        其中 DenseScore = 0.60 * Recall@50 + 0.25 * MRR + 0.15 * HitRatio
        
    解释：
    - Recall@50：Dense 检索前 50 条中包含了多少 ground truth（最重要）
    - MRR：第一个正确法条的排名（排序质量）
    - HitRatio：是否至少找到一个正确法条（保底奖励）
    """
    
    def __init__(
        self,
        format_weight: float = 0.05,       # 格式权重（最低）
        diversity_weight: float = 0.20,    # 多样性权重
        dense_weight: float = 0.75,        # Dense 检索效果权重（最高）
        min_queries: int = 3,
        max_queries: int = 8,
        dense_top_k: int = 50,             # Dense 检索 top-k
    ):
        self.format_weight = format_weight
        self.diversity_weight = diversity_weight
        self.dense_weight = dense_weight
        self.min_queries = min_queries
        self.max_queries = max_queries
        self.dense_top_k = dense_top_k
        
        # 懒加载检索器
        self._retriever = None
    
    @property
    def retriever(self):
        if self._retriever is None:
            self._retriever = SimpleRetriever()
        return self._retriever
    
    def _parse_queries(self, output: str) -> List[str]:
        """解析 LLM 输出的查询列表"""
        text = strip_code_fences(output)
        if not text:
            return []
        
        # 尝试 JSON 数组解析
        arr = json_raw_decode_from(text, '[')
        if isinstance(arr, list):
            queries = []
            for item in arr:
                if isinstance(item, str):
                    q = item.strip()
                    if q and 4 <= len(q) <= 100:
                        queries.append(q)
            return queries
        
        # 尝试按行解析
        queries = []
        for line in text.split('\n'):
            line = line.strip()
            if not line:
                continue
            # 去掉编号前缀
            line = re.sub(r'^[\d]+[.)：:\s]+', '', line)
            line = re.sub(r'^[-*•]\s*', '', line)
            line = line.strip('" \'')
            if line and 4 <= len(line) <= 100:
                queries.append(line)
        
        return queries[:10]  # 最多取 10 个
    
    def _evaluate_format(self, queries: List[str]) -> float:
        """评估格式正确性"""
        if not queries:
            return 0.0
        
        n = len(queries)
        if self.min_queries <= n <= self.max_queries:
            return 1.0
        elif n < self.min_queries:
            # 太少：线性惩罚
            return 0.3 + 0.5 * (n / self.min_queries)
        else:
            # 太多：轻微惩罚
            return 0.8
    
    def _evaluate_diversity(self, queries: List[str]) -> float:
        """
        评估查询多样性
        
        使用基于字符的 Jaccard 距离，而不是完全相同的字符串
        """
        if not queries:
            return 0.0
        
        if len(queries) == 1:
            return 0.5  # 只有一个查询，多样性中等
        
        # 计算每对查询之间的 Jaccard 距离
        distances = []
        for i in range(len(queries)):
            for j in range(i + 1, len(queries)):
                set1 = set(queries[i])
                set2 = set(queries[j])
                if set1 or set2:
                    jaccard = len(set1 & set2) / len(set1 | set2)
                    distances.append(1 - jaccard)  # 距离 = 1 - 相似度
        
        if not distances:
            return 0.5
        
        # 平均多样性
        diversity = np.mean(distances)
        
        # 奖励查询数量适中
        if 4 <= len(queries) <= 6:
            diversity = min(1.0, diversity * 1.1)
        
        return min(1.0, max(0.0, diversity))
    
    def _evaluate_dense_retrieval(self, queries: List[str], ground_truth_laws: Set[str]) -> float:
        """
        评估 Dense 阶段的检索效果
        
        这是最重要的指标！
        QueryGen 的目标就是帮助 Dense Retriever 找到更多的 ground truth 法条
        """
        if not queries or not ground_truth_laws:
            return 0.0
        
        try:
            # 合并所有查询的检索结果
            merged_ids, _ = self.retriever.merge_results(queries, top_k=self.dense_top_k)
            
            if not merged_ids:
                return 0.0
            
            # 计算 Dense 阶段的指标
            # 使用 Recall@50 作为主要指标（与实际 pipeline 一致）
            recall_50 = compute_recall_at_k(merged_ids, ground_truth_laws, 50)
            
            # MRR：第一个正确法条的排名（排序质量）
            mrr = compute_mrr(merged_ids, ground_truth_laws)
            
            # Hit Ratio：至少找到一个正确法条（保底）
            hit_ratio = compute_hit_ratio(merged_ids, ground_truth_laws)
            
            # 综合得分
            # Recall 最重要，MRR 次之，Hit Ratio 保底
            score = 0.60 * recall_50 + 0.25 * mrr + 0.15 * hit_ratio
            
            return score
        
        except Exception as e:
            logger.warning(f"[QueryGenReward] Retrieval failed: {e}")
            return 0.0
    
    def __call__(
        self, 
        completions: List[str], 
        case_fact: List[str],
        ground_truth_laws: List[str] = None,
        **kwargs
    ) -> List[float]:
        rewards = []
        
        for i, (output, fact) in enumerate(zip(completions, case_fact)):
            # 解析真实标签
            gt_laws = set()
            if ground_truth_laws and i < len(ground_truth_laws):
                try:
                    gt_laws = set(str(x) for x in json.loads(ground_truth_laws[i]))
                except (json.JSONDecodeError, TypeError):
                    pass
            
            # 解析查询
            queries = self._parse_queries(output)
            
            # 计算格式分数
            format_score = self._evaluate_format(queries)
            
            # 计算多样性分数
            diversity_score = self._evaluate_diversity(queries)
            
            # 计算 Dense 检索效果
            if gt_laws and queries:
                dense_score = self._evaluate_dense_retrieval(queries, gt_laws)
            else:
                # 没有 ground truth 或没有查询，给低分
                dense_score = 0.2 if queries else 0.0
            
            # 综合奖励
            reward = (
                self.format_weight * format_score + 
                self.diversity_weight * diversity_score +
                self.dense_weight * dense_score
            )
            rewards.append(reward)
        
        return rewards


# ============== LawSelect 奖励函数（改进版 v3）==============
class LawSelectReward(ORM):
    """
    LawSelect 任务的奖励函数
    
    核心思想：
    - LawSelect 的作用是从候选法条中选出正确的法条，并且排序正确
    - 全量召回率最重要：必须选出所有 ground truth
    - MRR 和 NDCG 评估排序质量
    - 精确率避免选太多无关法条
    
    奖励公式：
        Reward = 0.05 * Format + 0.45 * Recall + 0.30 * RankingScore + 0.20 * Precision
        
        其中 RankingScore = 0.6 * MRR + 0.4 * NDCG
        
    解释：
    - Recall：选中的法条中包含了多少 ground truth（最重要）
    - MRR：第一个正确法条的排名
    - NDCG：综合排序质量
    - Precision：选中的法条中有多少是正确的
    """
    
    def __init__(
        self,
        format_weight: float = 0.05,
        recall_weight: float = 0.45,
        ranking_weight: float = 0.30,
        precision_weight: float = 0.20,
    ):
        self.format_weight = format_weight
        self.recall_weight = recall_weight
        self.ranking_weight = ranking_weight
        self.precision_weight = precision_weight
    
    def _parse_selected_laws(self, output: str) -> List[str]:
        """解析 LLM 选定的法条 ID（保持顺序）"""
        text = strip_code_fences(output)
        if not text:
            return []
        
        selected_ids = []
        seen = set()
        
        # 尝试解析 JSON 对象
        obj = json_raw_decode_from(text, '{')
        if isinstance(obj, dict):
            selected = obj.get("selected_articles", obj.get("selected", []))
            if isinstance(selected, list):
                for item in selected:
                    law_id = None
                    if isinstance(item, dict):
                        law_id = item.get("law_id")
                        if law_id is None:
                            law_id = item.get("idx")
                    elif isinstance(item, (str, int)):
                        law_id = item
                    
                    if law_id:
                        law_id = str(law_id)
                        if law_id not in seen:
                            selected_ids.append(law_id)
                            seen.add(law_id)
        
        # 如果 JSON 解析失败，尝试正则匹配
        if not selected_ids:
            matches = re.findall(r'"law_id"\s*:\s*"?(\d+)"?', text)
            for m in matches:
                if m not in seen:
                    selected_ids.append(m)
                    seen.add(m)
        
        return selected_ids
    
    def _evaluate_format(self, selected_ids: List[str], output: str) -> float:
        """评估格式正确性"""
        if not selected_ids:
            # 尝试了但没解析出来
            if '{' in output and 'selected' in output.lower():
                return 0.3
            return 0.0
        
        n = len(selected_ids)
        if 1 <= n <= 15:
            return 1.0
        elif n > 15:
            return 0.7  # 选太多，惩罚
        return 0.5
    
    def __call__(
        self, 
        completions: List[str], 
        ground_truth_laws: List[str],
        **kwargs
    ) -> List[float]:
        rewards = []
        
        for output, gt_json in zip(completions, ground_truth_laws):
            # 解析真实标签
            try:
                gt_laws = set(str(x) for x in json.loads(gt_json))
            except (json.JSONDecodeError, TypeError):
                gt_laws = set()
            
            # 解析模型输出（保持顺序）
            selected_ids = self._parse_selected_laws(output)
            
            # 计算格式分数
            format_score = self._evaluate_format(selected_ids, output)
            
            if not gt_laws:
                # 没有标签，给中性分
                rewards.append(0.5)
                continue
            
            if not selected_ids:
                # 没有输出，给最低分
                rewards.append(self.format_weight * format_score)
                continue
            
            # 计算全量召回率（最重要）
            recall = compute_recall(selected_ids, gt_laws)
            
            # 计算排序质量
            mrr = compute_mrr(selected_ids, gt_laws)
            ndcg = compute_ndcg(selected_ids, gt_laws)
            ranking_score = 0.6 * mrr + 0.4 * ndcg
            
            # 计算精确率
            precision = compute_precision(selected_ids, gt_laws)
            
            # 综合奖励
            reward = (
                self.format_weight * format_score +
                self.recall_weight * recall +
                self.ranking_weight * ranking_score +
                self.precision_weight * precision
            )
            
            rewards.append(reward)
        
        return rewards


# ============== 综合奖励函数 ==============
class AgentCombinedReward(ORM):
    """Agent 综合奖励函数"""
    
    def __init__(self):
        self.query_gen_reward = QueryGenReward()
        self.law_select_reward = LawSelectReward()
    
    def __call__(
        self, 
        completions: List[str],
        task_type: List[str],
        case_fact: List[str] = None,
        ground_truth_laws: List[str] = None,
        **kwargs
    ) -> List[float]:
        rewards = []
        
        for i, (output, t_type) in enumerate(zip(completions, task_type)):
            if t_type == "query_gen":
                fact = case_fact[i] if case_fact else ""
                gt = ground_truth_laws[i] if ground_truth_laws else "[]"
                reward = self.query_gen_reward([output], [fact], [gt])[0]
            elif t_type == "law_select":
                gt = ground_truth_laws[i] if ground_truth_laws else "[]"
                reward = self.law_select_reward([output], [gt])[0]
            else:
                logger.warning(f"Unknown task type: {t_type}")
                reward = 0.0
            
            rewards.append(reward)
        
        return rewards


# 注册奖励函数
orms["query_gen_reward"] = QueryGenReward
orms["law_select_reward"] = LawSelectReward
orms["agent_combined_reward"] = AgentCombinedReward
