"""
Hybrid Fusion Agent - 融合 MRAG 和 Agent 两条检索路线

核心思想：
1. MRAG 路线：fact → Dense → top-K
2. Agent 路线：fact → QueryGen → Dense → top-K  
3. RRF 融合两路结果
4. Reranker 对合并后的候选统一重排
5. LLM Select（使用与训练一致的统一提示词）

改进点：
- 去掉来源标记 [BOTH]/[MRAG]/[AGENT]，与训练数据保持一致
- 使用统一的提示词模板和格式化函数
- 统一的文本截断长度（300 字符）
"""

import json
import argparse
import math
import gc
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass, field

import numpy as np
from tqdm import tqdm
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)

# 复用 law_agent.py 中的组件
from mrag.agent.law_agent import (
    QueryGenerator,
    DenseRetriever,
    LawReranker,
    LawSelector,  # 复用统一的 LawSelector
    SearchResult,
    SelectionResult,
    RejectedResult,
    strip_code_fences,
    json_raw_decode_from,
)

# 导入共享提示词和配置
from mrag.agent.prompts import (
    LAWSELECT_SYSTEM_PROMPT,
    LAWSELECT_USER_TEMPLATE,
    MAX_LAW_TEXT_LENGTH,
    MAX_FACT_LENGTH,
    truncate_law_text,
    truncate_fact,
    format_candidate_law,
)

try:
    from vllm import LLM, SamplingParams
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False


# ============== 来源信息（仅用于内部统计，不传递给 LLM）==============
class SourceInfo:
    """来源信息，仅用于统计，不传递给 LLM"""
    BOTH = "BOTH"      # 双重命中
    MRAG = "MRAG"      # 仅 MRAG 检索到
    AGENT = "AGENT"    # 仅 Agent 检索到


@dataclass
class HybridSearchResult:
    """带来源信息的检索结果（来源信息仅用于统计）"""
    law_id: str
    law_name: str
    law_text: str
    score: float
    source: str = ""            # 来源（BOTH/MRAG/AGENT），仅用于统计
    mrag_rank: int = -1         # MRAG 排名 (-1 表示未检索到)
    agent_rank: int = -1        # Agent 排名 (-1 表示未检索到)
    fusion_score: float = 0.0   # RRF 融合分数

    @property
    def full_text(self) -> str:
        return f"{self.law_name}：{self.law_text}"
    
    def to_search_result(self) -> SearchResult:
        """转换为标准的 SearchResult"""
        return SearchResult(
            law_id=self.law_id,
            law_name=self.law_name,
            law_text=self.law_text,
            score=self.score,
        )


@dataclass
class HybridAgentOutput:
    """Hybrid Agent 输出"""
    query_id: str
    fact: str
    generated_queries: List[str]
    mrag_count: int                              # MRAG 路线候选数
    agent_count: int                             # Agent 路线候选数
    both_count: int                              # 双重命中数
    reranked_candidates: List[HybridSearchResult]  # 重排后的候选
    selected_laws: List[SelectionResult]
    rejected_laws: List[RejectedResult]


# ============== Hybrid Fusion Agent ==============
class HybridFusionAgent:
    """
    混合融合 Agent
    
    Pipeline:
    1. MRAG 路线: fact → Dense → top-K
    2. Agent 路线: fact → QueryGen → Dense → top-K
    3. RRF 融合
    4. Reranker 统一重排
    5. LLM Select（使用统一提示词，不含来源标记）
    """
    
    def __init__(
        self,
        llm_model_path: str,
        law_corpus_path: str,
        dense_model_path: str,
        reranker_model_path: str,
        device: str = "cuda",
        use_vllm: bool = True,
        tensor_parallel_size: int = 1,
        gpu_memory_util: float = 0.5,
        querygen_model_path: Optional[str] = None,
        lawselect_model_path: Optional[str] = None,
        rrf_k: int = 60,  # RRF 参数
        fusion_mode: str = "weighted_rrf",  # 融合模式: rrf, weighted_rrf, mrag_priority
    ):
        self.rrf_k = rrf_k
        self.fusion_mode = fusion_mode
        self.device = device
        self.use_vllm = use_vllm
        self.tensor_parallel_size = tensor_parallel_size
        self.gpu_memory_util = gpu_memory_util
        
        # 确定模型路径
        self.qg_model = querygen_model_path if querygen_model_path else llm_model_path
        self.ls_model = lawselect_model_path if lawselect_model_path else llm_model_path
        
        print(f"[HybridAgent] QueryGen 模型: {self.qg_model}")
        print(f"[HybridAgent] LawSelect 模型: {self.ls_model}")
        print(f"[HybridAgent] RRF k={rrf_k}, 融合模式={fusion_mode}")
        
        # QueryGenerator (for Agent path)
        self.query_generator = QueryGenerator(
            model_path=self.qg_model,
            device=device,
            use_vllm=use_vllm,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_util=gpu_memory_util,
        )
        
        # Dense Retriever (shared)
        self.dense_retriever = DenseRetriever(
            law_corpus_path=law_corpus_path,
            dense_model_path=dense_model_path,
            device=device,
        )
        
        # Reranker (shared)
        self.reranker = LawReranker(
            model_path=reranker_model_path,
            device=device,
        )
        
        # LawSelector (使用统一的 LawSelector，延迟加载)
        self.law_selector = None
    
    def _load_law_selector(self):
        """延迟加载 LawSelector"""
        if self.law_selector is not None:
            return
        
        print(f"\n[HybridAgent] 释放 QueryGen 模型，准备加载 LawSelect 模型...")
        
        # 释放 QueryGenerator
        if hasattr(self.query_generator, 'llm') and self.query_generator.llm is not None:
            del self.query_generator.llm
            self.query_generator.llm = None
        if hasattr(self.query_generator, 'model') and self.query_generator.model is not None:
            del self.query_generator.model
            self.query_generator.model = None
        
        gc.collect()
        torch.cuda.empty_cache()
        
        if torch.cuda.is_available():
            free_mem = torch.cuda.mem_get_info()[0] / 1024**3
            total_mem = torch.cuda.mem_get_info()[1] / 1024**3
            print(f"[HybridAgent] 释放后显存: {free_mem:.1f}/{total_mem:.1f} GiB 可用")
        
        # 使用统一的 LawSelector（与 law_agent.py 相同）
        print(f"[HybridAgent] 加载 LawSelect 模型: {self.ls_model}")
        self.law_selector = LawSelector(
            model_path=self.ls_model,
            device=self.device,
            use_vllm=self.use_vllm,
            tensor_parallel_size=self.tensor_parallel_size,
            gpu_memory_util=self.gpu_memory_util,
        )
    
    def _mrag_dense_search(self, fact: str, top_k: int = 50) -> List[SearchResult]:
        """MRAG 路线: 直接用 fact 作为 query 进行 Dense 检索"""
        query = fact[:1500] if len(fact) > 1500 else fact
        return self.dense_retriever.search([query], top_k=top_k)
    
    def _agent_dense_search(
        self, queries: List[str], top_k: int = 50
    ) -> List[SearchResult]:
        """Agent 路线: 用生成的 queries 进行 Dense 检索"""
        if not queries:
            return []
        return self.dense_retriever.search(queries, top_k=top_k)
    
    def _rrf_fusion(
        self,
        mrag_results: List[SearchResult],
        agent_results: List[SearchResult],
        top_k: int = 80,
        fusion_mode: str = "weighted_rrf",  # 融合模式
    ) -> List[HybridSearchResult]:
        """
        改进的融合策略
        
        支持五种融合模式:
        1. "rrf": 标准 RRF（两路权重相等）
        2. "weighted_rrf": 加权 RRF（MRAG 权重更高 + 双重命中加成）
        3. "mrag_priority": MRAG 优先（按分数混合，MRAG 权重极高）
        4. "mrag_first": MRAG 完全优先（保留 MRAG 完整排序，Agent 只追加补充）
        5. "agent_first": Agent 完全优先（保留 Agent 完整排序，MRAG 只追加补充）；当 Agent 优于 MRAG 时推荐
        
        来源信息仅用于统计，不传递给 LLM
        """
        # 构建法条 ID 到排名的映射
        mrag_ranks: Dict[str, int] = {}
        agent_ranks: Dict[str, int] = {}
        
        for rank, r in enumerate(mrag_results, 1):
            if r.law_id not in mrag_ranks:
                mrag_ranks[r.law_id] = rank
        
        for rank, r in enumerate(agent_results, 1):
            if r.law_id not in agent_ranks:
                agent_ranks[r.law_id] = rank
        
        # 收集所有唯一的法条 ID
        all_law_ids: Set[str] = set(mrag_ranks.keys()) | set(agent_ranks.keys())
        
        # 构建法条信息映射
        law_info: Dict[str, SearchResult] = {}
        for r in mrag_results + agent_results:
            if r.law_id not in law_info:
                law_info[r.law_id] = r
        
        # ============ mrag_first 模式：完全保留 MRAG 排序，Agent 只追加 ============
        if fusion_mode == "mrag_first":
            fusion_results: List[HybridSearchResult] = []
            mrag_law_ids = set()
            
            # Step 1: 完整保留 MRAG 结果（保持原始排序）
            for rank, r in enumerate(mrag_results, 1):
                if r.law_id in mrag_law_ids:
                    continue
                mrag_law_ids.add(r.law_id)
                agent_rank = agent_ranks.get(r.law_id, -1)
                source = SourceInfo.BOTH if agent_rank > 0 else SourceInfo.MRAG
                
                # 使用大基数保证 MRAG 结果在前
                # MRAG 排名 1 → score = 1000, 排名 100 → score = 901
                score = 1000.0 - rank + 1
                
                fusion_results.append(HybridSearchResult(
                    law_id=r.law_id,
                    law_name=r.law_name,
                    law_text=r.law_text,
                    score=r.score,
                    source=source,
                    mrag_rank=rank,
                    agent_rank=agent_rank,
                    fusion_score=score,
                ))
            
            # Step 2: 追加 Agent 独有的法条（MRAG 没有检索到的）
            agent_unique: List[HybridSearchResult] = []
            for rank, r in enumerate(agent_results, 1):
                if r.law_id in mrag_law_ids:
                    continue  # 已在 MRAG 结果中
                if r.law_id in [x.law_id for x in agent_unique]:
                    continue  # 去重
                
                # Agent 独有的法条，分数基数更低，按 Agent 排名
                # Agent 排名 1 → score = 100, 排名 100 → score = 1
                score = 100.0 - rank + 1
                
                agent_unique.append(HybridSearchResult(
                    law_id=r.law_id,
                    law_name=r.law_name,
                    law_text=r.law_text,
                    score=r.score,
                    source=SourceInfo.AGENT,
                    mrag_rank=-1,
                    agent_rank=rank,
                    fusion_score=score,
                ))
            
            # 按 Agent 排名排序后追加
            agent_unique.sort(key=lambda x: x.agent_rank)
            fusion_results.extend(agent_unique)
            
            return fusion_results[:top_k]
        
        # ============ agent_first 模式：完全保留 Agent 排序，MRAG 只追加（当 Agent > MRAG 时推荐）============
        if fusion_mode == "agent_first":
            fusion_results = []
            agent_law_ids = set()
            
            # Step 1: 完整保留 Agent 结果（保持原始排序）
            for rank, r in enumerate(agent_results, 1):
                if r.law_id in agent_law_ids:
                    continue
                agent_law_ids.add(r.law_id)
                mrag_rank = mrag_ranks.get(r.law_id, -1)
                source = SourceInfo.BOTH if mrag_rank > 0 else SourceInfo.AGENT
                score = 1000.0 - rank + 1
                fusion_results.append(HybridSearchResult(
                    law_id=r.law_id,
                    law_name=r.law_name,
                    law_text=r.law_text,
                    score=r.score,
                    source=source,
                    mrag_rank=mrag_rank,
                    agent_rank=rank,
                    fusion_score=score,
                ))
            
            # Step 2: 追加 MRAG 独有的法条
            mrag_unique: List[HybridSearchResult] = []
            for rank, r in enumerate(mrag_results, 1):
                if r.law_id in agent_law_ids:
                    continue
                if r.law_id in [x.law_id for x in mrag_unique]:
                    continue
                score = 100.0 - rank + 1
                mrag_unique.append(HybridSearchResult(
                    law_id=r.law_id,
                    law_name=r.law_name,
                    law_text=r.law_text,
                    score=r.score,
                    source=SourceInfo.MRAG,
                    mrag_rank=rank,
                    agent_rank=-1,
                    fusion_score=score,
                ))
            mrag_unique.sort(key=lambda x: x.mrag_rank)
            fusion_results.extend(mrag_unique)
            return fusion_results[:top_k]
        
        # ============ 其他融合模式 ============
        # MRAG 路线质量更稳定，给予更高权重
        MRAG_WEIGHT = 2.0 if fusion_mode == "weighted_rrf" else 1.0
        AGENT_WEIGHT = 0.3 if fusion_mode == "weighted_rrf" else 1.0
        BOTH_BONUS = 0.03 if fusion_mode == "weighted_rrf" else 0.0
        
        # 计算融合分数并标记来源
        fusion_results: List[HybridSearchResult] = []
        
        for law_id in all_law_ids:
            mrag_rank = mrag_ranks.get(law_id, -1)
            agent_rank = agent_ranks.get(law_id, -1)
            
            # 确定来源（仅用于统计）
            if mrag_rank > 0 and agent_rank > 0:
                source = SourceInfo.BOTH
            elif mrag_rank > 0:
                source = SourceInfo.MRAG
            else:
                source = SourceInfo.AGENT
            
            # ============ 融合分数计算 ============
            if fusion_mode == "mrag_priority":
                # MRAG 优先模式：MRAG 结果排在前面，Agent 只补充
                if mrag_rank > 0:
                    score = 1.0 / mrag_rank + (0.01 if agent_rank > 0 else 0)
                else:
                    score = 0.01 / (self.rrf_k + agent_rank)
            else:
                # RRF 或加权 RRF 模式
                score = 0.0
                if mrag_rank > 0:
                    score += MRAG_WEIGHT * 1.0 / (self.rrf_k + mrag_rank)
                if agent_rank > 0:
                    score += AGENT_WEIGHT * 1.0 / (self.rrf_k + agent_rank)
                
                if source == SourceInfo.BOTH:
                    score += BOTH_BONUS
            
            orig_score = law_info[law_id].score
            
            fusion_results.append(HybridSearchResult(
                law_id=law_id,
                law_name=law_info[law_id].law_name,
                law_text=law_info[law_id].law_text,
                score=orig_score,
                source=source,
                mrag_rank=mrag_rank,
                agent_rank=agent_rank,
                fusion_score=score,
            ))
        
        # 按融合分数排序
        fusion_results.sort(key=lambda x: x.fusion_score, reverse=True)
        
        return fusion_results[:top_k]
    
    def _rerank_hybrid(
        self,
        fact: str,
        candidates: List[HybridSearchResult],
        top_k: int = 25,
    ) -> List[HybridSearchResult]:
        """对融合后的候选进行重排"""
        if not candidates:
            return []
        
        # 转换为 SearchResult 进行重排
        search_results = [c.to_search_result() for c in candidates]
        
        reranked = self.reranker.rerank(fact, search_results, top_k=top_k)
        
        # 恢复来源信息
        id_to_hybrid: Dict[str, HybridSearchResult] = {c.law_id: c for c in candidates}
        
        hybrid_reranked: List[HybridSearchResult] = []
        for r in reranked:
            orig = id_to_hybrid[r.law_id]
            hybrid_reranked.append(HybridSearchResult(
                law_id=r.law_id,
                law_name=r.law_name,
                law_text=r.law_text,
                score=r.score,  # 使用 reranker 分数
                source=orig.source,
                mrag_rank=orig.mrag_rank,
                agent_rank=orig.agent_rank,
                fusion_score=orig.fusion_score,
            ))
        
        return hybrid_reranked
    
    def retrieve(
        self,
        query_ids: List[str],
        facts: List[str],
        dense_top_k: int = 50,
        fusion_top_k: int = 80,
        rerank_top_k: int = 20,
        batch_size: int = 8,
        min_selected: int = 5,
        skip_llm_select: bool = False,  # 跳过 LLM Select，直接用 Reranker 结果
    ) -> List[HybridAgentOutput]:
        """执行 Hybrid Fusion Pipeline
        
        Args:
            skip_llm_select: 如果为 True，跳过 LLM Select 阶段，直接输出 Reranker top-K
                            这可以避免 LLM 过滤导致的信息损失
        """
        print("\n" + "=" * 60)
        print(f"[HybridAgent] 开始处理 {len(facts)} 个样本")
        print("=" * 60 + "\n")
        
        # Step 1: Agent QueryGen
        print("[Step 1/5] 生成检索查询 (Agent QueryGen)...")
        query_results = self.query_generator.generate(facts, batch_size=batch_size)
        
        # Step 2 & 3: 两路 Dense 检索 + RRF 融合
        print(f"[Step 2/5] MRAG Dense 检索 (top-{dense_top_k})...")
        print(f"[Step 3/5] Agent Dense 检索 (top-{dense_top_k}) + RRF 融合...")
        
        all_merged_candidates: List[List[HybridSearchResult]] = []
        stats_mrag = []
        stats_agent = []
        stats_both = []
        
        for i, (fact, qr) in enumerate(tqdm(
            zip(facts, query_results), total=len(facts), desc="Dense + RRF"
        )):
            # MRAG 路线
            mrag_cands = self._mrag_dense_search(fact, top_k=dense_top_k)
            
            # Agent 路线
            queries = qr.queries if qr.queries else [fact[:500]]
            agent_cands = self._agent_dense_search(queries, top_k=dense_top_k)
            
            # 融合（使用配置的融合模式）
            merged = self._rrf_fusion(
                mrag_cands, agent_cands, 
                top_k=fusion_top_k, 
                fusion_mode=self.fusion_mode
            )
            
            # 统计
            both_count = sum(1 for c in merged if c.source == SourceInfo.BOTH)
            mrag_only = sum(1 for c in merged if c.source == SourceInfo.MRAG)
            agent_only = sum(1 for c in merged if c.source == SourceInfo.AGENT)
            
            stats_mrag.append(mrag_only)
            stats_agent.append(agent_only)
            stats_both.append(both_count)
            
            all_merged_candidates.append(merged)
        
        # 统计融合效果
        avg_both = sum(stats_both) / len(stats_both) if stats_both else 0
        avg_mrag = sum(stats_mrag) / len(stats_mrag) if stats_mrag else 0
        avg_agent = sum(stats_agent) / len(stats_agent) if stats_agent else 0
        print(f"[HybridAgent] 平均来源分布: BOTH={avg_both:.1f}, MRAG={avg_mrag:.1f}, AGENT={avg_agent:.1f}")
        
        # Step 4: Reranker
        print(f"[Step 4/5] Reranker 重排 (top-{rerank_top_k})...")
        all_reranked: List[List[HybridSearchResult]] = []
        
        for fact, merged in tqdm(
            zip(facts, all_merged_candidates), total=len(facts), desc="Reranking"
        ):
            reranked = self._rerank_hybrid(fact, merged, top_k=rerank_top_k)
            all_reranked.append(reranked)
        
        # Step 5: LLM Select 或直接输出 Reranker 结果
        outputs: List[HybridAgentOutput] = []
        
        if skip_llm_select:
            # 跳过 LLM Select，直接使用 Reranker 结果
            print("[Step 5/5] 跳过 LLM-Select，直接使用 Reranker 结果...")
            
            for i, qid in enumerate(query_ids):
                reranked_cands = all_reranked[i]
                
                # 将 Reranker 结果转换为 SelectionResult
                selected_laws = []
                for cand in reranked_cands:
                    # 根据来源和分数计算置信度
                    conf = 0.7 if cand.source == SourceInfo.BOTH else 0.6
                    conf += 0.2 * (1 / (1 + math.exp(-cand.score)))
                    conf = min(0.95, max(0.5, conf))
                    
                    selected_laws.append(SelectionResult(
                        law_id=cand.law_id,
                        law_name=cand.law_name,
                        reason=f"Reranker 分数: {cand.score:.3f}",
                        confidence=conf,
                    ))
                
                outputs.append(HybridAgentOutput(
                    query_id=qid,
                    fact=facts[i],
                    generated_queries=query_results[i].queries,
                    mrag_count=stats_mrag[i] + stats_both[i],
                    agent_count=stats_agent[i] + stats_both[i],
                    both_count=stats_both[i],
                    reranked_candidates=reranked_cands,
                    selected_laws=selected_laws,
                    rejected_laws=[],  # 跳过 LLM Select 时没有 rejected
                ))
        else:
            # 使用 LLM Select
            print("[Step 5/5] 筛选相关法条 (LLM-Select)...")
            self._load_law_selector()
            
            # 转换为 SearchResult 列表供 LawSelector 使用
            search_results_list = [
                [c.to_search_result() for c in candidates]
                for candidates in all_reranked
            ]
            
            selection_results = self.law_selector.select(
                facts, search_results_list, batch_size=min(batch_size, 4)
            )
            
            fallback_count = 0
            
            for i, qid in enumerate(query_ids):
                selected_laws = selection_results[i][0]
                rejected_laws = selection_results[i][1]
                reranked_cands = all_reranked[i]
                
                # Fallback 策略
                if len(selected_laws) < min_selected and reranked_cands:
                    fallback_count += 1
                    existing_ids = {s.law_id for s in selected_laws}
                    rejected_ids = {r.law_id for r in rejected_laws}
                    
                    # 优先添加双重命中的法条
                    sorted_cands = sorted(
                        reranked_cands, 
                        key=lambda c: (c.source == SourceInfo.BOTH, c.score), 
                        reverse=True
                    )
                    
                    for cand in sorted_cands:
                        if cand.law_id not in existing_ids and cand.law_id not in rejected_ids:
                            conf = 0.7 if cand.source == SourceInfo.BOTH else 0.6
                            conf += 0.15 * (1 / (1 + math.exp(-cand.score)))
                            conf = min(0.85, max(0.6, conf))
                            
                            selected_laws.append(SelectionResult(
                                law_id=cand.law_id,
                                law_name=cand.law_name,
                                reason=f"由 Reranker 推荐（分数: {cand.score:.3f}）",
                                confidence=conf,
                            ))
                            existing_ids.add(cand.law_id)
                        if len(selected_laws) >= min_selected:
                            break
                
                outputs.append(HybridAgentOutput(
                    query_id=qid,
                    fact=facts[i],
                    generated_queries=query_results[i].queries,
                    mrag_count=stats_mrag[i] + stats_both[i],
                    agent_count=stats_agent[i] + stats_both[i],
                    both_count=stats_both[i],
                    reranked_candidates=reranked_cands,
                    selected_laws=selected_laws,
                    rejected_laws=rejected_laws,
                ))
            
            if fallback_count > 0:
                print(f"[HybridAgent] Fallback 触发: {fallback_count}/{len(facts)} 个样本")
        
        print("\n[HybridAgent] 处理完成！")
        return outputs


def convert_to_mrag_format(
    outputs: List[HybridAgentOutput],
    output_path: str,
) -> None:
    """将 Hybrid Agent 输出转换为 TREC 格式"""
    with open(output_path, "w", encoding="utf-8") as f:
        for output in outputs:
            rank = 1
            for selected in output.selected_laws:
                f.write(
                    f"{output.query_id}\tQ0\t{selected.law_id}\t{rank}\t{selected.confidence:.6f}\thybrid_agent\n"
                )
                rank += 1
    
    print(f"[HybridAgent] 已保存检索结果到: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid Fusion Agent")
    
    # 必需参数
    parser.add_argument("--llm_model", type=str, required=True)
    parser.add_argument("--law_corpus", type=str, required=True)
    parser.add_argument("--dense_model", type=str, required=True)
    parser.add_argument("--reranker_model", type=str, required=True)
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    
    # 可选模型路径
    parser.add_argument("--querygen_model", type=str, default=None)
    parser.add_argument("--lawselect_model", type=str, default=None)
    
    # 检索参数
    parser.add_argument("--dense_top_k", type=int, default=50)
    parser.add_argument("--fusion_top_k", type=int, default=80)
    parser.add_argument("--rerank_top_k", type=int, default=20)
    parser.add_argument("--min_selected", type=int, default=5)
    parser.add_argument("--rrf_k", type=int, default=60)
    parser.add_argument("--fusion_mode", type=str, default="rrf",
                        choices=["rrf", "weighted_rrf", "mrag_priority", "mrag_first", "agent_first"],
                        help="融合模式: rrf, weighted_rrf, mrag_priority, mrag_first(MRAG优先+Agent补充), agent_first(Agent优先+MRAG补充，Agent优于MRAG时推荐)")
    parser.add_argument("--batch_size", type=int, default=8)
    
    # 其他参数
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--use_vllm", action="store_true")
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--gpu_memory_util", type=float, default=0.5)
    parser.add_argument("--save_details", action="store_true")
    parser.add_argument("--skip_llm_select", action="store_true",
                        help="跳过 LLM Select，直接输出 Reranker 结果（可提高 Recall）")
    
    args = parser.parse_args()
    
    # 加载输入数据
    print(f"[Main] 加载输入数据: {args.input_file}")
    data_items: List[Dict[str, str]] = []
    with open(args.input_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            data_items.append({
                "text_id": str(item.get("text_id", "")),
                "text": item.get("text", "")
            })
    print(f"[Main] 加载了 {len(data_items)} 个样本")
    
    # 初始化 Agent
    print(f"[Main] 融合模式: {args.fusion_mode}")
    agent = HybridFusionAgent(
        llm_model_path=args.llm_model,
        law_corpus_path=args.law_corpus,
        dense_model_path=args.dense_model,
        reranker_model_path=args.reranker_model,
        device=args.device,
        use_vllm=args.use_vllm,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_util=args.gpu_memory_util,
        querygen_model_path=args.querygen_model,
        lawselect_model_path=args.lawselect_model,
        rrf_k=args.rrf_k,
        fusion_mode=args.fusion_mode,
    )
    
    # 执行检索
    query_ids = [item["text_id"] for item in data_items]
    facts = [item["text"] for item in data_items]
    
    outputs = agent.retrieve(
        query_ids=query_ids,
        facts=facts,
        dense_top_k=args.dense_top_k,
        fusion_top_k=args.fusion_top_k,
        rerank_top_k=args.rerank_top_k,
        min_selected=args.min_selected,
        batch_size=args.batch_size,
        skip_llm_select=args.skip_llm_select,
    )
    
    # 转换并保存结果
    convert_to_mrag_format(outputs, args.output_file)
    
    # 保存详细输出
    if args.save_details:
        detail_path = args.output_file.replace(".tsv", "_details.json")
        details = []
        for output in outputs:
            # 统计来源分布
            source_stats = {"BOTH": 0, "MRAG": 0, "AGENT": 0}
            for c in output.reranked_candidates:
                if c.source == SourceInfo.BOTH:
                    source_stats["BOTH"] += 1
                elif c.source == SourceInfo.MRAG:
                    source_stats["MRAG"] += 1
                else:
                    source_stats["AGENT"] += 1
            
            details.append({
                "query_id": output.query_id,
                "fact": output.fact[:500] + "..." if len(output.fact) > 500 else output.fact,
                "generated_queries": output.generated_queries,
                "source_distribution": source_stats,
                "reranked_candidates": [
                    {
                        "law_id": c.law_id,
                        "law_name": c.law_name,
                        "source": c.source,
                        "mrag_rank": c.mrag_rank,
                        "agent_rank": c.agent_rank,
                        "rerank_score": c.score,
                    }
                    for c in output.reranked_candidates[:10]
                ],
                "selected_laws": [
                    {
                        "law_id": s.law_id,
                        "law_name": s.law_name,
                        "reason": s.reason,
                        "confidence": s.confidence,
                    }
                    for s in output.selected_laws
                ],
                "rejected_laws": [
                    {
                        "law_id": r.law_id,
                        "law_name": r.law_name,
                        "reason": r.reason,
                    }
                    for r in output.rejected_laws
                ],
            })
        
        with open(detail_path, "w", encoding="utf-8") as f:
            json.dump(details, f, ensure_ascii=False, indent=2)
        print(f"[Main] 详细输出已保存到: {detail_path}")
    
    print("[Main] 完成！")


if __name__ == "__main__":
    main()
