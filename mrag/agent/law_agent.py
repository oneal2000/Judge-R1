"""
LLM-based 法条检索 Agent (精简版)

四阶段 Pipeline:
(A) LLM Planning (QueryGen): 基于 Fact 生成检索查询
(B) Dense Retriever: 检索 top-50 候选法条
(C) Reranker: Cross-encoder 重排，取 top-10
(D) LLM-Select: LLM 从候选法条中筛选最相关的法条

改进点：
- 训练和推理使用完全一致的提示词模板
- 统一的文本截断长度（300 字符）
- 统一的候选法条格式化
"""

import json
import argparse
import re
import math
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, field

import numpy as np
from tqdm import tqdm

import torch
from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
)
import faiss

try:
    from vllm import LLM, SamplingParams
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False
    print("警告: vLLM 未安装，将使用 transformers 进行推理")

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


# =========================================================
# 通用工具函数
# =========================================================

META_HINTS = (
    "首先", "你是一名", "用户要求", "案件事实回顾", "事实回顾", "要求：", "输出：",
    "请以JSON", "例如：", "分析：", "下面", "综上", "因此"
)


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


def is_meta_or_junk_query(q: str) -> bool:
    """过滤明显不是检索 query 的内容"""
    if not q:
        return True
    if any(h in q for h in META_HINTS):
        return True
    if len(q) > 90 or len(q) < 4:
        return True
    return False


def dedup_keep_order(items: List[str]) -> List[str]:
    seen = set()
    out = []
    for x in items:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


# ============== 数据结构定义 ==============
@dataclass
class QueryGenResult:
    """查询生成结果"""
    queries: List[str] = field(default_factory=list)
    raw_output: str = ""


@dataclass
class SearchResult:
    """检索结果"""
    law_id: str
    law_name: str
    law_text: str
    score: float

    @property
    def full_text(self) -> str:
        return f"{self.law_name}：{self.law_text}"


@dataclass
class SelectionResult:
    """筛选结果 - 包含完整的法条信息"""
    law_id: str
    law_name: str
    reason: str
    confidence: float


@dataclass
class RejectedResult:
    """被排除的法条 - 包含完整的法条信息"""
    law_id: str
    law_name: str
    reason: str


@dataclass
class AgentOutput:
    """Agent 输出"""
    query_id: str
    fact: str
    generated_queries: List[str]
    candidate_laws: List[SearchResult]
    selected_laws: List[SelectionResult]
    rejected_laws: List[RejectedResult]


# ============== (A) LLM Planning (QueryGen) ==============
class QueryGenerator:
    """基于 LLM 生成检索查询"""

    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        use_vllm: bool = True,
        tensor_parallel_size: int = 1,
        gpu_memory_util: float = 0.5,
    ):
        self.model_path = model_path
        self.device = device
        self.use_vllm = bool(use_vllm and VLLM_AVAILABLE)

        print(f"[QueryGenerator] Loading model from {model_path}...")

        if self.use_vllm:
            self.llm = LLM(
                model=model_path,
                tensor_parallel_size=tensor_parallel_size,
                trust_remote_code=True,
                max_model_len=4096,
                gpu_memory_utilization=gpu_memory_util,
            )
            try:
                self.tokenizer = self.llm.get_tokenizer()
            except Exception:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    model_path, trust_remote_code=True, use_fast=False
                )
            self.sampling_params = SamplingParams(
                temperature=0.1,
                top_p=0.9,
                max_tokens=256,
                stop=["\n\n\n"],
            )
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path, trust_remote_code=True, use_fast=False
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True,
            )

    def build_prompt(self, fact: str) -> str:
        # 使用统一的截断函数
        fact_truncated = truncate_fact(fact)
        content = QUERYGEN_USER_TEMPLATE.format(fact=fact_truncated)
        messages = [
            {"role": "system", "content": QUERYGEN_SYSTEM_PROMPT},
            {"role": "user", "content": content},
        ]
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    def parse_queries(self, raw_output: str) -> List[str]:
        """解析 LLM 生成的查询列表"""
        if not raw_output:
            return []

        text = strip_code_fences(raw_output)
        if not text:
            return []

        # 尝试 JSON 数组解析
        arr = json_raw_decode_from(text, '[')
        if isinstance(arr, list):
            queries = []
            for item in arr:
                if isinstance(item, str):
                    q = item.strip()
                    if q and not is_meta_or_junk_query(q):
                        queries.append(q)
            if queries:
                return dedup_keep_order(queries)

        # 尝试按行解析
        queries = []
        for line in text.split('\n'):
            line = line.strip()
            if not line:
                continue
            line = re.sub(r'^[\d]+[.)：:\s]+', '', line)
            line = re.sub(r'^[-*•]\s*', '', line)
            line = line.strip('" \'')
            if line and not is_meta_or_junk_query(line):
                queries.append(line)

        return dedup_keep_order(queries) if queries else []

    def generate(self, facts: List[str], batch_size: int = 8) -> List[QueryGenResult]:
        """批量生成检索查询"""
        prompts = [self.build_prompt(fact) for fact in facts]
        results: List[QueryGenResult] = []

        if self.use_vllm:
            outputs = self.llm.generate(prompts, self.sampling_params)
            for out in outputs:
                raw_text = out.outputs[0].text
                queries = self.parse_queries(raw_text)
                results.append(QueryGenResult(queries=queries, raw_output=raw_text))
            return results

        for prompt in tqdm(prompts, desc="Generating queries"):
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            with torch.no_grad():
                output_ids = self.model.generate(
                    **inputs, max_new_tokens=256, temperature=0.1, do_sample=True,
                )
            raw_text = self.tokenizer.decode(
                output_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True,
            )
            queries = self.parse_queries(raw_text)
            results.append(QueryGenResult(queries=queries, raw_output=raw_text))

        return results


# ============== (B) Dense Retriever ==============
class DenseRetriever:
    """纯 Dense Retriever"""

    def __init__(
        self,
        law_corpus_path: str,
        dense_model_path: str,
        device: str = "cuda",
    ):
        self.device = device

        # 加载法条库
        print(f"[DenseRetriever] Loading law corpus from {law_corpus_path}...")
        self.laws: List[Dict[str, str]] = []
        self.law_id_to_idx: Dict[str, int] = {}

        with open(law_corpus_path, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                item = json.loads(line)
                law_id = str(item["text_id"])
                self.laws.append({"id": law_id, "name": item["name"], "text": item["text"]})
                self.law_id_to_idx[law_id] = idx

        print(f"[DenseRetriever] Loaded {len(self.laws)} laws")

        # 初始化 Dense Retriever
        print(f"[DenseRetriever] Loading dense model from {dense_model_path}...")
        self.dense_tokenizer = AutoTokenizer.from_pretrained(dense_model_path)
        self.dense_model = AutoModel.from_pretrained(dense_model_path).to(self.device)
        self.dense_model.eval()

        # 构建 FAISS 索引
        self.build_dense_index()

    def encode_texts(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """编码文本为向量"""
        all_embeddings: List[np.ndarray] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            inputs = self.dense_tokenizer(
                batch, padding=True, truncation=True, max_length=400, return_tensors="pt",
            ).to(self.device)

            with torch.no_grad():
                outputs = self.dense_model(**inputs)
                embeddings = outputs.last_hidden_state[:, 0, :].detach().cpu().numpy()

            all_embeddings.append(embeddings)

        embeddings = np.vstack(all_embeddings)
        embeddings = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-12)
        return embeddings

    def build_dense_index(self) -> None:
        """构建 FAISS 索引"""
        print("[DenseRetriever] Building FAISS index...")
        texts = [f"{law['name']}：{law['text']}" for law in self.laws]
        law_embeddings = self.encode_texts(texts)
        dim = law_embeddings.shape[1]
        self.faiss_index = faiss.IndexFlatIP(dim)
        self.faiss_index.add(law_embeddings.astype(np.float32))
        print(f"[DenseRetriever] FAISS index built with {self.faiss_index.ntotal} vectors")

    def search(self, queries: List[str], top_k: int = 50) -> List[SearchResult]:
        """Dense 检索，返回 top_k 候选"""
        n = len(self.laws)
        combined_scores = np.zeros(n, dtype=np.float32)

        for query in queries:
            query = (query or "").strip()
            if not query:
                continue

            query_embedding = self.encode_texts([query])
            k = min(top_k * 2, n)  # 取更多再融合
            scores, indices = self.faiss_index.search(query_embedding.astype(np.float32), k)

            for idx, score in zip(indices[0], scores[0]):
                combined_scores[int(idx)] = max(combined_scores[int(idx)], float(score))

        top_indices = np.argsort(combined_scores)[::-1][:top_k]
        results: List[SearchResult] = []
        for idx in top_indices:
            law = self.laws[int(idx)]
            results.append(
                SearchResult(
                    law_id=law["id"],
                    law_name=law["name"],
                    law_text=law["text"],
                    score=float(combined_scores[int(idx)]),
                )
            )
        return results


# ============== (C) Reranker ==============
class LawReranker:
    """Cross-encoder Reranker
    
    注意: BERT 模型的 position embeddings 最大只支持 512 个 token，
    不能设置超过 512 的 max_len，否则会报错。
    """

    def __init__(self, model_path: str, device: str = "cuda", max_len: int = 512):
        self.device = device
        self.max_len = max_len  # BERT 最大支持 512

        print(f"[LawReranker] Loading reranker from {model_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.model.to(device)
        self.model.eval()
        print(f"[LawReranker] Loaded successfully")

    def _smart_truncate(self, query: str, passage: str) -> Tuple[str, str]:
        """智能截断，确保 query 和 passage 都能在 max_len 内保留足够信息。"""
        special_tokens = 3
        min_passage_tokens = 150
        available = self.max_len - special_tokens
        
        query_tokens = self.tokenizer.tokenize(query)
        passage_tokens = self.tokenizer.tokenize(passage)
        
        query_len = len(query_tokens)
        passage_len = len(passage_tokens)
        total_len = query_len + passage_len
        
        if total_len <= available:
            return query, passage
        
        max_passage_allocation = min(passage_len, available - 50)
        passage_allocation = min(max_passage_allocation, max(min_passage_tokens, passage_len))
        query_allocation = available - passage_allocation
        
        if query_len > query_allocation:
            query_tokens = query_tokens[:query_allocation]
            query = self.tokenizer.convert_tokens_to_string(query_tokens)
        
        if passage_len > passage_allocation:
            passage_tokens = passage_tokens[:passage_allocation]
            passage = self.tokenizer.convert_tokens_to_string(passage_tokens)
        
        return query, passage

    def rerank(
        self,
        query: str,
        candidates: List[SearchResult],
        top_k: int = 10,
        batch_size: int = 32,
    ) -> List[SearchResult]:
        """对候选法条进行重排"""
        if not candidates:
            return []

        pairs = []
        for c in candidates:
            passage = f"{c.law_name}：{c.law_text}"
            truncated_query, truncated_passage = self._smart_truncate(query, passage)
            pairs.append((truncated_query, truncated_passage))

        all_scores = []
        for i in range(0, len(pairs), batch_size):
            batch_pairs = pairs[i:i + batch_size]
            queries_batch = [p[0] for p in batch_pairs]
            passages_batch = [p[1] for p in batch_pairs]

            inputs = self.tokenizer(
                queries_batch,
                passages_batch,
                padding=True,
                truncation=True,
                max_length=self.max_len,
                return_tensors="pt",
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)
                scores = outputs.logits.squeeze(-1).cpu().numpy()

            if len(scores.shape) == 0:
                all_scores.append(float(scores))
            else:
                all_scores.extend(scores.tolist())

        scored_candidates = list(zip(candidates, all_scores))
        scored_candidates.sort(key=lambda x: x[1], reverse=True)

        results = []
        for cand, score in scored_candidates[:top_k]:
            results.append(
                SearchResult(
                    law_id=cand.law_id,
                    law_name=cand.law_name,
                    law_text=cand.law_text,
                    score=float(score),
                )
            )
        return results


# ============== (D) LLM-Select ==============
class LawSelector:
    """基于 LLM 的法条筛选器"""

    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        use_vllm: bool = True,
        tensor_parallel_size: int = 1,
        gpu_memory_util: float = 0.5,
    ):
        self.model_path = model_path
        self.device = device
        self.use_vllm = bool(use_vllm and VLLM_AVAILABLE)

        print(f"[LawSelector] Loading model from {model_path}...")

        if self.use_vllm:
            self.llm = LLM(
                model=model_path,
                tensor_parallel_size=tensor_parallel_size,
                trust_remote_code=True,
                max_model_len=8192,
                gpu_memory_utilization=gpu_memory_util,
            )
            try:
                self.tokenizer = self.llm.get_tokenizer()
            except Exception:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    model_path, trust_remote_code=True, use_fast=False
                )

            self.sampling_params = SamplingParams(
                temperature=0.1, top_p=0.95, max_tokens=2048, stop=["\n\n\n"],
            )
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path, trust_remote_code=True, use_fast=False
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True,
            )

    def build_prompt(self, fact: str, candidates: List[SearchResult]) -> str:
        # 使用统一的格式化函数
        law_list = "\n".join([
            format_candidate_law(
                idx=i,
                law_id=c.law_id,
                law_name=c.law_name,
                law_text=c.law_text,
                max_text_length=MAX_LAW_TEXT_LENGTH  # 使用统一的截断长度
            )
            for i, c in enumerate(candidates)
        ])

        # 使用统一的截断函数
        fact_truncated = truncate_fact(fact)
        
        content = LAWSELECT_USER_TEMPLATE.format(
            fact=fact_truncated,
            num_candidates=len(candidates),
            candidate_laws=law_list
        )
        messages = [
            {"role": "system", "content": LAWSELECT_SYSTEM_PROMPT},
            {"role": "user", "content": content},
        ]
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    def _find_candidate_by_law_id(
        self, law_id: str, candidates: List[SearchResult]
    ) -> Optional[SearchResult]:
        """根据 law_id 查找候选法条"""
        for c in candidates:
            if c.law_id == law_id:
                return c
        return None

    def _find_candidate_by_idx(
        self, idx: int, candidates: List[SearchResult]
    ) -> Optional[SearchResult]:
        """根据 idx (1-based) 查找候选法条"""
        i = idx - 1
        if 0 <= i < len(candidates):
            return candidates[i]
        return None

    def parse_selection(
        self, raw_output: str, candidates: List[SearchResult],
    ) -> Tuple[List[SelectionResult], List[RejectedResult]]:
        """解析 LLM 输出的筛选结果"""
        if not raw_output:
            return [], []

        text = strip_code_fences(raw_output)
        if not text:
            return [], []

        selected_results: List[SelectionResult] = []
        rejected_results: List[RejectedResult] = []

        obj = json_raw_decode_from(text, "{")
        if isinstance(obj, dict):
            selected = obj.get("selected_articles", obj.get("selected", []))
            rejected = obj.get("rejected_articles", obj.get("rejected", []))

            if isinstance(selected, list):
                for item in selected:
                    if not isinstance(item, dict):
                        continue

                    cand = None
                    law_id = item.get("law_id")
                    idx = item.get("idx")

                    if law_id:
                        cand = self._find_candidate_by_law_id(str(law_id), candidates)
                    if cand is None and idx is not None:
                        try:
                            cand = self._find_candidate_by_idx(int(idx), candidates)
                        except (ValueError, TypeError):
                            pass

                    if cand is not None:
                        reason = str(item.get("reason", "")).strip()
                        conf = item.get("confidence", 0.8)
                        try:
                            conf_f = float(conf)
                        except Exception:
                            conf_f = 0.8
                        conf_f = max(0.0, min(1.0, conf_f))

                        selected_results.append(SelectionResult(
                            law_id=cand.law_id,
                            law_name=cand.law_name,
                            reason=reason if reason else "与本案事实相关",
                            confidence=conf_f,
                        ))

            if isinstance(rejected, list):
                for item in rejected:
                    if not isinstance(item, dict):
                        continue

                    cand = None
                    law_id = item.get("law_id")
                    idx = item.get("idx")

                    if law_id:
                        cand = self._find_candidate_by_law_id(str(law_id), candidates)
                    if cand is None and idx is not None:
                        try:
                            cand = self._find_candidate_by_idx(int(idx), candidates)
                        except (ValueError, TypeError):
                            pass

                    if cand is not None:
                        reason = str(item.get("reason", "")).strip()
                        rejected_results.append(RejectedResult(
                            law_id=cand.law_id,
                            law_name=cand.law_name,
                            reason=reason if reason else "与本案事实不相关",
                        ))

            # 去重
            if selected_results:
                seen = set()
                deduped = []
                for s in selected_results:
                    if s.law_id not in seen:
                        seen.add(s.law_id)
                        deduped.append(s)
                selected_results = deduped

            if rejected_results:
                seen = set()
                deduped = []
                for r in rejected_results:
                    if r.law_id not in seen:
                        seen.add(r.law_id)
                        deduped.append(r)
                rejected_results = deduped

        return selected_results, rejected_results

    def select(
        self,
        facts: List[str],
        candidates_list: List[List[SearchResult]],
        batch_size: int = 4,
    ) -> List[Tuple[List[SelectionResult], List[RejectedResult]]]:
        """批量筛选法条"""
        prompts = [
            self.build_prompt(fact, candidates)
            for fact, candidates in zip(facts, candidates_list)
        ]

        results: List[Tuple[List[SelectionResult], List[RejectedResult]]] = []

        if self.use_vllm:
            outputs = self.llm.generate(prompts, self.sampling_params)
            for out, candidates in zip(outputs, candidates_list):
                raw_text = out.outputs[0].text
                selected, rejected = self.parse_selection(raw_text, candidates)
                results.append((selected, rejected))
            return results

        for prompt, candidates in tqdm(
            list(zip(prompts, candidates_list)), desc="Selecting laws"
        ):
            inputs = self.tokenizer(
                prompt, return_tensors="pt", truncation=True, max_length=7000
            ).to(self.device)
            with torch.no_grad():
                output_ids = self.model.generate(
                    **inputs, max_new_tokens=2048, temperature=0.1, do_sample=True,
                )
            raw_text = self.tokenizer.decode(
                output_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True,
            )
            selected, rejected = self.parse_selection(raw_text, candidates)
            results.append((selected, rejected))

        return results


# ============== 主 Agent 类 ==============
class LawRetrievalAgent:
    """
    法条检索 Agent
    Pipeline: QueryGen → Dense Retriever (top-50) → Reranker (top-10) → LLM-Select
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
    ):
        self.qg_model = querygen_model_path if querygen_model_path else llm_model_path
        self.ls_model = lawselect_model_path if lawselect_model_path else llm_model_path
        self.use_different_models = (self.qg_model != self.ls_model)
        
        self.device = device
        self.use_vllm = use_vllm
        self.tensor_parallel_size = tensor_parallel_size
        self.gpu_memory_util = gpu_memory_util
        
        print(f"[Agent] QueryGen 模型: {self.qg_model}")
        print(f"[Agent] LawSelect 模型: {self.ls_model}")
        print(f"[Agent] 采用串行加载策略以节省 GPU 内存")
        
        self.query_generator = QueryGenerator(
            model_path=self.qg_model,
            device=device,
            use_vllm=use_vllm,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_util=gpu_memory_util,
        )

        self.dense_retriever = DenseRetriever(
            law_corpus_path=law_corpus_path,
            dense_model_path=dense_model_path,
            device=device,
        )

        self.reranker = LawReranker(
            model_path=reranker_model_path,
            device=device,
        )

        self.law_selector = None
    
    def _load_law_selector(self):
        """延迟加载 LawSelector"""
        if self.law_selector is not None:
            return
        
        print(f"\n[Agent] 释放 QueryGen 模型，准备加载 LawSelect 模型...")
        
        if hasattr(self.query_generator, 'llm') and self.query_generator.llm is not None:
            print("[Agent] 释放 QueryGen vLLM 实例...")
            del self.query_generator.llm
            self.query_generator.llm = None
        if hasattr(self.query_generator, 'model') and self.query_generator.model is not None:
            print("[Agent] 释放 QueryGen HF 模型...")
            del self.query_generator.model
            self.query_generator.model = None
        
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        
        if torch.cuda.is_available():
            free_mem = torch.cuda.mem_get_info()[0] / 1024**3
            total_mem = torch.cuda.mem_get_info()[1] / 1024**3
            print(f"[Agent] 释放后显存: {free_mem:.1f}/{total_mem:.1f} GiB 可用")
        
        print(f"[Agent] 加载 LawSelect 模型: {self.ls_model}")
        self.law_selector = LawSelector(
            model_path=self.ls_model,
            device=self.device,
            use_vllm=self.use_vllm,
            tensor_parallel_size=self.tensor_parallel_size,
            gpu_memory_util=self.gpu_memory_util,
        )

    def retrieve(
        self,
        query_ids: List[str],
        facts: List[str],
        dense_top_k: int = 50,
        rerank_top_k: int = 10,
        batch_size: int = 8,
        min_selected: int = 3,
    ) -> List[AgentOutput]:
        """执行完整 pipeline"""
        print("\n" + "=" * 60)
        print(f"[Agent] 开始处理 {len(facts)} 个样本")
        print("=" * 60 + "\n")

        # Step 1: 生成检索查询
        print("[Step 1/4] 生成检索查询 (QueryGen)...")
        query_results = self.query_generator.generate(facts, batch_size=batch_size)

        # Step 2: Dense 检索 top-50
        print(f"[Step 2/4] Dense 检索候选法条 (top-{dense_top_k})...")
        dense_candidates_list: List[List[SearchResult]] = []
        for i, qr in enumerate(tqdm(query_results, desc="Dense Retrieval")):
            queries = qr.queries if qr.queries else [facts[i][:500]]
            candidates = self.dense_retriever.search(queries, top_k=dense_top_k)
            dense_candidates_list.append(candidates)

        # Step 3: Reranker 重排取 top-10
        print(f"[Step 3/4] Reranker 重排 (top-{rerank_top_k})...")
        reranked_candidates_list: List[List[SearchResult]] = []
        for i, (fact, candidates) in enumerate(
            tqdm(zip(facts, dense_candidates_list), desc="Reranking", total=len(facts))
        ):
            reranked = self.reranker.rerank(fact, candidates, top_k=rerank_top_k)
            reranked_candidates_list.append(reranked)

        # Step 4: LLM 筛选相关法条
        print("[Step 4/4] 筛选相关法条 (LLM-Select)...")
        
        self._load_law_selector()
        
        selection_results = self.law_selector.select(
            facts, reranked_candidates_list, batch_size=min(batch_size, 4)
        )

        # 组装输出
        outputs: List[AgentOutput] = []
        fallback_count = 0
        
        for i, (qid, fact) in enumerate(zip(query_ids, facts)):
            selected_laws = selection_results[i][0]
            rejected_laws = selection_results[i][1]
            reranked_candidates = reranked_candidates_list[i]

            # Fallback 策略
            if len(selected_laws) < min_selected and reranked_candidates:
                fallback_count += 1
                existing_ids = {s.law_id for s in selected_laws}
                rejected_ids = {r.law_id for r in rejected_laws}
                
                for cand in reranked_candidates:
                    if cand.law_id not in existing_ids and cand.law_id not in rejected_ids:
                        conf = 0.6 + 0.25 * (1 / (1 + math.exp(-cand.score)))
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

            outputs.append(
                AgentOutput(
                    query_id=qid,
                    fact=fact,
                    generated_queries=query_results[i].queries,
                    candidate_laws=reranked_candidates,
                    selected_laws=selected_laws,
                    rejected_laws=rejected_laws,
                )
            )
        
        if fallback_count > 0:
            print(f"[Agent] Fallback 触发: {fallback_count}/{len(facts)} 个样本使用了 Reranker 补充")

        print("\n[Agent] 处理完成！")
        return outputs


# ============== 工具函数：转换为 MRAG 格式 ==============
def convert_to_mrag_format(
    outputs: List[AgentOutput],
    law_corpus: Dict[str, str],
    output_path: str,
) -> None:
    """将 Agent 输出转换为 MRAG 数据格式（TREC 格式）"""
    with open(output_path, "w", encoding="utf-8") as f:
        for output in outputs:
            rank = 1
            for selected in output.selected_laws:
                f.write(
                    f"{output.query_id}\tQ0\t{selected.law_id}\t{rank}\t{selected.confidence:.6f}\tlaw_agent\n"
                )
                rank += 1

    print(f"[MRAG] 已保存检索结果到: {output_path}")


# ============== 命令行接口 ==============
def main() -> None:
    parser = argparse.ArgumentParser(description="LLM-based 法条检索 Agent")

    parser.add_argument("--llm_model", type=str, required=True)
    parser.add_argument("--law_corpus", type=str, required=True)
    parser.add_argument("--dense_model", type=str, required=True)
    parser.add_argument("--reranker_model", type=str, required=True)
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)

    parser.add_argument("--querygen_model", type=str, default=None)
    parser.add_argument("--lawselect_model", type=str, default=None)

    parser.add_argument("--gpu_memory_util", type=float, default=0.5)
    parser.add_argument("--dense_top_k", type=int, default=50)
    parser.add_argument("--rerank_top_k", type=int, default=10)
    parser.add_argument("--min_selected", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--use_vllm", action="store_true")
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--save_details", action="store_true")

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
            data_items.append({"text_id": str(item.get("text_id", "")), "text": item.get("text", "")})
    print(f"[Main] 加载了 {len(data_items)} 个样本")

    # 加载法条库
    law_corpus: Dict[str, str] = {}
    with open(args.law_corpus, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            law_corpus[str(item["text_id"])] = f"{item['name']}：{item['text']}"

    # 初始化 Agent
    agent = LawRetrievalAgent(
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
    )

    # 执行检索
    query_ids = [item["text_id"] for item in data_items]
    facts = [item["text"] for item in data_items]

    outputs = agent.retrieve(
        query_ids=query_ids,
        facts=facts,
        dense_top_k=args.dense_top_k,
        rerank_top_k=args.rerank_top_k,
        min_selected=args.min_selected,
        batch_size=args.batch_size,
    )

    # 转换并保存结果
    convert_to_mrag_format(outputs, law_corpus, args.output_file)

    # 保存详细输出
    if args.save_details:
        detail_path = args.output_file.replace(".tsv", "_details.json")
        details = [
            {
                "query_id": output.query_id,
                "fact": output.fact[:500] + "..." if len(output.fact) > 500 else output.fact,
                "generated_queries": output.generated_queries,
                "selected_laws": [
                    {
                        "law_id": s.law_id,
                        "law": s.law_name,
                        "reason": s.reason,
                        "confidence": s.confidence
                    }
                    for s in output.selected_laws
                ],
                "rejected_laws": [
                    {
                        "law_id": r.law_id,
                        "law": r.law_name,
                        "reason": r.reason
                    }
                    for r in output.rejected_laws
                ],
            }
            for output in outputs
        ]

        with open(detail_path, "w", encoding="utf-8") as f:
            json.dump(details, f, ensure_ascii=False, indent=2)
        print(f"[Main] 详细输出已保存到: {detail_path}")

    print("[Main] 完成！")


if __name__ == "__main__":
    main()
