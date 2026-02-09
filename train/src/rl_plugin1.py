import os
import re
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Any, List

import numpy as np
from swift.plugin import ORM, orms
from swift.utils import get_logger

logger = get_logger()

ROOT = Path(__file__).resolve().parents[2]

# ============== 环境变量配置（支持用户覆盖）==============
# 优先级: 函数参数 > 环境变量 > 默认相对路径
DEFAULT_BERT_MODEL_PATH = os.environ.get("BERT_MODEL_PATH", "bert-base-chinese")
EVAL_DIR = ROOT / "evaluation"
if str(EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(EVAL_DIR))
SEGMENT_DIR = EVAL_DIR / "segment"
if str(SEGMENT_DIR) not in sys.path:
    sys.path.insert(0, str(SEGMENT_DIR))

import jieba  
from bert_score import BERTScorer, score  

from crime_extraction import get_crime  
from judge_extraction import calc_amt_sum, calc_time_sum  
from law_extraction import get_penalcode_index_from_text  
from data_segment_xingshi import DataSegmentXingshi  


def strip_think(text: str) -> str:
    """
    去除 thinking 模型的思考过程，只保留最终输出。
    
    支持格式: <think>...</think>
    也处理只有 <think> 没有 </think> 的截断情况。
    """
    if '</think>' in text:
        return text.split('</think>', 1)[-1].strip()
    # 处理只有 <think> 没有闭合的情况（模型输出被截断）

    # 方法2: 正则替换各种思考标记
    patterns = [
        (r'<think>.*?</think>', ''),
    ]

    result = text
    for pattern, replacement in patterns:
        result = re.sub(pattern, replacement, result, flags=re.DOTALL)
        
    return text.strip()


def _recall_prec_f1(expected: List[str], actual: List[str]) -> tuple[float, float, float]:
    """计算 Recall, Precision, F1"""
    exp_set = set(expected)
    act_set = set(actual)
    tp = len(exp_set & act_set)
    recall = tp / len(exp_set) if exp_set else 0.0
    precision = tp / len(act_set) if act_set else 0.0
    f1 = (2 * recall * precision / (recall + precision)) if (recall + precision) else 0.0
    return recall, precision, f1


def _percent_for_judge(exp_val: int, act_val: int) -> float:
    """刑期/罚金的匹配分数"""
    if exp_val == 0 and act_val == 0:
        return 1.0

    if (exp_val >= 0 and act_val < 0) or (exp_val < 0 and act_val >= 0):
        return 0.0

    if (exp_val - 10000) * (act_val - 10000) < 0:
        return 0.0

    x = abs(exp_val - act_val) / max(exp_val, act_val, 1)
    return max(0.0, 1 - x)


def _safe_calc_time(text: str) -> int:
    """安全计算刑期"""
    try:
        return calc_time_sum(text)
    except Exception as exc:
        logger.warning(f"calc_time_sum failed, return 0. err={exc}")
        return 0


def _safe_calc_amt(text: str) -> int:
    """安全计算罚金"""
    try:
        return calc_amt_sum(text)
    except Exception as exc:
        logger.warning(f"calc_amt_sum failed, return 0. err={exc}")
        return 0

class LegalDocRewardImproved(ORM):
    """
    改进版奖励函数 v2
    
    主要改进：
    1. 移除冗余指标（只用F1，不用Recall+Precision+F1）
    2. 调整权重分配：法律准确性 60%，文本质量 30%，思考质量 10%
    3. 对非 thinking 模型不惩罚思考质量
    4. v2: 法律指标内部加权（法条35% > 罪名30% > 刑期20% > 罚金15%）
    """
    
    def __init__(
        self, 
        bert_model_path: str | None = None,
        legal_weight: float = 0.60,      # 法律准确性权重
        text_weight: float = 0.30,       # 文本质量权重
        thinking_weight: float = 0.10,   # 思考质量权重
        # v2: 法律指标内部权重（总和=1.0）
        law_article_weight: float = 0.35,   # 法条引用最重要
        crime_weight: float = 0.30,         # 罪名判定次之
        time_weight: float = 0.20,          # 刑期
        amount_weight: float = 0.15,        # 罚金（基础分较低，降低权重）
    ):
        # 优先级: 参数传入 > 环境变量 > 默认路径（已在 DEFAULT_BERT_MODEL_PATH 处理）
        self.bert_model_path = bert_model_path or DEFAULT_BERT_MODEL_PATH
        self.segmenter = DataSegmentXingshi(punctuation_replace=True)
        self.legal_weight = legal_weight
        self.text_weight = text_weight
        self.thinking_weight = thinking_weight
        # 法律指标内部权重
        self.law_article_weight = law_article_weight
        self.crime_weight = crime_weight
        self.time_weight = time_weight
        self.amount_weight = amount_weight
        # reference 特征缓存（仅缓存 reference 侧，不改变奖励定义）
        self.ref_cache_size = int(os.environ.get("RL_REF_CACHE_SIZE", "16384"))
        self._ref_feature_cache: "OrderedDict[str, dict[str, Any]]" = OrderedDict()
        # 复用 BERTScorer 实例，避免每次 score 都重复初始化模型/Tokenizer
        self.bert_batch_size = int(os.environ.get("RL_BERT_BATCH_SIZE", "16"))
        self._bertscorer = BERTScorer(
            model_type=self.bert_model_path,
            num_layers=12,
        )
    
    def _split_sections(self, text: str) -> tuple[str, str]:
        try:
            parsed = self.segmenter.parse(text)
            return parsed.get("reason", ""), parsed.get("judgment", "")
        except Exception as exc:
            logger.warning(f"segmenter.parse failed, return empty sections. err={exc}")
            return "", ""
    
    def _bert_f1(self, ref_list: List[str], hyp_list: List[str]) -> List[float]:
        if not ref_list or not hyp_list:
            return [0.0] * len(hyp_list)
        # 常驻 scorer 优先；失败时降级到函数接口，保证健壮性
        try:
            _, _, f1 = self._bertscorer.score(
                hyp_list,
                ref_list,
                batch_size=self.bert_batch_size,
            )
        except Exception as exc:
            logger.warning(f"BERTScorer.score failed, fallback to score(). err={exc}")
            _, _, f1 = score(
                hyp_list,
                ref_list,
                model_type=self.bert_model_path,
                num_layers=12,
            )
        return f1.tolist()

    def _put_ref_cache(self, ref: str, feat: dict[str, Any]) -> None:
        self._ref_feature_cache[ref] = feat
        self._ref_feature_cache.move_to_end(ref)
        while len(self._ref_feature_cache) > self.ref_cache_size:
            self._ref_feature_cache.popitem(last=False)

    def _get_ref_features(self, ref: str) -> dict[str, Any]:
        cached = self._ref_feature_cache.get(ref)
        if cached is not None:
            self._ref_feature_cache.move_to_end(ref)
            return cached

        r_reason, r_judge = self._split_sections(ref)
        feat: dict[str, Any] = {
            "reason": r_reason,
            "judgment": r_judge,
            "reason_cut": " ".join(jieba.cut(r_reason)) if r_reason else "",
            "judgment_cut": " ".join(jieba.cut(r_judge)) if r_judge else "",
            "crime_set": get_crime(ref),
            "law_set": get_penalcode_index_from_text(ref),
            "time": _safe_calc_time(ref),
            "amt": _safe_calc_amt(ref),
        }
        self._put_ref_cache(ref, feat)
        return feat
    
    def _extract_think(self, text: str) -> str | None:
        """提取思考内容"""
        if '<think>' not in text or '</think>' not in text:
            return None
        
        match = re.search(r'<think>(.*?)</think>', text, flags=re.DOTALL)
        if match:
            return match.group(1).strip()
        return None
    
    def _evaluate_thinking_quality(self, think_content: str | None, has_any_think: bool) -> float:
        """
        评估思考质量
        
        改进：如果整个 batch 都没有思考标记（非 thinking 模型），则跳过思考评分
        """
        # 如果是非 thinking 模型（整个 batch 都没有思考），返回中性分数
        if not has_any_think:
            return 0.5  # 中性分数，不奖励也不惩罚
        
        if think_content is None:
            return 0.0  # thinking 模型但没有生成思考
        
        score_val = 0.0
        think_len = len(think_content)
        
        # 1. 长度合理性 (50%)
        if 50 < think_len < 1000:
            if 100 <= think_len <= 600:
                score_val += 0.50
            elif 50 < think_len < 100 or 600 < think_len < 1000:
                score_val += 0.30
        elif think_len >= 1000:
            score_val += max(0.0, 0.30 - (think_len - 1000) / 3000)
        
        # 2. 不要过度重复 (50%)
        dash_lines = think_content.count('\n-')
        if dash_lines <= 5:
            score_val += 0.50
        elif dash_lines <= 10:
            score_val += 0.30
        elif dash_lines <= 20:
            score_val += 0.10
        
        return min(1.0, score_val)
    
    def __call__(self, completions: List[str], reference_document: List[str], **kwargs) -> List[float]:
        rewards: List[float] = []
        
        # 提取思考内容
        think_contents = [self._extract_think(c) for c in completions]
        
        # 判断是否为 thinking 模型（如果有任何一个样本有思考标记）
        has_any_think = any(tc is not None for tc in think_contents)
        
        # 评估思考质量
        thinking_scores = [
            self._evaluate_thinking_quality(tc, has_any_think) 
            for tc in think_contents
        ]
        
        # 去除思考过程
        clean_completions = [strip_think(c) for c in completions]
        
        # 预计算 BERTScore（批量计算更高效）
        reason_refs, reason_hyps, judge_refs, judge_hyps = [], [], [], []
        reason_idx_map, judge_idx_map = [], []
        ref_feats: List[dict[str, Any]] = []
        
        for idx, (gen, ref) in enumerate(zip(clean_completions, reference_document)):
            g_reason, g_judge = self._split_sections(gen)
            rf = self._get_ref_features(ref)
            ref_feats.append(rf)
            if g_reason and rf["reason"]:
                reason_hyps.append(" ".join(jieba.cut(g_reason)))
                reason_refs.append(rf["reason_cut"])
                reason_idx_map.append(idx)
            if g_judge and rf["judgment"]:
                judge_hyps.append(" ".join(jieba.cut(g_judge)))
                judge_refs.append(rf["judgment_cut"])
                judge_idx_map.append(idx)
        
        reason_berts = self._bert_f1(reason_refs, reason_hyps)
        judge_berts = self._bert_f1(judge_refs, judge_hyps)
        reason_bert_map = dict(zip(reason_idx_map, reason_berts))
        judge_bert_map = dict(zip(judge_idx_map, judge_berts))
        
        # 计算每个样本的奖励
        for idx, gen in enumerate(clean_completions):
            rf = ref_feats[idx]
            
            # ========== 1. 法律准确性指标 (权重 60%) ==========
            # v2: 使用加权而非平均（法条35% > 罪名30% > 刑期20% > 罚金15%）
            
            # 刑期准确率
            g_time = _safe_calc_time(gen)
            r_time = rf["time"]
            time_score = _percent_for_judge(r_time, g_time)
            
            # 罚金准确率
            g_amt = _safe_calc_amt(gen)
            r_amt = rf["amt"]
            amt_score = _percent_for_judge(r_amt, g_amt)
            
            # 罪名 F1（只用 F1，不用 Recall/Precision）
            _, _, conv_f1 = _recall_prec_f1(rf["crime_set"], get_crime(gen))
            
            # 法条 F1（只用 F1）
            _, _, ref_f1 = _recall_prec_f1(
                rf["law_set"],
                get_penalcode_index_from_text(gen),
            )
            
            # v2: 加权计算法律准确性
            legal_score = (
                self.law_article_weight * ref_f1 +      # 法条 35%
                self.crime_weight * conv_f1 +           # 罪名 30%
                self.time_weight * time_score +         # 刑期 20%
                self.amount_weight * amt_score          # 罚金 15%
            )
            
            # ========== 2. 文本质量指标 (权重 30%) ==========
            text_metrics = []
            
            # BERTScore (比 METEOR 更准确，只用 BERTScore)
            reason_bert = reason_bert_map.get(idx, 0.0)
            judge_bert = judge_bert_map.get(idx, 0.0)
            text_metrics.append(reason_bert)
            text_metrics.append(judge_bert)
            
            text_score = float(np.mean(text_metrics)) if text_metrics else 0.0
            
            # ========== 3. 思考质量 (权重 10%) ==========
            thinking_score = thinking_scores[idx]
            
            # ========== 4. 最终奖励 = 加权组合 ==========
            final_reward = (
                self.legal_weight * legal_score +
                self.text_weight * text_score +
                self.thinking_weight * thinking_score
            )
            
            rewards.append(final_reward)
            
            # # 日志记录（前3个样本）
            # if idx < 3:
            #     logger.info(
            #         f"Sample {idx}: legal={legal_score:.3f} (time={time_score:.2f}, amt={amt_score:.2f}, "
            #         f"crime_f1={conv_f1:.2f}, law_f1={ref_f1:.2f}), "
            #         f"text={text_score:.3f}, thinking={thinking_score:.3f}, "
            #         f"final={final_reward:.3f}"
            #     )
        
        return rewards


# 注册改进版奖励函数
orms["legal_doc_reward"] = LegalDocRewardImproved

