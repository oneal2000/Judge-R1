import json
import re
import sys
from pathlib import Path
from typing import List

import numpy as np
from swift.plugin import ORM, orms
from swift.utils import get_logger

logger = get_logger()

ROOT = Path(__file__).resolve().parents[2]
EVAL_DIR = ROOT / "evaluation"
if str(EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(EVAL_DIR))
SEGMENT_DIR = EVAL_DIR / "segment"
if str(SEGMENT_DIR) not in sys.path:
    sys.path.insert(0, str(SEGMENT_DIR))

import jieba  
from bert_score import score  
from nltk.translate.meteor_score import meteor_score  

from crime_extraction import get_crime  
from judge_extraction import calc_amt_sum, calc_time_sum  
from law_extraction import get_penalcode_index_from_text  
from data_segment_xingshi import DataSegmentXingshi  


def strip_think(text: str) -> str:
    """
    去除 thinking 模型的思考过程，只保留最终输出
    
    支持格式:
    - <think>...</think>
    """
    # 方法1: 如果有 </think> 标记，取其后内容
    if '</think>' in text:
        return text.split('</think>', 1)[-1].strip()
    
    # 方法2: 正则替换各种思考标记
    patterns = [
        (r'<think>.*?</think>', ''),
    ]
    
    result = text
    for pattern, replacement in patterns:
        result = re.sub(pattern, replacement, result, flags=re.DOTALL)
    
    return result.strip()


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
        self.bert_model_path = bert_model_path or "/data-share/chenxuanyi/LLM/bert-base-chinese"
        self.segmenter = DataSegmentXingshi(punctuation_replace=True)
        self.legal_weight = legal_weight
        self.text_weight = text_weight
        self.thinking_weight = thinking_weight
        # 法律指标内部权重
        self.law_article_weight = law_article_weight
        self.crime_weight = crime_weight
        self.time_weight = time_weight
        self.amount_weight = amount_weight
    
    def _split_sections(self, text: str) -> tuple[str, str]:
        parsed = self.segmenter.parse(text)
        return parsed.get("reason", ""), parsed.get("judgment", "")
    
    def _bert_f1(self, ref_list: List[str], hyp_list: List[str]) -> List[float]:
        if not ref_list or not hyp_list:
            return [0.0] * len(hyp_list)
        _, _, f1 = score(
            hyp_list,
            ref_list,
            model_type=self.bert_model_path,
            num_layers=12,
        )
        return f1.tolist()
    
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
        
        for idx, (gen, ref) in enumerate(zip(clean_completions, reference_document)):
            g_reason, g_judge = self._split_sections(gen)
            r_reason, r_judge = self._split_sections(ref)
            if g_reason and r_reason:
                reason_hyps.append(" ".join(jieba.cut(g_reason)))
                reason_refs.append(" ".join(jieba.cut(r_reason)))
                reason_idx_map.append(idx)
            if g_judge and r_judge:
                judge_hyps.append(" ".join(jieba.cut(g_judge)))
                judge_refs.append(" ".join(jieba.cut(r_judge)))
                judge_idx_map.append(idx)
        
        reason_berts = self._bert_f1(reason_refs, reason_hyps)
        judge_berts = self._bert_f1(judge_refs, judge_hyps)
        reason_bert_map = dict(zip(reason_idx_map, reason_berts))
        judge_bert_map = dict(zip(judge_idx_map, judge_berts))
        
        # 计算每个样本的奖励
        for idx, (gen, ref) in enumerate(zip(clean_completions, reference_document)):
            
            # ========== 1. 法律准确性指标 (权重 60%) ==========
            # v2: 使用加权而非平均（法条35% > 罪名30% > 刑期20% > 罚金15%）
            
            # 刑期准确率
            g_time = _safe_calc_time(gen)
            r_time = _safe_calc_time(ref)
            time_score = _percent_for_judge(r_time, g_time)
            
            # 罚金准确率
            g_amt = _safe_calc_amt(gen)
            r_amt = _safe_calc_amt(ref)
            amt_score = _percent_for_judge(r_amt, g_amt)
            
            # 罪名 F1（只用 F1，不用 Recall/Precision）
            _, _, conv_f1 = _recall_prec_f1(get_crime(ref), get_crime(gen))
            
            # 法条 F1（只用 F1）
            _, _, ref_f1 = _recall_prec_f1(
                get_penalcode_index_from_text(ref),
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

# class LegalDocReward(ORM):
#     """
#     Reward = mean of metrics listed in the paper figure:
#       Penalty Acc.: prison/fine alignment
#       Convicting Acc.: recall/precision/F1 on crimes
#       Referencing Acc.: recall/precision/F1 on law articles
#       Reasoning: METEOR + BERTScore (reasoning section)
#       Judgment: METEOR + BERTScore (judgment section)
#     Higher score => higher reward.
#     """

#     def __init__(self, bert_model_path: str | None = None):
#         self.bert_model_path = bert_model_path or "/data-share/chenxuanyi/LLM/bert-base-chinese"
#         self.segmenter = DataSegmentXingshi(punctuation_replace=True)

#     def _split_sections(self, text: str) -> tuple[str, str]:
#         parsed = self.segmenter.parse(text)
#         return parsed.get("reason", ""), parsed.get("judgment", "")

#     def _meteor(self, ref: str, hyp: str) -> float:
#         ref_tokens = list(jieba.cut(ref))
#         hyp_tokens = list(jieba.cut(hyp))
#         try:
#             return float(meteor_score([ref_tokens], hyp_tokens))
#         except Exception as exc:  # pragma: no cover - defensive
#             logger.warning(f"METEOR failed, return 0.0. err={exc}")
#             return 0.0

#     def _bert_f1(self, ref_list: List[str], hyp_list: List[str]) -> List[float]:
#         if not ref_list or not hyp_list:
#             return [0.0] * len(hyp_list)
#         _, _, f1 = score(
#             hyp_list,
#             ref_list,
#             model_type=self.bert_model_path,
#             num_layers=12,
#         )
#         return f1.tolist()

#     def __call__(self, completions: List[str], reference_document: List[str], **kwargs) -> List[float]:
#         rewards: List[float] = []
        
#         # 对 thinking 模型的输出去除思考过程
#         clean_completions = [strip_think(c) for c in completions]
        
#         # Vectorized BERTScore for reasoning/judgment
#         reason_refs, reason_hyps, judge_refs, judge_hyps = [], [], [], []
#         reason_idx_map, judge_idx_map = [], []

#         for idx, (gen, ref) in enumerate(zip(clean_completions, reference_document)):
#             g_reason, g_judge = self._split_sections(gen)
#             r_reason, r_judge = self._split_sections(ref)
#             if g_reason and r_reason:
#                 reason_hyps.append(" ".join(jieba.cut(g_reason)))
#                 reason_refs.append(" ".join(jieba.cut(r_reason)))
#                 reason_idx_map.append(idx)
#             if g_judge and r_judge:
#                 judge_hyps.append(" ".join(jieba.cut(g_judge)))
#                 judge_refs.append(" ".join(jieba.cut(r_judge)))
#                 judge_idx_map.append(idx)

#         reason_berts = self._bert_f1(reason_refs, reason_hyps)
#         judge_berts = self._bert_f1(judge_refs, judge_hyps)
#         reason_bert_map = dict(zip(reason_idx_map, reason_berts))
#         judge_bert_map = dict(zip(judge_idx_map, judge_berts))

#         for idx, (gen, ref) in enumerate(zip(clean_completions, reference_document)):
#             metrics = []

#             # Penalty Acc. 使用安全版本防止异常中断
#             g_time = _safe_calc_time(gen)
#             r_time = _safe_calc_time(ref)
#             metrics.append(_percent_for_judge(r_time, g_time))

#             g_amt = _safe_calc_amt(gen)
#             r_amt = _safe_calc_amt(ref)
#             metrics.append(_percent_for_judge(r_amt, g_amt))

#             # Convicting Acc.
#             conv_rec, conv_prec, conv_f1 = _recall_prec(get_crime(ref), get_crime(gen))
#             metrics.extend([conv_rec, conv_prec, conv_f1])

#             # Referencing Acc.
#             ref_rec, ref_prec, ref_f1 = _recall_prec(
#                 get_penalcode_index_from_text(ref),
#                 get_penalcode_index_from_text(gen),
#             )
#             metrics.extend([ref_rec, ref_prec, ref_f1])

#             # Reasoning Section metrics
#             g_reason, g_judge = self._split_sections(gen)
#             r_reason, r_judge = self._split_sections(ref)
#             metrics.append(self._meteor(r_reason, g_reason) if g_reason and r_reason else 0.0)
#             metrics.append(reason_bert_map.get(idx, 0.0))

#             # Judgment Section metrics
#             metrics.append(self._meteor(r_judge, g_judge) if g_judge and r_judge else 0.0)
#             metrics.append(judge_bert_map.get(idx, 0.0))

#             # Final reward: mean of available metrics
#             reward = float(np.mean(metrics)) if metrics else 0.0
#             rewards.append(reward)

#         return rewards


# # Register for swift rlhf
# orms["legal_doc_reward"] = LegalDocReward

# class LegalDocRewardWithThinking(ORM):
#     """
#     增强版奖励函数：在原有基础上添加思考质量评估
    
#     思考质量评估包括：
#     1. 是否生成思考过程（Thinking模型应该思考）
#     2. 思考长度合理性（不要太短或太长）
#     3. 思考内容相关性（包含法律关键词）
#     4. 避免噪音（不包含元分析内容）
#     """
    
#     def __init__(self, bert_model_path: str | None = None, thinking_weight: float = 0.15):
#         self.bert_model_path = bert_model_path or "/data-share/chenxuanyi/LLM/bert-base-chinese"
#         self.segmenter = DataSegmentXingshi(punctuation_replace=True)
#         self.thinking_weight = thinking_weight  # 思考质量的权重
    
#     def _split_sections(self, text: str) -> tuple[str, str]:
#         parsed = self.segmenter.parse(text)
#         return parsed.get("reason", ""), parsed.get("judgment", "")
    
#     def _meteor(self, ref: str, hyp: str) -> float:
#         ref_tokens = list(jieba.cut(ref))
#         hyp_tokens = list(jieba.cut(hyp))
#         try:
#             return float(meteor_score([ref_tokens], hyp_tokens))
#         except Exception as exc:
#             logger.warning(f"METEOR failed, return 0.0. err={exc}")
#             return 0.0
    
#     def _bert_f1(self, ref_list: List[str], hyp_list: List[str]) -> List[float]:
#         if not ref_list or not hyp_list:
#             return [0.0] * len(hyp_list)
#         _, _, f1 = score(
#             hyp_list,
#             ref_list,
#             model_type=self.bert_model_path,
#             num_layers=12,
#         )
#         return f1.tolist()
    
#     def _extract_think(self, text: str) -> str | None:
#         """提取思考内容"""
#         if '<think>' not in text or '</think>' not in text:
#             return None
        
#         match = re.search(r'<think>(.*?)</think>', text, flags=re.DOTALL)
#         if match:
#             return match.group(1).strip()
#         return None
    
#     def _evaluate_thinking_quality(self, think_content: str | None) -> float:
#         """
#         评估思考质量
        
#         返回 0.0-1.0 的分数：
#         - 没有思考：0.0
#         - 有思考但质量差：0.2-0.5
#         - 有思考且质量好：0.6-1.0
#         """
#         if think_content is None:
#             return 0.0  # 没有思考
        
#         score = 0.0
#         think_len = len(think_content)
        
#         # 1. 长度合理性 (30%)
#         if 50 < think_len < 800:
#             # 理想长度 100-500
#             if 100 <= think_len <= 500:
#                 score += 0.30
#             elif 50 < think_len < 100 or 500 < think_len < 800:
#                 score += 0.20
#         elif think_len >= 800:
#             # 太长扣分
#             score += max(0.0, 0.20 - (think_len - 800) / 2000)
#         dash_lines = think_content.count('\n-')
#         if dash_lines <= 5:
#             score += 0.10
#         elif dash_lines <= 10:
#             score += 0.05
        
#         return min(1.0, score)
    
#     def __call__(self, completions: List[str], reference_document: List[str], **kwargs) -> List[float]:
#         rewards: List[float] = []
        
#         # 提取思考内容（在去除之前）
#         think_contents = [self._extract_think(c) for c in completions]
        
#         # 评估思考质量
#         thinking_scores = [self._evaluate_thinking_quality(tc) for tc in think_contents]
        
#         # 对最终输出去除思考过程进行评估
#         clean_completions = [strip_think(c) for c in completions]
        
#         # 原有的判决书质量评估（与原版相同）
#         reason_refs, reason_hyps, judge_refs, judge_hyps = [], [], [], []
#         reason_idx_map, judge_idx_map = [], []
        
#         for idx, (gen, ref) in enumerate(zip(clean_completions, reference_document)):
#             g_reason, g_judge = self._split_sections(gen)
#             r_reason, r_judge = self._split_sections(ref)
#             if g_reason and r_reason:
#                 reason_hyps.append(" ".join(jieba.cut(g_reason)))
#                 reason_refs.append(" ".join(jieba.cut(r_reason)))
#                 reason_idx_map.append(idx)
#             if g_judge and r_judge:
#                 judge_hyps.append(" ".join(jieba.cut(g_judge)))
#                 judge_refs.append(" ".join(jieba.cut(r_judge)))
#                 judge_idx_map.append(idx)
        
#         reason_berts = self._bert_f1(reason_refs, reason_hyps)
#         judge_berts = self._bert_f1(judge_refs, judge_hyps)
#         reason_bert_map = dict(zip(reason_idx_map, reason_berts))
#         judge_bert_map = dict(zip(judge_idx_map, judge_berts))
        
#         for idx, (gen, ref) in enumerate(zip(clean_completions, reference_document)):
#             metrics = []
            
#             # Penalty Acc.
#             g_time = _safe_calc_time(gen)
#             r_time = _safe_calc_time(ref)
#             metrics.append(_percent_for_judge(r_time, g_time))
            
#             g_amt = _safe_calc_amt(gen)
#             r_amt = _safe_calc_amt(ref)
#             metrics.append(_percent_for_judge(r_amt, g_amt))
            
#             # Convicting Acc.
#             conv_rec, conv_prec, conv_f1 = _recall_prec(get_crime(ref), get_crime(gen))
#             metrics.extend([conv_rec, conv_prec, conv_f1])
            
#             # Referencing Acc.
#             ref_rec, ref_prec, ref_f1 = _recall_prec(
#                 get_penalcode_index_from_text(ref),
#                 get_penalcode_index_from_text(gen),
#             )
#             metrics.extend([ref_rec, ref_prec, ref_f1])
            
#             # Reasoning Section
#             g_reason, g_judge = self._split_sections(gen)
#             r_reason, r_judge = self._split_sections(ref)
#             metrics.append(self._meteor(r_reason, g_reason) if g_reason and r_reason else 0.0)
#             metrics.append(reason_bert_map.get(idx, 0.0))
            
#             # Judgment Section
#             metrics.append(self._meteor(r_judge, g_judge) if g_judge and r_judge else 0.0)
#             metrics.append(judge_bert_map.get(idx, 0.0))
            
#             # 基础奖励：原有指标的平均
#             base_reward = float(np.mean(metrics)) if metrics else 0.0
            
#             # 思考质量奖励（加权混合）
#             thinking_reward = thinking_scores[idx]
            
#             # 最终奖励 = (1-w) * 基础奖励 + w * 思考奖励
#             final_reward = (1 - self.thinking_weight) * base_reward + self.thinking_weight * thinking_reward
            
#             rewards.append(final_reward)
            
#             # 日志记录（调试用）
#             if idx < 3:  # 只记录前3个样本
#                 logger.info(
#                     f"Sample {idx}: base={base_reward:.3f}, thinking={thinking_reward:.3f}, "
#                     f"final={final_reward:.3f}, has_think={think_contents[idx] is not None}"
#                 )
        
#         return rewards

# orms["legal_doc_reward"] = LegalDocRewardWithThinking

