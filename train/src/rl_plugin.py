import json
import sys
from pathlib import Path
from typing import List

import numpy as np
from swift.plugin import ORM, orms
from swift.utils import get_logger


logger = get_logger()

ROOT = Path(__file__).resolve().parent[2]
EVAL_DIR = ROOT / "evaluation"
if str(EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(EVAL_DIR))
SEGMENT_DIR = EVAL_DIR / "segment"
if str(SEGMENT_DIR) not in sys.path:
    sys.path.insert(0, str(SEGMENT_DIR))

import jieba  # noqa: E402
from bert_score import score  # noqa: E402
from nltk.translate.meteor_score import meteor_score  # noqa: E402

from crime_extraction import get_crime  # noqa: E402
from judge_extraction import calc_amt_sum, calc_time_sum  # noqa: E402
from law_extraction import get_penalcode_index_from_text  # noqa: E402
from data_segment_xingshi import DataSegmentXingshi  # noqa: E402

SFT_BASE_METRICS = {
    "pen_prison": 0.6299,
    "pen_fine":   0.4984,
    "conv_rec":   0.9022,
    "conv_prec":  0.8985,
    "conv_f1":    0.9004,
    "ref_rec":    0.5863,
    "ref_prec":   0.7687,
    "ref_f1":     0.6653,
    "reason_met": 0.5167,
    "reason_bert":0.8747,
    "judg_met":   0.6426,
    "judg_bert":  0.9045,
}

METRIC_WEIGHTS = {
    "pen_prison": 1.2,
    "pen_fine":   1.5,
    "conv_rec":   1.0,
    "conv_prec":  1.0,
    "conv_f1":    1.0,
    "ref_rec":    1.5,
    "ref_prec":   1.2,
    "ref_f1":     1.2,
    "reason_met": 1.5,
    "reason_bert":1.0,
    "judg_met":   1.2,
    "judg_bert":  1.0,
}


def _shape_metric(value: float, key: str) -> float:
    """
    对单个 metric 做 shaping：
    - 用 (value - base) 表示相对 SFT 的提升/下降；
    - delta > 0: 放大一点，鼓励超过 SFT；
    - delta < 0: 略缩小惩罚，避免噪声导致过度修正；
    - base 越低（短板），正向放大系数越大。
    """
    base = SFT_BASE_METRICS.get(key, 0.0)
    if not np.isfinite(value):
        value = base
    delta = value - base

    # base 越低，说明这里越是短板，对「超过 SFT」的提升放大越多
    alpha_pos = 1.5 * (1.0 - base)   # 0~1 之间
    # base 越高，说明这里本来就强，对落在 base 之下的惩罚稍微大一点
    alpha_neg = 0.5 * base

    if delta >= 0:
        shaped = (1.0 + alpha_pos) * delta
    else:
        shaped = (1.0 - alpha_neg) * delta
    return shaped * METRIC_WEIGHTS.get(key, 1.0)

def _recall_prec(expected: List[str], actual: List[str]) -> tuple[float, float, float]:
    exp_set = set(expected)
    act_set = set(actual)
    tp = len(exp_set & act_set)
    recall = tp / len(exp_set) if exp_set else 0.0
    precision = tp / len(act_set) if act_set else 0.0
    f1 = (2 * recall * precision / (recall + precision)) if (recall + precision) else 0.0
    return recall, precision, f1


def _percent_for_judge(exp_val: int, act_val: int) -> float:
    # 完全匹配且都为 0
    if exp_val == 0 and act_val == 0:
        return 1.0

    # 符号不一致：一个非负，一个负数
    if (exp_val >= 0 and act_val < 0) or (exp_val < 0 and act_val >= 0):
        return 0.0

    # 是否跨过 10000 这个分段阈值
    if (exp_val - 10000) * (act_val - 10000) < 0:
        return 0.0

    # 相对误差打分
    x = abs(exp_val - act_val) / max(exp_val, act_val, 1)
    return max(0.0, 1 - x)


def _safe_calc_time(text: str) -> int:
    """包装 calc_time_sum，异常时返回 0，避免 reward 崩溃。"""
    try:
        return calc_time_sum(text)
    except Exception as exc:  # pragma: no cover
        logger.warning(f"calc_time_sum failed, return 0. text_head={text[:50]!r}, err={exc}")
        return 0


def _safe_calc_amt(text: str) -> int:
    """包装 calc_amt_sum，异常时返回 0，避免 reward 崩溃。"""
    try:
        return calc_amt_sum(text)
    except Exception as exc:  # pragma: no cover
        logger.warning(f"calc_amt_sum failed, return 0. text_head={text[:50]!r}, err={exc}")
        return 0


class LegalDocReward(ORM):
    def __init__(self, bert_model_path: str | None = None):
        self.bert_model_path = bert_model_path or "/data-share/chenxuanyi/LLM/bert-base-chinese"
        self.segmenter = DataSegmentXingshi(punctuation_replace=True)

    def _split_sections(self, text: str) -> tuple[str, str]:
        parsed = self.segmenter.parse(text)
        return parsed.get("reason", ""), parsed.get("judgment", "")

    def _meteor(self, ref: str, hyp: str) -> float:
        ref_tokens = list(jieba.cut(ref))
        hyp_tokens = list(jieba.cut(hyp))
        try:
            return float(meteor_score([ref_tokens], hyp_tokens))
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning(f"METEOR failed, return 0.0. err={exc}")
            return 0.0

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

    def __call__(self, completions: List[str], reference_document: List[str], **kwargs) -> List[float]:
        rewards: List[float] = []
        # Vectorized BERTScore for reasoning/judgment
        reason_refs, reason_hyps, judge_refs, judge_hyps = [], [], [], []
        reason_idx_map, judge_idx_map = [], []

        for idx, (gen, ref) in enumerate(zip(completions, reference_document)):
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

        for idx, (gen, ref) in enumerate(zip(completions, reference_document)):
            components = {}

            # Penalty Acc.（刑期 + 罚金）
            g_time = _safe_calc_time(gen)
            r_time = _safe_calc_time(ref)
            components["pen_prison"] = _percent_for_judge(r_time, g_time)

            g_amt = _safe_calc_amt(gen)
            r_amt = _safe_calc_amt(ref)
            components["pen_fine"] = _percent_for_judge(r_amt, g_amt)

            # Convicting Acc.（罪名）
            conv_rec, conv_prec, conv_f1 = _recall_prec(get_crime(ref), get_crime(gen))
            components["conv_rec"] = conv_rec
            components["conv_prec"] = conv_prec
            components["conv_f1"] = conv_f1

            # Referencing Acc.（法条）
            ref_rec, ref_prec, ref_f1 = _recall_prec(
                get_penalcode_index_from_text(ref),
                get_penalcode_index_from_text(gen),
            )
            components["ref_rec"] = ref_rec
            components["ref_prec"] = ref_prec
            components["ref_f1"] = ref_f1

            # Reasoning / Judgment 文本指标
            g_reason, g_judge = self._split_sections(gen)
            r_reason, r_judge = self._split_sections(ref)

            if g_reason and r_reason:
                components["reason_met"] = self._meteor(r_reason, g_reason)
                components["reason_bert"] = reason_bert_map.get(idx, 0.0)
            else:
                components["reason_met"] = 0.0
                components["reason_bert"] = 0.0

            if g_judge and r_judge:
                components["judg_met"] = self._meteor(r_judge, g_judge)
                components["judg_bert"] = judge_bert_map.get(idx, 0.0)
            else:
                components["judg_met"] = 0.0
                components["judg_bert"] = 0.0

            shaped_sum = 0.0
            for key, val in components.items():
                shaped_sum += _shape_metric(val, key)

            rewards.append(float(shaped_sum))

        return rewards


# Register for swift rlhf
orms["legal_doc_reward"] = LegalDocReward
