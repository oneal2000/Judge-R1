"""
生成 RL 训练数据（用于 swift rlhf）

支持两种模式:
1. 简单模式: 只有案件事实 -> 判决书
2. MRAG 模式: 案件事实 + 相关法条 + 相似案例 -> 判决书

Prompt 格式与 sft_data.py 完全一致，确保训练和推理的一致性。
"""

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Optional
from difflib import SequenceMatcher


# ============ 配置常量（与 sft_data.py 保持一致）============
SYSTEM_PROMPT = "你是一个法律助理，提供帮助。"
MAX_LAWS = 10         # 统一使用前10条法条（与 Agent LawSelect 输出对齐）
MAX_CASE_LENGTH = 2048  # 案例最大长度


def are_strings_similar(str1: str, str2: str, threshold: float = 0.85) -> bool:
    """判断两个字符串是否相似度高于给定阈值"""
    return SequenceMatcher(None, str1, str2).ratio() > threshold


def build_simple_prompt(fact: str) -> str:
    """构建简单 prompt（无 MRAG）
    
    与 sft_data.py 完全一致的 prompt 格式
    """
    return f"""任务背景: 根据以下提供的案件事实，生成一份完整的刑法判决书。判决书需包括案件事实、法律分析、裁判理由以及最终裁判结论。
本案件事实：{fact}
本案件的完整判决书为："""


def build_mrag_prompt(
    fact: str,
    laws: List[str],
    case_qw: Optional[str] = None,
) -> str:
    """构建 MRAG 增强的 prompt
    
    与 sft_data.py 完全一致的 prompt 格式
    
    Args:
        fact: 案件事实
        laws: 相关法条列表（取前 MAX_LAWS 条）
        case_qw: 相似案例的判决书全文
    """
    # 处理案例
    relevant_qw = "无相关判决书"
    if case_qw:
        if len(case_qw) <= MAX_CASE_LENGTH:
            relevant_qw = f"相关案例判决书：{case_qw}"
        else:
            relevant_qw = f"相关案例判决书：{case_qw[:MAX_CASE_LENGTH]}..."
    
    # 处理法条（统一取前 MAX_LAWS 条）
    law_texts = []
    for i, law in enumerate(laws[:MAX_LAWS], 1):
        law_texts.append(f"{i}. {law}")
    relevant_laws = "\n".join(law_texts) if law_texts else "无相关法条"
    
    return f"""任务背景: 根据以下提供的相关案例、法律条款和案件事实，生成一份完整的刑法判决书。判决书需包括案件事实、法律分析、裁判理由以及最终裁判结论。

判决书的格式方面，请参考相关案例中的格式。
{relevant_qw}

以下是与本案件相关的法律条款：
{relevant_laws}

请根据以上内容和下面的案件事实描述，为这个案件生成一份刑事判决书，结构完整，参考提供给你的判决书的格式（需包含案件事实陈述、法律分析、裁判理由及裁判结论等部分）。不超过两千字。
本案件事实：{fact}
本案件的完整判决书为："""


def build_messages(fact: str, laws: List[str] = None, case_qw: str = None) -> list:
    """构建消息列表（swift rlhf 格式）"""
    if laws or case_qw:
        user_prompt = build_mrag_prompt(fact, laws or [], case_qw)
    else:
        user_prompt = build_simple_prompt(fact)
    
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


# ============ MRAG 检索结果加载函数 ============

def build_law_id_text_mapping(law_corpus_path: str) -> Dict[str, str]:
    """构建法条 text_id 到 text 的映射"""
    id_to_text = {}
    with open(law_corpus_path, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            text_id = str(item['text_id'])
            name_n_text = f"{item['name']}：{item['text']}"
            id_to_text[text_id] = name_n_text
    return id_to_text


def build_case_id_mapping(case_corpus_path: str) -> Dict[str, tuple]:
    """构建案例 text_id 到 (text, qw) 的映射"""
    id_to_data = {}
    with open(case_corpus_path, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            text_id = str(item.get('text_id', ''))
            text = item.get('text', '')
            qw = item.get('qw', '') or item.get('fd', '')
            id_to_data[text_id] = (text, qw)
    return id_to_data


def load_law_results(runfile_path: str, law_corpus_path: str) -> Dict[str, List[str]]:
    """从法条检索结果文件加载"""
    id_to_text = build_law_id_text_mapping(law_corpus_path)
    query_to_laws = {}
    
    with open(runfile_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            query_id = parts[0]
            law_id = parts[2]
            law_text = id_to_text.get(law_id, "")
            
            if query_id not in query_to_laws:
                query_to_laws[query_id] = []
            if law_text:
                query_to_laws[query_id].append(law_text)
    
    return query_to_laws


def load_case_results(
    runfile_path: str, 
    case_corpus_path: str,
    query_texts: Dict[str, str],
) -> Dict[str, List[str]]:
    """从案例检索结果文件加载，过滤相似案例"""
    id_to_data = build_case_id_mapping(case_corpus_path)
    query_to_qws = {}
    
    with open(runfile_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            query_id = parts[0]
            case_id = parts[2]
            
            if case_id not in id_to_data:
                continue
            
            case_text, qw = id_to_data[case_id]
            query = query_texts.get(query_id, "")
            
            # 过滤与查询过于相似的案例（避免数据泄露）
            if query and are_strings_similar(query, case_text):
                continue
            
            if query_id not in query_to_qws:
                query_to_qws[query_id] = []
            if qw:
                query_to_qws[query_id].append(qw)
    
    return query_to_qws


def select_case_qw(case_qws: List[str]) -> Optional[str]:
    """选择合适的案例判决书"""
    if not case_qws:
        return None
    
    suitable_qws = [qw for qw in case_qws if len(qw) <= MAX_CASE_LENGTH]
    if suitable_qws:
        return suitable_qws[0]
    else:
        return min(case_qws, key=len)


def convert(
    src: Path, 
    dst_dir: Path, 
    max_samples: int | None, 
    seed: int,
    law_results: Dict[str, List[str]] = None,
    case_results: Dict[str, List[str]] = None,
    use_mrag: bool = False,
) -> Path:
    rng = random.Random(seed)
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / "train.jsonl"

    with src.open("r", encoding="utf-8") as fin:
        try:
            data = json.load(fin)
        except json.JSONDecodeError:
            fin.seek(0)
            data = [json.loads(line) for line in fin if line.strip()]

    if max_samples:
        rng.shuffle(data)
        data = data[:max_samples]

    count = 0
    with dst.open("w", encoding="utf-8") as fout:
        for idx, item in enumerate(data):
            fact = item.get("text") or item.get("input") or ""
            reference = item.get("fd") or item.get("output") or ""
            text_id = str(item.get("text_id", idx))
            if not fact or not reference:
                continue

            # MRAG 支持
            laws = []
            case_qw = None
            if use_mrag:
                if law_results:
                    laws = law_results.get(text_id, [])
                if case_results:
                    case_qws = case_results.get(text_id, [])
                    case_qw = select_case_qw(case_qws)

            fout.write(
                json.dumps(
                    {
                        "id": text_id,
                        "messages": build_messages(fact, laws, case_qw),
                        "reference_document": reference,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            count += 1

    print(f"Wrote {count} samples to {dst}")
    return dst


def main():
    parser = argparse.ArgumentParser(description="Build RL dataset jsonl for GRPO training.")
    parser.add_argument("--src", type=Path, default=Path("data/train.json"))
    parser.add_argument("--dst_dir", type=Path, default=Path("data/rl_train"))
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    
    # MRAG 参数
    parser.add_argument("--use_mrag", action="store_true")
    parser.add_argument("--law_runfile", type=Path, default=None)
    parser.add_argument("--case_runfile", type=Path, default=None)
    parser.add_argument("--law_corpus", type=Path, default=Path("data/law_corpus.jsonl"))
    parser.add_argument("--case_corpus", type=Path, default=None)
    
    args = parser.parse_args()

    print(f"========================================")
    print(f"  生成 RL 训练数据")
    print(f"  MRAG 模式: {args.use_mrag}")
    print(f"  源文件: {args.src}")
    print(f"  输出目录: {args.dst_dir}")
    print(f"========================================")

    # 加载 MRAG 检索结果
    law_results = None
    case_results = None
    
    if args.use_mrag:
        if args.law_runfile and args.law_runfile.exists():
            print(f"Loading law results from {args.law_runfile}...")
            law_results = load_law_results(str(args.law_runfile), str(args.law_corpus))
            print(f"Loaded laws for {len(law_results)} queries")
        
        if args.case_runfile and args.case_corpus and args.case_runfile.exists():
            print(f"Loading case results from {args.case_runfile}...")
            query_texts = {}
            with args.src.open("r", encoding="utf-8") as f:
                try:
                    data = json.load(f)
                except json.JSONDecodeError:
                    f.seek(0)
                    data = [json.loads(line) for line in f if line.strip()]
            for idx, item in enumerate(data):
                text_id = str(item.get("text_id", idx))
                text = item.get("text") or item.get("input") or ""
                query_texts[text_id] = text
            
            case_results = load_case_results(
                str(args.case_runfile), 
                str(args.case_corpus),
                query_texts
            )
            print(f"Loaded cases for {len(case_results)} queries")

    convert(
        args.src, 
        args.dst_dir, 
        args.max_samples, 
        args.seed,
        law_results,
        case_results,
        args.use_mrag,
    )
    
    print("✅ RL 数据生成完成")


if __name__ == "__main__":
    main()
