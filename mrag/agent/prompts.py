"""
Agent 共享提示词模板

训练 (gen_agent_rl_data.py) 和推理 (law_agent.py, hybrid_agent.py) 统一使用此模块
确保训练和推理使用完全一致的提示词，避免分布偏移

配置常量也在此统一管理
"""

# ============== 配置常量 ==============
MAX_LAW_TEXT_LENGTH = 300  # 法条文本截断长度，训练和推理统一
MAX_FACT_LENGTH = 2000     # 案件事实截断长度
MAX_QUERIES = 8            # QueryGen 最大查询数
MIN_QUERIES = 5            # QueryGen 最小查询数


# ============== QueryGen 提示词 ==============
QUERYGEN_SYSTEM_PROMPT = "你是一个专业的法律助手。你必须严格遵循输出格式要求。"

QUERYGEN_USER_TEMPLATE = """请根据以下案件事实，生成 5-8 个用于检索相关法条的查询。

案件事实：
{fact}

要求：
1) 每个查询聚焦不同法律要点（罪名构成要件、未遂/既遂、共犯、量刑情节、罚金/追缴等）
2) 查询简洁，包含关键法律术语
3) 必须只输出一个 JSON 数组，数组元素为字符串
4) 必须生成至少 5 个查询

输出格式示例：
[
  "砸车窗后对车内财物的秘密窃取 行为构成 要件 数额较大",
  "同日多次在公共道路旁砸破车窗盗窃 现金合计11400元 量刑区间",
  "盗窃罪 构成要件 以非法占有为目的 盗窃公私财物 数额较大",
  "认罪认罚从宽 如实供述 坦白情节 从轻处罚",
  "罚金刑 附加刑 盗窃罪 罚金数额确定",
  "退赔被害人损失 获得谅解 量刑情节"
]

现在开始输出（只输出 JSON 数组）："""


# ============== LawSelect 提示词 ==============
# 训练和所有推理模式（普通/Hybrid）统一使用此提示词
LAWSELECT_SYSTEM_PROMPT = "你是一名资深法官。你必须严格遵循输出格式要求。"

LAWSELECT_USER_TEMPLATE = """请根据案件事实，从候选法条中筛选与本案真正相关的法条。

案件事实：
{fact}

候选法条（共 {num_candidates} 条，每条有编号和 law_id）：
{candidate_laws}

筛选原则（按重要性排序）：
1) 【必选】主要罪名的定罪条文（如盗窃罪选刑法第264条，故意伤害选第234条）
2) 【必选】主刑条文：有期徒刑、拘役、管制等刑期相关（如刑法第45条、第42条、第38条）
3) 【必选】附加刑条文：罚金刑（刑法第52条、第53条）、没收财产（第59条）
4) 【按需】量刑情节：自首（第67条）、坦白（第67条第3款）、从犯（第27条）、累犯（第65条）、缓刑（第72条、第73条）
5) 【按需】数罪并罚（第69条）、追缴退赔（第64条）等程序性条文
6) 【排除】与本案罪名、情节明显无关的条文

⚠️ 重要：宁可多选也不要漏选！遗漏重要法条会导致判决书错误。通常每个案件需要 20-30 条法条。

输出要求：
1) 只输出一个 JSON 对象，不要输出其他内容
2) 对于每条法条，必须提供 law_id、law（法条名称）、reason（详细理由）、confidence（0-1的置信度）

输出格式：
{{
  "selected_articles": [
    {{"law_id": "264", "law": "刑法第264条", "reason": "被告人以秘密方式窃取车内财物，金额11400元，符合盗窃罪'数额较大'的构成要件。", "confidence": 0.95}},
    {{"law_id": "45", "law": "刑法第45条", "reason": "确定有期徒刑的刑期幅度（六个月以上十五年以下）。", "confidence": 0.90}},
    {{"law_id": "52", "law": "刑法第52条", "reason": "盗窃罪可并处罚金，本案需依据此条确定罚金幅度。", "confidence": 0.88}},
    {{"law_id": "53", "law": "刑法第53条", "reason": "规定罚金缴纳方式。", "confidence": 0.85}},
    {{"law_id": "67", "law": "刑法第67条第3款", "reason": "被告人归案后如实供述犯罪事实，依法可从轻处罚。", "confidence": 0.85}},
    {{"law_id": "64", "law": "刑法第64条", "reason": "违法所得应予追缴或退赔。", "confidence": 0.80}}
  ],
  "rejected_articles": [
    {{"law_id": "263", "law": "刑法第263条", "reason": "本案系秘密窃取而非暴力胁迫当场取财，不构成抢劫罪。"}}
  ]
}}

现在开始输出（只输出 JSON）："""


# ============== 工具函数 ==============
def truncate_law_text(law_text: str, max_length: int = MAX_LAW_TEXT_LENGTH) -> str:
    """统一的法条文本截断函数"""
    if len(law_text) > max_length:
        return law_text[:max_length] + "..."
    return law_text


def truncate_fact(fact: str, max_length: int = MAX_FACT_LENGTH) -> str:
    """统一的案件事实截断函数"""
    if len(fact) > max_length:
        return fact[:max_length]
    return fact


def format_candidate_law(
    idx: int, 
    law_id: str, 
    law_name: str, 
    law_text: str,
    max_text_length: int = MAX_LAW_TEXT_LENGTH
) -> str:
    """统一的候选法条格式化函数
    
    Args:
        idx: 序号（从 0 开始，会自动 +1 显示）
        law_id: 法条 ID
        law_name: 法条名称
        law_text: 法条内容
        max_text_length: 最大文本长度
    
    Returns:
        格式化的字符串，如 "1. [law_id=264] 刑法第264条：盗窃公私财物..."
    """
    truncated_text = truncate_law_text(law_text, max_text_length)
    return f"{idx + 1}. [law_id={law_id}] {law_name}：{truncated_text}"
