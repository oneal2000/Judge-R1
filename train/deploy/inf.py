"""
统一的 vLLM 推理脚本

支持两种模式:
1. 使用预先格式化的数据（推荐）- 确保训练和推理的 prompt 完全一致
2. 在线构建 prompt（兼容旧版本）

使用方法:
  # 使用预先格式化的数据（推荐）
  python inf.py --model_path MODEL --dataset_path data/test_sft.json --output_path out.json --mode sft
  
  # MRAG 模式
  python inf.py --model_path MODEL --dataset_path data/test_sft_mrag.json --output_path out.json --mode sft
"""

import json
import os
import re
import argparse
from tqdm import tqdm
from pathlib import Path
from typing import Dict, List, Optional
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import logging

logger = logging.getLogger(__name__)

# ============ 配置常量（与训练数据保持一致）============
SYSTEM_PROMPT = "你是一个法律助理，提供帮助。"
MAX_LAWS = 10         # 统一使用前10条法条（与 SFT/RL 训练数据对齐）
MAX_CASE_LENGTH = 2048  # 案例最大长度


def parse_arguments():
    parser = argparse.ArgumentParser(description="Unified vLLM inference for Direct, ICL, SFT, and RL.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--dataset_path", type=str, default="data/test.json", help="Path to dataset")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save output")
    parser.add_argument("--mode", type=str, required=True, choices=["direct", "icl", "sft", "rl"], 
                        help="Inference mode to select correct prompt and params")
    parser.add_argument("--tensor_parallel_size", type=int, default=1, help="Number of GPUs")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.50, help="GPU memory utilization")
    
    # 数据格式选项
    parser.add_argument("--use_formatted_data", action="store_true", default=True,
                        help="Use pre-formatted data (recommended for consistency)")
    
    return parser.parse_args()


def strip_think(text: str, model_path: str = "") -> str:
    """去除 thinking 模型的思考过程
    
    支持多种格式:
    - Qwen3-Thinking: </think> 标记
    - LegalOne: [最终回答] 标记
    """
    # Qwen3-Thinking 格式
    if "</think>" in text:
        return text.split("</think>", 1)[1].strip()
    
    # LegalOne 格式 - 多种可能的标记
    legalone_markers = ["[最终回答]", "【最终回答】", "[Final Answer]", "最终回答：", "最终回答:"]
    for marker in legalone_markers:
        if marker in text:
            return text.split(marker, 1)[1].strip()
    
    return text.strip()


def is_thinking_model(model_path: str) -> bool:
    """判断是否为 Thinking 模型（Qwen3-Thinking）"""
    model_path_lower = model_path.lower()
    return ("qwen3" in model_path_lower or "thinking" in model_path_lower) and "legalone" not in model_path_lower


def is_legalone_model(model_path: str) -> bool:
    """判断是否为 LegalOne 模型"""
    return "legalone" in model_path.lower()


# ================= Prompt 构建函数（与训练数据完全一致）=================

def build_simple_prompt(fact: str) -> str:
    """构建简单 prompt（无 MRAG）- 与 sft_data.py 完全一致"""
    return f"""任务背景: 根据以下提供的案件事实，生成一份完整的刑法判决书。判决书需包括案件事实、法律分析、裁判理由以及最终裁判结论。
本案件事实：{fact}
本案件的完整判决书为："""


def build_prompt_direct(tokenizer, fact: str, model_path: str = "") -> str:
    """适用于 Direct 的 Prompt"""
    input_content = f"""
案件事实：{fact}
请根据上面提供的事实描述，生成一篇完整且具有法律效力的中文的刑事判决书。生成的文书必须结构严谨、逻辑清晰；确保文书所有部分均符合真实司法文书的写作规范，语言应正式、客观、清晰
"""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": input_content}
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    if is_thinking_model(model_path) or is_thinking_model(tokenizer.name_or_path):
        text = text + "<think>\n"

    if tokenizer.bos_token:
        return f"{tokenizer.bos_token}{text}"
    return text


def build_prompt_icl(tokenizer, fact: str, model_path: str = "") -> str:
    """适用于 ICL (Few-shot) 的 Prompt"""
    input_content = f"""
案件事实：{fact}
请根据上面提供的事实描述，生成一篇完整且具有法律效力的中文的刑事判决书。生成的文书必须结构严谨、逻辑清晰，并且要包含以下个部分：

1. 开头
2. **事实描述**  
- 请直接复述以下提供的事实描述，不得进行删减或改动：

3. **司法理由**  
- 根据上述事实描述，结合相关刑法条款、法律原则和司法解释，详细论述案件的法律分析，以"本院认为"开头。  
- 分析内容应包括：  
     - 对证据的评估  
     - 犯罪构成要件的论证  
     - 相关法律条文的引用及其适用说明  
- 请确保推理过程严谨、论证充分，为判决结果提供充分法律依据。

4. **判决结果**  
- 在此部分明确给出法院的最终判决，以"判决如下"开头。  
- 判决内容应具体包括处罚措施（如刑期、罚金、附加刑等）及其法律依据，确保与前述司法理由相呼应，文书整体逻辑连贯。

**注意：**  确保文书所有部分均符合真实司法文书的写作规范，语言应正式、客观、清晰。判决书的格式可以参考示例中的格式。
示例：
刑法判决书:  
许帅伟妨害公务一审刑事判决书 河南省登封市人民法院 刑事判决书 （2017）豫0185刑初1244号 公诉机关河南省登封市人民检察院。 被告人许帅伟，男，1979年8月24日出生于河南省登封市，汉族，中专文化程度，中共党员，郑州市嵩阳煤机制造有限公司员工，住登封市。因涉嫌犯妨害公务罪于2017年8月15日被登封市公安局刑事拘留，同年8月22日被取保候审。 指定辩护人景俊娜，河南群达律师事务所律师。 登封市人民检察院以登检刑诉［2017］616号起诉书指控被告人许帅伟犯妨害公务罪，于2017年11月17日向本院提起公诉，并建议本院适用速裁程序审理。依据《全国人民代表大会常务委员会关于授权最高人民法院、最高人民检察院在部分地区开展刑事案件认罪认罚从宽制度试点工作的决定》，本院决定适用速裁程序，实行独任审判，公开开庭审理了本案。登封市人民检察院检察员梁书伟出庭支持公诉。被告人许帅伟及其指定辩护人均到庭参加了诉讼。现已审理终结。 登封市人民检察院指控，2017年8月14日17时许，被告人 许帅伟在登封市少林办事处耿庄登封市第二游泳馆游泳期间，与他人发生争执并打架，登封市公安局少林派出所民警乔登辉带领辅警曹某和常某现场处警，期间，被告人许帅伟抗拒民警执法，并拿起杯具砸伤辅警曹某。经鉴定，曹某面部的损伤程度构成轻微伤。 为证明该指控事实，公诉机关向本院提供了被告人许帅伟的供述；证人乔登辉、常某、曹某、裴某等人的证言；辨认笔录；现场勘验检查笔录及照片；登封市公安局处警表；鉴定书；视听资料；血醇鉴定意见书；收到条、撤诉书；公安机关出具的户籍证明、无前科证明及到案经过等证据。据此认为被告人许帅伟的行为已构成妨害公务罪，并建议本院对被告人许帅伟判处拘役三至五个月。 经法庭审理，查明的事实、证据与指控的事实、证据相同，被告人许帅伟及其指定辩护人对起诉书指控的事实、罪名、证据及量刑建议均不持异议，且被告人签字具结。本院予以确认。 本院认为，被告人许帅伟以暴力方法阻碍公安机关依法执行公务，其行为已构成妨害公务罪。公诉机关指控被告人许帅伟犯妨害公务罪的事实、罪名及理由成立，量刑建议适当，依法予以支持。被告人许帅伟归案后，如实供述自己的罪行，可以从轻处罚；其自愿认罪，已赔偿受伤辅警经济损失，可酌情从轻处罚。与之相对应的辩护意见成立，本院予以采纳。 根据《中华人民共和国刑法》第二百七十七条第一款和第五款、第六十七条第三款之规定，判决如下： 被告人许帅伟犯妨害公务罪，判处拘役三个月。（刑期自判决执行之日起计算。判决执行前先行羁押的，羁押一日折抵刑期一日。即自2017年11月17日起至2018年2月8日止。原被羁押的8日已予折抵。） 如不服本判决，可在接到判决书的第二日起十日内，通过本院或直接向河南省郑州市中级人民法院提出上诉。书面上诉的，应当提交上诉状正本一份，副本十五份。 审判员杨海洋 二〇一七年十一月十七日 书记员张宇

请参考以上示例，根据案件事实生成一份刑事判决书，结构完整，需包含案件事实陈述、法律分析、裁判理由及裁判结论等部分。不超过两千字。
"""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": input_content}
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    if is_thinking_model(model_path) or is_thinking_model(tokenizer.name_or_path):
        text = text + "<think>\n"
    
    if tokenizer.bos_token:
        return f"{tokenizer.bos_token}{text}"
    return text


def build_prompt_from_formatted(tokenizer, input_prompt: str, model_path: str = "") -> str:
    """使用预先格式化的 prompt 构建最终输入
    
    这是推荐的方式，确保训练和推理的 prompt 完全一致
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": input_prompt}
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    if is_thinking_model(model_path) or is_thinking_model(tokenizer.name_or_path):
        text = text + "<think>\n"
    
    if tokenizer.bos_token:
        return f"{tokenizer.bos_token}{text}"
    return text


# ================= 主逻辑 =================

def main():
    args = parse_arguments()
    
    print(f"[INFO] Mode: {args.mode.upper()}")
    print(f"[INFO] Reading dataset from {args.dataset_path}...")
    
    # 读取数据
    data_items = []
    with open(args.dataset_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if not line.strip():
                continue
            try:
                one = json.loads(line.strip())
                text_id = str(one.get("text_id", idx))
                
                # 检查是否为预先格式化的数据
                if "input" in one:
                    # 预先格式化的数据（推荐）
                    input_prompt = one["input"]
                    fact = None  # 不需要
                else:
                    # 原始数据格式
                    fact = one.get("text") or one.get("Fact")
                    input_prompt = None
                
                exp_ans = one.get("output") or one.get("fd")
                
                if input_prompt or fact:
                    data_items.append({
                        "text_id": text_id, 
                        "input_prompt": input_prompt,
                        "fact": fact, 
                        "exp_ans": exp_ans
                    })
            except Exception as e:
                print(f"[WARN] Failed to parse line {idx}: {e}")
                continue
    
    print(f"[INFO] Loaded {len(data_items)} samples.")
    
    # 检测数据格式
    has_formatted = data_items[0].get("input_prompt") is not None if data_items else False
    if has_formatted:
        print(f"[INFO] 使用预先格式化的数据（推荐）")
    else:
        print(f"[INFO] 使用原始数据格式，将在线构建 prompt")
    
    # 设置 max_model_len
    # 检测是否为 MRAG 数据（通过文件名或内容判断）
    is_mrag = "mrag" in args.dataset_path.lower()
    if is_mrag:
        max_model_len = 8096  # MRAG: 4000输入 + 3072输出 + buffer
        print(f"[INFO] MRAG 模式: max_model_len={max_model_len}")
    else:
        max_model_len = 5120   # 标准: 1500输入 + 3072输出 + buffer
        print(f"[INFO] 标准模式: max_model_len={max_model_len}")

    # 初始化 vLLM
    gpu_mem_util = getattr(args, 'gpu_memory_utilization', 0.50)
    print(f"[INFO] Initializing vLLM with TP={args.tensor_parallel_size}, gpu_mem_util={gpu_mem_util}...")
    llm = LLM(
        model=args.model_path,
        tensor_parallel_size=args.tensor_parallel_size,
        trust_remote_code=True,
        gpu_memory_utilization=gpu_mem_util,
        max_model_len=max_model_len,
    )
    tokenizer = llm.get_tokenizer()
    
    # 如果 tokenizer 没有 chat_template，尝试从 jinja 文件加载
    if tokenizer.chat_template is None:
        jinja_path = Path(args.model_path) / "chat_template.jinja"
        if jinja_path.exists():
            print(f"[INFO] Loading chat_template from {jinja_path}")
            tokenizer.chat_template = jinja_path.read_text()
        else:
            # 使用默认的 Qwen3 chat template
            print(f"[WARN] No chat_template found, using default Qwen3 template")
            tokenizer.chat_template = """{% for message in messages %}{% if loop.first and messages[0]['role'] != 'system' %}{{ '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n' }}{% endif %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"""

    # 构建 Prompts
    prompts = []
    for item in data_items:
        if has_formatted and args.mode in ["sft", "rl"]:
            # 使用预先格式化的数据（推荐）
            p = build_prompt_from_formatted(tokenizer, item["input_prompt"], args.model_path)
        elif args.mode == "icl":
            p = build_prompt_icl(tokenizer, item["fact"], args.model_path)
        elif args.mode == "direct":
            p = build_prompt_direct(tokenizer, item["fact"], args.model_path)
        else:
            # SFT/RL 使用原始数据时，使用简单 prompt
            if item["fact"]:
                input_prompt = build_simple_prompt(item["fact"])
                p = build_prompt_from_formatted(tokenizer, input_prompt, args.model_path)
            else:
                p = build_prompt_from_formatted(tokenizer, item["input_prompt"], args.model_path)
        prompts.append(p)

    # 设置采样参数
    temperature = 0.7
    max_tokens = 4096
    
    if args.mode == "rl":
        temperature = 0.6

    sampling_params = SamplingParams(
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=0.9,
        repetition_penalty=1.1
    )
    print(f"[INFO] Sampling Params: temp={temperature}, max_tokens={max_tokens}")

    # 批量推理
    print("[INFO] Starting generation...")
    outputs = llm.generate(prompts, sampling_params)

    # 处理与保存
    test_result = []
    for i, output in enumerate(outputs):
        full_res = output.outputs[0].text
        doc_res = strip_think(full_res, args.model_path)
        
        entry = {
            "text_id": data_items[i]["text_id"],
            "gen_ans": doc_res,
            "gen_full": full_res,
            "exp_ans": data_items[i]["exp_ans"]
        }
        test_result.append(entry)

    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(test_result, f, ensure_ascii=False, indent=4)
    
    print(f"[DONE] Saved to {args.output_path}")


if __name__ == "__main__":
    main()
