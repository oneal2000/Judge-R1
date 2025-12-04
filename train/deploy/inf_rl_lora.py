import argparse
import inspect
import json
import os
from typing import Dict, Any

import torch
from peft import LoraConfig, PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def strip_think(text: str) -> str:
    marker = "</think>"
    if marker in text:
        return text.split(marker, 1)[1].strip()
    return text


def load_lora_config(adapter_path: str) -> LoraConfig:
    """Load adapter_config.json but drop unknown fields for older peft versions."""
    with open(os.path.join(adapter_path, "adapter_config.json"), "r", encoding="utf-8") as f:
        raw_cfg: Dict[str, Any] = json.load(f)
    allowed = set(inspect.signature(LoraConfig.__init__).parameters)
    cleaned = {k: v for k, v in raw_cfg.items() if k in allowed}
    return LoraConfig(**cleaned)


def build_prompt(fact: str) -> str:
    input_content = (
        "任务背景: 根据以下提供的案件事实，生成一份完整的刑法判决书。判决书需包括案件事实、"
        "法律分析、裁判理由以及最终裁判结论。\n"
        f"本案件事实：{fact}\n"
        "本案件的完整判决书为："
    )
    return input_content


def generate(model, tokenizer, fact: str) -> str:
    messages = [
        {"role": "system", "content": "你是一个法律助理，提供帮助。"},
        {"role": "user", "content": build_prompt(fact)},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([text], return_tensors="pt").to(model.device)
    with torch.no_grad():
        out_ids = model.generate(
            **inputs,
            max_new_tokens=4096,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1,
        )
    gen = tokenizer.decode(out_ids[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    return strip_think(gen)


def parse_args():
    parser = argparse.ArgumentParser(description="Inference with base + LoRA RL adapter")
    parser.add_argument("--base_model", type=str, required=True, help="Path to base model (SFT)")
    parser.add_argument("--adapter", type=str, required=True, help="Path to LoRA adapter folder")
    parser.add_argument("--dataset_path", type=str, default="../data/test.json", help="Input dataset (json lines)")
    parser.add_argument("--output_path", type=str, default="../outputs/qwen3_rl_lora_raw.json", help="Output json file")
    return parser.parse_args()


def main():
    args = parse_args()
    lora_cfg = load_lora_config(args.adapter)

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model, dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
    )
    model = PeftModel.from_pretrained(model, args.adapter, config=lora_cfg)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model,
        trust_remote_code=True,
        use_fast=False,
    )

    results = []
    with open(args.dataset_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(tqdm(f, desc="Generating"), start=1):
            item = json.loads(line.strip())
            fact = item.get("input") or item.get("text") or item.get("Fact") or ""
            exp_ans = item.get("output") or item.get("fd") or ""
            if not fact:
                continue
            gen_ans = generate(model, tokenizer, fact)
            # 与其他推理脚本保持一致，边生成边输出，便于观察进度和内容
            print(f"[{idx}] Generated answer:\n{gen_ans}\n", flush=True)
            results.append({"gen_ans": gen_ans, "exp_ans": exp_ans})

    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Results saved to {args.output_path}")


if __name__ == "__main__":
    main()
