import json
import os
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse
import torch

def parse_arguments():
    parser = argparse.ArgumentParser(description='Run legal document generation with specified model suffix.')
    parser.add_argument('--suffix', type=str, required=True, help='Suffix of the model path')
    parser.add_argument('--base_model_path', type=str, default=None, help='Base model path (for LoRA adapters)')
    parser.add_argument('--dataset_path', type=str, default="../data/test.json", help='Path to the dataset')
    parser.add_argument('--output_path', type=str, default="../output/finetune/output.json", help='Path to save the output')
    return parser.parse_args()

def strip_think(text: str) -> str:
    marker = "</think>"
    if marker in text:
        return text.split(marker, 1)[1].strip()
    return text

# Function to generate reasoning given a question
def generate_reasoning(fact):
    input_content = f"""
任务背景: 根据以下提供的案件事实，生成一份完整的刑法判决书。判决书需包括案件事实、法律分析、裁判理由以及最终裁判结论。
本案件事实：{fact}
本案件的完整判决书为：
"""
    messages = [
        {"role": "system", "content": "你是一个法律助理，提供帮助。"},
        {"role": "user", "content": input_content}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=4096,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.1
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return strip_think(response)

# Function to process a dataset
def process_dataset(dataset_path, output_path):
    # Load the dataset
    test_result = []
    with open(dataset_path, "r") as f:
        for line in tqdm(f):
            one = json.loads(line.strip())
            fact = one.get("input") or one.get("text") or one.get("Fact")
            exp_ans = one.get("output") or one.get("fd")
            gen_ans = generate_reasoning(fact)
            print(f"Generated answer: {gen_ans}")

            # Update data and add to results list
            entry = {
                "gen_ans": gen_ans,
                "exp_ans": exp_ans
            }
            test_result.append(entry)

    # Save the results
    with open(output_path, 'w') as f:
        json.dump(test_result, f, ensure_ascii=False, indent=4)
    print(f"Results saved to {output_path}")

def main():
    args = parse_arguments()
    model_name = args.suffix
    base_model = args.base_model_path or model_name

    global model, tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    # 如果 suffix 目录是 LoRA 适配器，尝试加载适配器权重
    adapter_candidate = os.path.join(model_name, "pytorch_model.bin")
    if os.path.isdir(adapter_candidate):
        adapter_candidate = os.path.join(adapter_candidate, "pytorch_model.bin")
    if os.path.isfile(adapter_candidate):
        state_dict = torch.load(adapter_candidate, map_location="cpu")
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        print("LoRA adapter loaded. Missing keys (first 5):", missing[:5], "Unexpected keys:", unexpected[:5])

    tokenizer = AutoTokenizer.from_pretrained(
        base_model,
        trust_remote_code=True,
        use_fast=False,
    )

    process_dataset(args.dataset_path, args.output_path)

if __name__ == "__main__":
    main()
