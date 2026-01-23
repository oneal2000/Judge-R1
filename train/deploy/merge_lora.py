import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import os

def merge_lora(base_model_path, adapter_dir, output_dir):
    print(f"Loading base model from {base_model_path}")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    
    print(f"Loading LoRA adapter from {adapter_dir}")
    model = PeftModel.from_pretrained(base_model, adapter_dir)
    
    print("Merging weights...")
    model = model.merge_and_unload()
    
    print(f"Saving merged model to {output_dir}")
    model.save_pretrained(output_dir)

    # 【修改版】尝试保存 tokenizer，如果失败则提示手动复制
    try:
        print("Try saving tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
        tokenizer.save_pretrained(output_dir)
    except Exception as e:
        print(f"Warning: Tokenizer saving failed ({e}).")
        print("Please manually copy tokenizer files (tokenizer.json, vocab.json, etc.) from base_model_path to output_dir.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str, required=True)
    parser.add_argument("--adapter_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()
    
    merge_lora(args.base_model, args.adapter_dir, args.output_dir)