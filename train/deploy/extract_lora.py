import torch
import os
import argparse
from transformers import AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, TaskType

def extract_and_save_adapter(base_model_path, mixed_checkpoint_dir, output_adapter_dir,
                              lora_rank=64, lora_alpha=128, target_modules=None):
    """
    从 DeepSpeed 保存的混合权重中提取 LoRA adapter
    
    参数必须与训练时完全一致！
    """
    if target_modules is None:
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
    
    print(f"[1/4] Loading Base Model from: {base_model_path}")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        device_map="cpu",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )

    print(f"[2/4] Initializing LoRA Config")
    print(f"      rank={lora_rank}, alpha={lora_alpha}")
    print(f"      target_modules={target_modules}")
    
    config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=0.05,
        target_modules=target_modules,
        bias="none"
    )
    model = get_peft_model(base_model, config)

    print(f"[3/4] Loading Mixed Weights from: {mixed_checkpoint_dir}")
    
    # 尝试多种加载方式
    loaded = False
    
    # 检查分片文件可能的位置
    possible_dirs = [
        mixed_checkpoint_dir,
        os.path.join(mixed_checkpoint_dir, "pytorch_model.bin"),  # 新增：子目录情况
    ]
    
    for check_dir in possible_dirs:
        if loaded:
            break
            
        # 方式1: 直接加载单个 pytorch_model.bin 文件
        bin_path = os.path.join(check_dir, "pytorch_model.bin")
        if os.path.isfile(bin_path):
            print(f"   -> Loading from {bin_path}")
            state_dict = torch.load(bin_path, map_location="cpu")
            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            print(f"   -> Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}")
            loaded = True
            break
        
        # 方式2: 尝试 safetensors
        safetensor_path = os.path.join(check_dir, "model.safetensors")
        if os.path.isfile(safetensor_path):
            from safetensors.torch import load_file
            print(f"   -> Loading from {safetensor_path}")
            state_dict = load_file(safetensor_path)
            model.load_state_dict(state_dict, strict=False)
            loaded = True
            break
        
        # 方式3: 尝试分片加载（检查 index.json 文件）
        index_file = os.path.join(check_dir, "pytorch_model.bin.index.json")
        if os.path.isfile(index_file):
            try:
                from transformers.modeling_utils import load_sharded_checkpoint
                print(f"   -> Loading sharded checkpoint from {check_dir}")
                load_sharded_checkpoint(model, check_dir, strict=False)
                print("   -> Loaded sharded checkpoint successfully.")
                loaded = True
                break
            except Exception as e:
                print(f"   -> Sharded loading failed: {e}")
    
    if not loaded:
        raise RuntimeError(f"Cannot load weights from {mixed_checkpoint_dir}")

    print(f"[4/4] Saving Extracted Adapter to: {output_adapter_dir}")
    os.makedirs(output_adapter_dir, exist_ok=True)
    model.save_pretrained(output_adapter_dir)
    print("Done! Now you can use merge_lora.py")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", required=True)
    parser.add_argument("--mixed_dir", required=True, help="Directory containing pytorch_model.bin")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--lora_rank", type=int, default=64)
    parser.add_argument("--lora_alpha", type=int, default=128)
    parser.add_argument("--target_modules", type=str, default="q_proj,k_proj,v_proj,o_proj",
                        help="Comma-separated list of target modules")
    args = parser.parse_args()
    
    target_modules = [m.strip() for m in args.target_modules.split(",")]
    
    extract_and_save_adapter(
        args.base_model, 
        args.mixed_dir, 
        args.output_dir,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        target_modules=target_modules
    )