import argparse
import time
from tqdm import tqdm
import torch.optim as optim
import json
import os

import torch
import deepspeed
from torch.utils.data import DistributedSampler
from transformers import AutoModelForCausalLM, AutoTokenizer

from data import LegalAidDataset, LegalAidCollator

# (可选) 显存打印
from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo
from deepspeed.runtime.utils import see_memory_usage
from peft import LoraConfig, get_peft_model, TaskType


def get_argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", type=str, default="")
    parser.add_argument("--max_src", type=int, default=2048)
    parser.add_argument("--max_tar", type=int, default=6144)
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--tokenizer_name_or_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--workers", type=int, default=4)

    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=5e-5)

    # LoRA 开关：默认 True，但你命令行 --disable_lora 会关掉
    parser.add_argument("--enable_lora", action="store_true", default=True)
    parser.add_argument("--disable_lora", action="store_false", dest="enable_lora")

    parser.add_argument("--local_rank", type=int, default=-1, help="Reserved for deepspeed")

    return parser


def print_gpu_utilization(gpu_idx: int = 0):
    nvmlInit()
    h = nvmlDeviceGetHandleByIndex(gpu_idx)
    info = nvmlDeviceGetMemoryInfo(h)
    print(f"[GPU{gpu_idx}] used={info.used//1024**2}MB total={info.total//1024**2}MB")


def ensure_pad_token(tokenizer, model):
    """
    不要 tokenizer.pad_token_id = 0 这种硬设。
    如果 tokenizer 没有 pad，就用 eos 作为 pad（对 CausalLM 常见做法）。
    """
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token is None:
            raise ValueError("Tokenizer has no pad_token_id and no eos_token; cannot set pad.")
        tokenizer.pad_token = tokenizer.eos_token

    # 确保 model config 也同步
    model.config.pad_token_id = tokenizer.pad_token_id


def maybe_apply_lora(model, enable_lora: bool):
    if not enable_lora:
        return model

    # 显式指定 Qwen 的所有线性层，效果通常比自动查找更好
    # target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=128,            
        lora_alpha=256,   
        lora_dropout=0.05,
        target_modules=target_modules,
        bias="none"
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


def main():
    # deepspeed distributed init
    deepspeed.init_distributed(dist_backend="nccl")

    parser = get_argument_parser()
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()

    print("[INFO] loading model...")
    t0 = time.time()

    # 用 torch_dtype，不要 dtype
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16
    )

    # 省显存：梯度检查点
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    # 可选：LoRA
    model = maybe_apply_lora(model, args.enable_lora)

    # 只训练 requires_grad 的参数（全参时就是全部）
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())

    print(f"[INFO] model loaded in {time.time() - t0:.2f}s")

    # tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name_or_path,
        trust_remote_code=True,
        use_fast=False
    )

    #  正确设置 pad token（别设 0）
    ensure_pad_token(tokenizer, model)

    print("[INFO] tokenizer special ids:",
          "bos=", tokenizer.bos_token_id,
          "eos=", tokenizer.eos_token_id,
          "pad=", tokenizer.pad_token_id)

    # dataset & collator
    train_dataset = LegalAidDataset(
        data_path=args.train_path,
        tokenizer=tokenizer,
        max_src=args.max_src,
        max_tar=args.max_tar
    )
    data_collator = LegalAidCollator(
        tokenizer=tokenizer,
        max_src=args.max_src,
        max_tar=args.max_tar
    )

    # sampler: shuffle across epochs
    train_sampler = DistributedSampler(train_dataset, shuffle=True)

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.per_device_train_batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        sampler=train_sampler,
        collate_fn=data_collator
    )
    import json as json_lib

    # 动态计算训练步数
    world_size = torch.distributed.get_world_size()
    steps_per_epoch = len(train_dataloader)
    total_steps = steps_per_epoch * args.num_train_epochs
    warmup_steps = max(1, int(total_steps * 0.1))  # 3% warmup

    # 加载并修改 DeepSpeed 配置
    with open(args.deepspeed_config, 'r') as f:
        ds_config = json_lib.load(f)

    # 同步 gradient_accumulation_steps
    ds_config["gradient_accumulation_steps"] = args.gradient_accumulation_steps
    ds_config["train_batch_size"] = (
        args.per_device_train_batch_size * world_size * args.gradient_accumulation_steps
    )

    # 动态设置 scheduler 步数
    if "scheduler" in ds_config:
        ds_config["scheduler"]["params"]["total_num_steps"] = total_steps
        ds_config["scheduler"]["params"]["warmup_num_steps"] = warmup_steps
        ds_config["scheduler"]["params"]["warmup_max_lr"] = args.learning_rate

    print(f"[INFO] Dynamic DeepSpeed config: total_steps={total_steps}, warmup_steps={warmup_steps}")

    # 保存临时配置文件
    import tempfile
    temp_config = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
    json_lib.dump(ds_config, temp_config)
    temp_config.close()
    args.deepspeed_config = temp_config.name

    
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    # deepspeed engine init
    optimizer = optim.AdamW(
        model_parameters,
        lr=args.learning_rate,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.01
    )

    model_engine, optimizer, _, _ = deepspeed.initialize(
        args=args,
        model=model,
        optimizer=optimizer
    )

    print_gpu_utilization(0)

    os.makedirs(args.output_dir, exist_ok=True)
    loss_log_path = os.path.join(args.output_dir, "train_loss.jsonl")
    print(f"[INFO] Training loss will be saved to: {loss_log_path}")
    
    # 如果文件存在则清空，从头开始记录（或者根据需求保留）
    with open(loss_log_path, "w", encoding="utf-8") as f:
        pass 

    model_engine.train()

    for epoch in range(args.num_train_epochs):
        # ✅ 关键：每个 epoch 设置随机种子/打散
        train_sampler.set_epoch(epoch)

        for step, batch in enumerate(tqdm(train_dataloader), start=1):
            input_ids = batch["input_ids"].to(model_engine.device)
            labels = batch["labels"].to(model_engine.device)
            attention_mask = batch["attention_mask"].to(model_engine.device)

            outputs = model_engine(
                input_ids=input_ids,
                labels=labels,
                attention_mask=attention_mask
            )
            loss = outputs.loss

            if step % 20 == 0:
                current_loss = loss.item()
                print(f"[epoch={epoch}] step={step} loss={loss.item():.4f}")
                see_memory_usage("mem", force=True)
                print_gpu_utilization(0)

                global_step = step + epoch * len(train_dataloader)
                log_entry = {
                    "epoch": epoch,
                    "step": step,
                    "global_step": global_step,
                    "loss": current_loss,
                    "timestamp": time.time()
                }
                
                with open(loss_log_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(log_entry) + "\n")

            model_engine.backward(loss)
            model_engine.step()

        torch.cuda.empty_cache()
        model_engine.save_checkpoint(args.output_dir)
        print(f"[INFO] saved checkpoint epoch={epoch} to {args.output_dir}")
        
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
        print(f"[Rank {torch.distributed.get_rank()}] Training completed successfully!")

if __name__ == "__main__":
    main()
