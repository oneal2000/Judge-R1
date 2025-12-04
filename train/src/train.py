import json
import torch
import argparse
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

from tensorboardX import SummaryWriter
import os
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM
from data import LegalAidDataset,LegalAidCollator
from arguments import ModelArguments, DataTrainingArguments, FinetuneArguments as TrainingArguments
# from models.modeling_baichuan import BaiChuanForCausalLM
import torch
import deepspeed
import argparse
import time
from tqdm import tqdm
from torch.utils.data import DistributedSampler
from peft import LoraConfig, get_peft_model, TaskType


def get_argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", type=str, default='')
    parser.add_argument("--max_src", type=int, default=150)
    parser.add_argument("--max_tar", type=int, default=512)
    parser.add_argument("--model_name_or_path", type=str, default="data_dir")
    parser.add_argument("--tokenizer_name_or_path", type=str, default="data_dir")
    parser.add_argument("--ds_config", type=str, default="data_dir")
    parser.add_argument("--output_dir", type=str, default="data_dir")
    parser.add_argument("--tensorboard_dir", type=str, default="data_dir")
    parser.add_argument("--workers", type=int, default=1)


    parser.add_argument("--per_device_train_batch_size", type=int,default=4)
    parser.add_argument("--num_train_epochs", type=int,default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int,default=4)
    parser.add_argument("--save_steps", type=int,default=4)
    parser.add_argument("--learning_rate", type=float,default=1e-5)
    parser.add_argument("--remove_unused_columns", type=bool,default=4)
    parser.add_argument("--logging_steps", type=int,default=4)
    # LoRA 开关，默认启用，可通过 --disable_lora 关闭以做全参 SFT
    parser.add_argument("--enable_lora", action="store_true", default=True)
    parser.add_argument("--disable_lora", action="store_false", dest="enable_lora")

    parser.add_argument("--local_rank", type=int, default=-1,
                    help="Reserved for deepspeed framework")

    return parser

from deepspeed.runtime.utils import see_memory_usage
from pynvml import *

def print_gpu_utilization():
    # 打印显存使用情况
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(handle)
    print(f"GPU memory occupied: {info.used//1024**2} MB.")
    
def main(): 
    # deepspeed initialize
    deepspeed.init_distributed(dist_backend='nccl')
    parser = get_argument_parser()
    parser = deepspeed.add_config_arguments(parser)
    cmd_args = parser.parse_args()
    print('loading model')
    start_time = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        cmd_args.model_name_or_path,
        trust_remote_code=True,
        dtype=torch.bfloat16  # align with DeepSpeed bf16 config
    )
    model.gradient_checkpointing_enable()   # 梯度检查点 节省显存
    model.enable_input_require_grads()    
    model.is_parallelizable = True    # gpu并行
    model.model_parallel = True     # gpu并行

    if cmd_args.enable_lora:
        # LoRA 适配：动态匹配线性层名称，避免命名不一致
        target_candidates = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj",
                             "up_proj", "down_proj", "wqkv", "wo", "w1", "w2", "w3"]
        found_targets = set()
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear):
                lname = name.lower()
                for t in target_candidates:
                    if t in lname:
                        found_targets.add(t)
        if not found_targets:
            found_targets = set(target_candidates)
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=8,
            lora_alpha=16,
            lora_dropout=0.05,
            target_modules=list(found_targets),
            bias="none"
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())

    end_time = time.time()
    execution_time = end_time - start_time
    print("loading model successfully with", execution_time, " seconds")


    # print('creating model engine')
    # deepspeed 初始化engine 优化器
    model_engine, optimizer, _, _ = deepspeed.initialize(   
                                                            args=cmd_args,
                                                            model=model,
                                                            model_parameters=model_parameters
                                                        )
    print_gpu_utilization()
    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        cmd_args.tokenizer_name_or_path,
        trust_remote_code=True,
        use_fast=False,
    )
    tokenizer.pad_token_id = 0

    # # Configure LoRA
    # lora_config = LoraConfig(
    #     task_type=TaskType.CAUSAL_LM,  # LoRA for causal language modeling
    #     inference_mode=False,          # Training mode
    #     r=16,                          # Rank of the LoRA updates
    #     lora_alpha=32,                 # Scaling factor
    #     lora_dropout=0.05,               # Dropout for LoRA layers
    #     target_modules=["down_proj", "gate_proj", "up_proj"]
    # )
    # model.enable_input_require_grads()
    # model = get_peft_model(model, lora_config)

    # load training data
    train_dataset = LegalAidDataset(
        data_path = cmd_args.train_path,        
        tokenizer = tokenizer,
        max_src = cmd_args.max_src,
        max_tar=cmd_args.max_tar
    )
    
    data_collator = LegalAidCollator(
        tokenizer=tokenizer,
        max_src = cmd_args.max_src,
        max_tar=cmd_args.max_tar
    )

    train_sampler = DistributedSampler(train_dataset)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=cmd_args.per_device_train_batch_size, shuffle=False, num_workers=cmd_args.workers, pin_memory=True, sampler=train_sampler,collate_fn=data_collator)

    # if cmd_args.tensorboard_dir != "data_dir":
    # writer = SummaryWriter(cmd_args.tensorboard_dir)


    model_engine.train()
    for i in range(cmd_args.num_train_epochs):
        step = 0
        for batch in tqdm(train_dataloader):
            step += 1

            input_ids = batch["input_ids"].to(model_engine.device)
            labels = batch["labels"].to(model_engine.device)
            attention_mask = batch["attention_mask"].to(model_engine.device)

            # forward
            loss = model_engine(
                input_ids=input_ids,
                labels=labels,
                attention_mask = attention_mask
            ).loss
            # if cmd_args.tensorboard_dir != "data_dir":
            # writer.add_scalar('Loss', loss, step)
            print(f"{loss=}")
            see_memory_usage(f"Before", force=True)
            print_gpu_utilization()
            model_engine.backward(loss)
            model_engine.step()
            see_memory_usage(f"Afer", force=True)
            print_gpu_utilization()


        torch.cuda.empty_cache()
        model_engine.save_checkpoint(cmd_args.output_dir)



if __name__ == "__main__":
    main()
