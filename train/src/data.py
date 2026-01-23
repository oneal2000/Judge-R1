import random
from typing import List, Dict
import json
import copy
import torch
from torch.utils.data import Dataset
from transformers import DefaultDataCollator
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

IGNORE_INDEX = -100

class LegalAidDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_tar, max_src):
        self.max_target = max_tar
        self.max_source = max_src
        self.tokenizer = tokenizer
        self.dataset = []
        
        with open(data_path, "r", encoding="utf-8") as fh:
            for line in tqdm(fh):
                item = json.loads(line.strip())
                if not item:
                    continue
                
                input_content = f"""
任务背景: 根据以下提供的案件事实，生成一份完整的刑法判决书。判决书需包括案件事实、法律分析、裁判理由以及最终裁判结论。
本案件事实：{item['input']}
本案件的完整判决书为：
"""
                messages = [
                    {"role": "system", "content": "你是一个法律助理，提供帮助。"},
                    {"role": "user", "content": input_content}
                ]
                
                train_input = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
                self.dataset.append({"input": train_input, "output": item["output"]})

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        example = self.dataset[idx]
        
        # 1. Source 处理（添加 BOS token）
        source_txt = f"{self.tokenizer.bos_token}{example['input']}" if self.tokenizer.bos_token else example['input']
        tokenized_source = self.tokenizer(
            source_txt,
            max_length=self.max_source,
            truncation=True,  # 先按max_source截断
            add_special_tokens=False,
        )
        src_ids = tokenized_source['input_ids']

        # 2. Target 处理
        target_txt = f"{example['output']}{self.tokenizer.eos_token}"
        tokenized_target = self.tokenizer(
            target_txt,
            max_length=self.max_source,  # 先不限制长度
            truncation=True,
            add_special_tokens=False,
        )
        tgt_ids = tokenized_target['input_ids']

        # 3. 截断策略（与 JuDGE 一致）
        # 优先保留完整的 input，只截断 output 的前面部分（保留后半部分）
        total_max_len = self.max_source
        
        if len(src_ids) + len(tgt_ids) > total_max_len:
            # 计算目标长度：优先保留 src_ids 完整
            len_src = len(src_ids)
            len_tgt = total_max_len - len_src
            
            if len_tgt <= 0:
                # 如果 src 已经超过 max_len，需要截断 src
                len_src = total_max_len // 2
                len_tgt = total_max_len - len_src
                src_ids = src_ids[-len_src:]  # 从尾部保留
            
            # 截断 tgt_ids 的前面部分，保留后半部分（判决结论更重要）
            tgt_ids = tgt_ids[-len_tgt:]

        # 4. 构造 input_ids 和 labels
        input_ids = torch.tensor(src_ids + tgt_ids, dtype=torch.long)
        labels = torch.tensor(
            [IGNORE_INDEX] * len(src_ids) + copy.deepcopy(tgt_ids),
            dtype=torch.long
        )

        return {'input_ids': input_ids, 'labels': labels}


class LegalAidCollator(DefaultDataCollator):
    def __init__(self, tokenizer, max_src, max_tar):
        self.tokenizer = tokenizer

    def __call__(self, instances: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        input_ids_list = [ins['input_ids'] for ins in instances]
        labels_list = [ins['labels'] for ins in instances]

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        input_ids = pad_sequence(
            input_ids_list, 
            batch_first=True, 
            padding_value=self.tokenizer.pad_token_id
        )
        # 注意：labels 使用 IGNORE_INDEX 作为 padding，而不是 pad_token_id
        labels = pad_sequence(
            labels_list, 
            batch_first=True, 
            padding_value=IGNORE_INDEX
        )
        attention_mask = input_ids.ne(self.tokenizer.pad_token_id)

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }