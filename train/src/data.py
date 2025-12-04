import random
from dataclasses import dataclass
from typing import List, Dict, Tuple
import json
import copy
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import DefaultDataCollator
from torch.nn.utils.rnn import pad_sequence

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = '[PAD]'
DEFAULT_EOS_TOKEN = '</s>'
DEFAULT_BOS_TOKEN = '<s>'
DEFAULT_UNK_TOKEN = '<unk>'


def find_sublist(sub, full):
    """Return start index of sub in full, or -1 if not found."""
    n, m = len(full), len(sub)
    if m == 0 or m > n:
        return -1
    for i in range(n - m + 1):
        if full[i:i + m] == sub:
            return i
    return -1


class CLM(Dataset):
    def __init__(self, data_path,tokenizer,max_tar,max_src):

        self.max_target = max_tar
        self.max_source = max_src
        self.tokenizer = tokenizer


        self.dataset = []
        with open(data_path, "r", encoding="utf-8") as fh:
            for i, line in tqdm(enumerate(fh)):
                sample = json.loads(line.strip())                
                self.dataset.append(
                   sample["content"]
                )
            

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:  
        example = self.dataset[idx]
        source_txt = f"{example}{self.tokenizer.eos_token}"
        tokenized_source = self.tokenizer(
            source_txt,
            max_length=self.max_source,
            truncation=True,
            add_special_tokens=False,
        )
        src_ids = tokenized_source['input_ids']

        input_ids = torch.tensor(src_ids)

        data_dict = {'input_ids': input_ids, 'labels': input_ids}
        return data_dict

class LegalAidDataset(Dataset):
    def __init__(self, data_path,tokenizer,max_tar,max_src):

        self.max_target = max_tar
        self.max_source = max_src
        self.tokenizer = tokenizer
        self.think_ids = tokenizer("</think>", add_special_tokens=False)["input_ids"]


        self.dataset = []
        with open(data_path, "r", encoding="utf-8") as fh:
            for i, line in tqdm(enumerate(fh)):
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
                    add_generation_prompt=True  # Set to True if you want to add a generation prompt
                )
                # print(train_input)   
                self.dataset.append(
                    {"input": train_input,"output": item["output"]})
            

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:  
        example = self.dataset[idx]
        source_txt = f"{self.tokenizer.bos_token}{example['input']}"
        tokenized_source = self.tokenizer(
            source_txt,
            max_length=self.max_source,
            truncation=True,
            add_special_tokens=False,
        )
        src_ids = tokenized_source['input_ids']


        # target_length = self.max_target - len(src_ids)

        # if target_length == 0:
        #     len_src_ids = int(len(src_ids)/2)
        #     src_ids = src_ids[0:len_src_ids]
        #     target_length = self.max_target - len(src_ids)
        
        target_txt = f"{example['output']}{self.tokenizer.eos_token}"
        tokenized_target = self.tokenizer(
            target_txt,
            max_length=self.max_source,
            truncation=True,
            add_special_tokens=False,
        )

        tgt_ids = tokenized_target['input_ids']


        # if len(src_ids) + len(tgt_ids) > self.max_source:

        #     aaa = len(src_ids)
        #     bbb = len(tgt_ids)
        #     ccc = self.max_source

        #     len_src = aaa * (ccc/(aaa + bbb))
        #     len_src = int(len_src)

        #     len_tgt = bbb * (ccc/(aaa + bbb))
        #     len_tgt = int(len_tgt)

        #     # src_ids = src_ids[0:len_src]
        #     # tgt_ids = tgt_ids[0:len_tgt-1] + [tgt_ids[-1]]
            
        #     # 改成从尾部截取指定长度，因为重要的信息都在尾部
        #     src_ids = src_ids[-len_src:]
        #     tgt_ids = tgt_ids[-len_tgt:]

        if len(src_ids) + len(tgt_ids) > self.max_source: # 改成保留input不截断，如果截就从output的开头截断从而保留output的后半部分
            aaa = len(src_ids)
            bbb = len(tgt_ids)
            ccc = self.max_source

            # 计算目标长度
            len_src = aaa
            len_tgt = ccc - len_src if ccc > len_src else 0

            if len_src + len_tgt > ccc:
                len_tgt = ccc - len_src
            
            # 保留 src_ids，不截断
            src_ids = src_ids
            
            # 截断 tgt_ids 的前面部分
            tgt_ids = tgt_ids[-len_tgt:]

        # 构造标签：确保与截断后的 tgt_ids 同步长度
        tgt_label = copy.deepcopy(tgt_ids)
        # 如果包含思维链，训练时忽略 </think> 之前的部分
        think_pos = find_sublist(self.think_ids, tgt_ids)
        if think_pos != -1:
            mask_len = think_pos + len(self.think_ids)
            tgt_label[:mask_len] = [-100] * mask_len

        # 其他处理

        input_ids = torch.tensor(src_ids + tgt_ids)

        labels = torch.tensor(
            [-100 for _ in range(len(src_ids))] + tgt_label)
        data_dict = {'input_ids': input_ids, 'labels': labels}
        return data_dict
    
class LegalAidCollator(DefaultDataCollator):
    def __init__(self, tokenizer,max_src,max_tar):
        self.max_tar = max_src
        self.max_src = max_tar
        self.tokenizer = tokenizer

    def __post_init__(self):
        super().__post_init__()
        self.rng = random.Random()

    def __call__(self, instances: List[Dict[str,
                                 torch.Tensor]]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances]
                                  for key in ('input_ids', 'labels'))
        # print('2')

        # Pad sequences to be of equal length
        # print(f'-----------------------{self.tokenizer.pad_token_id}------------------------')
        input_ids = pad_sequence(input_ids,
                                 batch_first=True,
                                 padding_value=self.tokenizer.pad_token_id)
        labels = pad_sequence(
            labels, batch_first=True, padding_value=-100  # ignore padding in loss
        ) 

        # Construct attention mask based on padded input IDs
        attention_mask = input_ids.ne(self.tokenizer.pad_token_id)

        # Return collated batch as dictionary
        data_dict = {'input_ids': input_ids, 'attention_mask': attention_mask}
        if labels is not None:
            data_dict['labels'] = labels

        return data_dict
