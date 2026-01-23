# Copyright 2021 Reranker Author. All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import random
from dataclasses import dataclass
import warnings

import datasets
from typing import Union, List, Tuple, Dict

import torch
from torch.utils.data import Dataset

from .arguments import DataArguments, RerankerTrainingArguments
from transformers import PreTrainedTokenizer, BatchEncoding
from transformers import DataCollatorWithPadding

# 抑制 tokenizer 溢出 tokens 警告（作为备用方案）
warnings.filterwarnings("ignore", message=".*overflowing tokens.*")


class GroupedTrainDataset(Dataset):
    query_columns = ['qid', 'query']
    document_columns = ['pid', 'passage']

    def __init__(
            self,
            args: DataArguments,
            path_to_tsv: Union[List[str], str],
            tokenizer: PreTrainedTokenizer,
            train_args: RerankerTrainingArguments = None,
    ):
        # 修复：移除 ignore_verifications 参数（新版本 datasets 库不支持）
        self.nlp_dataset = datasets.load_dataset(
            'json',
            data_files=path_to_tsv,
            features=datasets.Features({
                'qry': {
                    'qid': datasets.Value('string'),
                    'query': [datasets.Value('int32')],
                },
                'pos': [{
                    'pid': datasets.Value('string'),
                    'passage': [datasets.Value('int32')],
                }],
                'neg': [{
                    'pid': datasets.Value('string'),
                    'passage': [datasets.Value('int32')],
                }]}
            )
        )['train']

        self.tok = tokenizer
        self.SEP = [self.tok.sep_token_id]
        self.args = args
        self.total_len = len(self.nlp_dataset)
        self.train_args = train_args

        # 修复：检查分布式是否已初始化
        if train_args is not None and train_args.collaborative:
            import torch.distributed as dist
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            if not dist.is_initialized():
                raise RuntimeError("Distributed training requires process group to be initialized")
            self.world_size = dist.get_world_size()
            self.rank = dist.get_rank()
            chunk_size = int(self.args.train_group_size / self.world_size)
            self.chunk_start = self.rank * chunk_size
            self.chunk_end = self.chunk_start + chunk_size

    def create_one_example(self, qry_encoding: List[int], doc_encoding: List[int]):
        """
        创建训练样本，智能截断确保总长度不超过 max_len
        这样可以避免 transformers 的警告，从根源解决问题
        """
        # 特殊 tokens: [CLS] query [SEP] passage [SEP] = 3个
        special_tokens_count = 3
        max_content_len = self.args.max_len - special_tokens_count
        
        # 计算总长度
        total_len = len(qry_encoding) + len(doc_encoding)
        
        # 如果超过限制，智能截断
        if total_len > max_content_len:
            # 优先保留 query，但不超过 70% 的空间
            max_qry_len = int(max_content_len * 0.7)
            min_qry_len = min(64, len(qry_encoding))  # query 至少保留 64 tokens
            
            # 确定 query 长度
            qry_len = min(max_qry_len, len(qry_encoding))
            # 剩余空间给 passage
            doc_len = max_content_len - qry_len
            
            # 如果 passage 太短，调整 query 长度
            if doc_len < 32:  # passage 至少需要 32 tokens
                doc_len = 32
                qry_len = max_content_len - doc_len
                # 确保 query 不会太短
                if qry_len < min_qry_len:
                    qry_len = min_qry_len
                    doc_len = max_content_len - qry_len
            
            qry_encoding = qry_encoding[:qry_len]
            doc_encoding = doc_encoding[:doc_len]
        
        # 现在总长度肯定不超过限制，不需要自动截断
        item = self.tok.encode_plus(
            qry_encoding,
            doc_encoding,
            truncation=False,  # 关键：已经手动控制长度，不需要自动截断
            max_length=self.args.max_len,
            padding=False,
        )
        return item

    def __len__(self):
        return self.total_len

    def __getitem__(self, item) -> List[BatchEncoding]:
        group = self.nlp_dataset[item]
        examples = []
        group_batch = []
        _, qry = (group['qry'][k] for k in self.query_columns)
        _, pos_psg = [
            random.choice(group['pos'])[k] for k in self.document_columns]
        examples.append((qry, pos_psg))

        if len(group['neg']) < self.args.train_group_size - 1:
            negs = random.choices(group['neg'], k=self.args.train_group_size - 1)
        else:
            negs = random.sample(group['neg'], k=self.args.train_group_size - 1)

        for neg_entry in negs:
            _, neg_psg = [neg_entry[k] for k in self.document_columns]
            examples.append((qry, neg_psg))

        # collaborative mode, split the group
        if self.train_args is not None and self.train_args.collaborative:
            examples = examples[self.chunk_start: self.chunk_end]

        for e in examples:
            group_batch.append(self.create_one_example(*e))
        return group_batch


class PredictionDataset(Dataset):
    columns = [
        'qid', 'pid', 'qry', 'psg'
    ]

    def __init__(self, path_to_json: List[str], tokenizer: PreTrainedTokenizer, max_len=128):
        self.nlp_dataset = datasets.load_dataset(
            'json',
            data_files=path_to_json,
        )['train']
        self.tok = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.nlp_dataset)

    def __getitem__(self, item):
        qid, pid, qry, psg = (self.nlp_dataset[item][f] for f in self.columns)
        
        # 智能截断，确保总长度不超过 max_len
        special_tokens_count = 3
        max_content_len = self.max_len - special_tokens_count
        total_len = len(qry) + len(psg)
        
        if total_len > max_content_len:
            # 优先保留 query，但不超过 70% 的空间
            max_qry_len = int(max_content_len * 0.7)
            min_qry_len = min(64, len(qry))  # query 至少保留 64 tokens
            
            # 确定 query 长度
            qry_len = min(max_qry_len, len(qry))
            # 剩余空间给 passage
            doc_len = max_content_len - qry_len
            
            # 如果 passage 太短，调整 query 长度
            if doc_len < 32:  # passage 至少需要 32 tokens
                doc_len = 32
                qry_len = max_content_len - doc_len
                # 确保 query 不会太短
                if qry_len < min_qry_len:
                    qry_len = min_qry_len
                    doc_len = max_content_len - qry_len
            
            qry = qry[:qry_len]
            psg = psg[:doc_len]
        
        # 现在总长度肯定不超过限制，不需要自动截断
        return self.tok.encode_plus(
            qry,
            psg,
            truncation=False,  # 关键：已经手动控制长度
            max_length=self.max_len,
            padding=False,
        )


@dataclass
class GroupCollator(DataCollatorWithPadding):
    """
    Wrapper that does conversion from List[Tuple[encode_qry, encode_psg]] to List[qry], List[psg]
    and pass batch separately to the actual collator.
    Abstract out data detail for the model.
    """

    def __call__(
            self, features
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        if isinstance(features[0], list):
            features = sum(features, [])
        return super().__call__(features)