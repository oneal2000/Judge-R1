# Copyright 2021 Reranker Author. All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional

import torch
import torch.functional as F
import copy
from transformers import AutoModelForSequenceClassification, AutoTokenizer,\
    PreTrainedModel, PreTrainedTokenizer
from transformers.modeling_outputs import SequenceClassifierOutput, BaseModelOutputWithPooling
from torch import nn
import torch.distributed as dist

from .arguments import ModelArguments, DataArguments, \
    RerankerTrainingArguments as TrainingArguments
import logging

logger = logging.getLogger(__name__)


class Reranker(nn.Module):
    def __init__(self, hf_model: PreTrainedModel, model_args: ModelArguments, data_args: DataArguments,
                 train_args: TrainingArguments):
        super().__init__()
        self.hf_model = hf_model
        self.model_args = model_args
        self.train_args = train_args
        self.data_args = data_args

        self.cross_entropy = nn.CrossEntropyLoss(reduction='mean')

        self.register_buffer(
            'target_label',
            torch.zeros(self.train_args.per_device_train_batch_size, dtype=torch.long)
        )

        # 修复：只在分布式已初始化时才获取 world_size
        if train_args.local_rank >= 0 and dist.is_initialized():
            self.world_size = dist.get_world_size()
        else:
            self.world_size = 1  # 单GPU训练时 world_size = 1

        # 修复：添加 Transformers Trainer 需要的属性
        self._keys_to_ignore_on_save = None
        # 为了兼容性，添加 config 属性（指向 hf_model 的 config）
        if hasattr(hf_model, 'config'):
            self.config = hf_model.config

    def forward(self, batch):
        ranker_out: SequenceClassifierOutput = self.hf_model(**batch, return_dict=True)
        logits = ranker_out.logits

        if self.model_args.temperature is not None:
            assert self.model_args.temperature > 0
            logits = logits / self.model_args.temperature

        if self.train_args.collaborative:
            logits = self.dist_gather_tensor(logits)
            logits = logits.view(
                self.world_size,
                self.train_args.per_device_train_batch_size,
                -1  # chunk
            )
            logits = logits.transpose(0, 1).contiguous()

        if self.training:
            scores = logits.view(
                self.train_args.per_device_train_batch_size,
                self.data_args.train_group_size
            )
            loss = self.cross_entropy(scores, self.target_label)

            return SequenceClassifierOutput(
                loss=loss,
                **ranker_out,
            )
        else:
            return ranker_out

    @classmethod
    def from_pretrained(
            cls, model_args: ModelArguments, data_args: DataArguments, train_args: TrainingArguments,
            *args, **kwargs
    ):
        hf_model = AutoModelForSequenceClassification.from_pretrained(*args, **kwargs)
        reranker = cls(hf_model, model_args, data_args, train_args)
        return reranker

    def save_pretrained(self, output_dir: str):
        self.hf_model.save_pretrained(output_dir)

    def dist_gather_tensor(self, t: Optional[torch.Tensor]):
        if t is None:
            return None

        # 修复：只在分布式已初始化时才进行 gather
        if not dist.is_initialized():
            return t

        all_tensors = [torch.empty_like(t) for _ in range(self.world_size)]
        dist.all_gather(all_tensors, t)
        all_tensors[self.train_args.local_rank] = t
        all_tensors = torch.cat(all_tensors, dim=0)

        return all_tensors


class RerankerDC(Reranker):
    def compute_grad(self, scores: torch.Tensor):
        scores = scores.view(
            self.train_args.per_device_train_batch_size,
            self.data_args.train_group_size
        ).detach().requires_grad_()
        loss = self.cross_entropy(scores, self.target_label)
        loss.backward()

        return loss.detach(), scores.grad

    def forward(self, batch, grad_tensor: torch.Tensor = None):
        ranker_out: SequenceClassifierOutput = self.hf_model(**batch, return_dict=True)
        logits = ranker_out.logits

        if self.training:
            if grad_tensor is not None:
                return torch.dot(logits.float().flatten(), grad_tensor.flatten())
            else:
                return logits

        else:
            return ranker_out


class RerankerForInference(nn.Module):
    def __init__(
            self,
            hf_model: Optional[PreTrainedModel] = None,
            tokenizer: Optional[PreTrainedTokenizer] = None
    ):
        super().__init__()
        self.hf_model = hf_model
        self.tokenizer = tokenizer

    def tokenize(self, *args, **kwargs):
        return self.tokenizer(*args, **kwargs)

    def forward(self, batch):
        return self.hf_model(**batch)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str):
        hf_model = AutoModelForSequenceClassification.from_pretrained(
            pretrained_model_name_or_path)
        hf_tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path)

        hf_model.eval()
        return cls(hf_model, hf_tokenizer)

    def load_pretrained_model(self, pretrained_model_name_or_path, *model_args, **kwargs):
        self.hf_model = AutoModelForSequenceClassification.from_pretrained(
            pretrained_model_name_or_path, *model_args, **kwargs
        )

    def load_pretrained_tokenizer(self, pretrained_model_name_or_path, *inputs, **kwargs):
        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path, *inputs, **kwargs
        )