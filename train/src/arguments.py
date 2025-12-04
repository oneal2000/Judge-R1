#!/usr/bin/python
# -*- encoding: utf-8 -*-

from dataclasses import dataclass, field
from typing import Optional, Union
import os
from transformers import TrainingArguments

@dataclass
class DataTrainingArguments: # 与数据处理和预处理相关的参数
    """
    Arguments control input data path, mask behaviors
    """
    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field( # 数据集的配置名称
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    train_dir: str = field( # 训练数据目录路径
        default=None, metadata={"help": "Path to train directory"}
    )
    train_path: str = field( # 训练数据文件路径
        default=None, metadata={"help": "Path to train data"}
    )
    train_file: Optional[str] = field( # 训练文件（一个txt）
        default=None, metadata={"help": "The input training data file (a text file)."}
    )
    overwrite_cache: bool = field( # 是否覆盖缓存的训练和评估数据集
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    preprocessing_num_workers: Optional[int] = field( # 用于数据预处理的进程数，增加该值可以加速数据加载和预处理。
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": "Whether to pad all samples to `max_seq_length`. "
                    "If False, will pad the samples dynamically when batching to the maximum length in the batch."
        },
    )
    ignore_pad_token_for_loss: bool = field( # 在损失计算中是否忽略填充的标签部分。通常用于忽略序列末尾的填充部分。
        default=True,
        metadata={
            "help": "Whether to ignore the tokens corresponding to padded labels in the loss computation or not."
        },
    )
    source_prefix: Optional[str] = field( # 在每个源文本前添加的前缀，通常用于T5模型等多任务模型中。
        default="", metadata={"help": "A prefix to add before every source text (useful for T5 models)."}
    )
    forced_bos_token: Optional[str] = field( # 强制生成的第一个token为指定的token。对于多语言模型（如mBART），可以指定目标语言的token。
        default=None,
        metadata={
            "help": (
                "The token to force as the first generated token after the decoder_start_token_id."
                "Useful for multilingual models like mBART where the first generated token"
                "needs to be the target language token (Usually it is the target language token)"
            )
        },
    )




@dataclass
class ModelArguments: # 模型配置、解码器头部配置等
    """
    Arguments control model config, decoder head config
    """

    model_name_or_path: Optional[str] = field( # 预训练模型的名称或路径
        default=None,
        metadata={
            "help": "The model checkpoint for weights initialization."
                    "Don't set if you want to train a model from scratch."
        },
    )
    model_type: Optional[str] = field( # 模型的类型（如bert、gpt-2等）
        default=None,
    )
    config_name: Optional[str] = field( # 如果配置文件不同于模型名称，可以指定预训练配置名称或路径。
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field( # 如果分词器文件不同于模型名称，可以指定预训练分词器名称或路径。
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field( # 用于存储从huggingface.co下载的预训练模型的缓存目录。
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field( # 是否使用tokenizers库支持的快速分词器。                 
        default=False,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    use_auth_token: bool = field( # 是否使用huggingface-cli login生成的认证token（必要时使用私有模型）。
        default=False,
        metadata={
            "help": (
                "Will use the token generated when running `huggingface-cli login` (necessary to use this script "
                "with private models)."
            )
        },
    )
    resize_position_embeddings: Optional[bool] = field( # 如果max_source_length超过模型的位置嵌入的大小，是否自动调整嵌入的大小。
        default=None,
        metadata={
            "help": (
                "Whether to automatically resize the position embeddings if `max_source_length` exceeds "
                "the model's position embeddings."
            )
        },
    )
    quantization_bit: Optional[int] = field( # 用于模型量化的位数。
        default=None
    )
    pre_seq_len: Optional[int] = field( # 前缀序列的长度，用于特定的预训练模型。
        default=None
    )
    prefix_projection: bool = field( # 是否使用前缀投影（可能用于P-tuning或前缀调优方法）。
        default=False
    )
    ptuning_checkpoint: str = field( # P-tuning V2检查点路径，用于微调预训练模型。
        default=None, metadata={"help": "Path to p-tuning v2 checkpoints"}
    )
    ds_config: str = field( # DeepSpeed的配置文件路径，用于控制DeepSpeed相关的设置。
        default=None, metadata={"help": "s"}
    )




@dataclass
class FinetuneArguments(TrainingArguments): # 设置与微调过程相关的参数
    lora_rank: int = field( # 用于LoRA（低秩适应）的秩数。
        default=8
    )
    max_len: int = field( # 输入数据的最大长度
        default=700
    )
    max_input_len: int = field( # 模型输入的最大长度
        default=350
    )
    warmup_ratio: float = field( # 热身阶段的学习率比例，用于控制学习率从零到峰值的增长时间
        default=0.1
    )
    remove_unused_columns: bool = field( # 是否在训练过程中移除未使用的列，通常用于加速训练
        default=False
    )
    local_rank: int = field( # 用于分布式训练的local_rank，控制当前进程的分布式训练索引
        default=-1, metadata={"help": "For distributed training: local_rank"}
    )




