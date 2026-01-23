# Copyright 2021 Reranker Author. All rights reserved.
# Code structure inspired by HuggingFace run_glue.py in the transformers library.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
import numpy as np

from reranker import Reranker, RerankerDC
from reranker import RerankerTrainer, RerankerDCTrainer
from reranker.data import GroupedTrainDataset, PredictionDataset, GroupCollator
from reranker.arguments import ModelArguments, DataArguments, \
    RerankerTrainingArguments as TrainingArguments

from transformers import AutoConfig, AutoTokenizer
from transformers import (
    HfArgumentParser,
    set_seed,
)

logger = logging.getLogger(__name__)


def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
     # Parse known and unknown args separately
    model_args, data_args, training_args, unknown_args = parser.parse_args_into_dataclasses(return_remaining_strings=True)
    
    # You might want to manually handle the `--local-rank` argument here if needed
    if '--local-rank' in unknown_args:
        unknown_args.remove('--local-rank')
    
    # Or, you can print/log unknown arguments for further inspection
    if unknown_args:
        print(f"Warning: Some arguments were not recognized: {unknown_args}")

    if (
            os.path.exists(training_args.output_dir)
            and os.listdir(training_args.output_dir)
            and training_args.do_train
            and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
        )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )
    logger.info("Training/evaluation parameters %s", training_args)
    logger.info("Model parameters %s", model_args)
    logger.info("Data parameters %s", data_args)

    # Set seed
    set_seed(training_args.seed)

    num_labels = 1

    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=num_labels,
        cache_dir=model_args.cache_dir,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=False,
    )

    _model_class = RerankerDC if training_args.distance_cache else Reranker

    model = _model_class.from_pretrained(
        model_args, data_args, training_args,
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
    )

    # Get datasets
    if training_args.do_train:
        train_dataset = GroupedTrainDataset(
            data_args, data_args.train_path, tokenizer=tokenizer, train_args=training_args
        )
    else:
        train_dataset = None


    # Initialize our Trainer
    _trainer_class = RerankerDCTrainer if training_args.distance_cache else RerankerTrainer
    trainer = _trainer_class(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=GroupCollator(tokenizer),
    )

    # Training
    if training_args.do_train:
        # 修复：预训练模型路径不应该作为 checkpoint 恢复
        # 只在 training_args.resume_from_checkpoint 明确指定时才恢复训练
        # 或者检查是否是真正的 checkpoint 目录（包含 trainer_state.json）
        resume_from_checkpoint = None
        if training_args.resume_from_checkpoint:
            # 如果明确指定了 resume_from_checkpoint，使用它
            resume_from_checkpoint = training_args.resume_from_checkpoint
        elif os.path.isdir(model_args.model_name_or_path):
            # 检查是否是 checkpoint 目录（包含 trainer_state.json）
            trainer_state_path = os.path.join(model_args.model_name_or_path, "trainer_state.json")
            if os.path.exists(trainer_state_path):
                # 这是真正的 checkpoint 目录
                resume_from_checkpoint = model_args.model_name_or_path
                logger.info(f"Found checkpoint at {resume_from_checkpoint}, resuming training...")
            # 否则是预训练模型目录，不恢复训练
        
        trainer.train(resume_from_checkpoint=resume_from_checkpoint)
        trainer.save_model()
        # For convenience, we also re-save the tokenizer to the same directory,
        # so that you can share your model easily on huggingface.co/models =)
        if trainer.is_world_process_zero():
            tokenizer.save_pretrained(training_args.output_dir)

    if training_args.do_eval:
        trainer.evaluate()

    if training_args.do_predict:
        logging.info("*** Prediction ***")

        # 修复：如果 score 文件已存在，删除它（允许覆盖）
        if os.path.exists(data_args.rank_score_path):
            if os.path.isfile(data_args.rank_score_path):
                logger.info(f'Score file {data_args.rank_score_path} already exists, will be overwritten')
                os.remove(data_args.rank_score_path)
            else:
                raise ValueError(f'Should specify a file name')
        else:
            score_dir = os.path.split(data_args.rank_score_path)[0]
            if not os.path.exists(score_dir):
                logger.info(f'Creating score directory {score_dir}')
                os.makedirs(score_dir)

        test_dataset = PredictionDataset(
            data_args.pred_path, tokenizer=tokenizer,
            max_len=data_args.max_len,
        )
        assert data_args.pred_id_file is not None

        pred_qids = []
        pred_pids = []
        with open(data_args.pred_id_file, 'r', encoding='utf-8') as f:
            for l in f:
                # 兼容 tab 和空格分隔
                parts = l.strip().split('\t') if '\t' in l else l.strip().split()
                if len(parts) >= 2:
                    q, p = parts[0], parts[1]
                    pred_qids.append(q)
                    pred_pids.append(p)

        pred_scores = trainer.predict(test_dataset=test_dataset).predictions
        print(len(pred_qids), len(pred_scores))
        if trainer.is_world_process_zero():
            assert len(pred_qids) == len(pred_scores)
            
            # 修复：确保 pred_scores 是展平的 numpy 数组，并转换为标量
            pred_scores = np.array(pred_scores).flatten()
            
            with open(data_args.rank_score_path, "w") as writer:
                for qid, pid, score in zip(pred_qids, pred_pids, pred_scores):
                    # 确保 score 是标量值
                    score_value = float(score.item() if hasattr(score, 'item') else score)
                    writer.write(f'{qid}\t{pid}\t{score_value}\n')


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()