# Copyright 2021 Reranker Author. All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""
将检索结果转换为 Reranker 推理格式
"""
import warnings
import os

# 抑制 transformers 警告
warnings.filterwarnings("ignore", message=".*overflowing tokens.*")
warnings.filterwarnings("ignore", message=".*not returned for the setting.*")
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

from transformers import AutoTokenizer
from argparse import ArgumentParser
from tqdm import tqdm
import json
from collections import defaultdict

parser = ArgumentParser()

parser.add_argument('--save_to', required=True)
parser.add_argument('--tokenizer', required=True)
parser.add_argument('--generate_id_to')
parser.add_argument('--truncate', type=int, default=512)
parser.add_argument('--q_truncate', type=int, default=512)  # 增大查询截断长度
parser.add_argument('--qry_file', required=True, help='Query file (can be train or test)')
parser.add_argument('--law_data_file', required=True)
parser.add_argument('--run_file_train', required=True)
args = parser.parse_args()

tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)

# 加载查询数据和法条数据
law_data, qry_data = [], []
with open(args.qry_file, 'r', encoding='utf-8') as f1, \
     open(args.law_data_file, 'r', encoding='utf-8') as f2:
    for l1 in f1:
        tmp = json.loads(l1.strip())
        qry_data.append(tmp)
    for l2 in f2:
        tmp = json.loads(l2.strip())
        law_data.append(tmp)

did_2_body = {int(x['text_id']): x['text'] for x in law_data}
qid_2_body = {str(x['text_id']): x['text'] for x in qry_data}

# 加载检索结果
rankings = defaultdict(list)
with open(args.run_file_train, 'r', encoding='utf-8') as f:
    for l in f:
        # 兼容 tab 和空格分隔
        parts = l.strip().split('\t') if '\t' in l else l.strip().split()
        if len(parts) < 3:
            continue
        qid = str(parts[0])
        pid = int(parts[2])
        rankings[qid].append(pid)

print(f"Loaded {len(rankings)} queries from runfile")

with open(args.save_to, 'w', encoding='utf-8') as jfile:
    all_ids = []

    for data_item in tqdm(qry_data):
        qry_id = str(data_item['text_id'])
        
        # 检查是否有该查询的检索结果
        if qry_id not in rankings:
            continue
        
        # 检查是否有该查询的文本
        if qry_id not in qid_2_body:
            continue
        
        q_body = qid_2_body[qry_id]
        pids = rankings[qry_id]
        
        for pid in pids:
            # 检查法条是否存在
            if pid not in did_2_body:
                continue
            
            all_ids.append((qry_id, pid))
            
            p_body = did_2_body[pid]
            qry_encoded = tokenizer.encode(
                q_body,
                truncation=True,
                max_length=args.q_truncate,
                add_special_tokens=False,
                padding=False,
            )
            doc_encoded = tokenizer.encode(
                p_body,
                truncation=True,
                max_length=args.truncate,
                add_special_tokens=False,
                padding=False
            )
            entry = {
                'qid': qry_id,  # 修复：使用 qry_id 而不是 qid
                'pid': pid,
                'qry': qry_encoded,
                'psg': doc_encoded,
            }
            jfile.write(json.dumps(entry) + '\n')

    print(f"Generated {len(all_ids)} query-passage pairs")

    if args.generate_id_to is not None:
        with open(args.generate_id_to, 'w', encoding='utf-8') as id_file:
            for qry_id, doc_id in all_ids:
                id_file.write(f'{qry_id}\t{doc_id}\n')
        print(f"Saved IDs to {args.generate_id_to}")

print("DONE!")