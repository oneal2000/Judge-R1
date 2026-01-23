# Copyright 2021 Reranker Author. All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""
从检索结果构建 Reranker 训练数据
- 正例：真实相关的法条
- 负例：检索到但实际不相关的法条
"""
import warnings
import os

warnings.filterwarnings("ignore", message=".*overflowing tokens.*")
warnings.filterwarnings("ignore", message=".*not returned for the setting.*")
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

from argparse import ArgumentParser
from transformers import AutoTokenizer
import json
from collections import defaultdict
import random
from tqdm import tqdm


parser = ArgumentParser()
parser.add_argument('--tokenizer_name', required=True)

parser.add_argument('--truncate', type=int, default=200)
parser.add_argument('--q_truncate', type=int, default=512)
parser.add_argument('--sample_from_top', type=int, default=100)
parser.add_argument('--n_sample', type=int, default=10)
parser.add_argument('--random', action='store_true')

parser.add_argument('--run_file_train', required=True)
parser.add_argument('--output_train_file', required=True)
parser.add_argument('--qry_train_file', required=True)
parser.add_argument('--law_data_file', required=True)
args = parser.parse_args()


def read_qrel():
    """将query和相关的doc存储成dict的形式，形式为 qid: [docid1, docid2, ...]"""
    qrel = {}
    with open(args.qry_train_file, 'r', encoding='utf-8') as f:
        for line in f:
            content = json.loads(line.strip())
            qid = str(content['text_id'])  # 统一为字符串
            qrel[qid] = [int(x) for x in content['la']]  # 法条ID为整数
    return qrel


qrel = read_qrel()
rankings = defaultdict(list)
no_judge = set()

# 读取检索结果
with open(args.run_file_train, 'r', encoding='utf-8') as f:
    for l in f:
        # 兼容 tab 和空格分隔
        parts = l.strip().split('\t') if '\t' in l else l.strip().split()
        if len(parts) < 3:
            continue
        qid = str(parts[0])
        pid = int(parts[2])
        
        if qid not in qrel:
            no_judge.add(qid)
            continue
        
        # 只添加不相关的法条作为负例
        if pid not in qrel[qid]:
            rankings[qid].append(pid)

print(f'{len(no_judge)} queries not judged and skipped', flush=True)
print(f'{len(rankings)} queries with negative samples', flush=True)

# 加载数据
law_data, qry_train_data = [], []
with open(args.qry_train_file, 'r', encoding='utf-8') as f1, \
     open(args.law_data_file, 'r', encoding='utf-8') as f2:
    for l1 in f1:
        tmp = json.loads(l1.strip())
        qry_train_data.append(tmp)
    for l2 in f2:
        tmp = json.loads(l2.strip())
        law_data.append(tmp)

did_2_body = {int(x['text_id']): x['text'] for x in law_data}
qid_2_body = {str(x['text_id']): x['text'] for x in qry_train_data}

tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, use_fast=True)

queries = list(rankings.keys())
print(f'Processing {len(queries)} queries...', flush=True)

with open(args.output_train_file, 'w', encoding='utf-8') as f:
    for qid in tqdm(queries):
        # 跳过没有正例的 query
        if qid not in qrel or not qrel[qid]:
            continue
        
        # 选取负例
        negs = rankings[qid][:args.sample_from_top]
        if args.random:
            random.shuffle(negs)
        negs = negs[:args.n_sample]
        
        # 如果没有负例，跳过
        if not negs:
            continue
        
        # 编码负例
        neg_encoded = []
        for neg in negs:
            if neg not in did_2_body:
                continue
            body = did_2_body[neg]
            encoded_neg = tokenizer.encode(
                body,
                add_special_tokens=False,
                max_length=args.truncate,
                truncation=True
            )
            neg_encoded.append({
                'passage': encoded_neg,
                'pid': str(neg),
            })
        
        # 如果没有有效负例，跳过
        if not neg_encoded:
            continue
        
        # 编码正例
        pos_encoded = []
        for pos in qrel[qid]:
            if pos not in did_2_body:
                continue
            body = did_2_body[pos]
            encoded_pos = tokenizer.encode(
                body,
                add_special_tokens=False,
                max_length=args.truncate,
                truncation=True
            )
            pos_encoded.append({
                'passage': encoded_pos,
                'pid': str(pos),
            })
        
        # 如果没有有效正例，跳过
        if not pos_encoded:
            continue
        
        # 编码查询
        if qid not in qid_2_body:
            continue
        q_body = qid_2_body[qid]
        query_dict = {
            'qid': qid,
            'query': tokenizer.encode(
                q_body,
                add_special_tokens=False,
                max_length=args.q_truncate,
                truncation=True),
        }
        
        item_set = {
            'qry': query_dict,
            'pos': pos_encoded,
            'neg': neg_encoded,
        }
        f.write(json.dumps(item_set) + '\n')

print("DONE!")
