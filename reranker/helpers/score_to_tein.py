# Copyright 2021 Reranker Author. All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""
将 Reranker 评分结果转换为 TREC 格式
"""
import argparse
from collections import defaultdict
import re

parser = argparse.ArgumentParser()
parser.add_argument('--score_file', required=True)
parser.add_argument('--reranker_run_file', required=True)
parser.add_argument('--part', required=True)
args = parser.parse_args()

run_id = f"lr_reranker_{args.part}"
with open(args.score_file, 'r', encoding='utf-8') as f:
    lines = f.readlines()

all_scores = defaultdict(dict)

for line in lines:
    if len(line.strip()) == 0:
        continue
    # 兼容 tab 和空格分隔
    parts = line.strip().split('\t') if '\t' in line else line.strip().split()
    if len(parts) < 3:
        continue
    qid, did, score = parts[0], parts[1], parts[2]
    
    # 修复：处理可能的数组格式字符串，如 '[1.09375]'
    # 如果 score 是数组格式，提取其中的数值
    if score.startswith('[') and score.endswith(']'):
        # 提取方括号中的内容
        score = score[1:-1]
    
    # 尝试转换为 float
    try:
        score = float(score)
    except ValueError:
        # 如果仍然无法转换，尝试提取数字
        numbers = re.findall(r'-?\d+\.?\d*', score)
        if numbers:
            score = float(numbers[0])
        else:
            print(f"Warning: Could not parse score '{score}' for query {qid}, doc {did}, skipping...")
            continue
    
    all_scores[qid][did] = score

print(f"Loaded scores for {len(all_scores)} queries")

qq = list(all_scores.keys())

with open(args.reranker_run_file, 'w', encoding='utf-8') as f:
    for qid in qq:
        score_list = sorted(list(all_scores[qid].items()), key=lambda x: x[1], reverse=True)
        for rank, (did, score) in enumerate(score_list):
            f.write(f'{qid}\t0\t{did}\t{rank+1}\t{score}\t{run_id}\n')

print(f"TREC format file saved to {args.reranker_run_file}")