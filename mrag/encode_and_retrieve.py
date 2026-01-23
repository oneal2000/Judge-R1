"""
编码法条库并执行检索
"""
import os,sys
import json
import torch
import numpy as np
import faiss
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import argparse
from pathlib import Path

class DenseRetriever:
    """Dense Retriever 用于编码和检索"""
    
    def __init__(self, model_path, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModel.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()
    
    @torch.no_grad()
    def encode(self, texts, batch_size=32, max_length=256, show_progress=True):
        """编码文本列表"""
        all_embeddings = []
        
        iterator = range(0, len(texts), batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc="Encoding")
        
        for i in iterator:
            batch_texts = texts[i:i+batch_size]
            encoding = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors='pt'
            )
            
            input_ids = encoding['input_ids'].to(self.device)
            attention_mask = encoding['attention_mask'].to(self.device)
            
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            embeddings = outputs.last_hidden_state[:, 0, :]  # [CLS] token
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
            
            all_embeddings.append(embeddings.cpu().numpy())
        
        return np.vstack(all_embeddings)

def encode_corpus(args):
    """编码法条库"""
    print(f"Loading model from {args.model_path}")
    retriever = DenseRetriever(args.model_path, device=args.device)
    
    # 加载法条库
    print(f"Loading law corpus from {args.law_corpus}")
    law_ids = []
    law_texts = []
    with open(args.law_corpus, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            law_ids.append(item['text_id'])
            law_texts.append(f"{item['name']}：{item['text']}")
    
    print(f"Loaded {len(law_texts)} laws")
    
    # 编码
    print("Encoding law corpus...")
    embeddings = retriever.encode(law_texts, batch_size=args.batch_size, max_length=args.max_length)
    
    # 保存
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    np.save(output_dir / 'law_embeddings.npy', embeddings)
    with open(output_dir / 'law_ids.json', 'w') as f:
        json.dump(law_ids, f)
    
    print(f"Saved embeddings to {output_dir}")
    return embeddings, law_ids

def retrieve(args):
    """执行法条检索"""
    print(f"Loading model from {args.model_path}")
    retriever = DenseRetriever(args.model_path, device=args.device)
    
    # 加载预编码的法条
    embeddings_dir = Path(args.embeddings_dir)
    print(f"Loading precomputed embeddings from {embeddings_dir}")
    law_embeddings = np.load(embeddings_dir / 'law_embeddings.npy')
    with open(embeddings_dir / 'law_ids.json', 'r') as f:
        law_ids = json.load(f)
    
    # 构建 FAISS 索引
    print("Building FAISS index...")
    dimension = law_embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)  # 内积相似度（因为已经 L2 归一化）
    index.add(law_embeddings.astype(np.float32))
    
    # 加载查询
    print(f"Loading queries from {args.queries}")
    queries = []
    query_ids = []
    with open(args.queries, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            query_ids.append(item['query_id'])
            queries.append(item['query'])
    
    print(f"Loaded {len(queries)} queries")
    
    # 编码查询
    print("Encoding queries...")
    query_embeddings = retriever.encode(queries, batch_size=args.batch_size, max_length=512)
    
    # 检索
    print(f"Retrieving top-{args.top_k} laws...")
    scores, indices = index.search(query_embeddings.astype(np.float32), args.top_k)
    
    # 保存检索结果
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # TREC 格式: query_id Q0 doc_id rank score run_name
    with open(output_path, 'w') as f:
        for i, query_id in enumerate(query_ids):
            for rank, (idx, score) in enumerate(zip(indices[i], scores[i])):
                doc_id = law_ids[idx]
                f.write(f"{query_id}\tQ0\t{doc_id}\t{rank+1}\t{score:.6f}\tbge_retriever\n")
    
    print(f"Saved retrieval results to {output_path}")

def encode_case_corpus(args):
    """编码案例库"""
    print(f"Loading model from {args.model_path}")
    retriever = DenseRetriever(args.model_path, device=args.device)
    
    # 加载案例库
    print(f"Loading case corpus from {args.case_corpus}")
    case_ids = []
    case_texts = []  # 使用 text 字段（案件事实）进行检索
    case_fds = []    # 保存完整判决书，用于后续使用
    
    with open(args.case_corpus, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            case_ids.append(item['text_id'])
            case_texts.append(item['text'])  # 案件事实用于检索
            case_fds.append(item.get('fd', ''))  # 完整判决书
    
    print(f"Loaded {len(case_texts)} cases")
    
    # 编码（使用案件事实）
    print("Encoding case corpus...")
    embeddings = retriever.encode(case_texts, batch_size=args.batch_size, max_length=args.max_length)
    
    # 保存
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    np.save(output_dir / 'case_embeddings.npy', embeddings)
    with open(output_dir / 'case_ids.json', 'w') as f:
        json.dump(case_ids, f)
    # 保存案例完整判决书映射（可选，用于后续生成）
    with open(output_dir / 'case_fds.json', 'w') as f:
        json.dump(case_fds, f)
    
    print(f"Saved embeddings to {output_dir}")
    return embeddings, case_ids

def retrieve_cases(args):
    """执行案例检索"""
    print(f"Loading model from {args.model_path}")
    retriever = DenseRetriever(args.model_path, device=args.device)
    
    # 加载预编码的案例
    embeddings_dir = Path(args.embeddings_dir)
    print(f"Loading precomputed embeddings from {embeddings_dir}")
    case_embeddings = np.load(embeddings_dir / 'case_embeddings.npy')
    with open(embeddings_dir / 'case_ids.json', 'r') as f:
        case_ids = json.load(f)
    
    # 构建 FAISS 索引
    print("Building FAISS index...")
    dimension = case_embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(case_embeddings.astype(np.float32))
    
    # 加载查询
    print(f"Loading queries from {args.queries}")
    queries = []
    query_ids = []
    with open(args.queries, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            query_ids.append(item['query_id'])
            queries.append(item['query'])
    
    print(f"Loaded {len(queries)} queries")
    
    # 编码查询
    print("Encoding queries...")
    query_embeddings = retriever.encode(queries, batch_size=args.batch_size, max_length=512)
    
    # 检索
    print(f"Retrieving top-{args.top_k} cases...")
    scores, indices = index.search(query_embeddings.astype(np.float32), args.top_k)
    
    # 保存检索结果
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # TREC 格式: query_id Q0 doc_id rank score run_name
    with open(output_path, 'w') as f:
        for i, query_id in enumerate(query_ids):
            for rank, (idx, score) in enumerate(zip(indices[i], scores[i])):
                doc_id = case_ids[idx]
                f.write(f"{query_id}\tQ0\t{doc_id}\t{rank+1}\t{score:.6f}\tbge_case_retriever\n")
    
    print(f"Saved retrieval results to {output_path}")

# 在 main() 函数中添加子命令：
def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='mode', help='Mode: encode, retrieve, encode_case, retrieve_case')
    
    # 编码法条库子命令
    encode_parser = subparsers.add_parser('encode', help='Encode law corpus')
    encode_parser.add_argument('--model_path', type=str, required=True)
    encode_parser.add_argument('--law_corpus', type=str, required=True)
    encode_parser.add_argument('--output_dir', type=str, required=True)
    encode_parser.add_argument('--batch_size', type=int, default=32)
    encode_parser.add_argument('--max_length', type=int, default=256)
    encode_parser.add_argument('--device', type=str, default='cuda')
    
    # 检索法条子命令
    retrieve_parser = subparsers.add_parser('retrieve', help='Retrieve laws for queries')
    retrieve_parser.add_argument('--model_path', type=str, required=True)
    retrieve_parser.add_argument('--embeddings_dir', type=str, required=True)
    retrieve_parser.add_argument('--queries', type=str, required=True)
    retrieve_parser.add_argument('--output_file', type=str, required=True)
    retrieve_parser.add_argument('--top_k', type=int, default=10)
    retrieve_parser.add_argument('--batch_size', type=int, default=32)
    retrieve_parser.add_argument('--device', type=str, default='cuda')
    
    # 编码案例库子命令
    encode_case_parser = subparsers.add_parser('encode_case', help='Encode case corpus')
    encode_case_parser.add_argument('--model_path', type=str, required=True)
    encode_case_parser.add_argument('--case_corpus', type=str, required=True)
    encode_case_parser.add_argument('--output_dir', type=str, required=True)
    encode_case_parser.add_argument('--batch_size', type=int, default=32)
    encode_case_parser.add_argument('--max_length', type=int, default=512)  # 案例事实可能较长
    encode_case_parser.add_argument('--device', type=str, default='cuda')
    
    # 检索案例子命令
    retrieve_case_parser = subparsers.add_parser('retrieve_case', help='Retrieve cases for queries')
    retrieve_case_parser.add_argument('--model_path', type=str, required=True)
    retrieve_case_parser.add_argument('--embeddings_dir', type=str, required=True)
    retrieve_case_parser.add_argument('--queries', type=str, required=True)
    retrieve_case_parser.add_argument('--output_file', type=str, required=True)
    retrieve_case_parser.add_argument('--top_k', type=int, default=10)
    retrieve_case_parser.add_argument('--batch_size', type=int, default=32)
    retrieve_case_parser.add_argument('--device', type=str, default='cuda')
    
    args = parser.parse_args()
    
    if args.mode == 'encode':
        encode_corpus(args)
    elif args.mode == 'retrieve':
        retrieve(args)
    elif args.mode == 'encode_case':
        encode_case_corpus(args)
    elif args.mode == 'retrieve_case':
        retrieve_cases(args)
    else:
        parser.print_help()

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        import traceback
        print(f"错误: {e}", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)