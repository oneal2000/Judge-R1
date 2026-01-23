"""
使用chinese-roberta-wwm 训练法条检索模型
基于对比学习 + Hard Negatives
"""
import os
import json
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from tqdm import tqdm
import argparse
from pathlib import Path

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# ==================== Hard Negatives 挖掘 ====================

def mine_hard_negatives(
    model_path: str,
    train_data_path: str,
    law_corpus_path: str,
    device: torch.device,
    top_k: int = 50,
    num_hard_negs: int = 7,
    batch_size: int = 32,
    max_query_len: int = 512,
    max_passage_len: int = 512
):
    """
    用预训练模型挖掘 Hard Negatives
    返回: {query_text: [hard_neg_text1, hard_neg_text2, ...]}
    """
    print("=" * 50)
    print("挖掘 Hard Negatives...")
    print("=" * 50)
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    encoder = AutoModel.from_pretrained(model_path).to(device)
    encoder.eval()
    
    # 加载法条库
    law_dict = {}
    law_ids = []
    law_texts = []
    with open(law_corpus_path, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            law_id = str(item['text_id'])
            law_text = f"{item['name']}：{item['text']}"
            law_dict[law_id] = law_text
            law_ids.append(law_id)
            law_texts.append(law_text)
    
    print(f"加载 {len(law_texts)} 条法条")
    
    # 编码法条库
    print("编码法条库...")
    law_embeddings = []
    with torch.no_grad():
        for i in tqdm(range(0, len(law_texts), batch_size), desc="编码法条"):
            batch_texts = law_texts[i:i+batch_size]
            encoding = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=max_passage_len,
                return_tensors='pt'
            ).to(device)
            
            outputs = encoder(**encoding)
            emb = outputs.last_hidden_state[:, 0, :]
            emb = F.normalize(emb, p=2, dim=1)
            law_embeddings.append(emb.cpu())
    
    law_embeddings = torch.cat(law_embeddings, dim=0)  # [num_laws, dim]
    
    # 加载训练数据，提取唯一的 query 和对应的正例
    print("加载训练数据...")
    train_data = []
    with open(train_data_path, 'r', encoding='utf-8') as f:
        for line in f:
            train_data.append(json.loads(line))
    
    # 构建 query -> positives 映射（合并同一 query 的所有正例）
    query_to_positives = {}
    for item in train_data:
        query = item['query']
        pos = item['positives'][0]
        if query not in query_to_positives:
            query_to_positives[query] = set()
        query_to_positives[query].add(pos)
    
    unique_queries = list(query_to_positives.keys())
    print(f"共 {len(unique_queries)} 个唯一 query")
    
    # 为每个 query 检索并提取 hard negatives
    print("检索 Hard Negatives...")
    query_to_hard_negs = {}
    
    with torch.no_grad():
        for i in tqdm(range(0, len(unique_queries), batch_size), desc="检索"):
            batch_queries = unique_queries[i:i+batch_size]
            
            encoding = tokenizer(
                batch_queries,
                padding=True,
                truncation=True,
                max_length=max_query_len,
                return_tensors='pt'
            ).to(device)
            
            outputs = encoder(**encoding)
            query_emb = outputs.last_hidden_state[:, 0, :]
            query_emb = F.normalize(query_emb, p=2, dim=1).cpu()
            
            # 计算相似度并取 top-k
            scores = torch.matmul(query_emb, law_embeddings.T)
            top_scores, top_indices = torch.topk(scores, k=min(top_k, scores.size(1)), dim=1)
            
            for j, query in enumerate(batch_queries):
                positives_set = query_to_positives[query]
                
                hard_negs = []
                for idx in top_indices[j].tolist():
                    law_text = law_texts[idx]
                    # 排除正例
                    if law_text not in positives_set:
                        hard_negs.append(law_text)
                        if len(hard_negs) >= num_hard_negs:
                            break
                
                query_to_hard_negs[query] = hard_negs
    
    print(f"为 {len(query_to_hard_negs)} 个 query 挖掘了 hard negatives")
    
    # 清理显存
    del encoder, law_embeddings
    torch.cuda.empty_cache()
    
    return query_to_hard_negs

# ==================== 数据集 ====================

class DenseRetrieverDataset(Dataset):
    """Dense Retriever 训练数据集（支持 hard negatives）"""
    
    def __init__(self, data_path, tokenizer, max_query_len=512, max_passage_len=512,
                 hard_negatives_dict=None):
        self.tokenizer = tokenizer
        self.max_query_len = max_query_len
        self.max_passage_len = max_passage_len
        self.hard_negatives_dict = hard_negatives_dict or {}
        
        self.data = []
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                self.data.append(json.loads(line))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        query = item['query']
        positive = item['positives'][0]
        
        # 优先使用 hard negatives，否则用原始随机 negatives
        if query in self.hard_negatives_dict and self.hard_negatives_dict[query]:
            negatives = self.hard_negatives_dict[query]
        else:
            negatives = item['negatives']
        
        return {
            'query': query,
            'positive': positive,
            'negatives': negatives
        }

def collate_fn(batch, tokenizer, max_query_len, max_passage_len):
    """批处理函数"""
    queries = [item['query'] for item in batch]
    positives = [item['positive'] for item in batch]
    
    all_negatives = []
    for item in batch:
        all_negatives.extend(item['negatives'])
    
    passages = positives + all_negatives
    
    query_encoding = tokenizer(
        queries,
        padding=True,
        truncation=True,
        max_length=max_query_len,
        return_tensors='pt'
    )
    
    passage_encoding = tokenizer(
        passages,
        padding=True,
        truncation=True,
        max_length=max_passage_len,
        return_tensors='pt'
    )
    
    return {
        'query_input_ids': query_encoding['input_ids'],
        'query_attention_mask': query_encoding['attention_mask'],
        'passage_input_ids': passage_encoding['input_ids'],
        'passage_attention_mask': passage_encoding['attention_mask'],
    }

# ==================== 模型 ====================

class BiEncoder(nn.Module):
    """双塔编码器模型"""
    
    def __init__(self, model_name_or_path):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name_or_path)
    
    def encode(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        embeddings = outputs.last_hidden_state[:, 0, :]
        embeddings = F.normalize(embeddings, p=2, dim=1)
        return embeddings
    
    def forward(self, query_input_ids, query_attention_mask, 
                passage_input_ids, passage_attention_mask):
        query_emb = self.encode(query_input_ids, query_attention_mask)
        passage_emb = self.encode(passage_input_ids, passage_attention_mask)
        return query_emb, passage_emb

def compute_loss(query_emb, passage_emb, temperature=0.02):
    """计算 InfoNCE Loss"""
    batch_size = query_emb.size(0)
    scores = torch.matmul(query_emb, passage_emb.T) / temperature
    labels = torch.arange(batch_size, device=query_emb.device)
    loss = F.cross_entropy(scores, labels)
    return loss

# ==================== 训练 ====================

def train(args):
    set_seed(args.seed)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 第一步：挖掘 Hard Negatives
    hard_negatives_dict = mine_hard_negatives(
        model_path=args.model_name_or_path,
        train_data_path=args.train_data,
        law_corpus_path=args.law_corpus,
        device=device,
        top_k=50,
        num_hard_negs=args.num_hard_negs,
        batch_size=32,
        max_query_len=args.max_query_len,
        max_passage_len=args.max_passage_len
    )
    
    # 第二步：加载模型和数据
    print(f"Loading model from {args.model_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    model = BiEncoder(args.model_name_or_path)
    model.to(device)
    
    print(f"Loading training data with hard negatives...")
    dataset = DenseRetrieverDataset(
        args.train_data, 
        tokenizer,
        max_query_len=args.max_query_len,
        max_passage_len=args.max_passage_len,
        hard_negatives_dict=hard_negatives_dict
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda batch: collate_fn(
            batch, tokenizer, args.max_query_len, args.max_passage_len
        ),
        num_workers=4,
        pin_memory=True
    )
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    total_steps = len(dataloader) * args.num_epochs
    warmup_steps = int(total_steps * 0.1)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    
    scaler = torch.cuda.amp.GradScaler()
    
    print(f"\n{'='*50}")
    print(f"开始训练")
    print(f"{'='*50}")
    print(f"Epochs: {args.num_epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Temperature: {args.temperature}")
    print(f"Hard negatives per query: {args.num_hard_negs}")
    print(f"{'='*50}\n")
    
    model.train()
    best_loss = float('inf')
    
    for epoch in range(args.num_epochs):
        epoch_loss = 0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.num_epochs}")
        
        for batch in progress_bar:
            query_input_ids = batch['query_input_ids'].to(device, non_blocking=True)
            query_attention_mask = batch['query_attention_mask'].to(device, non_blocking=True)
            passage_input_ids = batch['passage_input_ids'].to(device, non_blocking=True)
            passage_attention_mask = batch['passage_attention_mask'].to(device, non_blocking=True)
            
            with torch.amp.autocast('cuda'):
                query_emb, passage_emb = model(
                    query_input_ids, query_attention_mask,
                    passage_input_ids, passage_attention_mask
                )
                loss = compute_loss(query_emb, passage_emb, temperature=args.temperature)
            
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            
            epoch_loss += loss.item()
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch+1} average loss: {avg_loss:.4f}")
        
        if avg_loss < best_loss:
            best_loss = avg_loss
    
    # 只保存最终模型
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nSaving final model to {output_dir}")
    model.encoder.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    print(f"✅ Training completed! Best loss: {best_loss:.4f}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name_or_path', type=str, required=True)
    parser.add_argument('--train_data', type=str, required=True)
    parser.add_argument('--law_corpus', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_epochs', type=int, default=5)
    parser.add_argument('--learning_rate', type=float, default=2e-5)
    parser.add_argument('--max_query_len', type=int, default=512)
    parser.add_argument('--max_passage_len', type=int, default=512)
    parser.add_argument('--temperature', type=float, default=0.02)
    parser.add_argument('--num_hard_negs', type=int, default=7)
    parser.add_argument('--seed', type=int, default=42)
    
    args = parser.parse_args()
    train(args)

if __name__ == '__main__':
    main()