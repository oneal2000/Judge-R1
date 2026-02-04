# JuDGE_RL: 基于强化学习的高质量法律文书生成系统

## 项目概述

JuDGE_RL (Judicial Document Generation with Reinforcement Learning) 是一个端到端的中文刑事判决书自动生成系统。本项目通过结合 **检索增强生成 (RAG)**、**LLM-based 检索代理** 和 **强化学习 (GRPO)** 三种技术，实现高质量、法律准确的判决书自动生成。

### 核心特点

- **双路检索架构**: MRAG (Multi-Retrieval Augmented Generation) + LLM-based Agent
- **端到端 RL 优化**: 使用 GRPO 算法同时优化检索和生成两个阶段
- **多维度奖励函数**: 法律准确性 + 文本质量 + 推理质量
- **支持 Thinking 模型**: 充分利用 Qwen3-4B-Thinking 的 Chain-of-Thought 能力

---

## 系统架构

```
┌─────────────────────────────────────────────────────────────────────┐
│                         JuDGE_RL 系统架构                            │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌─────────────┐     ┌─────────────────────────────────────────┐   │
│  │  案件事实   │────▶│           检索增强模块                   │   │
│  └─────────────┘     │  ┌─────────────┐  ┌─────────────────┐   │   │
│                      │  │    MRAG     │  │  LLM-based Agent │   │   │
│                      │  │             │  │                  │   │   │
│                      │  │ Dense + Re- │  │ QueryGen + Dense │   │   │
│                      │  │   ranker    │  │ + Rerank + Law-  │   │   │
│                      │  │             │  │     Select       │   │   │
│                      │  └─────────────┘  └─────────────────┘   │   │
│                      │         │                  │             │   │
│                      │         └────────┬─────────┘             │   │
│                      │                  ▼                       │   │
│                      │         相关法条 + 相似案例               │   │
│                      └─────────────────────────────────────────┘   │
│                                         │                           │
│                                         ▼                           │
│                      ┌─────────────────────────────────────────┐   │
│                      │         判决书生成模块                    │   │
│                      │                                         │   │
│                      │  Qwen3-4B-Thinking (SFT + GRPO)         │   │
│                      │                                         │   │
│                      │  奖励函数:                               │   │
│                      │  - 法律准确性 (罪名、法条、刑期、罚金)    │   │
│                      │  - 文本质量 (BERTScore, METEOR)          │   │
│                      │  - 推理质量 (格式、长度、无重复)          │   │
│                      └─────────────────────────────────────────┘   │
│                                         │                           │
│                                         ▼                           │
│                      ┌─────────────────────────────────────────┐   │
│                      │              判决书输出                   │   │
│                      │  (案件事实 + 法律分析 + 裁判理由 + 结论)  │   │
│                      └─────────────────────────────────────────┘   │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 目录结构

```
JuDGE_RL/
├── bash/                           # Shell 脚本
│   ├── agent/                      # LLM-based Agent 相关
│   │   ├── train_agent_rl_querygen.sh    # QueryGen RL 训练
│   │   ├── train_agent_rl_lawselect.sh   # LawSelect RL 训练
│   │   ├── run_law_agent_pipeline.sh     # Agent 推理流程
│   │   ├── run_hybrid_agent.sh           # Hybrid 融合流程
│   │   ├── eval_agent_rl_ablation.sh     # 消融实验
│   │   └── prepare_agent_rl_data.sh      # 准备 Agent RL 训练数据
│   ├── retriever/                  # Dense Retriever 相关
│   │   ├── kfold_train_retriever.sh      # K-Fold 训练
│   │   ├── encode_corpus.sh              # 编码语料库
│   │   └── eval_retriever.sh             # 评测检索效果
│   ├── reranker/                   # Reranker 相关
│   │   ├── kfold_train_reranker.sh       # K-Fold 训练
│   │   └── run_reranker.sh               # 运行重排序
│   ├── legalone/                   # LegalOne-4B 基线实验
│   ├── data_train.sh               # 生成训练/测试数据
│   ├── train_sft.sh                # SFT 训练
│   ├── train_rl.sh                 # RL (GRPO) 训练
│   ├── gen.sh                      # 判决书生成
│   ├── convert.sh                  # 格式转换
│   └── eval.sh                     # 评测
│
├── config/                         # 配置文件
│   ├── ds_zero2_config.json        # DeepSpeed ZeRO-2 配置
│   └── ds_zero3_config.json        # DeepSpeed ZeRO-3 配置
│
├── data/                           # 数据文件
│   ├── train.json                  # 原始训练集
│   ├── test.json                   # 原始测试集
│   ├── law_corpus.jsonl            # 法条语料库
│   ├── case_corpus.jsonl           # 案例语料库
│   ├── train_sft.json              # SFT 训练集（标准模式）
│   ├── train_sft_mrag.json         # SFT 训练集（MRAG 模式）
│   ├── test_sft.json               # SFT 测试集（标准模式）
│   └── test_sft_mrag.json          # SFT 测试集（MRAG 模式）
│
├── evaluation/                     # 评测模块
│   ├── calc.py                     # 法律准确性评测 (罪名、法条、刑期)
│   ├── calc_rel.py                 # 文本质量评测 (METEOR, BERTScore)
│   ├── crime_extraction.py         # 罪名提取
│   ├── law_extraction.py           # 法条提取
│   └── judge_extraction.py         # 刑期/罚金提取
│
├── mrag/                           # MRAG 检索模块
│   ├── agent/                      # LLM-based Agent
│   │   ├── law_agent.py            # Agent 主流程
│   │   ├── hybrid_agent.py         # Hybrid 融合流程
│   │   ├── prompts.py              # 统一提示词模板
│   │   └── gen_agent_rl_data.py    # 生成 Agent RL 训练数据
│   ├── train_retriever.py          # Dense Retriever 训练
│   ├── encode_and_retrieve.py      # 编码与检索
│   └── eval_retriever.py           # 检索评测
│
├── reranker/                       # Reranker 模块
│   ├── reranker/                   # 核心代码
│   │   ├── modeling.py             # 模型定义
│   │   ├── trainer.py              # 训练器
│   │   └── data.py                 # 数据处理
│   └── run_reranker.py             # 运行脚本
│
├── train/                          # 训练模块
│   ├── src/
│   │   ├── rl_plugin1.py           # 判决书生成 RL 奖励函数
│   │   ├── agent_rl_plugin.py      # Agent RL 奖励函数
│   │   └── data.py                 # 数据处理
│   └── deploy/
│       ├── inf.py                  # vLLM 推理脚本
│       └── merge_lora.py           # LoRA 合并
│
├── script/                         # 数据生成脚本
│   ├── sft_data.py                 # 生成 SFT 数据
│   └── rl_data.py                  # 生成 RL 数据
│
├── outputs/                        # 推理输出
├── output/                         # 模型输出
└── result/                         # 评测结果
```

---

## 环境配置

### 依赖环境

本项目需要 **三个** conda 环境：

| 环境名 | 用途 | 说明 |
|--------|------|------|
| `swift` | SFT/RL 训练 | 基于 ms-swift 框架，用于模型微调和强化学习训练 |
| `judge` | 检索训练与评测 | 用于 Dense Retriever、Reranker 训练及结果评测 |
| `vllm` | 推理加速（可选） | 用于高效推理，可与 swift 环境配合使用 |

```bash
# 1. swift 环境 (用于 SFT/RL 训练)
conda create -n swift python=3.10
conda activate swift
pip install -r requirements_swift.txt

# 2. judge 环境 (用于检索训练和评测)
conda create -n judge python=3.10
conda activate judge
pip install -r requirements_judge.txt

# 3. vllm 环境 (用于推理加速，可选)
conda create -n vllm python=3.10
conda activate vllm
pip install -r requirements_vllm.txt
```

### 硬件要求

| 任务 | 最低配置 | 推荐配置 |
|------|----------|----------|
| SFT (LoRA) | 1x A100 40GB | 2x A100 80GB |
| SFT (Full) | 4x A100 40GB | 4x A100 80GB |
| RL (GRPO) | 1x A100 80GB | 4x A100 80GB |
| Agent RL (3B) | 4x A100 40GB | 4x A100 80GB |
| Agent RL (7B) | 4x A100 80GB | 8x A100 80GB |
| 推理 (vLLM) | 1x A100 40GB | 1x A100 80GB |

> **注意**: GRPO 训练需要同时加载 policy model 和 reference model，显存需求较高。单卡 80GB 可以训练 3B 模型（需设置 `num_generations=4~8`），多卡推荐使用 DeepSpeed ZeRO。

---

## 模型下载

本项目需要以下预训练模型，请提前下载：

### 必需模型

| 模型 | 用途 | 下载链接 |
|------|------|----------|
| Qwen2.5-3B-Instruct | 判决书生成基座模型 | [HuggingFace](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct) |
| chinese-roberta-wwm-ext | Dense Retriever / Reranker | [HuggingFace](https://huggingface.co/hfl/chinese-roberta-wwm-ext) |

### 可选模型（用于对比实验）

| 模型 | 用途 | 下载链接 |
|------|------|----------|
| Qwen3-4B-Thinking | Thinking 模型实验 | [HuggingFace](https://huggingface.co/Qwen/Qwen3-4B) |
| Qwen2.5-7B-Instruct | Agent (QueryGen/LawSelect) | [HuggingFace](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct) |

### 下载方式

```bash
# 方式 1: 使用 huggingface-cli
pip install huggingface_hub
huggingface-cli download Qwen/Qwen2.5-3B-Instruct --local-dir /path/to/models/Qwen2.5-3B-Instruct

# 方式 2: 使用 modelscope (国内用户推荐)
pip install modelscope
modelscope download --model qwen/Qwen2.5-3B-Instruct --local_dir /path/to/models/Qwen2.5-3B-Instruct

# 方式 3: 使用 git lfs
git lfs install
git clone https://huggingface.co/Qwen/Qwen2.5-3B-Instruct /path/to/models/Qwen2.5-3B-Instruct
```

下载完成后，修改相关脚本中的模型路径，或设置环境变量：

```bash
export BASE_MODEL="/path/to/models/Qwen2.5-3B-Instruct"
export RETRIEVER_MODEL="/path/to/models/chinese-roberta-wwm-ext"
```

---

## 完整复现流程

以下是从 clone 项目到完成全部实验的完整步骤。

### Step 0: 克隆项目

```bash
git clone https://github.com/xxx/JuDGE_RL.git
cd JuDGE_RL
```

### Step 1: 环境配置

```bash
# 创建三个环境
conda create -n swift python=3.10 -y
conda create -n judge python=3.10 -y
conda create -n vllm python=3.10 -y

# 安装依赖
conda activate swift && pip install -r requirements_swift.txt
conda activate judge && pip install -r requirements_judge.txt
conda activate vllm && pip install -r requirements_vllm.txt
```

### Step 2: 数据准备

项目已包含以下原始数据文件（在 `data/` 目录下）：

| 文件 | 说明 |
|------|------|
| `train.json` | 原始训练集 |
| `test.json` | 原始测试集 |
| `law_corpus.jsonl` | 法条语料库 |
| `case_corpus.jsonl` | 案例语料库 |

生成训练所需的数据：

```bash
conda activate swift

# 生成 SFT 训练数据（标准模式）
bash bash/data_train.sh

# 生成 SFT 训练数据（MRAG 模式，需先完成检索）
USE_MRAG=true bash bash/data_train.sh
```

### Step 3: 检索模型训练

```bash
conda activate judge

# 3.1 训练 Dense Retriever (K-Fold)
bash bash/retriever/kfold_train_retriever.sh

# 3.2 编码语料库
bash bash/retriever/encode_corpus.sh

# 3.3 训练 Reranker (K-Fold)
bash bash/reranker/kfold_train_reranker.sh

# 3.4 运行完整检索流程（Dense + Rerank）
bash bash/reranker/run_reranker.sh
```

### Step 4: 判决书生成模型训练

```bash
conda activate swift

# 4.1 SFT 训练（LoRA）
bash bash/train_sft.sh

# 4.2 合并 LoRA 权重
bash bash/loramerge.sh

# 4.3 RL 训练（GRPO）
# 注意：需要足够显存，建议单卡 80GB 或多卡
bash bash/train_rl.sh
```

### Step 5: 推理与评测

```bash
# 5.1 生成判决书
conda activate swift  # 或 vllm 环境
USE_MRAG=true bash bash/gen.sh

# 5.2 格式转换
USE_MRAG=true bash bash/convert.sh

# 5.3 评测
conda activate judge
USE_MRAG=true bash bash/eval.sh

# 5.4 查看结果
cat result/eval_summary_mrag.txt
```

### Step 6: Agent RL 实验（可选）

```bash
conda activate swift

# 6.1 准备 Agent RL 训练数据
bash bash/agent/prepare_agent_rl_data.sh

# 6.2 训练 QueryGen
bash bash/agent/train_agent_rl_querygen.sh

# 6.3 训练 LawSelect
bash bash/agent/train_agent_rl_lawselect.sh

# 6.4 运行 Agent 推理
bash bash/agent/run_law_agent_pipeline.sh

# 6.5 消融实验评测
bash bash/agent/eval_agent_rl_ablation.sh
```

---

## 项目文件说明

### 仓库包含的文件

以下文件已包含在仓库中，clone 后即可使用：

```
data/
├── train.json              # 原始训练集
├── test.json               # 原始测试集
├── law_corpus.jsonl        # 法条语料库（约 500 条刑法条文）
└── case_corpus.jsonl       # 案例语料库
```

### 需要生成的文件

以下文件需要通过脚本生成（已在 `.gitignore` 中排除）：

| 目录/文件 | 说明 | 生成命令 |
|-----------|------|----------|
| `data/train_sft.json` | SFT 训练数据 | `bash bash/data_train.sh` |
| `data/train_sft_mrag.json` | MRAG 模式 SFT 数据 | `USE_MRAG=true bash bash/data_train.sh` |
| `data/rl_train/` | RL 训练数据 | `bash bash/data_train.sh` |
| `output/` | 模型 checkpoint | 训练过程生成 |
| `outputs/` | 推理结果 | 推理脚本生成 |
| `result/` | 评测结果 | 评测脚本生成 |
| `mrag/kfold_*/` | 检索中间产物 | 检索训练生成 |
| `reranker/result/` | Reranker 结果 | Reranker 训练生成 |

---

## 数据格式说明

### train.json / test.json

```json
{
  "id": "case_001",
  "fact": "被告人张某于2023年1月1日...",
  "laws": ["第二百三十四条", "第六十七条"],
  "crimes": ["故意伤害罪"],
  "imprisonment": "有期徒刑三年",
  "fine": "无",
  "reference_document": "经审理查明，被告人张某..."
}
```

### law_corpus.jsonl

```json
{"id": "law_234", "title": "第二百三十四条", "content": "故意伤害他人身体的，处三年以下有期徒刑..."}
```

---

## 快速开始

### 1. 数据准备

```bash
# 生成标准模式训练数据
bash bash/data_train.sh

# 生成 MRAG 模式训练数据（需要先完成检索）
USE_MRAG=true bash bash/data_train.sh
```

### 2. 检索模型训练

```bash
# 训练 Dense Retriever
bash bash/retriever/kfold_train_retriever.sh

# 训练 Reranker
bash bash/reranker/kfold_train_reranker.sh

# 运行 MRAG 检索
bash bash/reranker/run_reranker.sh
```

### 3. Agent RL 训练（可选）

```bash
# 准备 Agent RL 训练数据
bash bash/agent/prepare_agent_rl_data.sh

# 训练 QueryGen
bash bash/agent/train_agent_rl_querygen.sh

# 训练 LawSelect
bash bash/agent/train_agent_rl_lawselect.sh
```

### 4. 判决书生成模型训练

```bash
# SFT 训练
bash bash/train_sft.sh

# LoRA 合并
bash bash/loramerge.sh

# RL 训练
bash bash/train_rl.sh
```

### 5. 推理与评测

```bash
# 生成判决书
USE_MRAG=true bash bash/gen.sh

# 格式转换
USE_MRAG=true bash bash/convert.sh

# 评测
USE_MRAG=true bash bash/eval.sh

# 查看结果
cat result/eval_summary_mrag.txt
```

---

## 核心模块详解

### 1. MRAG 检索模块

**流程**: 案件事实 → Dense Retriever → Reranker → Top-K 法条

```bash
# Dense Retriever: chinese-roberta-wwm 微调
# 训练目标: 对比学习 + Hard Negatives

# Reranker: chinese-roberta-wwm 微调  
# 训练目标: Pairwise Ranking Loss
```

### 2. LLM-based Agent

**流程**: 案件事实 → QueryGen → Dense → Reranker → LawSelect → 最终法条

| 组件 | 模型 | 作用 |
|------|------|------|
| QueryGen | Qwen2.5-7B-Instruct | 生成 5-8 个检索查询 |
| LawSelect | Qwen2.5-7B-Instruct | 从候选中筛选 20-30 条法条 |

**RL 奖励函数**:
- QueryGen: Recall@50 + MRR + HitRatio + Diversity
- LawSelect: Recall (70%) + RankingScore (20%) + Precision (5%) + Format (5%)

### 3. 判决书生成

**模型**: Qwen3-4B-Thinking

**训练流程**:
1. SFT: 使用参考判决书进行监督学习
2. GRPO: 使用多维度奖励函数进行强化学习

**奖励函数组成**:
- 法律准确性 (60%): 罪名 F1、法条 F1、刑期/罚金匹配
- 文本质量 (30%): BERTScore
- 推理质量 (10%): `<think>` 格式、长度、无重复

---

## 评测指标

### 检索评测

| 指标 | 说明 |
|------|------|
| Recall@K | Top-K 中召回的正确法条比例 |
| Precision@K | Top-K 中正确法条占比 |
| MRR | 第一个正确法条的倒数排名 |
| NDCG | 归一化折损累积增益 |

### 生成评测

| 指标 | 说明 |
|------|------|
| Crime F1 | 罪名识别准确率 |
| Law Article F1 | 法条引用准确率 |
| Prison Time Score | 刑期匹配度 |
| Fine Amount Score | 罚金匹配度 |
| METEOR | 文本相似度 |
| BERTScore | 语义相似度 |

---

## 实验结果

### 法条检索性能

| 方法 | Recall@5 | Recall@10 | Recall@20 | MRR |
|------|----------|-----------|-----------|-----|
| MRAG | 0.7606 | 0.9073 | 0.9588 | 0.9590 |
| Agent (Both RL) | 0.7146 | 0.8145 | 0.8145 | 0.9523 |

### 消融实验（Agent）

| 配置 | Recall@5 | Recall@10 | MRR |
|------|----------|-----------|-----|
| Baseline (无 RL) | 0.6384 | 0.6964 | 0.8676 |
| QueryGen RL only | 0.6602 | 0.7145 | 0.8733 |
| LawSelect RL only | 0.7061 | 0.8027 | 0.9471 |
| Both RL | 0.7146 | 0.8145 | 0.9523 |

---

## 常见问题

### Q1: CUDA OOM（显存不足）怎么办？

GRPO 训练需要同时加载 policy model 和 reference model，显存需求较高。

**解决方案（按优先级排序）**：

1. **减少 `num_generations`**（最有效）
   ```bash
   # 在 bash/train_rl.sh 中修改
   --num_generations 4   # 默认是 16，改为 4 或 8
   ```

2. **减少 `max_completion_length`**
   ```bash
   --max_completion_length 2048  # 默认是 4096
   ```

3. **使用 LoRA 训练代替全参数训练**
   ```bash
   EXPERIMENT=sft_lora bash bash/train_rl.sh
   ```

4. **启用 DeepSpeed ZeRO**
   ```bash
   # 添加 DeepSpeed 配置
   --deepspeed config/ds_zero2_config.json
   ```

5. **确保 GPU 没有被其他进程占用**
   ```bash
   # 检查 GPU 占用情况
   nvidia-smi
   
   # 查看占用 GPU 的进程
   fuser -v /dev/nvidia*
   ```

**显存估算（Qwen2.5-3B + GRPO）**：
| 配置 | 显存需求 |
|------|----------|
| full + num_generations=16 | ~60-70 GB |
| full + num_generations=8 | ~45-50 GB |
| full + num_generations=4 | ~35-40 GB |
| lora + num_generations=8 | ~25-30 GB |

### Q2: 训练不稳定？

1. 增加 `beta` (KL 惩罚系数): `--beta 0.1`
2. 减小学习率: `--learning_rate 1e-6`
3. 增加 `gradient_accumulation_steps`: `--gradient_accumulation_steps 32`
4. 检查奖励函数是否有 NaN 值

### Q3: 检索效果差？

1. 检查 Hard Negatives 数量是否足够（建议 15-30 个）
2. 增加训练 epochs
3. 使用 K-Fold 交叉验证确保泛化性
4. 检查语料库编码是否正确

### Q4: 如何使用 vLLM 加速推理？

```bash
# 方式 1: 启动 vLLM server
conda activate vllm
python -m vllm.entrypoints.openai.api_server \
    --model /path/to/model \
    --port 8000

# 方式 2: 在训练中使用 vLLM
USE_VLLM=true VLLM_PORT=8000 bash bash/train_rl.sh
```

### Q5: 如何恢复中断的训练？

```bash
# 找到最近的 checkpoint
ls output/rl_qwen2.5-3b_grpo_sft_full/

# 从 checkpoint 恢复
bash bash/train_rl.sh --resume_from_checkpoint output/rl_qwen2.5-3b_grpo_sft_full/v1-xxx/checkpoint-xxx
```

### Q6: vLLM colocate 模式训练结束后出现 Segfault？

使用 vLLM colocate 模式（`VLLM_MODE=colocate`）进行 Agent RL 训练时，训练**成功完成**后可能出现以下错误：

```
!!!!!!! Segfault encountered !!!!!!!
terminate called after throwing an instance of 'c10::Error'
  what():  Trying to free a pointer not allocated here
Signal 11 (SIGSEGV) received
```

**这是正常现象，不影响训练结果！**

**原因**：vLLM 使用自己的 `CuMemAllocator` 管理显存，但在进程退出时 PyTorch 尝试释放这些内存，导致内存分配器冲突。这是 vLLM + PyTorch 的已知兼容性问题，发生在训练**结束后的清理阶段**。

**判断训练是否成功**：
- 检查日志中是否有 `global_step/max_steps: 'xxx/xxx', percentage: '100.00%'`
- 检查是否有 `[INFO:swift] Saving model checkpoint to ...`
- 检查 checkpoint 文件是否已保存

**解决方案（可选）**：

```bash
# 方法 1: 忽略退出错误（推荐）
VLLM_MODE=colocate bash bash/agent/train_agent_rl_lawselect.sh || true

# 方法 2: 检查 checkpoint 是否存在来判断训练是否成功
VLLM_MODE=colocate bash bash/agent/train_agent_rl_lawselect.sh || {
    if ls output/agent_rl_lawselect_7b_v1_lora/*/checkpoint-* 1>/dev/null 2>&1; then
        echo "训练已完成，Segfault 发生在清理阶段，可忽略"
    else
        echo "训练失败"
        exit 1
    fi
}
```

### Q7: vLLM colocate 模式与 DeepSpeed 冲突？

vLLM colocate 模式与 DeepSpeed CPU Offload **不兼容**，同时使用会导致训练过程中的 Segfault：

```
!!!!!!! Segfault encountered !!!!!!!
File "...deepspeed/ops/csrc/includes/cpu_adam.h" in Adam_Optimizer::Step_AVX
```

**解决方案**：脚本已自动处理，当启用 `VLLM_MODE=colocate` 时会自动禁用 DeepSpeed。如果仍遇到问题，可手动设置：

```bash
VLLM_MODE=colocate DEEPSPEED_MODE=none bash bash/agent/train_agent_rl_lawselect.sh
```

---

## 引用

如果本项目对您的研究有帮助，请引用：

```bibtex
@misc{judge_rl_2026,
  title={JuDGE_RL: Judicial Document Generation with Reinforcement Learning},
  author={Chen Xuanyi},
  year={2026},
  howpublished={\url{https://github.com/xxx/JuDGE_RL}}
}
```

---

## License

本项目采用 MIT License。详见 [LICENSE](LICENSE) 文件。

---

## TODO

- [ ] 支持更多罪名类型
- [ ] 实现 Agent 和 MRAG 的智能融合（后置补充策略）
- [ ] 支持 vLLM + Swift 联合训练加速
- [ ] 添加民事判决书生成支持
- [ ] 开源预训练模型权重
