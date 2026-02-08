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
│   │   ├── train_rl_querygen.sh         # QueryGen RL 训练
│   │   ├── train_rl_lawselect.sh        # LawSelect RL 训练
│   │   ├── merge_agent_lora.sh          # Agent LoRA 合并（自动查找 checkpoint）
│   │   ├── prepare_agent_rl_data.sh     # 准备 Agent RL 训练数据
│   │   ├── run_law_agent_pipeline.sh    # Agent 推理流程
│   │   ├── run_hybrid_agent.sh          # Hybrid 融合流程（MRAG + Agent）
│   │   └── eval_ablation.sh             # Agent 消融实验（9 种配置）
│   ├── retriever/                  # Dense Retriever 相关
│   │   ├── kfold_train_retriever.sh     # K-Fold 训练（自动生成检索数据）
│   │   ├── encode_corpus.sh             # 编码语料库
│   │   ├── eval_retriever.sh            # 评测检索效果
│   │   ├── retrieve.sh                  # 运行检索
│   │   └── compare.sh                   # 对比结果
│   ├── reranker/                   # Reranker 相关
│   │   ├── kfold_train_reranker.sh      # K-Fold 训练
│   │   └── run_reranker.sh              # 运行重排序
│   ├── legalone/                   # LegalOne-4B 基线实验
│   │   ├── download_model.sh            # 下载 LegalOne-4B 模型
│   │   ├── train_sft.sh                 # LegalOne SFT 训练
│   │   ├── loramerge.sh                 # LegalOne LoRA 合并
│   │   ├── gen.sh                       # LegalOne 推理
│   │   ├── convert.sh                   # LegalOne 格式转换
│   │   └── eval.sh                      # LegalOne 评测
│   ├── data_train.sh               # 生成 SFT/RL 训练数据
│   ├── train_sft.sh                # SFT 训练（参数化: MODEL_NAME + USE_MRAG）
│   ├── train_rl.sh                 # RL (GRPO) 训练（参数化: MODEL_NAME + EXPERIMENT）
│   ├── loramerge.sh                # SFT LoRA 合并（参数化: MERGE_CONFIG）
│   ├── gen.sh                      # 判决书推理（参数化: MODEL_NAME + MODES）
│   ├── convert.sh                  # 格式转换
│   └── eval.sh                     # 评测
│
├── data/                           # 数据文件
│   ├── train.json                  # 原始训练集（仓库自带）
│   ├── test.json                   # 原始测试集（仓库自带）
│   ├── law_corpus.jsonl            # 法条语料库（仓库自带）
│   ├── case_corpus.jsonl           # 案例语料库（仓库自带）
│   ├── expected.jsonl              # 评测参考判决书（仓库自带）
│   ├── train_sft.json              # SFT 训练集 - 标准模式（生成）
│   ├── train_sft_mrag.json         # SFT 训练集 - MRAG 模式（生成）
│   ├── test_sft.json               # SFT 测试集 - 标准模式（生成）
│   └── test_sft_mrag.json          # SFT 测试集 - MRAG 模式（生成）
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
│   ├── gen_kfold_data.py           # 生成检索数据 (qrels, queries)
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
│   │   ├── ds_config.json          # DeepSpeed 配置
│   │   ├── data.py                 # SFT 数据处理
│   │   └── train.py                # SFT 训练主脚本
│   └── deploy/
│       ├── inf.py                  # vLLM 推理脚本
│       ├── merge_lora.py           # LoRA 合并
│       └── extract_lora.py         # LoRA 提取（DeepSpeed checkpoint）
│
├── script/                         # 数据生成脚本
│   ├── sft_data.py                 # 生成 SFT 数据
│   └── rl_data.py                  # 生成 RL 数据
│
├── outputs/                        # 推理输出（生成）
├── output/                         # 模型 checkpoint（生成）
└── result/                         # 评测结果（生成）
```

---

## 环境配置

### 依赖环境

本项目需要 **三个** conda 环境：

| 环境名 | 用途 | 说明 |
|--------|------|------|
| `swift` | SFT/RL 训练、推理 | 基于 ms-swift 框架，用于模型微调和强化学习训练 |
| `judge` | 检索训练与评测 | 用于 Dense Retriever、Reranker 训练及结果评测 |
| `vllm` | 推理加速（可选） | 用于高效推理，可与 swift 环境配合使用 |

```bash
# 1. swift 环境 (用于 SFT/RL 训练)
conda create -n swift python=3.10 -y
conda activate swift
pip install -r requirements_swift.txt

# 2. judge 环境 (用于检索训练和评测)
conda create -n judge python=3.10 -y
conda activate judge
pip install -r requirements_judge.txt

# 3. vllm 环境 (用于推理加速，可选)
conda create -n vllm python=3.10 -y
conda activate vllm
pip install -r requirements_vllm.txt
```

### 硬件要求

| 任务 | 最低配置 | 推荐配置 |
|------|----------|----------|
| SFT (LoRA) | 1x A100 40GB | 2x A100 80GB |
| SFT (Full) | 4x A100 40GB | 4x A100 80GB |
| RL (GRPO) | 1x A100 80GB | 4x A100 80GB |
| Agent RL (7B) | 4x A100 80GB | 4x A100 80GB |
| 推理 (vLLM) | 1x A100 40GB | 1x A100 80GB |

> **注意**: GRPO 训练需要同时加载 policy model 和 reference model，显存需求较高。

---

## 模型下载

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
| LegalOne-4B | 法律领域基线模型 | 需自行获取 |

```bash
# 使用 huggingface-cli 下载
pip install huggingface_hub
huggingface-cli download Qwen/Qwen2.5-3B-Instruct --local-dir /path/to/models/Qwen2.5-3B-Instruct

# 使用 modelscope (国内推荐)
pip install modelscope
modelscope download --model qwen/Qwen2.5-3B-Instruct --local_dir /path/to/models/Qwen2.5-3B-Instruct
```

---

## 脚本参数化参考

所有核心脚本均支持通过环境变量控制行为，无需手动注释/取消注释代码。

### train_sft.sh — SFT 训练

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `MODEL_NAME` | `qwen2` | `qwen3` = Qwen3-4B-Thinking, `qwen2` = Qwen2.5-3B |
| `USE_MRAG` | `false` | 是否使用 MRAG 训练数据 |
| `MODEL` | (自动) | 直接覆盖模型路径 |
| `OUT_DIR` | (自动) | 直接覆盖输出目录 |

```bash
bash bash/train_sft.sh                                  # Qwen2.5, 无MRAG
MODEL_NAME=qwen3 bash bash/train_sft.sh                 # Qwen3, 无MRAG
MODEL_NAME=qwen3 USE_MRAG=true bash bash/train_sft.sh   # Qwen3, MRAG
MODEL_NAME=qwen2 USE_MRAG=true bash bash/train_sft.sh   # Qwen2.5, MRAG
```

### train_rl.sh — RL (GRPO) 训练

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `MODEL_NAME` | `qwen2` | `qwen3` 或 `qwen2` |
| `EXPERIMENT` | `sft_full` | `sft_full` = 全参GRPO, `sft_lora` = LoRA GRPO |
| `USE_MRAG` | `false` | 是否使用 MRAG 数据 |
| `USE_VLLM` | `false` | 是否使用外部 vLLM |

```bash
bash bash/train_rl.sh                                       # Qwen2.5, sft_full
MODEL_NAME=qwen3 bash bash/train_rl.sh                      # Qwen3, sft_full
MODEL_NAME=qwen3 USE_MRAG=true bash bash/train_rl.sh        # Qwen3, MRAG
MODEL_NAME=qwen2 EXPERIMENT=sft_lora bash bash/train_rl.sh  # Qwen2.5, LoRA
```

### loramerge.sh — SFT LoRA 合并

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `MERGE_CONFIG` | `sft_qwen3_mrag` | 选择预设配置（见下表） |

可选 `MERGE_CONFIG` 值: `sft_qwen3_mrag` / `sft_qwen3` / `sft_qwen2_mrag` / `sft_qwen2`

```bash
bash bash/loramerge.sh                                  # Qwen3 SFT MRAG
MERGE_CONFIG=sft_qwen2 bash bash/loramerge.sh           # Qwen2.5 SFT
MERGE_CONFIG=sft_qwen2_mrag bash bash/loramerge.sh      # Qwen2.5 SFT MRAG
```

> **Agent RL 模型合并**请使用专用脚本: `bash bash/agent/merge_agent_lora.sh [querygen|lawselect]`

### gen.sh — 推理

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `MODEL_NAME` | `qwen3,qwen2` | 逗号分隔: `qwen3`, `qwen2`, `all` |
| `MODES` | `rl` | 逗号分隔: `direct`, `icl`, `sft`, `rl`, `all` |
| `USE_MRAG` | `false` | 是否使用 MRAG 测试集 |

```bash
bash bash/gen.sh                                           # 所有模型的 RL
MODES=all bash bash/gen.sh                                 # 所有模型 × 所有模式
MODEL_NAME=qwen3 MODES=sft,rl bash bash/gen.sh             # Qwen3 的 SFT+RL
MODEL_NAME=qwen2 MODES=direct bash bash/gen.sh             # Qwen2.5 Direct
USE_MRAG=true MODES=all bash bash/gen.sh                   # MRAG 模式全量推理
```

### merge_agent_lora.sh — Agent RL LoRA 合并

```bash
bash bash/agent/merge_agent_lora.sh querygen                    # 自动找最新 checkpoint
bash bash/agent/merge_agent_lora.sh lawselect                   # 自动找最新 checkpoint
bash bash/agent/merge_agent_lora.sh querygen checkpoint-1000    # 指定 checkpoint
```

---

## 完整复现流程

以下是从 clone 项目到完成**全部实验（含所有 baseline 和消融实验）**的完整步骤。

### 阶段 A：环境与数据准备

#### Step 1: 克隆项目 & 安装环境

```bash
git clone https://github.com/xxx/JuDGE_RL.git
cd JuDGE_RL

# 创建三个 conda 环境
conda create -n swift python=3.10 -y
conda create -n judge python=3.10 -y
conda create -n vllm python=3.10 -y

# 安装依赖
conda activate swift && pip install -r requirements_swift.txt
conda activate judge && pip install -r requirements_judge.txt
conda activate vllm  && pip install -r requirements_vllm.txt
```

#### Step 2: 生成标准模式训练数据

标准模式数据不依赖检索结果，可以先生成。

```bash
conda activate swift

# 生成 SFT + RL 训练数据（标准模式）
bash bash/data_train.sh
```

产出文件:
- `data/train_sft.json` — SFT 训练集
- `data/test_sft.json` — SFT 测试集
- `data/rl_train/train.jsonl` — RL 训练集

---

### 阶段 B：检索模型训练

```bash
conda activate judge
```

#### Step 3: 训练 Dense Retriever

```bash
# K-Fold 训练（自动调用 gen_kfold_data.py 生成 qrels/queries 数据）
bash bash/retriever/kfold_train_retriever.sh
```

#### Step 4: 编码语料库 & 运行检索

```bash
# 编码法条/案例语料库为向量
bash bash/retriever/encode_corpus.sh

# 运行 Dense 检索
bash bash/retriever/retrieve.sh
```

#### Step 5: 训练 Reranker & 重排序

```bash
# K-Fold 训练 Reranker
bash bash/reranker/kfold_train_reranker.sh

# 运行重排序（Dense 结果 → Reranked 结果）
bash bash/reranker/run_reranker.sh
```

#### Step 6: 评测检索效果（消融：Dense only vs Dense+Reranker）

```bash
bash bash/retriever/eval_retriever.sh
```

产出: `mrag/retriever_output/eval_dense_results.txt`, `eval_reranked_results.txt`

#### Step 7: 生成 MRAG 模式训练数据

检索完成后，可以生成 MRAG 增强的训练数据。

```bash
conda activate swift
USE_MRAG=true bash bash/data_train.sh
```

产出文件:
- `data/train_sft_mrag.json` — MRAG SFT 训练集
- `data/test_sft_mrag.json` — MRAG SFT 测试集
- `data/rl_train_mrag/train.jsonl` — MRAG RL 训练集

---

### 阶段 C：Agent RL 训练（用于 Hybrid 检索和 Agent 消融实验）

```bash
conda activate swift
```

#### Step 8: 准备 Agent RL 训练数据

```bash
bash bash/agent/prepare_agent_rl_data.sh
```

#### Step 9: 训练 QueryGen RL

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 bash bash/agent/train_rl_querygen.sh
```

#### Step 10: 训练 LawSelect RL

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 bash bash/agent/train_rl_lawselect.sh
```

#### Step 11: 合并 Agent LoRA 权重

```bash
# 合并 QueryGen（自动查找最新 checkpoint）
bash bash/agent/merge_agent_lora.sh querygen

# 合并 LawSelect（自动查找最新 checkpoint）
bash bash/agent/merge_agent_lora.sh lawselect
```

#### Step 12: Agent 消融实验（9 种配置的检索评测）

```bash
CUDA_VISIBLE_DEVICES=0 bash bash/agent/eval_ablation.sh
```

产出: `result/agent_ablation/` 目录下各配置的检索指标

---

### 阶段 D：判决书生成模型训练

以下对 **Qwen3-4B-Thinking** 和 **Qwen2.5-3B-Instruct** 两个模型分别进行 SFT → LoRA 合并 → RL 三阶段训练。

```bash
conda activate swift
```

#### Step 13: Qwen3 训练全流程

```bash
# 13.1 SFT（无 MRAG）
MODEL_NAME=qwen3 bash bash/train_sft.sh
# 产出: output/sft_qwen3-4b_lora/

# 13.2 SFT（MRAG）
MODEL_NAME=qwen3 USE_MRAG=true bash bash/train_sft.sh
# 产出: output/sft_qwen3-4b_lora_mrag/

# 13.3 合并 LoRA
MERGE_CONFIG=sft_qwen3 bash bash/loramerge.sh
# 产出: output/sft_qwen3-4b_lora/merge/

MERGE_CONFIG=sft_qwen3_mrag bash bash/loramerge.sh
# 产出: output/sft_qwen3-4b_lora_mrag/merge/

# 13.4 RL（无 MRAG，基于 SFT 模型）
MODEL_NAME=qwen3 bash bash/train_rl.sh
# 产出: output/rl_qwen3-4b_grpo_sft_full/

# 13.5 RL（MRAG）
MODEL_NAME=qwen3 USE_MRAG=true bash bash/train_rl.sh
# 产出: output/rl_qwen3-4b_grpo_sft_full_mrag/
```

#### Step 14: Qwen2.5 训练全流程

```bash
# 14.1 SFT（无 MRAG）
MODEL_NAME=qwen2 bash bash/train_sft.sh
# 产出: output/sft_qwen2.5-3b_lora/

# 14.2 SFT（MRAG）
MODEL_NAME=qwen2 USE_MRAG=true bash bash/train_sft.sh
# 产出: output/sft_qwen2.5-3b_lora_mrag/

# 14.3 合并 LoRA
MERGE_CONFIG=sft_qwen2 bash bash/loramerge.sh
# 产出: output/sft_qwen2.5-3b_lora/merge/

MERGE_CONFIG=sft_qwen2_mrag bash bash/loramerge.sh
# 产出: output/sft_qwen2.5-3b_lora_mrag/merge/

# 14.4 RL（无 MRAG）
MODEL_NAME=qwen2 bash bash/train_rl.sh
# 产出: output/rl_qwen2.5-3b_grpo_sft_full/

# 14.5 RL（MRAG）
MODEL_NAME=qwen2 USE_MRAG=true bash bash/train_rl.sh
# 产出: output/rl_qwen2.5-3b_grpo_sft_full_mrag/
```

---

### 阶段 E：LegalOne-4B 基线实验

```bash
conda activate swift
```

#### Step 15: LegalOne-4B 训练

```bash
# 15.1 下载模型
bash bash/legalone/download_model.sh

# 15.2 SFT（无 MRAG）
bash bash/legalone/train_sft.sh

# 15.3 SFT（MRAG）
USE_MRAG=true bash bash/legalone/train_sft.sh

# 15.4 合并 LoRA
bash bash/legalone/loramerge.sh
USE_MRAG=true bash bash/legalone/loramerge.sh
```

---

### 阶段 F：推理（所有模型 × 所有模式）

```bash
conda activate swift   # 或 vllm 环境
```

#### Step 16: Qwen3 + Qwen2.5 推理

```bash
# 16.1 标准模式：所有模型 × 所有模式 (direct, icl, sft, rl)
MODES=all bash bash/gen.sh

# 16.2 MRAG 模式：所有模型 × sft,rl 模式
USE_MRAG=true MODES=sft,rl bash bash/gen.sh

# 也可以单独运行某个模型/模式:
# MODEL_NAME=qwen3 MODES=rl USE_MRAG=true bash bash/gen.sh
```

#### Step 17: LegalOne-4B 推理

```bash
# 17.1 标准模式 (direct, icl, sft)
bash bash/legalone/gen.sh

# 17.2 MRAG 模式
USE_MRAG=true bash bash/legalone/gen.sh
```

---

### 阶段 G：格式转换 & 评测

```bash
# Step 18: 格式转换
conda activate swift

# Qwen3 + Qwen2.5
bash bash/convert.sh
USE_MRAG=true bash bash/convert.sh

# LegalOne-4B
bash bash/legalone/convert.sh
USE_MRAG=true bash bash/legalone/convert.sh

# Step 19: 评测
conda activate judge

# Qwen3 + Qwen2.5
bash bash/eval.sh
USE_MRAG=true bash bash/eval.sh

# LegalOne-4B
bash bash/legalone/eval.sh
USE_MRAG=true bash bash/legalone/eval.sh

# Step 20: 查看结果
cat result/eval_summary.txt          # 标准模式结果
cat result/eval_summary_mrag.txt     # MRAG 模式结果
cat result/eval_legalone_summary.txt # LegalOne 结果
```

---

## 完整消融实验矩阵

完成上述流程后，可以得到以下消融实验对比：

### 消融 1：基座模型对比（Qwen2.5-3B vs Qwen3-4B-Thinking vs LegalOne-4B）

| 模型 | Direct | ICL | SFT | SFT+RL |
|------|--------|-----|-----|--------|
| Qwen2.5-3B | `qwen25_direct` | `qwen25_icl` | `qwen25_sft` | `qwen25_rl` |
| Qwen3-4B-Thinking | `qwen3_direct` | `qwen3_icl` | `qwen3_sft` | `qwen3_rl` |
| LegalOne-4B | `legalone_direct` | `legalone_icl` | `legalone_sft` | — |

### 消融 2：训练阶段对比（SFT only vs SFT+GRPO）

对比 `*_sft` 和 `*_rl` 的结果。

### 消融 3：检索增强对比（无检索 vs MRAG Hybrid）

对比标准模式 (`*_sft`, `*_rl`) 和 MRAG 模式 (`*_sft_mrag`, `*_rl_mrag`) 的结果。

### 消融 4：检索组件对比（Dense only vs Dense+Reranker）

Step 6 的 `eval_retriever.sh` 直接输出 Dense 和 Reranked 的 Recall@K、Precision@K 对比。

### 消融 5：Agent 组件 2×2 矩阵（±QueryGen RL, ±LawSelect RL）

Step 12 的 `eval_ablation.sh` 输出 9 种配置的检索指标（含 Dense only、MRAG、Agent 消融、Hybrid 融合）：

|               | LawSelect 基座 | LawSelect RL |
|---------------|----------------|--------------|
| **QG 基座**    | Baseline (#4)   | LS RL only (#6) |
| **QG RL**      | QG RL only (#5) | Both RL (#7) |

外加: MRAG only (#1), No QueryGen (#2), No LawSelect (#3)

### 消融 6：检索来源对比（MRAG only vs Agent only vs Hybrid）

使用 `eval_ablation.sh` 的 #1 (MRAG only) 结果、#7 (Agent Both RL) 结果，以及 `run_hybrid_agent.sh` 的 Hybrid 结果进行对比。

---

## 核心模块详解

### 1. MRAG 检索模块

**流程**: 案件事实 → Dense Retriever → Reranker → Top-K 法条

- Dense Retriever: `chinese-roberta-wwm` 微调，对比学习 + Hard Negatives
- Reranker: `chinese-roberta-wwm` 微调，Pairwise Ranking Loss

### 2. LLM-based Agent

**流程**: 案件事实 → QueryGen → Dense → Reranker → LawSelect → 最终法条

| 组件 | 模型 | 作用 |
|------|------|------|
| QueryGen | Qwen2.5-7B-Instruct | 生成 5-8 个检索查询 |
| LawSelect | Qwen2.5-7B-Instruct | 从候选中筛选 20-30 条法条 |

**RL 奖励函数**:
- QueryGen: `0.60 * Recall@50 + 0.25 * MRR + 0.15 * HitRatio`
- LawSelect (v2): `0.45 * Recall@5 + 0.35 * Precision@5 + 0.15 * Recall@10 + 0.05 * QuantityBonus`

### 3. 判决书生成

**模型**: Qwen3-4B-Thinking / Qwen2.5-3B-Instruct

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
| Recall@K | Top-K 中召回的正确法条比例 (K=5, 10) |
| Precision@K | Top-K 中正确法条占比 (K=5, 10) |

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

## LoRA 合并说明

本项目有两个 LoRA 合并脚本，各司其职：

| 脚本 | 适用范围 | 特点 |
|------|----------|------|
| `bash/loramerge.sh` | SFT 模型 (Qwen3/Qwen2.5) | 含 extract_lora 步骤（处理 DeepSpeed 输出） |
| `bash/agent/merge_agent_lora.sh` | Agent RL 模型 (querygen/lawselect) | 自动查找最新 checkpoint（处理 ms-swift 输出） |
| `bash/legalone/loramerge.sh` | LegalOne-4B SFT 模型 | LegalOne 专用 |

**不要混用**：SFT 模型用 `loramerge.sh`，Agent RL 模型用 `merge_agent_lora.sh`。

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

2. **使用 LoRA 训练代替全参数训练**
   ```bash
   EXPERIMENT=sft_lora bash bash/train_rl.sh
   ```

3. **减少 `max_completion_length`**
   ```bash
   --max_completion_length 2048  # 默认是 4096
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

### Q3: 检索效果差？

1. 检查 Hard Negatives 数量是否足够（建议 15-30 个）
2. 增加训练 epochs
3. 使用 K-Fold 交叉验证确保泛化性

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

### Q6: vLLM colocate 模式训练后出现 Segfault？

使用 vLLM colocate 模式进行 Agent RL 训练时，训练**成功完成**后可能出现 Segfault。**这是正常现象，不影响训练结果。**

原因是 vLLM 的 `CuMemAllocator` 与 PyTorch 内存释放的兼容性问题。

判断训练是否成功：检查日志中是否有 `percentage: '100.00%'` 和 `Saving model checkpoint`。

---

## 引用

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
