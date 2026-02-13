# JuDGE_RL

**Retrieval-Grounded Reinforcement Learning for Judgment Document Generation**

## Abstract

Judgment document generation aims to draft a complete court judgment from a case fact description, requiring both accurate grounding in legal knowledge and multi-step legal reasoning. Prior retrieval-augmented generation (RAG) improves coverage of relevant statutes, but a fixed top-k retrieval strategy often introduces irrelevant articles, leading to over-citation and degraded legal faithfulness. To address this issue, we propose a retrieval-grounded framework that strengthens both evidence acquisition and long-form drafting. On the retrieval side, an LLM first plans multiple legally oriented queries based on the fact description. On the generation side, we apply retrieval-augmented supervised fine-tuning to stabilize document structure and employ multi-objective Group Relative Policy Optimization to jointly optimize legal correctness, writing quality, and reasoning-form compliance. Experiments on the JuDGE benchmark show consistent improvements over strong retrieval-augmented baselines, particularly in statute referencing precision and overall legal coherence.

## Overview

JuDGE_RL is an end-to-end system for automatic Chinese criminal judgment document generation. Given a case fact description, the system generates a complete judgment including factual findings, legal analysis, reasoning, and sentencing conclusions.

### Problem & Approach

Traditional RAG for judgment generation suffers from two issues:

1. **Low recall**: using only the case fact as query misses statutes related to sentencing circumstances, supplementary penalties, etc.
2. **Low precision**: fixed top-k retrieval introduces irrelevant statutes that mislead the generation model into over-citation.

Our framework addresses these through three layers:

| Layer | Method | Problem Solved |
|-------|--------|----------------|
| **Retrieval** | Dual-path retrieval (MRAG + LLM Agent) | Improve statute recall and precision |
| **SFT** | Retrieval-Augmented Supervised Fine-Tuning | Learn to use retrieved statutes to produce well-structured documents |
| **RL** | Multi-objective GRPO | Further optimize legal correctness, text quality, and reasoning form |

### Key Design Decisions

**Why dual-path retrieval?** MRAG (Dense Retriever + Reranker) achieves high recall, while the LLM Agent (QueryGen + LawSelect) achieves high precision. Fusion combines both strengths.

**Why GRPO over PPO/DPO?** GRPO (Group Relative Policy Optimization) does not require a separate critic model. It ranks multiple generations within the same batch. For long-form judgment generation, GRPO offers better memory efficiency and training stability than PPO.

**Why multi-objective reward?** Judgment quality cannot be measured by a single metric. Optimizing only text fluency (BERTScore) leads to legally inaccurate outputs; optimizing only statute F1 leads to outputs that list statutes without proper analysis. Multi-objective weighting forces the model to balance across dimensions.

## Architecture

```
Case Fact Description
    |
    +-------------------------------------+
    |                                     |
    v                                     v
+-------------------+     +-------------------------------+
|   MRAG Path       |     |    LLM Agent Path             |
|                   |     |                               |
| Fact -> Dense ->  |     | Fact -> QueryGen(LLM) ->      |
|   Reranker -> K   |     |   Dense -> Reranker ->        |
|                   |     |   LawSelect(LLM) -> refined   |
| Strength:         |     | Strength:                     |
|   High recall     |     |   High precision              |
+--------+----------+     +--------------+----------------+
         |                               |
         +---------------+---------------+
                         v
              Hybrid Fusion (RRF / Agent-First)
                         |
                         v
           Top-10 Statutes + Similar Cases
                         |
                         v
    +--------------------------------------+
    |     Judgment Generation Model        |
    |                                      |
    |  Stage 1: SFT (LoRA)                 |
    |    Supervised fine-tuning with       |
    |    reference judgments               |
    |                                      |
    |  Stage 2: GRPO (full-parameter)      |
    |    Reward = 0.60 * Legal Accuracy    |
    |           + 0.30 * Text Quality      |
    |           + 0.10 * Reasoning Form    |
    |                                      |
    |  Base: Qwen3-4B / Qwen2.5-3B         |
    +------------------+-------------------+
                       v
               Complete Judgment
    (Facts + Legal Analysis + Reasoning + Sentencing)
```

## Core Modules

### 1. MRAG Retrieval

Uses the case fact directly as query for two-stage retrieval:

```
Case Fact -> Dense Retriever (top-50) -> Reranker (top-10) -> Statutes
```

- **Dense Retriever**: Fine-tuned `chinese-roberta-wwm-ext` with contrastive learning + hard negatives, trained with K-Fold cross-validation.
- **Reranker**: Fine-tuned `chinese-roberta-wwm-ext` as cross-encoder with pairwise ranking loss.

### 2. LLM Agent Retrieval

LLM understands the case, plans multi-angle retrieval queries, and filters candidates:

```
Case Fact -> QueryGen (5-8 queries) -> Dense (top-50) -> Reranker (top-20) -> LawSelect (5-10 statutes)
```

| Component | Model | Role | RL Reward |
|-----------|-------|------|-----------|
| QueryGen | Qwen2.5-7B | Generate diverse legal queries | 0.05*Format + 0.25*Diversity + 0.70*DenseScore |
| LawSelect | Qwen2.5-7B | Filter truly relevant statutes | 0.45*R@5 + 0.35*P@5 + 0.15*R@10 + 0.05*Quantity |

**Why multiple queries?** A case involves multiple legal dimensions (crime definition, sentencing range, supplementary penalties, mitigating factors). A single query cannot cover all. For example, a theft case requires statutes for: theft elements (Art. 264), imprisonment range (Art. 45), fines (Art. 52/53), and voluntary confession (Art. 67).

**Why LawSelect?** Dense + Reranker only ranks by textual similarity. LLM can perform legal reasoning to distinguish "textually similar but inapplicable" statutes (e.g., distinguishing theft vs. robbery).

### 3. Hybrid Fusion

Two retrieval paths are merged via **output-level fusion**:

- **RRF (recommended)**: Reciprocal Rank Fusion weighted by source reliability.
- **Agent-First**: Keep all Agent outputs (filtered by LawSelect), supplement with MRAG results not covered by Agent.


### 4. Judgment Generation

**Stage 1 — SFT (LoRA)**:
- Fine-tune with reference judgments as labels.
- LoRA config: r=128, alpha=256, target modules: q/k/v/o_proj.
- Purpose: learn document structure and format.

**Stage 2 — GRPO (full-parameter)**:
- Further optimize on top of SFT model with multi-objective reward.
- Purpose: SFT learns "how to write", GRPO teaches "how to write correctly".

**Reward function** (`train/src/rl_plugin1.py`):

```
Total = 0.60 * Legal Accuracy + 0.30 * Text Quality + 0.10 * Reasoning Form

Legal Accuracy = 0.35 * Statute_F1      # Correct statute citations
               + 0.30 * Crime_F1        # Correct crime identification
               + 0.20 * Prison_Match    # Reasonable prison term
               + 0.15 * Fine_Match      # Reasonable fine amount

Text Quality   = BERTScore              # Computed separately for reasoning and sentencing sections

Reasoning Form = <think> format + length + no repetition   # Only for Thinking models
```

**Why Statute F1 has the highest weight (35%)?** Statute citation is the legal foundation of a judgment. Incorrect citations invalidate the judgment entirely. In contrast, minor variations in prison terms or fine amounts within a reasonable range are acceptable.

## Data Flow

```
data/train.json (raw: text_id + case facts + reference judgment)
    |
    +-- script/sft_data.py --> data/train_sft.json       (SFT: prompt + reference)
    +-- script/sft_data.py --> data/test_sft.json        (Inference: prompt only)
    +-- script/rl_data.py  --> data/rl_train/train.jsonl (RL: messages + reference)

# With retrieval results, MRAG-augmented versions are generated:
data/train.json + retrieval --> data/train_sft_mrag.json
                             --> data/test_sft_mrag.json
                             --> data/rl_train_mrag/train.jsonl
```

**Prompt consistency**: `sft_data.py`, `rl_data.py`, and `inf.py` share identical prompt templates, ensuring no distribution shift between training and inference.

## Directory Structure

```
JuDGE_RL/
├── bash/                           # Shell scripts
│   ├── agent/                      # LLM Agent
│   ├── retriever/                  # Dense Retriever
│   ├── reranker/                   # Reranker
│   ├── data_train.sh               # Generate SFT/RL training data
│   ├── train_sft.sh                # SFT training
│   ├── train_rl.sh                 # RL (GRPO) training
│   ├── loramerge.sh                # SFT LoRA merge
│   ├── gen.sh                      # Inference (9 modes)
│   ├── convert.sh                  # Format conversion
│   └── eval.sh                     # Evaluation
├── data/                           # Data files (included in repo)
├── evaluation/                     # Evaluation scripts
├── mrag/                           # Retrieval modules
│   └── agent/                      # LLM Agent (QueryGen, LawSelect, Fusion)
├── reranker/                       # Reranker module
├── train/                          # Training & inference
│   ├── src/                        # SFT training, RL reward functions
│   └── deploy/                     # vLLM inference, LoRA merge
└── script/                         # Data generation scripts
```

## Setup

### Requirements

Three conda environments are needed:

| Environment | Purpose |
|-------------|---------|
| `swift` | SFT/RL training, inference (based on ms-swift) |
| `judge` | Retriever/Reranker training, evaluation |
| `vllm` | Inference acceleration (optional) |

```bash
conda create -n swift python=3.10 -y && conda activate swift && pip install -r requirements_swift.txt
conda create -n judge python=3.10 -y && conda activate judge && pip install -r requirements_judge.txt
conda create -n vllm  python=3.10 -y && conda activate vllm  && pip install -r requirements_vllm.txt
```

### Models

| Model | Purpose | Link |
|-------|---------|------|
| Qwen2.5-3B-Instruct | Generation base model | [HuggingFace](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct) |
| chinese-roberta-wwm-ext | Retriever / Reranker | [HuggingFace](https://huggingface.co/hfl/chinese-roberta-wwm-ext) |
| Qwen3-4B | Thinking model experiments | [HuggingFace](https://huggingface.co/Qwen/Qwen3-4B) |
| Qwen2.5-7B-Instruct | Agent (QueryGen/LawSelect) | [HuggingFace](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct) |

### **>>> Path Configuration (IMPORTANT — Read Before Running) <<<**

**All model paths are centralized in a single file: `bash/paths.sh`.** Before running any script, you **must** edit this file to point to your local model directories:

```bash
# bash/paths.sh — edit these paths to match your environment

export QWEN3_MODEL_PATH="/path/to/Qwen3-4B"
export QWEN25_MODEL_PATH="/path/to/Qwen2.5-3B-Instruct"
export QWEN25_7B_MODEL_PATH="/path/to/Qwen2.5-7B-Instruct"
export ROBERTA_MODEL_PATH="/path/to/chinese-roberta-wwm-ext"
export BERT_MODEL_PATH="/path/to/bert-base-chinese"
```

Every shell script automatically sources `bash/paths.sh` and validates that the required model directory exists before proceeding. **If a path is wrong, the script will print a clear error message and exit.**

You can also override any path via environment variable without editing the file:

```bash
QWEN3_MODEL_PATH=/my/models/Qwen3-4B bash bash/train_sft.sh
```

### Trained Model Paths in gen.sh

`bash/paths.sh` manages paths for **downloaded base models** only. In addition, `bash/gen.sh` contains paths for **trained model outputs** — SFT merged models and RL checkpoints — which point to directories under `output/` created by the training scripts.

**SFT model paths** default to `output/sft_*/merge` and typically do not need manual adjustment, because the merge script (`loramerge.sh`) always writes to the same `merge/` subdirectory:

```bash
# gen.sh — SFT paths (usually no changes needed)
SFT_QWEN3="${SFT_QWEN3:-output/sft_qwen3-4b_lora/merge}"
SFT_MRAG_QWEN3="${SFT_MRAG_QWEN3:-output/sft_qwen3-4b_lora_mrag/merge}"
```

**RL checkpoint paths require manual configuration.** RL training (ms-swift GRPO) generates output directories with version timestamps, for example:

```
output/rl_qwen3-4b_grpo_sft_full/v19-20260116-061030/checkpoint-501
```

The path must point to a **specific checkpoint directory** with full model weights. The default values in `gen.sh` will NOT work unless your training happens to produce the exact same version directory names.

**How to find your checkpoint path** — after RL training completes, inspect the output:

```bash
ls output/rl_qwen3-4b_grpo_sft_full/
# v19-20260116-061030/

ls output/rl_qwen3-4b_grpo_sft_full/v19-20260116-061030/
# checkpoint-167/  checkpoint-334/  checkpoint-501/
```

Pick the final (or best) checkpoint and set the full path in `gen.sh`.

**Configuration methods** — edit `gen.sh` directly or override via environment variables:

```bash
# Method 1: Edit gen.sh directly
RL_SFT_QWEN3_PATH="${RL_SFT_QWEN3_PATH:-output/rl_qwen3-4b_grpo_sft_full/v19-20260116-061030/checkpoint-501}"

# Method 2: Override via environment variable at runtime
RL_SFT_QWEN3_PATH=output/rl_qwen3-4b_grpo_sft_full/v19-20260116-061030/checkpoint-501 \
RL_SFT_QWEN25_PATH=output/rl_qwen2.5-3b_grpo_sft_full/v17-20260117-091241/checkpoint-501 \
  MODES=sft_rl bash bash/gen.sh
```

**All RL path variables in gen.sh:**

| Variable | Description | Example Value |
|----------|-------------|---------------|
| `RL_BASE_QWEN3_PATH` | Qwen3 Base→RL | `output/rl_qwen3-4b_grpo_full/<version>/checkpoint-<step>` |
| `RL_BASE_QWEN25_PATH` | Qwen2.5 Base→RL | `output/rl_qwen2.5-3b_grpo_full/<version>/checkpoint-<step>` |
| `RL_SFT_QWEN3_PATH` | Qwen3 SFT→RL | `output/rl_qwen3-4b_grpo_sft_full/<version>/checkpoint-<step>` |
| `RL_SFT_QWEN25_PATH` | Qwen2.5 SFT→RL | `output/rl_qwen2.5-3b_grpo_sft_full/<version>/checkpoint-<step>` |
| `RL_BASE_MRAG_QWEN3_PATH` | Qwen3 Base→RL + MRAG | `output/rl_qwen3-4b_grpo_mrag_full/<version>/checkpoint-<step>` |
| `RL_BASE_MRAG_QWEN25_PATH` | Qwen2.5 Base→RL + MRAG | `output/rl_qwen2.5-3b_grpo_mrag_full/<version>/checkpoint-<step>` |
| `RL_SFT_MRAG_QWEN3_PATH` | Qwen3 SFT+MRAG→RL | `output/rl_qwen3-4b_grpo_sft_mrag_full/<version>/checkpoint-<step>` |
| `RL_SFT_MRAG_QWEN25_PATH` | Qwen2.5 SFT+MRAG→RL | `output/rl_qwen2.5-3b_grpo_sft_mrag_full/<version>/checkpoint-<step>` |

## Quick Start: Evaluation Only (No Training Required)

If you only want to **reproduce the main experiment results** without training, follow these steps. The pre-trained model checkpoint is available on Google Drive.

### Prerequisites

| Item | How to Get |
|------|------------|
| Model checkpoint (`JuDGE_RL.tar.gz`) | [Download from Google Drive](https://drive.google.com/file/d/1lquq4EePHRQWE8wOWdsFwEUpNzyiZolx/view?usp=sharing) (~8GB, SFT+MRAG+RL trained Qwen3-4B) |
| `bert-base-chinese` | Download from [HuggingFace](https://huggingface.co/google-bert/bert-base-chinese) (required for BERTScore in evaluation) |
| GPU | 1x GPU with >= 16GB VRAM (inference only) |

### Step 1: Environment Setup

Only two environments are needed for evaluation:

```bash
# For inference (vLLM)
conda create -n vllm python=3.10 -y && conda activate vllm
pip install -r requirements_vllm.txt

# For evaluation metrics (BERTScore, METEOR)
conda create -n judge python=3.10 -y && conda activate judge
pip install -r requirements_judge.txt
```

### Step 2: Download and Extract the Model Checkpoint

Download `JuDGE_RL.tar.gz` from Google Drive and extract it:

```bash
# Method 1: Download via browser, then extract
tar -xzf JuDGE_RL.tar.gz -C .

# Method 2: Download via gdown (pip install gdown)
gdown 1lquq4EePHRQWE8wOWdsFwEUpNzyiZolx
tar -xzf JuDGE_RL.tar.gz -C .
```

After extraction, check the model directory path and note it for the next step:

```bash

# Verify the directory contains model files
ls JuDGE_R1/release_model   # or your extracted directory name
# config.json  model-00001-of-00002.safetensors  model-00002-of-00002.safetensors
# model.safetensors.index.json  tokenizer.json  added_tokens.json  
#merges.txt  special_tokens_map.json  chat_template.jinja
#vocab.json  generation_config.json
```
### Step 3: Run Inference

```bash
cd Judge-R1
mkdir -p outputs
conda activate vllm
export CUDA_VISIBLE_DEVICES=0

# Main experiment: SFT+MRAG+RL model on MRAG test set
# Replace <MODEL_PATH> with your extracted model directory (e.g., checkpoint-501)
python train/deploy/inf.py \
    --model_path <MODEL_PATH> \
    --dataset_path data/test_sft_mrag.json \
    --output_path outputs/qwen3_sft_mrag_rl_raw.json \
    --mode rl \
    --tensor_parallel_size 1 \
    --gpu_memory_utilization 0.85
```

### Step 4: Convert Output Format

```bash
# Convert inference output to evaluation format
python -c "
import json
fd2id = {}
with open('data/test.json', 'r') as f:
    for line in f:
        obj = json.loads(line)
        fd2id[obj['fd']] = obj['text_id']

data = json.load(open('outputs/qwen3_sft_mrag_rl_raw.json', 'r'))
with open('outputs/qwen3_sft_mrag_rl.jsonl', 'w') as out:
    for item in data:
        cid = item.get('text_id') or fd2id.get(item.get('exp_ans'))
        gen = item.get('gen_ans')
        if cid and gen is not None:
            out.write(json.dumps({'id': cid, 'document': gen}, ensure_ascii=False) + '\n')
print('Converted to outputs/qwen3_sft_mrag_rl.jsonl')
"
```

### Step 5: Evaluate

```bash
conda activate judge
export BERT_MODEL_PATH="/path/to/bert-base-chinese"   # Required for BERTScore

cd evaluation

# Legal accuracy (Crime F1, Law Article F1, Prison Score, Fine Score)
python calc.py \
    --gen_file ../outputs/qwen3_sft_mrag_rl.jsonl \
    --exp_file ../data/expected.jsonl

# Text quality (METEOR, BERTScore)
python calc_rel.py \
    --gen_file ../outputs/qwen3_sft_mrag_rl.jsonl \
    --exp_file ../data/expected.jsonl
```

### Pre-computed Retrieval Results

The MRAG test set (`data/test_sft_mrag.json`) already contains pre-computed retrieval results embedded in each prompt (top-10 statutes + similar cases). No retrieval models are needed for evaluation.

To inspect the retrieval quality and explainability:

- **`mrag/retriever_output/ablation_both_rl_eval.txt`** — Retrieval evaluation metrics (Recall@K, MRR, etc.)
- **`mrag/retriever_output/ablation_both_rl_details.json`** — Per-case explainability: generated queries, selected statutes with reasons, rejected statutes with reasons

---

## Full Reproduction (Training from Scratch)

### Phase A: Data Preparation

```bash
conda activate swift
bash bash/data_train.sh                # Standard mode
```

### Phase B: Retrieval Model Training

```bash
conda activate judge
bash bash/retriever/kfold_train_retriever.sh   # Dense Retriever
bash bash/retriever/encode_corpus.sh           # Encode corpus
bash bash/retriever/retrieve.sh                # Run retrieval
bash bash/reranker/kfold_train_reranker.sh     # Reranker
bash bash/reranker/run_reranker.sh             # Run reranking
bash bash/retriever/eval_retriever.sh          # Evaluate retrieval

# Generate MRAG training data
conda activate swift
USE_MRAG=true bash bash/data_train.sh
```

### Phase C: Agent RL Training

```bash
conda activate swift
bash bash/agent/prepare_agent_rl_data.sh
CUDA_VISIBLE_DEVICES=0,1,2,3 bash bash/agent/train_rl_querygen.sh
CUDA_VISIBLE_DEVICES=0,1,2,3 bash bash/agent/train_rl_lawselect.sh
bash bash/agent/merge_agent_lora.sh querygen
bash bash/agent/merge_agent_lora.sh lawselect
CUDA_VISIBLE_DEVICES=0 bash bash/agent/eval_ablation.sh
```

### Phase D: Generation Model Training

```bash
conda activate swift

# Qwen3-4B
MODEL_NAME=qwen3 bash bash/train_sft.sh
MODEL_NAME=qwen3 USE_MRAG=true bash bash/train_sft.sh
MERGE_CONFIG=sft_qwen3 bash bash/loramerge.sh
MERGE_CONFIG=sft_qwen3_mrag bash bash/loramerge.sh
MODEL_NAME=qwen3 bash bash/train_rl.sh
MODEL_NAME=qwen3 USE_MRAG=true bash bash/train_rl.sh

# Qwen2.5-3B
MODEL_NAME=qwen2 bash bash/train_sft.sh
MODEL_NAME=qwen2 USE_MRAG=true bash bash/train_sft.sh
MERGE_CONFIG=sft_qwen2 bash bash/loramerge.sh
MERGE_CONFIG=sft_qwen2_mrag bash bash/loramerge.sh
MODEL_NAME=qwen2 bash bash/train_rl.sh
MODEL_NAME=qwen2 USE_MRAG=true bash bash/train_rl.sh
```

### Phase E: Inference

```bash
conda activate swift
MODES=all bash bash/gen.sh      # All models x all 9 modes
```

**9 inference modes:**

| Mode | Model | Data | Description |
|------|-------|------|-------------|
| `direct` | Base | Raw | Zero-shot |
| `icl` | Base | Raw | Few-shot |
| `sft` | SFT | Standard | Supervised fine-tuned |
| `mrag` | Base | MRAG | Base + retrieval |
| `rl` | Base->RL | Standard | RL only |
| `sft_mrag` | SFT+MRAG | MRAG | SFT + retrieval |
| `sft_rl` | SFT->RL | Standard | SFT + RL |
| `mrag_rl` | Base->RL | MRAG | RL + retrieval |
| `sft_mrag_rl` | SFT+MRAG->RL | MRAG | **Full pipeline (best)** |

### Phase F: Evaluation

```bash
conda activate swift
bash bash/convert.sh

conda activate judge
bash bash/eval.sh

cat result/eval_summary.txt
```

## Script Parameters

All scripts are controlled via environment variables.

### train_sft.sh

| Variable | Default | Options |
|----------|---------|---------|
| `MODEL_NAME` | `qwen2` | `qwen3` (Qwen3-4B), `qwen2` (Qwen2.5-3B) |
| `USE_MRAG` | `false` | `true` to use MRAG training data |

### train_rl.sh

| Variable | Default | Options |
|----------|---------|---------|
| `MODEL_NAME` | `qwen3` | `qwen3` or `qwen2` |
| `EXPERIMENT` | `sft_full` | `sft_full` (GRPO on SFT model), `base_full` (GRPO on base model) |
| `USE_MRAG` | `false` | `true` to use MRAG data |
| `USE_VLLM` | `false` | `true` to use external vLLM server |

### gen.sh

| Variable | Default | Options |
|----------|---------|---------|
| `MODEL_NAME` | `qwen3,qwen2` | Comma-separated model names |
| `MODES` | `all` | Comma-separated from the 9 modes above |

## Evaluation Metrics

### Legal Accuracy (`evaluation/calc.py`)

| Metric | Description |
|--------|-------------|
| Crime F1 | Crime identification F1 score |
| Law Article F1 | Statute citation F1 score |
| Prison Time Score | Sentence term matching (closer = better) |
| Fine Amount Score | Fine amount matching (closer = better) |

### Text Quality (`evaluation/calc_rel.py`)

| Metric | Description |
|--------|-------------|
| METEOR | Text similarity (segment-level: reasoning + sentencing) |
| BERTScore | Semantic similarity (segment-level: reasoning + sentencing) |

Evaluation first segments the judgment into "reasoning" and "sentencing" sections via `evaluation/segment/`, then computes metrics for each section separately.

## Ablation Experiments

| Experiment | Comparison | Output Files |
|------------|-----------|--------------|
| Base model | Qwen2.5 vs Qwen3 | `qwen25_*` / `qwen3_*` |
| Training stage | Direct -> ICL -> SFT -> SFT+RL | `*_direct` / `*_icl` / `*_sft` / `*_sft_rl` |
| Retrieval augmentation | w/o retrieval vs MRAG | `*_sft` vs `*_sft_mrag` |
| Retrieval + RL | SFT+RL vs SFT+MRAG+RL | `*_sft_rl` vs `*_sft_mrag_rl` |
| Retrieval components | Dense only vs Dense+Reranker | `eval_retriever.sh` output |
| Agent components | +/-QueryGen RL x +/-LawSelect RL | `eval_ablation.sh` output |
| Retrieval source | MRAG vs Agent vs Hybrid | `eval_ablation.sh` + `fuse_results.py` output |

## LoRA Merge Guide

| Script | Scope | Notes |
|--------|-------|-------|
| `bash/loramerge.sh` | SFT models (Qwen3/Qwen2.5) | Includes extract_lora step for DeepSpeed |
| `bash/agent/merge_agent_lora.sh` | Agent RL models | Auto-finds latest checkpoint (ms-swift) |

**Do not mix**: use `loramerge.sh` for SFT models, `merge_agent_lora.sh` for Agent RL models.


## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
