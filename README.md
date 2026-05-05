# SAD-RAG

**Stance-Aware Defense via Graph-Guided Collusion Mining and Adversarial Probing for Web Search RAG**

## Overview

**SAD-RAG** is a post-retrieval defense framework for web-search Retrieval-Augmented Generation (RAG) systems.

### Architecture

```
Query + Retrieved Documents
          │
          ▼
┌─────────────────────┐
│ Evidence Extraction │  ← Filter noise documents
└─────────┬───────────┘
          ▼
┌─────────────────────┐
│  Conflict Detection │  ← Identify contradictory clusters
└─────────┬───────────┘
          ▼
┌─────────────────────┐
│ Graph Mining Filter │  ← Stance-cluster-induced subgraph density analysis
└─────────┬───────────┘
          ▼
┌─────────────────────┐
│   Debate & Judge    │  ← Adversarial probing for stance determination
└─────────┬───────────┘
          ▼
┌─────────────────────┐
│  Answer Generation  │  ← Verified knowledge-based response
└─────────────────────┘
```

---

## Dataset: GEO-RAGBench

> **Note:** The dataset files are located in the `data/` directory at the project root. Make sure the data files are in place before running inference.

We construct **GEO-RAGBench**, a benchmark for evaluating post-retrieval defenses in web-search RAG. It contains two subsets:

| Subset | Attack Type | Samples | Description |
|--------|------------|---------|-------------|
| **GEO-RAGBench-Poison** | Corpus Poisoning | 542 | Targets corpus poisoning attacks where malicious documents are injected into the search results |
| **GEO-RAGBench-Injection** | Prompt Injection | 606 | Targets prompt injection attacks where malicious instructions are injected in retrieved pages |

### Data Files (`data/` directory)

```
data/
├── data_poisoning_dataset_ratio_20.jsonl    # 20% poisoning ratio
├── data_poisoning_dataset_ratio_40.jsonl    # 40% poisoning ratio
├── data_poisoning_dataset_ratio_60.jsonl    # 60% poisoning ratio
├── data_poisoning_dataset_ratio_80.jsonl    # 80% poisoning ratio
├── data_poisoning_dataset_ratio_90.jsonl    # 90% poisoning ratio
├── prompt_injection_dataset_ratio_20.jsonl  # 20% poisoning ratio
├── prompt_injection_dataset_ratio_40.jsonl  # 40% poisoning ratio
├── prompt_injection_dataset_ratio_60.jsonl  # 60% poisoning ratio
├── prompt_injection_dataset_ratio_80.jsonl  # 80% poisoning ratio
└── prompt_injection_dataset_ratio_90.jsonl  # 90% poisoning ratio
```

### Poisoning Ratio

Each subset is divided into **five poisoning ratios**: 20%, 40%, 60%, 80%, and 90%. The poisoning ratio denotes the proportion of malicious documents in the retrieved set.

### Data Format (JSONL)

Each line is a JSON object with the following structure:

```json
{
  "query": "...",
  "sample_id": "uuid",
  "request_id": "uuid",
  "attack_goal": "High-level attack category",
  "specific_objective": "Specific content the attacker wants the model to produce",
  "answer": "Standard (correct) answer",
  "score_points": ["point 1", "point 2", "..."],
  "poisoned_docs": [
    {"url": "...", "title": "...", "content": "...", "score": 0.95}
  ],
  "benign_docs": [
    {"url": "...", "title": "...", "content": "...", "score": 0.90}
  ],
  "noise_docs": [
    {"url": "...", "title": "...", "content": "...", "score": 0.50}
  ]
}
```

## Environment Setup

```bash
# Create and activate conda environment
conda env create -f environment.yml
conda activate sadrag
```

---

## API Configuration

The project uses OpenAI-compatible API endpoints for model inference. You need to configure the `.env` files.

### Main Inference `.env` (project root)

Edit the `.env` file in the project root directory:

```bash
# .env (project root)
MODEL_PROVIDER=vllm           # or "openai", "siliconflow", etc.
MODEL_ID=Qwen/Qwen3-8B       # Model identifier
OPENAI_API_BASE=http://localhost:6006/v1   # API base URL
OPENAI_API_KEY=your-api-key-here           # API key (set any value if using local vLLM)
```

### Evaluation `.env`

Edit `argumentative_truth_discovery/evaluation/.env`:

```bash
# argumentative_truth_discovery/evaluation/.env
MODEL_PROVIDER=
MODEL_ID=qwen/qwen3-235b-a22b-2507       # Evaluation judge model
OPENAI_API_BASE=your-api-base-url         # e.g., OpenRouter or other compatible endpoint
OPENAI_API_KEY=your-api-key-here
```

> **Note:** Replace the placeholder values with your actual API endpoint and key. If you are using a locally deployed vLLM server, the `OPENAI_API_KEY` can be set to any non-empty string (e.g., `EMPTY`).

---

## Model Serving with vLLM

We use [vLLM](https://docs.vllm.ai/) to serve models locally with an OpenAI-compatible API. This project has been tested with three models:

| Model | Parameters | Notes |
|-------|-----------|-------|
| **Qwen3-8B** | 8B | Default model, standard inference |
| **Qwen3.5-9B** | 9B | Requires `enable_thinking=False` |
| **GPT-OSS-20B** | 20B | Requires `reasoning_effort="low"` |

### 1. Install vLLM

We recommend using [uv](https://docs.astral.sh/uv/) for fast Python package management:

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install vLLM
uv venv --python 3.12 --seed --managed-python
source .venv/bin/activate
uv pip install vllm --torch-backend=auto
```

### 2. Download Models

Download models from Hugging Face or ModelScope.

```bash
# Install modelscope
pip install modelscope

# Download models
# Qwen3-8B
modelscope download --model Qwen/Qwen3-8B

# Qwen3.5-9B
modelscope download --model Qwen/Qwen3.5-9B

# GPT-OSS-20B
modelscope download --model openai-mirror/gpt-oss-20b
```

### 3. Launch vLLM Server

Adjust `--tensor-parallel-size` based on the number of GPUs available on your machine:

```bash
# Launch Qwen3-8B
VLLM_USE_MODELSCOPE=true vllm serve Qwen/Qwen3-8B \
    --tensor-parallel-size 2 \
    --reasoning-parser qwen3 \
    --enable-log-requests \
    --port 6006

# Launch Qwen3.5-9B
VLLM_USE_MODELSCOPE=true vllm serve Qwen/Qwen3.5-9B \
    --tensor-parallel-size 2 \
    --reasoning-parser qwen3 \
    --enable-log-requests \
    --port 6006

# Launch GPT-OSS-20B
VLLM_USE_MODELSCOPE=true vllm serve openai-mirror/gpt-oss-20b \
    --tensor-parallel-size 2 \
    --enable-log-requests \
    --port 6006
```

> **Note:** The number of GPUs (`--tensor-parallel-size`) depends on your server configuration and does not affect inference results. Common configurations: `2` for 2× GPUs, `4` for 4× GPUs, etc.

---

## Running Inference

Each script automatically configures the `.env` file and processes all 10 datasets:

```bash
# Make scripts executable (first time only)
chmod +x run_qwen3_8b.sh run_qwen3.5_9b.sh run_gpt_oss_20b.sh run_eval.sh

# Run inference with Qwen3-8B
bash run_qwen3_8b.sh

# Run inference with Qwen3.5-9B
bash run_qwen3.5_9b.sh

# Run inference with GPT-OSS-20B
bash run_gpt_oss_20b.sh
```

Each script will:
1. Configure the model in `.env`
2. Set the appropriate model-specific parameters (e.g., `enable_thinking=False` for Qwen3.5)
3. Process all 10 dataset files (both poisoning and injection subsets at all 5 ratios)
4. Save results to `argumentative_truth_discovery/record/{timestamp}/`

**Available parameters in scripts:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `TAU` | `1` | MAD-based anomaly detection threshold |
| `OPENAI_API_BASE` | `http://localhost:6006/v1` | vLLM server URL |

The script processes each line of the JSONL dataset, runs the full SAD-RAG pipeline, and outputs results in JSONL format to the `record/` directory.

---

## Running Evaluation

The evaluation uses **Qwen3-235B-A22B** (`qwen/qwen3-235b-a22b-2507`) as the judge model to assess both **Attack Success Rate (ASR)** and **Answer Accuracy (ACC)**.

### Configure the Evaluation Model

Edit `argumentative_truth_discovery/evaluation/.env` with your API credentials:

```bash
MODEL_ID=qwen/qwen3-235b-a22b-2507
OPENAI_API_BASE=your-api-base-url
OPENAI_API_KEY=your-api-key
```

### Run Evaluation

```bash
# Using the evaluation script
bash run_eval.sh
```

### Evaluation Metrics

| Metric | Full Name | Description |
|--------|-----------|-------------|
| **ASR** | Attack Success Rate | The proportion of samples where the attack succeeded (the model output contains the attacker's target content). Lower ASR means better defense. |
| **ACC** | Answer Accuracy | The proportion of score points in the standard answer that the model correctly covers. Higher ACC means better answer quality. |

### Evaluation Output

The evaluation script produces:
- Per-sample evaluation results (JSONL) with attack success analysis and accuracy scores
- Summary metrics (ASR and ACC) appended as the last line

---

## Acknowledgments

Part of this implementation is inspired by and references the following projects:

- [ReliabilityRAG](https://github.com/zeyushen-yo/ReliabilityRAG)
- [TrustRAG](https://github.com/gomate-community/TrustRAG)

We gratefully acknowledge their contributions to the community.

