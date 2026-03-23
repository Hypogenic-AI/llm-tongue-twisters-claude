# Datasets for LLM Tongue Twisters Research

This directory contains datasets for studying patterns and tokens that language models struggle to reproduce accurately.

## Dataset 1: Glitch Token Ground Truth (GlitchProber)

### Overview
- **Source**: GlitchProber paper (arXiv:2408.04905)
- **Location**: `glitch_tokens_ground_truth/`
- **Format**: CSV (index, token)
- **Description**: Curated lists of confirmed glitch tokens per model

### Files
| File | Model | Token Count |
|------|-------|-------------|
| Llama-2-7b-chat-glitch-tokens.csv | Llama-2-7B-Chat | 6,425 |
| Mistral-7B-Instruct-v0.1-glitch-tokens.csv | Mistral-7B-Instruct | 2,779 |
| Qwen-7B-Chat-glitch-tokens.csv | Qwen-7B-Chat | 32,141 |
| Yi-6B-Chat-glitch-tokens.csv | Yi-6B-Chat | 8,107 |
| gemma-2b-it-glitch-tokens.csv | Gemma-2B-IT | 27,962 |

### Loading
```python
import pandas as pd
df = pd.read_csv("datasets/glitch_tokens_ground_truth/Llama-2-7b-chat-glitch-tokens.csv")
print(f"Found {len(df)} glitch tokens")
print(df.head())
```

## Dataset 2: Magikarp Token Verification Results

### Overview
- **Source**: Fishing for Magikarp paper (arXiv:2405.05417)
- **Location**: `magikarp_verifications/`
- **Format**: JSONL.GZ (compressed JSON lines)
- **Description**: Per-token verification results including embedding indicators, categories, and decoded forms

### Files
| File | Model | Description |
|------|-------|-------------|
| EleutherAI_gpt_j_6b.jsonl.gz | GPT-J 6B | 50,400 tokens with indicators |
| meta_llama_Llama_2_7b_hf.jsonl.gz | Llama-2-7B | Token verification results |
| google_gemma_7b.jsonl.gz | Gemma-7B | Token verification results |

### Full dataset
88 model verification files are available in `code/magikarp/results/verifications/`.

### Loading
```python
import gzip, json
tokens = []
with gzip.open("datasets/magikarp_verifications/EleutherAI_gpt_j_6b.jsonl.gz", "rt") as f:
    for line in f:
        tokens.append(json.loads(line))
# Each token has: i, raw_vocab, category, decoded, indicators, indicator_names
undertrained = [t for t in tokens if t["category"] == "UNDECODEABLE"]
```

### Schema
Each JSONL record contains:
- `i`: Token index in vocabulary
- `raw_vocab`: Raw token string
- `category`: "OK", "UNDECODEABLE", "OK_SPECIAL"
- `decoded`: Decoded token string
- `indicators`: List of embedding-based indicator values
- `indicator_names`: Names of indicator metrics

## Dataset 3: GPT-4o Problematic Tokens

### Overview
- **Source**: Problematic Tokens paper (arXiv:2406.11214)
- **Location**: `problematic_tokens_gpt4o/`
- **Format**: JSON
- **Description**: Analysis of problematic Chinese and Korean tokens in GPT-4o tokenizer

### Files
- `tiktoken.json`: Full GPT-4o tokenizer vocabulary analysis (~4.8MB)
- `gpt4o_results.json`: GPT-4o test results on problematic tokens (~533KB)
- `jieba_tiktoken.json`: Chinese word segmentation analysis
- `korean.json`: Korean problematic token analysis

### Loading
```python
import json
with open("datasets/problematic_tokens_gpt4o/gpt4o_results.json") as f:
    results = json.load(f)
```

## Notes

- Large JSONL.GZ files are excluded from git tracking
- CSV and JSON files are small enough to be tracked
- For full Magikarp verification data (88 models), see `code/magikarp/results/verifications/`
- For experiment design, the GlitchProber ground truth CSVs provide ready-to-use token lists
