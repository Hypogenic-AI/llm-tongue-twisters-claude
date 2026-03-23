# Cloned Repositories

## Repo 1: magikarp (Cohere AI)
- **URL**: https://github.com/cohere-ai/magikarp
- **Purpose**: Automated detection of under-trained tokens in LLMs
- **Location**: `code/magikarp/`
- **Key files**:
  - `magikarp/` - Core detection library
  - `generate_results.py` - Run verification on any HuggingFace model
  - `results/verifications/` - Pre-computed results for 88 models (JSONL.GZ)
  - `results/reports/` - Human-readable reports per model
  - `results/summary.md` - Overview of all models
- **Dependencies**: Python, PyTorch, Transformers (managed via Poetry)
- **Notes**: Most useful repo - contains both detection code AND extensive pre-computed results

## Repo 2: GlitchMiner
- **URL**: https://github.com/wooozihui/GlitchMiner
- **Purpose**: Gradient-based discrete optimization for glitch token detection
- **Location**: `code/GlitchMiner/`
- **Key files**:
  - `glitchminer/` - Core mining framework
  - `baseline_code/` - Baseline comparison implementations
- **Dependencies**: PyTorch, Transformers
- **Notes**: Entropy-based loss function approach; 19% precision improvement over prior methods

## Repo 3: GlitchProber
- **URL**: https://github.com/LLM-Integrity-Guard/GlitchProber
- **Purpose**: Detection AND mitigation of glitch tokens via PCA + SVM
- **Location**: `code/GlitchProber/`
- **Key files**:
  - `GlitchProber.py` - Main detection/mitigation script
  - `GroundTruth/` - CSV files of confirmed glitch tokens for 5 models
  - `Tutorials.ipynb` - Usage tutorial
- **Dependencies**: See `requirements.txt`
- **Notes**: Ground truth CSVs are valuable as ready-to-use evaluation datasets

## Repo 4: LLMGPT4o (Problematic Tokens)
- **URL**: https://github.com/yeyimilk/LLMGPT4o
- **Purpose**: Analysis of tokenizer bias in GPT-4o, especially for Chinese/Korean
- **Location**: `code/LLMGPT4o/`
- **Key files**:
  - `src/handle_tiktoken.py` - Tokenizer analysis utilities
  - `src/data/tiktoken.json` - Full GPT-4o vocab analysis
  - `src/data/gpt4o_results.json` - Test results
  - `src/sentence_check.py` - Sentence-level evaluation
- **Dependencies**: tiktoken, openai
- **Notes**: Focused on multilingual tokenizer bias; good for cross-lingual experiments
