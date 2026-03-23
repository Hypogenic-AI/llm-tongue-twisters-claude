# Do Language Models Have Tongue Twisters?

Systematic investigation of strings and patterns that language models struggle to reproduce faithfully when asked to repeat them.

## Key Findings

- **Glitch tokens are solved**: Famous "tongue twister" tokens like `SolidGoldMagikarp` are perfectly reproduced by all 2025-era models tested (GPT-4.1, GPT-4.1-mini, GPT-4o-mini)
- **Models preserve misspellings**: Contrary to expectations, models faithfully reproduce misspelled text without auto-correcting, even at 50% word error density
- **Length is the new tongue twister**: All models fail at extreme text lengths (10k+ chars), especially with repetitive content. gpt-4.1-mini hallucinated output 5x the input length at 20k repetitive chars
- **Paradoxical clean/misspelled asymmetry**: GPT-4.1 refuses to reproduce long clean text (8k+ chars) but perfectly reproduces 25k chars of misspelled text, suggesting content-aware output filtering
- **Unicode handling is robust**: Zero-width characters, homoglyphs, combining marks, Braille, and mixed scripts are all faithfully preserved

## Project Structure

```
├── REPORT.md              # Full research report with methodology and results
├── planning.md            # Research plan and hypothesis decomposition
├── literature_review.md   # Literature review of glitch token research
├── resources.md           # Catalog of datasets, papers, and code
├── src/
│   ├── build_stimuli.py   # Constructs test stimulus sets
│   ├── run_experiments.py # V1: Token reproduction (glitch/control/adversarial)
│   ├── run_experiments_v2.py # V2: Long docs, embedded glitch tokens
│   ├── run_experiments_v3.py # V3: Cross-model comparison
│   ├── run_experiments_v4.py # V4: Misspelling density, consistency
│   ├── run_experiments_v5.py # V5: Length scaling, random strings
│   ├── run_experiments_v6_final.py # V6: Extreme length (8k-25k chars)
│   └── analyze_results.py # Analysis and visualization
├── results/
│   ├── raw_results*.json  # Raw experimental data
│   ├── plots/             # Visualizations
│   └── stimuli.json       # Generated test stimuli
├── datasets/              # Pre-gathered glitch token datasets
├── papers/                # Downloaded research papers
└── code/                  # Cloned baseline repositories
```

## Reproducing Results

```bash
# Setup
uv venv && source .venv/bin/activate
uv pip install openai httpx numpy pandas matplotlib seaborn scipy python-Levenshtein

# Set API key
export OPENAI_API_KEY="your-key"

# Run experiments (in order)
python src/build_stimuli.py
python src/run_experiments.py
python src/run_experiments_v2.py
python src/run_experiments_v3.py
python src/run_experiments_v4.py
python src/run_experiments_v5.py
python src/run_experiments_v6_final.py

# Analyze results
python src/analyze_results.py
```

## Models Tested
- GPT-4.1 (OpenAI, 2025)
- GPT-4.1-mini (OpenAI, 2025)
- GPT-4o-mini (OpenAI, 2024)

## Total Experiments
~500 API calls across 6 experiment rounds, testing single tokens, documents (clean and misspelled), adversarial strings, and extreme-length text reproduction.

See [REPORT.md](REPORT.md) for full methodology, results, and analysis.
