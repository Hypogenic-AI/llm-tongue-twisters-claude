# Resources Catalog

## Summary
This document catalogs all resources gathered for the research project "Do Language Models Have Tongue Twisters?" — investigating patterns and tokens that LLMs struggle to reproduce accurately.

## Papers
Total papers downloaded: 9

| Title | Authors | Year | File | Key Info |
|-------|---------|------|------|----------|
| Fishing for Magikarp | Land, Bartolo | 2024 | papers/2405.05417_fishing_for_magikarp.pdf | Automated under-trained token detection; 88 models |
| Glitch Tokens Taxonomy | Li, Liu, et al. | 2024 | papers/2404.09894_glitch_tokens_taxonomy.pdf | Token type + behavior taxonomy; GlitchHunter |
| Improbable Bigrams | Jang, Lee, et al. | 2024 | papers/2410.23684_improbable_bigrams.pdf | Token pair vulnerabilities; 33-77% hallucination |
| GlitchProber | Zhang, et al. | 2024 | papers/2408.04905_glitchprober.pdf | PCA+SVM detection; ground truth datasets |
| GlitchMiner | Wu, et al. | 2024 | papers/2410.15052_glitchminer.pdf | Gradient-based detection; entropy optimization |
| Problematic Tokens | Yang, Wang, et al. | 2024 | papers/2406.11214_problematic_tokens.pdf | GPT-4o Chinese/Korean tokenizer bias |
| Coercing LLMs | Geiping, Stein, et al. | 2024 | papers/2402.14020_coercing_llms.pdf | Security attack surface of glitch tokens |
| Robustness to Perturbations | Moradi, Samwald | 2021 | papers/2108.12237_robustness_perturbations.pdf | Model fragility to typos/misspellings |
| Reversal Curse | Berglund, et al. | 2023 | papers/2309.12288_reversal_curse.pdf | Directional generalization failure |

See papers/README.md for detailed descriptions.

## Datasets
Total datasets: 3 categories

| Name | Source | Size | Format | Location | Notes |
|------|--------|------|--------|----------|-------|
| GlitchProber Ground Truth | GlitchProber paper | 5 models, 6K-32K tokens each | CSV | datasets/glitch_tokens_ground_truth/ | Confirmed glitch tokens per model |
| Magikarp Verifications | Magikarp paper | 3 models (50K tokens each) | JSONL.GZ | datasets/magikarp_verifications/ | Full 88-model set in code/magikarp/ |
| GPT-4o Problematic Tokens | Problematic Tokens paper | ~200K tokens | JSON | datasets/problematic_tokens_gpt4o/ | Chinese/Korean analysis |

See datasets/README.md for detailed descriptions and loading instructions.

## Code Repositories
Total repositories cloned: 4

| Name | URL | Purpose | Location | Notes |
|------|-----|---------|----------|-------|
| magikarp | github.com/cohere-ai/magikarp | Under-trained token detection | code/magikarp/ | Pre-computed results for 88 models |
| GlitchMiner | github.com/wooozihui/GlitchMiner | Gradient-based glitch detection | code/GlitchMiner/ | AAAI 2026; entropy-based approach |
| GlitchProber | github.com/LLM-Integrity-Guard/GlitchProber | Detection + mitigation | code/GlitchProber/ | Ground truth CSVs for 5 models |
| LLMGPT4o | github.com/yeyimilk/LLMGPT4o | GPT-4o tokenizer bias analysis | code/LLMGPT4o/ | Multilingual focus |

See code/README.md for detailed descriptions.

## Resource Gathering Notes

### Search Strategy
- Used paper-finder tool with 5 query variations covering: glitch tokens, under-trained tokens, adversarial inputs, text reproduction, and tokenizer artifacts
- Supplemented with targeted web searches for specific arxiv papers and GitHub repositories
- Traced code repositories from paper references

### Selection Criteria
- **Papers**: Prioritized work directly studying tokens/patterns that cause LLM failures. Included related work on input perturbation robustness and generalization failures.
- **Datasets**: Focused on pre-existing glitch token lists and verification results that can serve as ground truth for experiments.
- **Code**: Cloned repos with detection tools, evaluation frameworks, and pre-computed results.

### Key Observations
1. The "glitch token" / "under-trained token" literature is relatively recent (2024) but growing rapidly
2. The original SolidGoldMagikarp discovery (LessWrong, Jan 2023) spawned multiple academic follow-ups
3. All major detection methods agree: the root cause is tokenizer-model training data mismatch
4. Under-trained tokens exist in ALL tested models (open and closed source)
5. The problem is worse for models with large vocabularies or borrowed tokenizers

### Gaps
- No existing benchmark specifically tests "tongue twister" reproduction of longer passages
- Limited work on how models handle documents with embedded difficult patterns
- Cross-model comparison on identical inputs is underexplored

## Recommendations for Experiment Design

Based on gathered resources:

1. **Primary datasets**: Use GlitchProber ground truth CSVs + Magikarp verification data as known-difficult tokens. Construct additional test cases using improbable bigrams methodology.

2. **Baseline methods**: Compare model performance on glitch tokens vs. frequency-matched normal tokens. Use multiple prompt templates (repeat, spell, define, code-embed).

3. **Evaluation metrics**: Exact match rate, edit distance (Levenshtein), hallucination rate, character-level accuracy.

4. **Code to adapt/reuse**:
   - Magikarp's `generate_results.py` for running verification on new models
   - GlitchProber's ground truth data as evaluation standard
   - Improbable bigrams methodology for constructing novel test cases
   - Build a custom evaluation harness that tests reproduction of single tokens, token pairs, and longer passages containing problematic patterns

5. **Novel experiment directions**:
   - Test whether models can reproduce paragraphs containing embedded glitch tokens
   - Measure how misspellings near glitch tokens compound the difficulty
   - Compare instruction-tuned vs. base models on reproduction tasks
   - Test whether models can detect their own reproduction failures
