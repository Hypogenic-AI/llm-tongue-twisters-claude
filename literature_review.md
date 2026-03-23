# Literature Review: Do Language Models Have Tongue Twisters?

## Research Area Overview

This review covers research on patterns, tokens, and strings that language models struggle to process, reproduce, or handle correctly. The phenomenon—analogous to "tongue twisters" for humans—arises primarily from the disconnect between tokenizer training and model training, creating tokens that exist in the vocabulary but were rarely or never seen during model training. These "glitch tokens" or "under-trained tokens" cause LLMs to produce garbled output, hallucinations, refusals, or completely unrelated text when asked to repeat or process them.

The field originated with the viral discovery of the "SolidGoldMagikarp" token (Rumbelow & Watkins, 2023, LessWrong) and has since grown into a systematic research area spanning detection methods, taxonomies, mitigation strategies, and security implications.

---

## Key Papers

### Paper 1: Fishing for Magikarp: Automatically Detecting Under-trained Tokens in Large Language Models
- **Authors**: Sander Land, Max Bartolo (Cohere)
- **Year**: 2024 (EMNLP 2024)
- **arXiv**: 2405.05417
- **Key Contribution**: First systematic, automated approach for detecting under-trained tokens across LLMs. Proposes a 3-step pipeline: (1) tokenizer analysis for partial UTF-8/unreachable/special tokens, (2) embedding-based indicators (L2 norm for untied models, cosine distance for tied), (3) prompting verification.
- **Methodology**: Analyzes token embeddings to find statistical outliers. Under-trained tokens have near-zero L2 norms due to weight decay. Verified by testing if tokens have <1% max output probability.
- **Models Tested**: 22 open-weight models (GPT-2, GPT-J, Llama2/3, Mistral, Gemma, Command R, Qwen, etc.) + closed-source (GPT-3.5/4, Claude 2/3)
- **Results**: Under-trained tokens found in ALL tested models (0.1-1% of vocabulary). Larger vocabularies have more issues (Qwen 72B: 2,047 undertrained; Gemma 7B: 800).
- **Datasets Used**: OLMo v1.7 training data (for frequency correlation), StarCoder2 tokenizer data
- **Code Available**: https://github.com/cohere-ai/magikarp (pre-computed results for 88 models)
- **Relevance**: Foundational paper. Provides both methodology and extensive datasets for our research.

### Paper 2: Glitch Tokens in Large Language Models: Categorization Taxonomy and Effective Detection
- **Authors**: Yuxi Li, Yi Liu, Gelei Deng, et al.
- **Year**: 2024 (FSE 2024)
- **arXiv**: 2404.09894
- **Key Contribution**: First comprehensive taxonomy of glitch token types AND model behaviors when encountering them. Proposes GlitchHunter, a clustering-based detection method.
- **Token Taxonomy** (5 types): (A) Word Tokens—concatenated English words, (B) Letter Tokens—nonsensical letter strings, (C) Character Tokens—non-alphabetic sequences, (D) Letter-Character Tokens—mixed, (E) Special Tokens—non-ASCII.
- **Behavior Taxonomy** (5 types): (A) Spelling Mistakes, (B) Incapacity/Refusal, (C) Hallucinatory Completion, (D) Question Repetition, (E) Random Characters.
- **Models**: 7 LLMs (GPT-3.5, GPT-4, Llama-2 7B/13B, Mistral-7B, Vicuna-13B, Text-Davinci-003)
- **Results**: 7,895 glitch tokens identified across 182,517 analyzed. GlitchHunter achieves 99.44% precision, 63.20% recall.
- **Code Available**: https://sites.google.com/view/glitchhunter-fse2024/
- **Relevance**: The behavior taxonomy directly maps types of "tongue twister" failures.

### Paper 3: Improbable Bigrams Expose Vulnerabilities of Incomplete Tokens in Byte-Level Tokenizers
- **Authors**: Eugene Jang, Kimin Lee, Jin-Woo Chung, et al.
- **Year**: 2024 (EMNLP 2025)
- **arXiv**: 2410.23684
- **Key Contribution**: Identifies "incomplete tokens" (containing stray bytes from byte-level BPE) as a distinct vulnerability class. Constructs "improbable bigrams"—OOD combinations that cause hallucinations even from well-trained tokens.
- **Methodology**: Analyzes UTF-8 byte structure to find prefix/suffix incomplete token pairs. Creates cross-script combinations (e.g., mixing Devanagari and Chinese) that are valid but never seen in training.
- **Models**: Llama 3.1, Exaone, Qwen2.5, Mistral-Nemo, Command-R
- **Results**: Hallucination rates of 33-77% with improbable bigrams vs. 0-20% baseline. Alternative tokenization reduces hallucinations by up to 90%.
- **Relevance**: Shows tongue-twister-like failures extend beyond single tokens to token *combinations*. Critical for experiment design.

### Paper 4: GlitchProber: Advancing Effective Detection and Mitigation of Glitch Tokens in Large Language Models
- **Authors**: Zhibo Zhang, et al.
- **Year**: 2024 (ASE 2024)
- **arXiv**: 2408.04905
- **Key Contribution**: Detection via PCA on attention patterns + SVM classification. Also proposes mitigation by rectifying intermediate layer values.
- **Results**: F1=0.86 for detection, 50% repair rate for glitch tokens.
- **Code Available**: https://github.com/LLM-Integrity-Guard/GlitchProber
- **Relevance**: Provides ground truth glitch token lists for 5 models (valuable as evaluation data).

### Paper 5: GlitchMiner: Mining Glitch Tokens via Gradient-based Discrete Optimization
- **Authors**: Zihui Wu, et al.
- **Year**: 2024 (AAAI 2026)
- **arXiv**: 2410.15052
- **Key Contribution**: Entropy-based loss function + gradient-guided local search to find tokens that maximize output uncertainty. Behavior-driven approach (vs. embedding-based).
- **Results**: 19.07% improvement in precision@1000 over prior methods on 10 LLMs.
- **Code Available**: https://github.com/wooozihui/GlitchMiner
- **Relevance**: Complementary detection approach; useful as baseline.

### Paper 6: Problematic Tokens: Tokenizer Bias in Large Language Models
- **Authors**: Jin Yang, Zhiqiang Wang, et al.
- **Year**: 2024
- **arXiv**: 2406.11214
- **Key Contribution**: Shows GPT-4o's tokenizer creates systematic failures for Chinese and Korean text due to undertrained tokens from misaligned vocabulary construction.
- **Key Finding**: Of 100 longest Chinese tokens in GPT-4o, only 3 are common words; rest are gambling/adult content scraped during tokenizer training.
- **Code Available**: https://github.com/yeyimilk/LLMGPT4o
- **Relevance**: Demonstrates tongue-twister effect in multilingual contexts.

### Paper 7: Coercing LLMs to Do and Reveal (Almost) Anything
- **Authors**: Jonas Geiping, Alex Stein, et al.
- **Year**: 2024
- **arXiv**: 2402.14020
- **Key Contribution**: Broad systematization of adversarial attacks on LLMs. Identifies glitch tokens as an active attack vector for jailbreaking, misdirection, and denial-of-service.
- **Relevance**: Security implications of tongue-twister tokens; motivates the research.

### Paper 8: Evaluating the Robustness of Neural Language Models to Input Perturbations
- **Authors**: Milad Moradi, Matthias Samwald
- **Year**: 2021 (EMNLP 2021)
- **arXiv**: 2108.12237
- **Key Contribution**: Shows BERT/XLNet/RoBERTa/ELMo are sensitive to character-level and word-level perturbations (typos, misspellings). Small changes cause significant performance drops.
- **Relevance**: Background on model fragility to input noise; related to misspelling-based tongue twisters.

### Paper 9: The Reversal Curse: LLMs trained on "A is B" fail to learn "B is A"
- **Authors**: Lukas Berglund, et al.
- **Year**: 2023
- **arXiv**: 2309.12288
- **Key Contribution**: Shows fundamental generalization failure where models cannot reverse learned relationships. GPT-4 answers "Tom Cruise's mother" 79% correctly but "Mary Lee Pfeiffer's son" only 33%.
- **Relevance**: Different type of "tongue twister"—information ordering creates reproducibility failures.

---

## Common Methodologies

### Detection Methods
- **Embedding Analysis**: L2 norm of input embeddings (untied) or cosine distance (tied) — Used in Magikarp, GlitchMiner
- **Clustering**: Iterative clustering in embedding space to find anomalous token groups — Used in GlitchHunter
- **Gradient-based Search**: Entropy maximization via gradient-guided optimization — Used in GlitchMiner
- **Attention Pattern Analysis**: PCA on attention + SVM classification — Used in GlitchProber
- **Prompting Verification**: Ask model to repeat tokens and check output probability — Used in Magikarp

### Evaluation Approaches
- **Repetition Tasks**: Ask model to repeat a token/string verbatim
- **Spelling Tasks**: Ask model to spell token letter by letter
- **Definition Tasks**: Ask model to define a token
- **Code Context**: Embed token in code snippet and check reproduction
- **Hallucination Rate**: Fraction of prompts where model fails all templates

---

## Standard Baselines
- **Random Baseline**: Compare detected glitch tokens against random vocabulary samples
- **Frequency-based**: Sort by training data frequency; lowest-frequency tokens as baseline
- **Prior Methods**: GlitchHunter (clustering), token probability analysis (Fell 2023)

## Evaluation Metrics
- **Precision@K**: Precision of top-K detected tokens being true glitch tokens
- **F1 Score**: For binary classification of tokens as glitch/normal
- **Hallucination Rate**: Fraction of prompts producing incorrect output
- **Max Output Probability**: Highest probability assigned to the target token in any prompt
- **Repair Rate**: Fraction of glitch tokens whose behavior is fixed after mitigation

---

## Datasets in the Literature

| Dataset | Papers | Purpose |
|---------|--------|---------|
| GlitchProber Ground Truth | GlitchProber | Confirmed glitch tokens for 5 models |
| Magikarp Verifications | Fishing for Magikarp | Per-token indicators for 88 models |
| GPT-4o Problematic Tokens | Problematic Tokens | Chinese/Korean undertrained tokens |
| Improbable Bigrams | Improbable Bigrams | Cross-script token combinations |
| Alpaca-52k, ShareGPT | GlitchHunter | General evaluation prompts |

---

## Gaps and Opportunities

1. **No unified benchmark**: Each paper uses its own evaluation methodology. No standard "tongue twister benchmark" exists.
2. **Beyond single tokens**: Improbable Bigrams shows multi-token combinations can be problematic. Little work on longer sequences.
3. **Long documents with misspellings**: The hypothesis mentions "long documents with small misspellings"—this specific scenario is underexplored. Most work focuses on single tokens or short strings.
4. **Closed-source models**: Limited systematic testing of GPT-4, Claude, etc. due to API constraints.
5. **Generative reproduction**: Most work tests whether models can *output* a token, not whether they can faithfully *reproduce* a longer passage containing difficult patterns.
6. **Cross-model comparison**: How do different model families compare on the same tongue-twister inputs?

---

## Recommendations for Our Experiment

### Recommended Datasets
1. **GlitchProber Ground Truth CSVs** — Ready-to-use lists of confirmed glitch tokens per model
2. **Magikarp Verification Results** — Rich per-token data with embedding indicators for 88 models
3. **Custom Construction** — Build improbable bigrams following the methodology from Paper 3
4. **Synthetic Long Documents** — Create documents with embedded glitch tokens and misspellings

### Recommended Baselines
1. **Random token selection** — Compare tongue-twister tokens against randomly sampled vocabulary
2. **Frequency-matched controls** — Tokens with similar training frequency but normal behavior
3. **Cross-model comparison** — Same inputs tested across multiple model families

### Recommended Metrics
1. **Exact match rate** — Does the model reproduce the input exactly?
2. **Edit distance** — How far is the model's output from the target?
3. **Hallucination rate** — Does the model produce completely unrelated output?
4. **Character-level accuracy** — Per-character comparison for partial matches

### Methodological Considerations
- Use multiple prompt templates (repeat, code context, definition) to reduce prompt sensitivity
- Test with greedy decoding for reproducibility
- Consider both single tokens AND multi-token combinations
- Include multilingual tokens to test cross-lingual robustness
- Design experiments that test the *reproduction* of longer text passages, not just single tokens
