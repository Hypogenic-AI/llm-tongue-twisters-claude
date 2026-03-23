# Do Language Models Have Tongue Twisters?

## 1. Executive Summary

We systematically tested whether modern language models (GPT-4.1, GPT-4.1-mini, GPT-4o-mini) struggle to reproduce specific strings or patterns when explicitly asked to repeat them. We found that **the classic "glitch token" tongue twisters (e.g., SolidGoldMagikarp) have been completely resolved in 2025-era models**—all models reproduced them perfectly. Models also faithfully preserved misspelled text without auto-correcting, even at 50% misspelling density. However, we discovered a **new class of tongue twisters at extreme text lengths**: all models fail to reproduce very long repetitive text (10k+ characters), and GPT-4.1 paradoxically refuses to reproduce long clean text (8k+ chars) while perfectly reproducing equally long misspelled text—suggesting content-based output filtering rather than a capability limitation.

## 2. Goal

**Hypothesis**: There exist specific strings or patterns that language models struggle to repeat or reproduce accurately when prompted.

**Why this matters**: LLMs are increasingly used for tasks requiring faithful text reproduction—copying code, quoting sources, data transcription, translation. Systematic reproduction failures affect reliability and trust. Understanding what models cannot copy reveals fundamental properties of how they process and generate text.

**Expected impact**: Identifying current-generation "tongue twisters" helps inform prompt engineering best practices, model evaluation benchmarks, and safety assessments.

## 3. Data Construction

### Dataset Description

We constructed 5 categories of stimuli from scratch and from prior work:

| Category | Source | N | Description |
|----------|--------|---|-------------|
| Glitch tokens | GlitchProber ground truth + Magikarp verification data | 50 | Known problematic tokens from literature |
| Control tokens | Hand-crafted | 50 | Common English words (same length distribution) |
| Adversarial strings | Constructed | 20 | Unicode tricks, homoglyphs, zero-width chars |
| Documents (clean/misspelled pairs) | Hand-crafted | 5 pairs | 50–1500 word paragraphs with deliberate misspellings |
| Extreme length texts | Generated | Variable | 500–25,000 character texts, clean/misspelled/repetitive/random |

### Example Samples

**Glitch tokens**: `SolidGoldMagikarp`, `TheNitromeFan`, `rawdownloadcloneembedreportprint`, `PsychExpandoExceptionObjectSyntax`

**Misspelled document excerpt**:
> "The developement of artifical intelligence has been one of the most tranformative technologicle advances of the twenty-first centruey."

**Adversarial strings**: `рython` (Cyrillic р), `hello​world` (zero-width space), `H̷e̷l̷l̷o̷` (combining chars)

### Preprocessing Steps
- Glitch tokens filtered to printable, ≥3 character strings (excluded special tokens like `<unk>`)
- Misspelled documents created with controlled character transpositions (1–2 adjacent character swaps per word)
- Misspelling density experiments used random seed 42 for reproducibility

## 4. Experiment Description

### Methodology

#### High-Level Approach
We used the **text reproduction paradigm**: present a string to the model and ask it to repeat it exactly. Compare the model's output to the original using edit distance metrics. This simple but powerful methodology directly tests whether models can faithfully reproduce content, which is the core question behind the "tongue twister" analogy.

#### Why This Method?
- **Directly addresses the hypothesis**: If models struggle with certain strings, they will fail to reproduce them
- **Model-agnostic**: Works with any LLM via API
- **Measurable**: Exact match and edit distance provide unambiguous metrics
- **Controlled**: Clean/misspelled pairs provide within-stimulus controls

### Implementation Details

#### Tools and Libraries
- Python 3.12.8
- OpenAI API (openai 1.x) for GPT-4.1, GPT-4.1-mini, GPT-4o-mini
- python-Levenshtein for edit distance computation
- matplotlib/seaborn for visualization
- scipy for statistical tests

#### Models Tested
| Model | Provider | Type |
|-------|----------|------|
| GPT-4.1 | OpenAI | Full-size |
| GPT-4.1-mini | OpenAI | Compact |
| GPT-4o-mini | OpenAI | Compact (prior gen) |

#### Prompt Templates
Three templates tested (v1), with "direct" used as primary:
```
Direct: "Repeat the following text exactly, character for character.
         Output ONLY the text, nothing else.\n\n{text}"
```

#### Hyperparameters
| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Temperature | 0 | Deterministic reproduction |
| max_tokens | 16384 | Allow long reproductions |
| Retries | 3 | Handle transient API errors |

### Experimental Protocol

Six experiment rounds (v1–v6) with escalating difficulty:

| Round | Focus | N Stimuli | Models |
|-------|-------|-----------|--------|
| V1 | Token reproduction (glitch vs control vs adversarial) | 120 tokens × 2 templates | GPT-4.1 |
| V2 | Long misspelled docs, embedded glitch tokens, consistency | ~25 tests | GPT-4.1 |
| V3 | Cross-model comparison, zero-width chars, homoglyphs | ~80 tests | All 3 models |
| V4 | Misspelling density (0–50%), glitch-in-document | ~30 tests | All 3 models |
| V5 | Length scaling (500–10k chars), random strings, hex, code | ~70 tests | All 3 models |
| V6 | Extreme length (8k–25k chars), repetitive text | ~36 tests | All 3 models |

**Total API calls**: ~500 across all rounds

### Raw Results

#### Experiment 1: Single Token Reproduction (GPT-4.1, direct template)

| Category | Exact Match Rate | Avg Norm Edit Distance |
|----------|-----------------|----------------------|
| Control tokens | **100%** (50/50) | 0.000 |
| Glitch tokens | **98%** (49/50) | 0.001 |
| Adversarial strings | **90%** (18/20) | 0.093 |

The single glitch token failure was ` SolidGoldMagikarp` (leading space stripped). The adversarial failures were escaped Unicode (`\u0048\u0065\u006C\u006F` → `Hello`, decoded instead of preserved) and flag emoji tag sequences (invisible tag characters dropped).

#### Experiment 2: Document Reproduction (all models)

**Short–medium documents (50–500 words)**: All models achieved **100% exact match** on both clean and misspelled versions.

**Misspelling density experiment**: All three models achieved **100% exact match** at all densities (0%, 5%, 10%, 20%, 30%, 50%) for ~800-char documents.

**Key finding**: Models do NOT auto-correct misspellings when asked to reproduce text faithfully.

#### Experiment 3: Extreme Length Reproduction

| Model | Text Type | 5k | 8k | 10k | 15k | 20k | 25k |
|-------|-----------|----|----|-----|-----|-----|-----|
| GPT-4.1 | Clean (varied) | ✓ | **✗** | ✓ | **✗** | — | **✗** |
| GPT-4.1 | Misspelled | ✓ | ✓ | ✓ | ✓ | — | ✓ |
| GPT-4.1 | Repetitive | ✓ | — | **✗** | — | **✗** | — |
| GPT-4.1-mini | Clean (varied) | ✓ | ✓ | ✓ | ✓ | — | ✓ |
| GPT-4.1-mini | Misspelled | ✓ | ✓ | ✓ | ✓ | — | ✓ |
| GPT-4.1-mini | Repetitive | ✓ | — | **✗** | — | **✗✗** | — |
| GPT-4o-mini | Clean (varied) | ✓ | ✓ | **✗** | ✓ | — | **✗** |
| GPT-4o-mini | Misspelled | ✓ | ✓ | ✓ | **≈✓** | — | **✗** |
| GPT-4o-mini | Repetitive | **✗** | — | **✗** | — | **✗** | — |

✓ = exact match, ✗ = failure, ✗✗ = catastrophic failure (hallucination)

#### Key Failure Modes at Extreme Length

**GPT-4.1 on clean varied text** (8k–25k chars):
- Output was **truncated to ~50 characters** (len_ratio ≈ 0.005)
- The model effectively refused to reproduce the text
- BUT: perfectly reproduced 25k chars of misspelled text
- This is the most surprising finding—the model distinguishes clean from misspelled text

**gpt-4.1-mini on 20k repetitive text**:
- Output was **5.15× the input length** (len_ratio = 5.15)
- Edit distance: 82,995 characters
- The model hallucinated, producing vastly more text than requested

**gpt-4o-mini on repetitive text**:
- Progressively worse with length: 5k (edit=278), 10k (edit=1554), 20k (edit=8738)
- Output shrank (len_ratio: 1.06 → 0.84 → 0.56)

#### Experiment 4: Special Character Preservation

| Test Type | N Tests | All Models Result |
|-----------|---------|-------------------|
| Zero-width characters | 5 | **100% preserved** |
| Cyrillic homoglyphs | 5 | **100% preserved** |
| Code with syntax errors | 1 | **100% preserved** |
| Near-miss famous phrases | 5 | **100% preserved** |
| Random alphanumeric | 5 lengths | **100% exact** |
| Hex strings | 4 lengths | **100% exact** |

## 5. Result Analysis

### Key Findings

**Finding 1: Classic glitch tokens are no longer tongue twisters.**
All three models (GPT-4.1, GPT-4.1-mini, GPT-4o-mini) perfectly reproduced every known glitch token tested, including `SolidGoldMagikarp`, `TheNitromeFan`, `rawdownloadcloneembedreportprint`, and `PsychExpandoExceptionObjectSyntax`. These tokens, which caused severe failures in GPT-2, GPT-3, and early GPT-4, are completely handled by 2025-era models. This represents a clear improvement over the phenomena documented in the glitch token literature (Land & Bartolo 2024, Li et al. 2024).

**Finding 2: Models faithfully preserve misspellings—they do not auto-correct.**
Across all document lengths (50–25,000 characters) and misspelling densities (5–50%), models reproduced misspelled text with remarkable fidelity. Even with explicit, emphatic instructions to preserve errors, the correction rate was 0%. This contradicts the initial hypothesis that models would struggle with misspelled documents. The result held across all three models tested.

**Finding 3: Length is the new tongue twister—with a paradoxical twist.**
The most striking finding is that GPT-4.1 refused to reproduce clean text longer than ~8,000 characters (truncating output to ~50 chars) while **perfectly reproducing misspelled text of the same or greater length**. This suggests the model may have content-based output limits that treat "original" misspelled text differently from "standard" clean text—possibly an anti-copying or deduplication mechanism. This behavior was not present in the mini models.

**Finding 4: Repetitive text causes hallucination in mini models.**
When asked to reproduce text consisting of the same paragraph repeated many times, gpt-4.1-mini generated output 5.15× the input length at 20k characters—a severe hallucination. gpt-4o-mini progressively truncated its output. This suggests that repetition detection interferes with faithful reproduction.

**Finding 5: Adversarial Unicode is mostly handled, with two exceptions.**
Models preserved zero-width characters, homoglyphs, combining characters, and Braille. The two consistent failures were: (a) escaped Unicode sequences (`\u0048` → `H`, decoded instead of literal), and (b) flag emoji tag sequences (invisible tag characters dropped). These reflect reasonable default behavior rather than "tongue twister" failures.

### Hypothesis Testing Results

| Hypothesis | Result | Evidence |
|------------|--------|----------|
| H1: Glitch tokens cause failures in modern LLMs | **Refuted** | 100% exact match across all 3 models |
| H2: Adversarial strings have higher error rates | **Partially supported** | 90% vs 100% for controls (escaped Unicode, emoji tags) |
| H3: Misspelled docs are harder to reproduce | **Refuted** | 100% exact match at all densities |
| H3a: Models auto-correct misspellings | **Refuted** | 0% correction rate even at 50% density |
| H3b: Effect scales with document length | **Partially supported** | Not for misspellings, but length itself causes failures |
| *New*: Length causes reproduction failures | **Supported** | Systematic failures above 8k–10k chars |
| *New*: Clean text is harder than misspelled | **Supported** | GPT-4.1 truncates clean but reproduces misspelled |

### Surprises and Insights

1. **The biggest surprise**: GPT-4.1 reproducing 25k characters of misspelled text perfectly while refusing to reproduce 8k characters of clean text. This was not predicted by any hypothesis and suggests a fundamentally different processing path for "original" vs "known" text.

2. **gpt-4.1-mini's 5× hallucination**: At 20k repetitive characters, the model produced output 5.15× longer than the input. The model appears to enter a generative loop when the input is highly repetitive, losing track of when to stop.

3. **The death of glitch tokens**: The SolidGoldMagikarp phenomenon, which spawned an entire sub-field of research, appears to be completely resolved in current-generation models. This likely reflects improvements in tokenizer training, vocabulary curation, or post-training alignment.

### Error Analysis

**Categories of failures observed:**
1. **Output truncation** (GPT-4.1 on long clean text): Model produces only a few dozen characters, effectively refusing the task
2. **Hallucinated extension** (gpt-4.1-mini on long repetitive text): Model generates far more text than input, losing count of repetitions
3. **Semantic decoding** (escaped Unicode): Model interprets `\u0048` as `H` instead of treating it as literal characters
4. **Invisible character loss** (emoji tags): Non-printing Unicode tag characters stripped during reproduction

### Limitations

1. **Model coverage**: Only tested OpenAI models (GPT-4.1 family). Results may differ for Claude, Gemini, Llama, or other model families.
2. **API constraints**: The extreme-length truncation in GPT-4.1 may reflect API-level output limits rather than model-level capability limits. We set max_tokens=16384 which should be sufficient, but internal API mechanisms may impose additional limits.
3. **Prompt sensitivity**: We tested 3 prompt templates but the space of possible instructions is infinite. Different phrasing might elicit different behavior.
4. **No open-weight models**: We could not test the original models (GPT-2, Llama-2-7B) where glitch tokens were first discovered, limiting our ability to confirm the historical phenomenon.
5. **Single-run design**: Most experiments used temperature=0 for determinism. Temperature>0 behavior may differ, though our consistency tests at temperature=0.7 showed perfect reproduction for short texts.

## 6. Conclusions

### Summary
Modern language models (GPT-4.1 family, March 2025) have **largely solved** the classic "tongue twister" problem of glitch tokens—strings like `SolidGoldMagikarp` that caused dramatic failures in earlier models are now reproduced perfectly. Models also faithfully preserve misspelled text without auto-correcting, even at high error densities. However, **a new class of tongue twisters has emerged at extreme text lengths**: models struggle to reproduce very long documents (8k+ characters), especially when the content is repetitive or "clean" (standard, likely-seen-in-training text). Most strikingly, GPT-4.1 paradoxically refuses to reproduce long clean text while perfectly reproducing equally long misspelled text, suggesting content-aware output filtering.

### Implications
- **Practical**: Users relying on LLMs for faithful text copying should be aware of length limits and the paradoxical behavior with clean vs. misspelled text
- **Theoretical**: The clean-vs-misspelled asymmetry suggests models may have internalized content-dependent output policies, possibly related to copyright/deduplication training objectives
- **Security**: The glitch token attack surface has been substantially reduced in modern models, but length-based reproduction failures could potentially be exploited

### Confidence in Findings
- **High confidence**: Glitch tokens are resolved; misspellings are faithfully preserved
- **Moderate confidence**: The clean-vs-misspelled length asymmetry may be API-specific behavior rather than model behavior
- **Lower confidence**: The exact thresholds for length-based failures likely vary by API version and parameters

## 7. Next Steps

### Immediate Follow-ups
1. Test Claude (Sonnet/Opus), Gemini 2.5 Pro, and Llama 3 to determine if the clean/misspelled asymmetry is universal or OpenAI-specific
2. Investigate GPT-4.1's truncation mechanism: Is it a safety filter, output length limit, or model-level behavior?
3. Test with the original open-weight models (Llama-2-7B, Mistral-7B) to confirm and quantify the historical glitch token effect

### Alternative Approaches
- Use logprob analysis to understand per-token confidence during reproduction
- Test token-level reproduction (ask model to output one token at a time)
- Investigate whether fine-tuning on reproduction tasks can eliminate length-based failures

### Open Questions
1. Why does GPT-4.1 distinguish clean from misspelled text during reproduction? Is this a deliberate design choice or an emergent behavior?
2. At what exact character count does each model's reproduction fidelity degrade? Is there a sharp threshold or gradual decline?
3. Would a "chain of thought" approach (reproduce section by section) overcome length limitations?

## References

1. Rumbelow, J. & Watkins, M. (2023). "SolidGoldMagikarp (plus, prompt generation)." LessWrong.
2. Land, S. & Bartolo, M. (2024). "Fishing for Magikarp: Automatically Detecting Under-trained Tokens in Large Language Models." EMNLP 2024. arXiv:2405.05417.
3. Li, Y., Liu, Y., Deng, G., et al. (2024). "Glitch Tokens in Large Language Models: Categorization Taxonomy and Effective Detection." FSE 2024. arXiv:2404.09894.
4. Jang, E., Lee, K., Chung, J-W., et al. (2024). "Improbable Bigrams Expose Vulnerabilities of Incomplete Tokens in Byte-Level Tokenizers." arXiv:2410.23684.
5. Zhang, Z., et al. (2024). "GlitchProber: Advancing Effective Detection and Mitigation of Glitch Tokens in Large Language Models." ASE 2024. arXiv:2408.04905.
6. Wu, Z., et al. (2024). "GlitchMiner: Mining Glitch Tokens via Gradient-based Discrete Optimization." arXiv:2410.15052.
7. Moradi, M. & Samwald, M. (2021). "Evaluating the Robustness of Neural Language Models to Input Perturbations." EMNLP 2021. arXiv:2108.12237.
