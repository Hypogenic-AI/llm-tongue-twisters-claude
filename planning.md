# Research Plan: Do Language Models Have Tongue Twisters?

## Motivation & Novelty Assessment

### Why This Research Matters
Language models are increasingly used for tasks requiring faithful text reproduction — copying code, quoting text, data entry, translation. If certain strings or patterns systematically cause reproduction failures, this has implications for reliability, safety, and trust. Understanding these "tongue twisters" helps identify blind spots in LLM capabilities.

### Gap in Existing Work
Prior work (Magikarp, GlitchHunter, GlitchProber) has extensively cataloged *single glitch tokens* in open-weight models. However:
1. No systematic study tests reproduction failures on **state-of-the-art closed-source models** (GPT-4.1, Claude) using the same stimuli
2. The "long documents with small misspellings" phenomenon is entirely unstudied
3. No work bridges single-token glitches to **passage-level reproduction** failures
4. No cross-model comparison on identical tongue-twister benchmarks exists for 2025-era models

### Our Novel Contribution
We conduct the first systematic study of "tongue twisters" for modern LLMs across three dimensions:
1. **Single glitch tokens**: Do known glitch tokens still cause failures in latest models?
2. **Constructed difficult strings**: Improbable bigrams, adversarial tokenizations
3. **Long document reproduction with misspellings**: A novel experiment testing whether models silently "correct" or garble text with embedded errors

### Experiment Justification
- **Experiment 1 (Glitch Token Reproduction)**: Tests whether the known phenomenon persists in SOTA models
- **Experiment 2 (Difficult String Construction)**: Tests multi-token combinations and adversarial strings
- **Experiment 3 (Long Document + Misspelling Reproduction)**: Tests the entirely novel hypothesis about passage-level reproduction failures

## Research Question
Do specific strings, tokens, or patterns exist that cause language models to fail at faithful reproduction when explicitly asked to repeat them?

## Hypothesis Decomposition
H1: Known glitch tokens (SolidGoldMagikarp, etc.) cause reproduction failures in modern LLMs
H2: Constructed adversarial strings (improbable bigrams, rare Unicode) cause higher error rates than normal text
H3: Long documents with small misspellings are harder for models to reproduce faithfully than identical documents without misspellings
H3a: Models tend to "auto-correct" misspellings rather than reproduce them faithfully
H3b: The effect scales with document length

## Proposed Methodology

### Approach
Use real LLM APIs (GPT-4.1 via OpenAI, and potentially OpenRouter models) to test reproduction across three stimulus categories. Measure exact match rate, edit distance, and error categorization.

### Experimental Steps

1. **Construct stimulus sets**:
   - Set A: 50 known glitch tokens from GlitchProber/Magikarp datasets
   - Set B: 50 normal control tokens (frequency-matched)
   - Set C: 20 constructed adversarial strings (improbable bigrams, rare Unicode)
   - Set D: 10 short paragraphs (50-100 words), clean vs. misspelled variants
   - Set E: 5 long passages (500-1000 words), clean vs. misspelled variants

2. **Prompt design**: Use 3 prompt templates per stimulus:
   - Direct: "Repeat the following text exactly: {text}"
   - Instruction: "Copy the following text character-for-character, including any errors: {text}"
   - Code: "Output exactly this string, preserving all characters: {text}"

3. **API calls**: Test on GPT-4.1 (primary), with temperature=0 for reproducibility

4. **Evaluation**: Exact match, Levenshtein distance, character-level accuracy, error categorization

### Baselines
- Normal English words (same frequency range)
- Clean documents (no misspellings) as control for misspelled versions
- Random character strings (to separate tokenization effects from randomness)

### Evaluation Metrics
- **Exact match rate**: Binary — did the model reproduce it perfectly?
- **Normalized edit distance**: Levenshtein distance / max(len(input), len(output))
- **Auto-correction rate**: For misspelled docs, did the model "fix" errors?
- **Hallucination rate**: Did the model produce completely unrelated output?

### Statistical Analysis Plan
- Fisher's exact test for exact match rates between conditions
- Mann-Whitney U test for edit distances between conditions
- Paired comparisons (clean vs. misspelled versions of same document)
- Effect sizes (Cohen's d) for continuous measures
- α = 0.05 with Bonferroni correction for multiple comparisons

## Expected Outcomes
- H1: Some glitch tokens still cause failures but fewer than in older models
- H2: Adversarial strings show higher error rates than controls
- H3: Models auto-correct misspellings rather than reproduce them, with effect increasing with document length

## Timeline and Milestones
- Planning: 15 min (this document)
- Implementation: 60 min
- Experiments: 45 min (API calls)
- Analysis: 30 min
- Documentation: 20 min

## Potential Challenges
- API rate limits → use exponential backoff
- Cost → limit to ~200-500 API calls total
- Glitch tokens may be "fixed" in latest models → interesting null result
- Long documents may exceed context limits → keep under 2000 tokens

## Success Criteria
- Complete data collection across all stimulus categories
- Statistical tests with clear results (positive or negative)
- At least one surprising or novel finding about reproduction failures
