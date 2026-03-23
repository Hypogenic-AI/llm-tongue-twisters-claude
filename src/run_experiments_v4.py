"""
V4: Focused experiments on the key findings.

Key findings so far:
- GPT-4.1: Perfect reproduction across all tests
- GPT-4.1-mini: 1 edit on very long misspelled doc (1 correction)
- GPT-4o-mini: 5 edits on very long misspelled doc
- All models: perfect on glitch tokens, homoglyphs, zero-width chars, near-miss phrases

This v4:
1. Saves full outputs for diff analysis
2. Tests even longer documents
3. Tests with increasing misspelling density
4. Tests documents that combine misspellings with glitch tokens
"""

import json
import os
import time
import random
import difflib
from datetime import datetime

import openai
import Levenshtein
import numpy as np

random.seed(42)
np.random.seed(42)

BASE = "/workspaces/llm-tongue-twisters-claude"
client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])

PROMPT = "Repeat the following text exactly, character for character. Output ONLY the text, nothing else.\n\n{text}"


def call_model(prompt, model="gpt-4o-mini", temperature=0, max_tokens=16384):
    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return resp.choices[0].message.content
        except Exception as e:
            print(f"  API error ({model}, attempt {attempt+1}): {e}")
            time.sleep(2 ** attempt)
    return None


def compute_metrics(expected, actual):
    if actual is None:
        return {"exact_match": False, "edit_distance": len(expected),
                "normalized_edit_distance": 1.0, "char_accuracy": 0.0}
    a = actual.strip()
    e = expected.strip()
    exact = e == a
    ed = Levenshtein.distance(e, a)
    ml = max(len(e), len(a), 1)
    matching = sum(1 for x, y in zip(e, a) if x == y)
    return {
        "exact_match": exact,
        "edit_distance": ed,
        "normalized_edit_distance": ed / ml,
        "char_accuracy": matching / max(len(e), 1),
        "length_ratio": len(a) / max(len(e), 1),
    }


def show_diff(expected, actual, label=""):
    """Show character-level differences."""
    if actual is None:
        print(f"  {label}: NO OUTPUT")
        return
    e = expected.strip()
    a = actual.strip()
    if e == a:
        return
    # Show unified diff of words
    e_words = e.split()
    a_words = a.split()
    diff = list(difflib.unified_diff(e_words, a_words, lineterm='', n=1))
    if diff:
        changes = [d for d in diff if d.startswith('+') or d.startswith('-')]
        changes = changes[:20]  # limit output
        for c in changes:
            if not c.startswith('+++') and not c.startswith('---'):
                print(f"    {c}")


def find_corrections(misspelled, output, clean):
    if output is None:
        return [], 0, 0, 0
    m_words = misspelled.split()
    c_words = clean.split()
    o_words = output.strip().split()
    corrections = []
    corr = pres = oth = 0
    for i, (m, c) in enumerate(zip(m_words, c_words)):
        if m != c and i < len(o_words):
            o = o_words[i]
            if o == c:
                corr += 1
                corrections.append({"pos": i, "misspelled": m, "clean": c, "output": o, "action": "corrected"})
            elif o == m:
                pres += 1
            else:
                oth += 1
                corrections.append({"pos": i, "misspelled": m, "clean": c, "output": o, "action": "other"})
    return corrections, corr, pres, oth


# === Documents with increasing misspelling density ===

BASE_TEXT = """Natural language processing has undergone a remarkable transformation in recent years, driven primarily by the development of large language models based on the transformer architecture. These models, trained on vast amounts of text data, have demonstrated surprising capabilities in understanding and generating human language. From simple text classification to complex reasoning tasks, language models continue to push the boundaries of what machines can achieve with language. However, significant challenges remain, including issues of bias, hallucination, and the fundamental question of whether these models truly understand the text they process or merely produce statistically likely continuations. The field continues to evolve at a rapid pace, with new architectures, training techniques, and applications emerging regularly."""

BASE_CLEAN_WORDS = BASE_TEXT.split()

def make_misspelled_version(text, density):
    """Create a version with misspellings at given density (fraction of words)."""
    words = text.split()
    n_to_misspell = int(len(words) * density)
    indices = random.sample(range(len(words)), min(n_to_misspell, len(words)))

    misspelled_words = list(words)
    for i in indices:
        w = words[i]
        if len(w) >= 4 and w.isalpha():
            # Swap two adjacent characters
            pos = random.randint(1, len(w) - 2)
            misspelled_words[i] = w[:pos] + w[pos+1] + w[pos] + w[pos+2:]
    return " ".join(misspelled_words)


# === Document with glitch tokens embedded ===

DOC_WITH_GLITCH = """The database contains several unusual usernames including SolidGoldMagikarp, TheNitromeFan, and StreamerBot. The system also tracks internal identifiers like cloneembedaliased and rawdownloadcloneembedreportprint. Each handler, such as ActionCodeHandler and InstoreAndOnline, processes specific request types. The user petertodd filed a bug report about the PsychExpandoExceptionObjectSyntax error that occurs when the guiActiveUn component fails to initialize properly."""


def main():
    print(f"{'='*70}")
    print(f"V4: Focused Analysis")
    print(f"Started: {datetime.now().isoformat()}")
    print(f"{'='*70}")

    all_results = []
    models = ["gpt-4o-mini", "gpt-4.1-mini", "gpt-4.1"]

    for model in models:
        print(f"\n{'='*70}")
        print(f"Model: {model}")
        print(f"{'='*70}")

        # Test 1: Varying misspelling density
        print(f"\n--- Misspelling Density Experiment ---")
        for density in [0.0, 0.05, 0.10, 0.20, 0.30, 0.50]:
            random.seed(42)  # Reset for consistency
            text = make_misspelled_version(BASE_TEXT, density) if density > 0 else BASE_TEXT
            prompt = PROMPT.format(text=text)
            output = call_model(prompt, model=model)
            metrics = compute_metrics(text, output)

            result = {"experiment": "density", "density": density, "model": model,
                      "stimulus": text, "output": (output or "").strip(),
                      "stimulus_len": len(text), **metrics}
            all_results.append(result)

            print(f"  density={density:.0%}: exact={metrics['exact_match']}, edit={metrics['edit_distance']}, "
                  f"norm_edit={metrics['normalized_edit_distance']:.4f}")
            if not metrics['exact_match']:
                show_diff(text, output, f"density={density}")

        # Test 2: Document with embedded glitch tokens
        print(f"\n--- Document with Embedded Glitch Tokens ---")
        prompt = PROMPT.format(text=DOC_WITH_GLITCH)
        output = call_model(prompt, model=model)
        metrics = compute_metrics(DOC_WITH_GLITCH, output)
        result = {"experiment": "glitch_in_doc", "model": model,
                  "stimulus": DOC_WITH_GLITCH, "output": (output or "").strip(), **metrics}
        all_results.append(result)
        print(f"  exact={metrics['exact_match']}, edit={metrics['edit_distance']}")
        if not metrics['exact_match']:
            show_diff(DOC_WITH_GLITCH, output, "glitch_doc")

        # Test 3: Repeat the SAME misspelled text 5 times (check consistency)
        print(f"\n--- Consistency: Same misspelled text 5x at temp=0 ---")
        random.seed(42)
        misspelled_text = make_misspelled_version(BASE_TEXT, 0.20)
        outputs = []
        for run in range(5):
            prompt = PROMPT.format(text=misspelled_text)
            output = call_model(prompt, model=model, temperature=0)
            outputs.append((output or "").strip())
        unique = set(outputs)
        exact_matches = sum(1 for o in outputs if o == misspelled_text.strip())
        result = {"experiment": "consistency_misspelled", "model": model,
                  "n_exact": exact_matches, "n_unique": len(unique), "stimulus": misspelled_text}
        all_results.append(result)
        print(f"  {exact_matches}/5 exact, {len(unique)} unique outputs")

        # Test 4: Very long text - 3x the base (concatenated with variations)
        print(f"\n--- Very Long Text (3x paragraph) ---")
        long_clean = (BASE_TEXT + "\n\n" +
                      BASE_TEXT.replace("Natural language processing", "Computer vision research") + "\n\n" +
                      BASE_TEXT.replace("Natural language processing", "Reinforcement learning"))
        random.seed(42)
        long_misspelled = make_misspelled_version(long_clean, 0.15)

        for version, text in [("clean", long_clean), ("misspelled", long_misspelled)]:
            prompt = PROMPT.format(text=text)
            output = call_model(prompt, model=model, max_tokens=16384)
            metrics = compute_metrics(text, output)
            result = {"experiment": "3x_long", "version": version, "model": model,
                      "stimulus_len": len(text), **metrics}
            all_results.append(result)
            print(f"  [{version}] ({len(text)} chars) exact={metrics['exact_match']}, "
                  f"edit={metrics['edit_distance']}, norm_edit={metrics['normalized_edit_distance']:.4f}")
            if not metrics['exact_match']:
                show_diff(text, output, f"3x_{version}")

    # Save all results
    out_path = os.path.join(BASE, "results/raw_results_v4.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)
    print(f"\nSaved {len(all_results)} results to {out_path}")
    print(f"Finished: {datetime.now().isoformat()}")


if __name__ == "__main__":
    main()
