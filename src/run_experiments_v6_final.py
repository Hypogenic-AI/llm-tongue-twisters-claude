"""
V6 Final: Extreme length push + comprehensive summary.

Push to 20k-50k chars to find the breaking point.
Also test the exact v3 math history document that caused gpt-4o-mini issues.
"""

import json
import os
import time
import random
import string
from datetime import datetime

import openai
import Levenshtein
import numpy as np

random.seed(42)
np.random.seed(42)

BASE = "/workspaces/llm-tongue-twisters-claude"
client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])

PROMPT = "Repeat the following text exactly, character for character. Output ONLY the text, nothing else.\n\n{text}"

PARAGRAPH = """The development of artificial intelligence has been one of the most transformative technological advances. Beginning with early neural network research, the field has gone through several waves of enthusiasm and disappointment. However, the current wave, driven by deep learning, has produced results that exceed even the most optimistic predictions."""


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
            time.sleep(2 ** (attempt + 1))
    return None


def compute_metrics(expected, actual):
    if actual is None:
        return {"exact_match": False, "edit_distance": -1,
                "normalized_edit_distance": 1.0, "char_accuracy": 0.0, "length_ratio": 0.0}
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


def make_long_text(target_chars, vary=True):
    """Generate text of target length."""
    text = ""
    i = 0
    while len(text) < target_chars:
        if vary:
            para = PARAGRAPH.replace("artificial intelligence",
                                     f"artificial intelligence (paragraph {i+1})")
        else:
            para = PARAGRAPH
        text += para + "\n\n"
        i += 1
    return text[:target_chars]


def make_misspelled(text, density=0.15):
    words = text.split()
    n = int(len(words) * density)
    indices = random.sample(range(len(words)), min(n, len(words)))
    for idx in indices:
        w = words[idx]
        if len(w) >= 4 and w.isalpha():
            pos = random.randint(1, len(w)-2)
            words[idx] = w[:pos] + w[pos+1] + w[pos] + w[pos+2:]
    return " ".join(words)


def main():
    print(f"{'='*70}")
    print(f"V6 Final: Extreme Length Tests")
    print(f"Started: {datetime.now().isoformat()}")
    print(f"{'='*70}")

    all_results = []
    models = ["gpt-4o-mini", "gpt-4.1-mini", "gpt-4.1"]

    for model in models:
        print(f"\n{'='*70}")
        print(f"Model: {model}")
        print(f"{'='*70}")

        # Test 1: Extreme length clean text
        print(f"\n--- Extreme Length (clean, varied paragraphs) ---")
        for target_len in [8000, 15000, 25000]:
            random.seed(42)
            text = make_long_text(target_len, vary=True)
            prompt = PROMPT.format(text=text)
            output = call_model(prompt, model=model)
            metrics = compute_metrics(text, output)
            result = {"experiment": "extreme_length_clean", "target_len": target_len,
                      "actual_len": len(text), "model": model, **metrics}
            all_results.append(result)
            print(f"  {target_len} chars: exact={metrics['exact_match']}, edit={metrics['edit_distance']}, "
                  f"len_ratio={metrics['length_ratio']:.4f}")

        # Test 2: Extreme length repetitive (same paragraph repeated)
        print(f"\n--- Extreme Length (repetitive, same paragraph) ---")
        for target_len in [5000, 10000, 20000]:
            random.seed(42)
            text = make_long_text(target_len, vary=False)
            prompt = PROMPT.format(text=text)
            output = call_model(prompt, model=model)
            metrics = compute_metrics(text, output)
            result = {"experiment": "extreme_length_repetitive", "target_len": target_len,
                      "actual_len": len(text), "model": model, **metrics}
            all_results.append(result)
            print(f"  {target_len} chars (repetitive): exact={metrics['exact_match']}, edit={metrics['edit_distance']}, "
                  f"len_ratio={metrics['length_ratio']:.4f}")

        # Test 3: Extreme length with misspellings
        print(f"\n--- Extreme Length (misspelled, density=15%) ---")
        for target_len in [8000, 15000, 25000]:
            random.seed(42)
            text = make_misspelled(make_long_text(target_len), 0.15)
            prompt = PROMPT.format(text=text)
            output = call_model(prompt, model=model)
            metrics = compute_metrics(text, output)
            result = {"experiment": "extreme_length_misspelled", "target_len": target_len,
                      "actual_len": len(text), "model": model, **metrics}
            all_results.append(result)
            print(f"  {target_len} chars (misspelled): exact={metrics['exact_match']}, edit={metrics['edit_distance']}, "
                  f"norm_edit={metrics['normalized_edit_distance']:.4f}")

        # Test 4: Long random string
        print(f"\n--- Long Random Strings ---")
        for length in [2000, 5000, 10000]:
            random.seed(42)
            text = ''.join(random.choice(string.ascii_letters + string.digits + ' ') for _ in range(length))
            prompt = PROMPT.format(text=text)
            output = call_model(prompt, model=model)
            metrics = compute_metrics(text, output)
            result = {"experiment": "long_random", "length": length, "model": model, **metrics}
            all_results.append(result)
            print(f"  {length} random chars: exact={metrics['exact_match']}, edit={metrics['edit_distance']}")

    # Save
    out_path = os.path.join(BASE, "results/raw_results_v6.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)
    print(f"\nSaved {len(all_results)} results to {out_path}")
    print(f"Finished: {datetime.now().isoformat()}")


if __name__ == "__main__":
    main()
