"""
V5: Extreme length tests and random string reproduction.

Push models to their limits:
1. Very long documents (5k, 10k, 20k chars)
2. Random character strings
3. Partially corrupted text (random char insertions)
4. Adversarial patterns: repeated tokens, code with syntax errors
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


# === Build extreme stimuli ===

PARAGRAPH = """The development of artificial intelligence has been one of the most transformative technological advances. Beginning with early neural network research, the field has gone through several waves of enthusiasm and disappointment. However, the current wave, driven by deep learning, has produced results that exceed even the most optimistic predictions."""

def make_long_text(target_chars, misspell_density=0.0):
    """Generate text of approximately target_chars length."""
    text = ""
    i = 0
    while len(text) < target_chars:
        para = PARAGRAPH.replace("artificial intelligence", f"artificial intelligence (section {i+1})")
        text += para + "\n\n"
        i += 1
    text = text[:target_chars]

    if misspell_density > 0:
        words = text.split()
        n_misspell = int(len(words) * misspell_density)
        indices = random.sample(range(len(words)), min(n_misspell, len(words)))
        for idx in indices:
            w = words[idx]
            if len(w) >= 4 and w.isalpha():
                pos = random.randint(1, len(w)-2)
                words[idx] = w[:pos] + w[pos+1] + w[pos] + w[pos+2:]
        text = " ".join(words)
    return text


def make_random_string(length, charset="alphanumeric"):
    """Generate a random string."""
    if charset == "alphanumeric":
        chars = string.ascii_letters + string.digits
    elif charset == "ascii":
        chars = string.printable.replace('\t', '').replace('\n', '').replace('\r', '').replace('\x0b', '').replace('\x0c', '')
    elif charset == "hex":
        chars = "0123456789abcdef"
    elif charset == "words":
        # Random "words" that look plausible
        vowels = "aeiou"
        consonants = "bcdfghjklmnpqrstvwxyz"
        words = []
        for _ in range(length // 6):  # ~6 chars per word
            word_len = random.randint(3, 8)
            word = ""
            for j in range(word_len):
                word += random.choice(consonants if j % 2 == 0 else vowels)
            words.append(word)
        return " ".join(words)
    else:
        chars = charset
    return "".join(random.choice(chars) for _ in range(length))


def make_code_with_errors():
    """Python code with deliberate syntax errors that should be preserved."""
    return '''def calculate_averge(numbers):
    """Calcualte the averge of a list of numbres."""
    if len(numbers) = 0:  # should be ==
        retrun None
    total = smu(numbers)  # should be sum
    coutn = len(numbers)
    averge = total / coutn
    retrun averge

def proccess_data(dta):
    """Proccess the input dta and retrun results."""
    resutls = []
    for itme in dta:
        if itme > 0
            resutls.append(calcualte_averge([itme]))
    retrun resutls

# Main executoin
if __name__ == "__mian__":
    data = [1, 2, 3, 4, 5]
    resutl = proccess_data(data)
    pirnt(f"Result: {resutl}")'''


def main():
    print(f"{'='*70}")
    print(f"V5: Extreme Length + Random String Tests")
    print(f"Started: {datetime.now().isoformat()}")
    print(f"{'='*70}")

    all_results = []
    models = ["gpt-4o-mini", "gpt-4.1-mini", "gpt-4.1"]

    for model in models:
        print(f"\n{'='*70}")
        print(f"Model: {model}")
        print(f"{'='*70}")

        # Test 1: Increasing document length (clean)
        print(f"\n--- Increasing Length (clean text) ---")
        for target_len in [500, 1000, 2000, 5000, 10000]:
            random.seed(42)
            text = make_long_text(target_len)
            prompt = PROMPT.format(text=text)
            output = call_model(prompt, model=model)
            metrics = compute_metrics(text, output)
            result = {"experiment": "length_clean", "target_len": target_len,
                      "actual_len": len(text), "model": model, **metrics}
            all_results.append(result)
            print(f"  {target_len} chars: exact={metrics['exact_match']}, edit={metrics['edit_distance']}, "
                  f"len_ratio={metrics['length_ratio']:.4f}")

        # Test 2: Increasing length (misspelled)
        print(f"\n--- Increasing Length (misspelled, density=20%) ---")
        for target_len in [500, 1000, 2000, 5000, 10000]:
            random.seed(42)
            text = make_long_text(target_len, misspell_density=0.20)
            prompt = PROMPT.format(text=text)
            output = call_model(prompt, model=model)
            metrics = compute_metrics(text, output)
            result = {"experiment": "length_misspelled", "target_len": target_len,
                      "actual_len": len(text), "model": model, **metrics}
            all_results.append(result)
            print(f"  {target_len} chars: exact={metrics['exact_match']}, edit={metrics['edit_distance']}, "
                  f"norm_edit={metrics['normalized_edit_distance']:.4f}")

        # Test 3: Random strings of varying lengths
        print(f"\n--- Random Alphanumeric Strings ---")
        for length in [10, 50, 100, 500, 1000]:
            random.seed(42)
            text = make_random_string(length, "alphanumeric")
            prompt = PROMPT.format(text=text)
            output = call_model(prompt, model=model)
            metrics = compute_metrics(text, output)
            result = {"experiment": "random_alphanum", "length": length, "model": model,
                      "stimulus": text[:200], "output": ((output or "").strip())[:200], **metrics}
            all_results.append(result)
            print(f"  {length} chars: exact={metrics['exact_match']}, edit={metrics['edit_distance']}")

        # Test 4: Random "word-like" strings
        print(f"\n--- Random Pseudo-Words ---")
        for length in [50, 200, 500, 1000]:
            random.seed(42)
            text = make_random_string(length, "words")
            prompt = PROMPT.format(text=text)
            output = call_model(prompt, model=model)
            metrics = compute_metrics(text, output)
            result = {"experiment": "random_words", "length": length, "model": model,
                      "stimulus": text[:200], "output": ((output or "").strip())[:200], **metrics}
            all_results.append(result)
            print(f"  ~{length} chars: exact={metrics['exact_match']}, edit={metrics['edit_distance']}")

        # Test 5: Code with syntax errors
        print(f"\n--- Code with Deliberate Errors ---")
        code = make_code_with_errors()
        prompt = PROMPT.format(text=code)
        output = call_model(prompt, model=model)
        metrics = compute_metrics(code, output)
        result = {"experiment": "code_errors", "model": model,
                  "stimulus": code, "output": (output or "").strip(), **metrics}
        all_results.append(result)
        print(f"  exact={metrics['exact_match']}, edit={metrics['edit_distance']}")
        if not metrics['exact_match'] and output:
            # Show specific changes
            code_lines = code.strip().split('\n')
            out_lines = output.strip().split('\n')
            for i, (cl, ol) in enumerate(zip(code_lines, out_lines)):
                if cl != ol:
                    print(f"    Line {i+1}: '{cl}' → '{ol}'")

        # Test 6: Hex string (looks like hash)
        print(f"\n--- Hex Strings (hash-like) ---")
        for length in [32, 64, 128, 256]:
            random.seed(42)
            text = make_random_string(length, "hex")
            prompt = PROMPT.format(text=text)
            output = call_model(prompt, model=model)
            metrics = compute_metrics(text, output)
            result = {"experiment": "hex_string", "length": length, "model": model, **metrics}
            all_results.append(result)
            print(f"  {length} chars: exact={metrics['exact_match']}, edit={metrics['edit_distance']}")

    # Save
    out_path = os.path.join(BASE, "results/raw_results_v5.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)
    print(f"\nSaved {len(all_results)} results to {out_path}")
    print(f"Finished: {datetime.now().isoformat()}")


if __name__ == "__main__":
    main()
