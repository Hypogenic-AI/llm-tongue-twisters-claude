"""
Run tongue twister experiments on LLMs via API.

Tests whether models can faithfully reproduce:
1. Known glitch tokens
2. Normal control tokens
3. Adversarial/unusual strings
4. Clean vs misspelled documents
"""

import json
import os
import time
import random
import sys
from datetime import datetime

import openai
import Levenshtein
import numpy as np

random.seed(42)
np.random.seed(42)

BASE = "/workspaces/llm-tongue-twisters-claude"

# API setup
client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])

# Also try OpenRouter for additional models
OPENROUTER_KEY = os.environ.get("OPENROUTER_API_KEY", "")
openrouter_client = None
if OPENROUTER_KEY:
    openrouter_client = openai.OpenAI(
        api_key=OPENROUTER_KEY,
        base_url="https://openrouter.ai/api/v1",
    )


def call_model(prompt, model="gpt-4.1", temperature=0, max_tokens=2048, provider="openai"):
    """Call an LLM API with retry logic."""
    cl = client if provider == "openai" else openrouter_client
    if cl is None:
        return None

    for attempt in range(3):
        try:
            resp = cl.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return resp.choices[0].message.content
        except Exception as e:
            print(f"  API error (attempt {attempt+1}): {e}")
            time.sleep(2 ** attempt)
    return None


def compute_metrics(expected, actual):
    """Compute reproduction quality metrics."""
    if actual is None:
        return {"exact_match": False, "edit_distance": len(expected), "normalized_edit_distance": 1.0,
                "char_accuracy": 0.0, "length_ratio": 0.0, "error": "api_failure"}

    exact = expected == actual
    edit_dist = Levenshtein.distance(expected, actual)
    max_len = max(len(expected), len(actual), 1)
    norm_edit = edit_dist / max_len

    # Character-level accuracy
    matching = sum(1 for a, b in zip(expected, actual) if a == b)
    char_acc = matching / max(len(expected), 1)

    length_ratio = len(actual) / max(len(expected), 1)

    return {
        "exact_match": exact,
        "edit_distance": edit_dist,
        "normalized_edit_distance": norm_edit,
        "char_accuracy": char_acc,
        "length_ratio": length_ratio,
    }


def analyze_misspelling_correction(original_clean, original_misspelled, model_output):
    """Analyze whether model corrected misspellings toward the clean version."""
    if model_output is None:
        return {"correction_tendency": None}

    # Distance from output to misspelled (what it should match)
    dist_to_misspelled = Levenshtein.distance(original_misspelled, model_output)
    # Distance from output to clean (what it might auto-correct toward)
    dist_to_clean = Levenshtein.distance(original_clean, model_output)

    # If output is closer to clean than to misspelled, model is "auto-correcting"
    correction_tendency = dist_to_misspelled - dist_to_clean  # positive = corrected toward clean

    return {
        "dist_to_misspelled": dist_to_misspelled,
        "dist_to_clean": dist_to_clean,
        "correction_tendency": correction_tendency,
        "auto_corrected": dist_to_clean < dist_to_misspelled,
    }


PROMPT_TEMPLATES = {
    "direct": "Repeat the following text exactly, character for character. Output ONLY the text, nothing else.\n\n{text}",
    "instruction": "Your task is to copy the following text exactly as written, preserving any errors, misspellings, or unusual formatting. Output ONLY the copied text.\n\n{text}",
    "strict": 'Please output the following string exactly as-is. Do not correct, modify, or explain it. Just output the exact string:\n\n"""\n{text}\n"""',
}


def run_token_experiment(tokens, category, model="gpt-4.1", provider="openai", template_name="direct"):
    """Test model's ability to reproduce individual tokens/strings."""
    template = PROMPT_TEMPLATES[template_name]
    results = []

    for i, token in enumerate(tokens):
        prompt = template.format(text=token)
        output = call_model(prompt, model=model, provider=provider)
        metrics = compute_metrics(token, output)

        result = {
            "category": category,
            "stimulus": token,
            "template": template_name,
            "model": model,
            "output": output,
            **metrics,
        }
        results.append(result)

        status = "✓" if metrics["exact_match"] else "✗"
        if not metrics["exact_match"] and output is not None:
            print(f"  {status} [{category}] '{token}' → '{output[:80]}...' (edit_dist={metrics['edit_distance']})")
        elif i % 10 == 0:
            print(f"  Progress: {i+1}/{len(tokens)}")

    return results


def run_document_experiment(documents, model="gpt-4.1", provider="openai", template_name="instruction"):
    """Test model's ability to reproduce clean vs misspelled documents."""
    template = PROMPT_TEMPLATES[template_name]
    results = []

    for doc in documents:
        for version in ["clean", "misspelled"]:
            text = doc[version]
            prompt = template.format(text=text)
            output = call_model(prompt, model=model, provider=provider, max_tokens=4096)
            metrics = compute_metrics(text, output)

            result = {
                "category": f"document_{version}",
                "doc_name": doc["name"],
                "doc_length": doc["length"],
                "stimulus_len": len(text),
                "template": template_name,
                "model": model,
                "stimulus": text,
                "output": output,
                **metrics,
            }

            # For misspelled version, also analyze correction tendency
            if version == "misspelled" and output is not None:
                correction = analyze_misspelling_correction(doc["clean"], doc["misspelled"], output)
                result.update(correction)

            results.append(result)

            status = "✓" if metrics["exact_match"] else "✗"
            print(f"  {status} [{doc['name']}_{version}] exact={metrics['exact_match']}, "
                  f"norm_edit={metrics['normalized_edit_distance']:.3f}, len_ratio={metrics['length_ratio']:.3f}")

    return results


def main():
    print(f"=" * 70)
    print(f"LLM Tongue Twisters Experiment")
    print(f"Started: {datetime.now().isoformat()}")
    print(f"=" * 70)

    # Load stimuli
    with open(os.path.join(BASE, "results/stimuli.json")) as f:
        stimuli = json.load(f)

    all_results = []
    models_to_test = [("gpt-4.1", "openai")]

    # Add OpenRouter models if key is available
    if openrouter_client:
        models_to_test.append(("anthropic/claude-sonnet-4", "openrouter"))

    for model, provider in models_to_test:
        print(f"\n{'='*70}")
        print(f"Testing model: {model} (via {provider})")
        print(f"{'='*70}")

        # Experiment 1: Glitch tokens
        print(f"\n--- Experiment 1: Glitch Token Reproduction ---")
        for template in ["direct", "strict"]:
            print(f"\n  Template: {template}")
            results = run_token_experiment(
                stimuli["glitch_tokens"], "glitch_token", model=model,
                provider=provider, template_name=template
            )
            all_results.extend(results)

        # Experiment 2: Control tokens
        print(f"\n--- Experiment 2: Control Token Reproduction ---")
        for template in ["direct", "strict"]:
            print(f"\n  Template: {template}")
            results = run_token_experiment(
                stimuli["control_tokens"], "control_token", model=model,
                provider=provider, template_name=template
            )
            all_results.extend(results)

        # Experiment 3: Adversarial strings
        print(f"\n--- Experiment 3: Adversarial String Reproduction ---")
        for template in ["direct", "strict"]:
            print(f"\n  Template: {template}")
            results = run_token_experiment(
                stimuli["adversarial_strings"], "adversarial", model=model,
                provider=provider, template_name=template
            )
            all_results.extend(results)

        # Experiment 4: Document reproduction (clean vs misspelled)
        print(f"\n--- Experiment 4: Document Reproduction ---")
        for template in ["instruction", "strict"]:
            print(f"\n  Template: {template}")
            results = run_document_experiment(
                stimuli["documents"], model=model,
                provider=provider, template_name=template
            )
            all_results.extend(results)

    # Save all results
    out_path = os.path.join(BASE, "results/raw_results.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)

    print(f"\n{'='*70}")
    print(f"Experiment complete. {len(all_results)} total results saved to {out_path}")
    print(f"Finished: {datetime.now().isoformat()}")

    # Quick summary
    for model, _ in models_to_test:
        model_results = [r for r in all_results if r["model"] == model]
        for cat in ["glitch_token", "control_token", "adversarial", "document_clean", "document_misspelled"]:
            cat_results = [r for r in model_results if r["category"] == cat]
            if cat_results:
                exact_rate = sum(1 for r in cat_results if r["exact_match"]) / len(cat_results)
                avg_edit = np.mean([r["normalized_edit_distance"] for r in cat_results])
                print(f"  {model} | {cat:25s} | exact_match={exact_rate:.1%} | avg_norm_edit={avg_edit:.3f}")


if __name__ == "__main__":
    main()
