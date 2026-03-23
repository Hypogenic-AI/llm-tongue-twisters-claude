"""
Extended tongue twister experiments with harder stimuli.

Key findings from v1:
- GPT-4.1 with 'direct' template reproduced almost all tokens perfectly
- The 'strict' template caused triple-quote wrapping artifacts
- Short misspelled documents were perfectly reproduced

This v2 tests:
1. Much longer documents with many misspellings
2. Documents with subtly wrong content (semantic "misspellings")
3. Multiple models for comparison
4. Glitch tokens as substrings within sentences
"""

import json
import os
import time
import random
from datetime import datetime

import openai
import Levenshtein
import numpy as np

random.seed(42)
np.random.seed(42)

BASE = "/workspaces/llm-tongue-twisters-claude"

client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])

OPENROUTER_KEY = os.environ.get("OPENROUTER_API_KEY", "")
openrouter_client = None
if OPENROUTER_KEY:
    openrouter_client = openai.OpenAI(
        api_key=OPENROUTER_KEY,
        base_url="https://openrouter.ai/api/v1",
    )


def call_model(prompt, model="gpt-4.1", temperature=0, max_tokens=4096, provider="openai"):
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
    if actual is None:
        return {"exact_match": False, "edit_distance": len(expected),
                "normalized_edit_distance": 1.0, "char_accuracy": 0.0, "error": "api_failure"}
    # Strip leading/trailing whitespace for fair comparison
    actual_stripped = actual.strip()
    expected_stripped = expected.strip()

    exact = expected_stripped == actual_stripped
    edit_dist = Levenshtein.distance(expected_stripped, actual_stripped)
    max_len = max(len(expected_stripped), len(actual_stripped), 1)
    norm_edit = edit_dist / max_len
    matching = sum(1 for a, b in zip(expected_stripped, actual_stripped) if a == b)
    char_acc = matching / max(len(expected_stripped), 1)
    length_ratio = len(actual_stripped) / max(len(expected_stripped), 1)

    return {
        "exact_match": exact,
        "edit_distance": edit_dist,
        "normalized_edit_distance": norm_edit,
        "char_accuracy": char_acc,
        "length_ratio": length_ratio,
        "output_stripped": actual_stripped,
    }


def find_misspelling_corrections(original_misspelled, model_output, original_clean):
    """Find specific words that the model corrected."""
    if model_output is None:
        return []

    misspelled_words = original_misspelled.split()
    clean_words = original_clean.split()
    output_words = model_output.strip().split()

    corrections = []
    for i, (m_word, c_word) in enumerate(zip(misspelled_words, clean_words)):
        if m_word != c_word and i < len(output_words):
            o_word = output_words[i]
            if o_word == c_word:
                corrections.append({"index": i, "misspelled": m_word, "clean": c_word, "output": o_word, "corrected": True})
            elif o_word == m_word:
                corrections.append({"index": i, "misspelled": m_word, "clean": c_word, "output": o_word, "corrected": False})
            else:
                corrections.append({"index": i, "misspelled": m_word, "clean": c_word, "output": o_word, "corrected": "other"})
    return corrections


# ===== HARD STIMULI =====

def build_long_misspelled_documents():
    """Build longer documents with more misspellings for harder testing."""
    docs = []

    # Very long document (~800 words) with dense misspellings
    clean_long = """The development of artificial intelligence has been one of the most transformative technological advances of the twenty-first century. Beginning with early neural network research in the nineteen fifties, the field has gone through several waves of enthusiasm and disappointment, often referred to as AI winters. However, the current wave, driven by deep learning and massive computational resources, has produced results that exceed even the most optimistic predictions of earlier decades.

Large language models represent perhaps the most visible achievement of modern artificial intelligence. These models, trained on billions of words of text from the internet, books, and other sources, can generate remarkably human-like text, answer questions, write code, and perform tasks that were considered impossible just a few years ago. The transformer architecture, introduced in the landmark paper "Attention Is All You Need" by Vaswani and colleagues in twenty seventeen, provided the foundation for these advances.

Despite their impressive capabilities, language models have significant limitations. They can produce confident-sounding but incorrect information, a phenomenon known as hallucination. They struggle with mathematical reasoning, often making errors in multi-step calculations. They cannot truly understand causality or maintain persistent memory across conversations. These limitations have important implications for how these systems should be deployed and trusted.

The ethical considerations surrounding artificial intelligence are equally complex. Questions about bias in training data, environmental impact of large-scale model training, potential job displacement, and the concentration of AI capabilities in a few large technology companies are actively debated. Researchers and policymakers around the world are working to establish frameworks for responsible AI development that balance innovation with safety and fairness.

Looking forward, the field continues to evolve rapidly. Multimodal models that can process text, images, audio, and video simultaneously are becoming more capable. Reasoning abilities are improving through techniques like chain-of-thought prompting and reinforcement learning from human feedback. The integration of AI systems with external tools and databases promises to address some of the current limitations around factual accuracy and real-time information access."""

    misspelled_long = """The developement of artifical intelligence has been one of the most tranformative technologicle advances of the twenty-first centruey. Begining with early nueral network reserach in the ninteen fifties, the feild has gone through several wavse of enthusiasm and disapointment, often refered to as AI wintres. However, the curent wave, dirven by deep leanring and masive computational resourses, has produced resutls that excede even the most optimisitc predictoins of eariler decades.

Large langauge models represnet perhaps the most visable acheivement of modern artifical intelligence. These modlse, trained on billoins of words of text from the intarnet, books, and other sourses, can generate remarkabely human-like text, answre questions, write cdoe, and preform tasks that were considred impossible just a few yaers ago. The transfomer architecture, introduced in the landmrak paper "Attention Is All You Nede" by Vaswani and colleaguse in twenty seventen, provided the foundaiton for these advacnes.

Despite thier impressive capabilites, langauge models have signficant limitatoins. They can produce confidnet-sounding but incorect information, a phenomonon known as hallucinatoin. They strugle with mathmatical reasoning, often makign errors in multi-step calculatoins. They canot truly undersatnd causality or maintain persistant memory accross conversations. These limitatoins have importnat implications for how these systmes should be deployde and trustde.

The ethcial considerations surounding artifical intelligence are equaly complex. Questoins about bais in training data, enviromental impact of large-scale model traning, potenial job displacment, and the concentratoin of AI capabilites in a few large technolgy companies are activley debated. Reserchers and policymakres around the world are workign to establsih frameworks for responsibe AI developement that balacne innovation with saftey and fariness.

Looking forwrad, the feild continues to evolev rapidly. Multimodl models that can processe text, images, auido, and video simultanously are becomign more capabel. Reasioning abilities are improvign through techniquse like chain-of-thougth prompting and reinforcment learning from human feedbakc. The integratoin of AI systmes with external tools and databses promises to addres some of the curent limitatoins around facutal accuracy and real-time informatoin access."""

    docs.append({
        "name": "long_ai_history",
        "clean": clean_long,
        "misspelled": misspelled_long,
        "length": "very_long",
        "n_misspellings": 95,  # approximate
    })

    # Document with VERY subtle misspellings (just one letter swaps)
    clean_subtle = """Python is a high-level programming language known for its clear syntax and readability. Created by Guido van Rossum and first released in nineteen ninety-one, Python has become one of the most popular languages in the world. It supports multiple programming paradigms, including procedural, object-oriented, and functional programming. Python's extensive standard library and rich ecosystem of third-party packages make it suitable for web development, data analysis, machine learning, scientific computing, and automation."""

    misspelled_subtle = """Python is a high-level programming langauge known for its clear syntax and readability. Created by Guido van Rossum and first released in nineteen ninety-one, Python has become one of the most popular langauges in the world. It supports multiple programming paradigms, including procedural, object-oriented, and functional programming. Python's extensive standard librayr and rich ecosystem of third-party packages make it suitable for web developemnt, data analysis, machine learning, scientific computing, and automation."""

    docs.append({
        "name": "subtle_python_desc",
        "clean": clean_subtle,
        "misspelled": misspelled_subtle,
        "length": "medium",
        "n_misspellings": 3,  # very few, very subtle
    })

    # Document with number/factual errors instead of spelling errors
    clean_facts = """The Earth orbits the Sun at an average distance of approximately 150 million kilometers. A year on Earth is approximately 365.25 days long, which is why we have leap years every four years. The Moon orbits the Earth at an average distance of about 384,400 kilometers, and it takes approximately 27.3 days to complete one orbit. Light from the Sun takes about 8 minutes and 20 seconds to reach Earth."""

    wrong_facts = """The Earth orbits the Sun at an average distance of approximately 150 million kilometers. A year on Earth is approximately 364.25 days long, which is why we have leap years every four years. The Moon orbits the Earth at an average distance of about 384,400 kilometers, and it takes approximately 27.3 days to complete one orbit. Light from the Sun takes about 8 minutes and 40 seconds to reach Earth."""

    docs.append({
        "name": "factual_errors",
        "clean": clean_facts,
        "misspelled": wrong_facts,  # reusing field name
        "length": "short",
        "n_misspellings": 2,  # factual errors: 364.25 vs 365.25, 40 vs 20
    })

    return docs


def build_embedded_glitch_sentences():
    """Glitch tokens embedded in normal sentences."""
    return [
        ("The username SolidGoldMagikarp was registered on the forum last Tuesday.",
         "SolidGoldMagikarp"),
        ("Please search for TheNitromeFan in the database and return all results.",
         "TheNitromeFan"),
        ("The function rawdownloadcloneembedreportprint handles file operations.",
         "rawdownloadcloneembedreportprint"),
        ("User petertodd submitted a pull request to the main branch.",
         "petertodd"),
        ("The class PsychExpandoExceptionObjectSyntax extends BaseException.",
         "PsychExpandoExceptionObjectSyntax"),
        ("Import the module cloneembedaliased from the utils package.",
         "cloneembedaliased"),
        ("The variable guiActiveUn controls the interface state.",
         "guiActiveUn"),
        ("Channel StreamerBot is currently broadcasting live content.",
         "StreamerBot"),
        ("The tag InstoreAndOnline indicates product availability.",
         "InstoreAndOnline"),
        ("Handler ActionCodeHandler processes verification requests.",
         "ActionCodeHandler"),
    ]


def run_experiment():
    print(f"{'='*70}")
    print(f"LLM Tongue Twisters V2 — Extended Experiments")
    print(f"Started: {datetime.now().isoformat()}")
    print(f"{'='*70}")

    all_results = []

    models = [("gpt-4.1", "openai")]
    if openrouter_client:
        models.append(("anthropic/claude-sonnet-4", "openrouter"))

    prompt_template = (
        "Repeat the following text exactly, character for character. "
        "Output ONLY the text, nothing else.\n\n{text}"
    )

    for model, provider in models:
        print(f"\n{'='*70}")
        print(f"Model: {model}")
        print(f"{'='*70}")

        # --- Test 1: Long misspelled documents ---
        print(f"\n--- Test 1: Long Document Reproduction (clean vs misspelled) ---")
        docs = build_long_misspelled_documents()

        for doc in docs:
            for version in ["clean", "misspelled"]:
                text = doc[version]
                prompt = prompt_template.format(text=text)
                output = call_model(prompt, model=model, provider=provider, max_tokens=8192)
                metrics = compute_metrics(text, output)

                result = {
                    "experiment": "long_document",
                    "category": f"document_{version}",
                    "doc_name": doc["name"],
                    "doc_length": doc["length"],
                    "n_misspellings": doc.get("n_misspellings", 0),
                    "stimulus_len": len(text),
                    "model": model,
                    "template": "direct",
                    **metrics,
                }

                # Correction analysis for misspelled versions
                if version == "misspelled":
                    corrections = find_misspelling_corrections(doc["misspelled"], output, doc["clean"])
                    n_corrected = sum(1 for c in corrections if c["corrected"] is True)
                    n_preserved = sum(1 for c in corrections if c["corrected"] is False)
                    n_other = sum(1 for c in corrections if c["corrected"] == "other")
                    result["n_words_corrected"] = n_corrected
                    result["n_words_preserved"] = n_preserved
                    result["n_words_other"] = n_other
                    result["correction_rate"] = n_corrected / max(n_corrected + n_preserved + n_other, 1)
                    result["corrections_detail"] = corrections[:20]  # sample

                    dist_to_clean = Levenshtein.distance(doc["clean"].strip(), (output or "").strip())
                    dist_to_misspelled = Levenshtein.distance(doc["misspelled"].strip(), (output or "").strip())
                    result["dist_to_clean"] = dist_to_clean
                    result["dist_to_misspelled"] = dist_to_misspelled
                    result["auto_corrected"] = dist_to_clean < dist_to_misspelled

                all_results.append(result)

                status = "✓" if metrics["exact_match"] else "✗"
                extra = ""
                if version == "misspelled" and "correction_rate" in result:
                    extra = f" correction_rate={result['correction_rate']:.1%}"
                print(f"  {status} [{doc['name']}_{version}] exact={metrics['exact_match']}, "
                      f"norm_edit={metrics['normalized_edit_distance']:.4f}{extra}")

        # --- Test 2: Embedded glitch tokens in sentences ---
        print(f"\n--- Test 2: Glitch Tokens Embedded in Sentences ---")
        sentences = build_embedded_glitch_sentences()

        for sentence, glitch_word in sentences:
            prompt = prompt_template.format(text=sentence)
            output = call_model(prompt, model=model, provider=provider)
            metrics = compute_metrics(sentence, output)

            # Check if glitch word specifically was preserved
            output_stripped = (output or "").strip()
            glitch_preserved = glitch_word in output_stripped

            result = {
                "experiment": "embedded_glitch",
                "category": "embedded_glitch",
                "stimulus": sentence,
                "glitch_word": glitch_word,
                "model": model,
                "template": "direct",
                "glitch_word_preserved": glitch_preserved,
                **metrics,
            }
            all_results.append(result)

            status = "✓" if metrics["exact_match"] else "✗"
            gw_status = "✓" if glitch_preserved else "✗"
            print(f"  {status} glitch_word={gw_status} '{glitch_word}' → exact={metrics['exact_match']}, edit={metrics['edit_distance']}")

        # --- Test 3: Repeated reproduction (consistency) ---
        print(f"\n--- Test 3: Reproduction Consistency (5 runs, temperature=0.7) ---")
        consistency_stimuli = [
            ("SolidGoldMagikarp", "glitch"),
            ("TheNitromeFan", "glitch"),
            ("computer", "control"),
            ("algorithm", "control"),
        ]

        for stimulus, cat in consistency_stimuli:
            outputs = []
            for run in range(5):
                prompt = prompt_template.format(text=stimulus)
                output = call_model(prompt, model=model, provider=provider, temperature=0.7)
                outputs.append(output)

            unique_outputs = set(o.strip() if o else None for o in outputs)
            exact_matches = sum(1 for o in outputs if o and o.strip() == stimulus)

            result = {
                "experiment": "consistency",
                "category": f"consistency_{cat}",
                "stimulus": stimulus,
                "model": model,
                "n_runs": 5,
                "n_exact_matches": exact_matches,
                "n_unique_outputs": len(unique_outputs),
                "outputs": [o.strip() if o else None for o in outputs],
            }
            all_results.append(result)
            print(f"  '{stimulus}' ({cat}): {exact_matches}/5 exact, {len(unique_outputs)} unique outputs")

        # --- Test 4: Instruction-following under pressure ---
        print(f"\n--- Test 4: Explicit Instruction to Preserve Errors ---")
        error_doc = docs[0]  # long AI document with many misspellings

        explicit_prompts = [
            # Increasingly emphatic instructions
            ("basic",
             "Copy the following text exactly as written:\n\n{text}"),
            ("emphatic",
             "IMPORTANT: Copy the following text EXACTLY as written. Do NOT correct any spelling errors. "
             "Preserve every single character exactly as it appears, including all misspellings and errors.\n\n{text}"),
            ("role_play",
             "You are a text copying machine. Your ONLY job is to reproduce text character-for-character. "
             "You must NOT fix spelling, grammar, or factual errors. Output the text below exactly as-is:\n\n{text}"),
        ]

        for prompt_name, template in explicit_prompts:
            text = error_doc["misspelled"]
            prompt = template.format(text=text)
            output = call_model(prompt, model=model, provider=provider, max_tokens=8192)
            metrics = compute_metrics(text, output)

            corrections = find_misspelling_corrections(error_doc["misspelled"], output, error_doc["clean"])
            n_corrected = sum(1 for c in corrections if c["corrected"] is True)
            n_preserved = sum(1 for c in corrections if c["corrected"] is False)
            n_other = sum(1 for c in corrections if c["corrected"] == "other")
            correction_rate = n_corrected / max(n_corrected + n_preserved + n_other, 1)

            result = {
                "experiment": "instruction_pressure",
                "category": "instruction_pressure",
                "prompt_name": prompt_name,
                "model": model,
                "doc_name": error_doc["name"],
                "n_words_corrected": n_corrected,
                "n_words_preserved": n_preserved,
                "n_words_other": n_other,
                "correction_rate": correction_rate,
                **metrics,
            }
            all_results.append(result)
            print(f"  [{prompt_name}] exact={metrics['exact_match']}, correction_rate={correction_rate:.1%}, "
                  f"corrected={n_corrected}, preserved={n_preserved}, other={n_other}")

    # Save results
    out_path = os.path.join(BASE, "results/raw_results_v2.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)
    print(f"\nSaved {len(all_results)} results to {out_path}")
    print(f"Finished: {datetime.now().isoformat()}")


if __name__ == "__main__":
    run_experiment()
