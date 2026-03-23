"""
V3: Test with multiple models (including smaller ones) and extreme stimuli.

GPT-4.1 aced everything. Now test:
1. GPT-4o-mini (smaller, cheaper, more likely to fail)
2. GPT-4.1-mini
3. Much longer documents (push context limits)
4. Adversarial edge cases that might trip up even strong models
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


def call_model(prompt, model="gpt-4o-mini", temperature=0, max_tokens=8192):
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


def find_corrections(misspelled, output, clean):
    """Count how many misspelled words got auto-corrected."""
    if output is None:
        return 0, 0, 0
    m_words = misspelled.split()
    c_words = clean.split()
    o_words = output.strip().split()
    corrected = preserved = other = 0
    for i, (m, c) in enumerate(zip(m_words, c_words)):
        if m != c and i < len(o_words):
            if o_words[i] == c:
                corrected += 1
            elif o_words[i] == m:
                preserved += 1
            else:
                other += 1
    return corrected, preserved, other


PROMPT = "Repeat the following text exactly, character for character. Output ONLY the text, nothing else.\n\n{text}"

# === Build extreme stimuli ===

# Very long document (1500+ words)
VERY_LONG_CLEAN = """The history of mathematics stretches back thousands of years, intertwining with the development of human civilization itself. From the earliest counting systems developed by ancient peoples to the sophisticated abstractions of modern mathematics, the journey reveals how deeply mathematical thinking is embedded in human cognition and culture.

The ancient Babylonians, around four thousand years ago, developed a sophisticated number system based on sixty, which is why we still divide hours into sixty minutes and circles into three hundred and sixty degrees. They could solve quadratic equations and had extensive tables of squares, cubes, and reciprocals. Their mathematics was primarily practical, driven by needs of agriculture, construction, and commerce.

Ancient Egypt contributed significantly to geometry, motivated by the annual flooding of the Nile which necessitated regular resurveying of agricultural land. The Rhind Papyrus, dating to around seventeen hundred before the common era, contains problems involving fractions, linear equations, and geometric calculations including approximations of the area of a circle.

Greek mathematics represented a revolutionary departure from the practical focus of earlier civilizations. The Greeks introduced the concept of mathematical proof, establishing a tradition of rigorous logical reasoning that remains the foundation of mathematics today. Euclid's Elements, compiled around three hundred before the common era, organized geometry into a deductive system built from a small number of axioms and postulates. This work remained the standard textbook for geometry for over two thousand years.

The contributions of Archimedes were particularly remarkable. He developed methods for calculating areas and volumes that anticipated integral calculus by nearly two thousand years. His method of exhaustion, used to find the area under a parabola, is a precursor to the limit concept that would later become central to analysis.

During the medieval period, Islamic mathematicians preserved and extended Greek mathematical knowledge while making significant original contributions. Al-Khwarizmi's work on algebra, from which the word itself derives, systematized methods for solving equations. His name also gave us the word algorithm. Omar Khayyam developed methods for solving cubic equations and worked on the binomial theorem.

Indian mathematicians made foundational contributions that often receive insufficient recognition. The development of the decimal place-value system, including the crucial concept of zero, originated in India. Brahmagupta provided rules for arithmetic with zero and negative numbers. The Kerala school of mathematics developed infinite series expansions for trigonometric functions centuries before European mathematicians.

The Renaissance brought mathematics back to prominence in European intellectual life. The solution of the cubic equation by Cardano and the quartic by Ferrari demonstrated that algebraic methods could solve problems that had resisted solution for centuries. The development of logarithms by Napier and the introduction of modern algebraic notation by Vieta and Descartes transformed mathematical practice.

The seventeenth century witnessed the creation of calculus, independently by Newton and Leibniz. This was perhaps the single most important development in the history of mathematics, providing powerful tools for analyzing change and motion. Newton used calculus to formulate his laws of motion and universal gravitation, while Leibniz developed the notation that we still use today.

The eighteenth and nineteenth centuries saw mathematics become increasingly abstract and rigorous. Euler made contributions to virtually every branch of mathematics, from number theory to topology. Gauss, often called the prince of mathematicians, made fundamental advances in algebra, analysis, geometry, and statistics. The development of non-Euclidean geometry by Lobachevsky, Bolyai, and Riemann showed that Euclid's parallel postulate was independent of his other axioms, opening up entirely new geometric universes."""

VERY_LONG_MISSPELLED = """The hisotry of mathemactics stretches back thosands of years, intertwining with the developement of human civilizaiton itself. From the earleist counting systmes developed by ancient peoples to the sophistacated abstractions of modrn mathematics, the journye reveals how deepyl mathematical thinkign is embeded in human cognitoin and cultrure.

The ancient Babylonains, around four thosand years ago, develoeped a sophistacated number systme based on sixty, which is why we sitll divide hours into sixty mintues and circles into three hundread and sixty degrees. They could sovle quadratic equatoins and had extensvie tables of squares, cubes, and reciporcals. Their mathemactics was primariliy practical, dirven by needs of agricultrue, constrcution, and commmerce.

Ancient Eygpt contributed signficantly to geometery, motivated by the annual floding of the Nile which necesitated regular resurveyign of agricultural land. The Rhind Papyrus, datign to around seventene hundred before the comon era, contians problems involving fractinos, linear equatoins, and geometirc calculations includign approximatoins of the area of a cirlce.

Greek mathemactics represented a revolutinoary departure from the pratical focus of eariler civilizaitons. The Greeks introduced the concpet of mathematcial proof, estalbishing a traditoin of rigoruos logical reasioning that remians the foundaiton of mathemactics today. Eucild's Elemtens, compiled around three hundread before the comon era, organized geometery into a deductve system built from a small numbr of axioms and postualtes. This work remaind the standrad textbook for geometery for over two thosand years.

The contributioins of Archimedes were particulraly remarkable. He develoeped methods for calculatign areas and volumes that anticpated integral calcualus by nearyl two thosand years. His metohd of exhaustoin, used to find the area under a parabola, is a precurser to the limit concpet that would later becmoe central to anlysis.

During the medeival period, Islamic mathematicans preserved and exetnded Greek mathematcial knowledge while makign signficant original contributioins. Al-Khwarizmi's work on algebre, from which the word itself derievs, systematiezd methods for solvign equations. His name aslo gave us the word algorithem. Omar Khayyam develoeped methods for solvign cubic equatoins and worked on the binomail theorem.

Indian mathematicans made foundatinal contributioins that often recieve insufficient recogniton. The developement of the decmial place-value systme, includign the crucail concept of zero, originated in India. Brahmagupta provdied rules for arithmetci with zero and negatvie numbers. The Kerela school of mathemactics develoeped infinite series expansoins for trigonometrc functions centruies before European mathematicans.

The Renaisance brought mathemactics back to prominnece in European intellecutal life. The soluton of the cubic equaiton by Cardano and the quartic by Ferrari demonstarted that algebraci methods could sovle problems that had resisted soluton for centruies. The developement of logaritmes by Napier and the introductoin of modrn algebraci notation by Vieta and Descrates transformed mathematcial practice.

The seventeeth century witnesed the creaiton of calcualus, independentyl by Newton and Leibniz. This was prehaps the single most importnat developement in the hisotry of mathemactics, providign powerful tools for analyzign change and moiton. Newton used calcualus to formualte his laws of moiton and universal gravitaiton, while Leibniz develoeped the notaiton that we sitll use today.

The eighteeth and ninteenth centuries saw mathemactics become increasinlgy abstract and rigoruos. Euler made contributioins to virtualy every branhc of mathemactics, from number theroy to topologey. Gauss, often caleld the prince of mathematicans, made fundamnetal advances in algebre, anlysis, geometery, and statsitcs. The developement of non-Eucldiean geometery by Lobachevsky, Bolyia, and Reimann showed that Eucild's paralell postulate was independnet of his other axioms, openign up entireyl new geometirc universes."""

# Tokens that look normal but have zero-width characters
ZERO_WIDTH_TESTS = [
    ("hello\u200Bworld", "zero-width space"),
    ("pass\u200Cword", "zero-width non-joiner"),
    ("test\u200Ding", "zero-width joiner"),
    ("nor\u2060mal", "word joiner"),
    ("data\uFEFFbase", "zero-width no-break space (BOM)"),
]

# Homoglyph attacks - visually similar but different characters
HOMOGLYPH_TESTS = [
    ("рython", "Cyrillic р instead of Latin p"),  # Cyrillic р
    ("hеllo", "Cyrillic е instead of Latin e"),   # Cyrillic е
    ("Gооgle", "Cyrillic оо instead of Latin oo"),  # Cyrillic о
    ("аpple", "Cyrillic а instead of Latin a"),    # Cyrillic а
    ("micrоsоft", "Cyrillic о instead of Latin o"),  # Cyrillic о
]

# Near-miss common phrases (subtle word-level errors)
NEAR_MISS_TESTS = [
    "The quick brown fox jumps over the lazy dogs",  # dogs plural
    "To be or not to be, that is a question",  # a instead of the
    "I think, therefore I exist",  # exist instead of am
    "All that glisters is not gold",  # glisters (actually correct Shakespeare!)
    "Elementary, my dear Watson, elementary",  # extra 'elementary'
]


def main():
    print(f"{'='*70}")
    print(f"V3: Multi-model + Extreme Stimuli")
    print(f"Started: {datetime.now().isoformat()}")
    print(f"{'='*70}")

    all_results = []
    models = ["gpt-4.1-mini", "gpt-4o-mini", "gpt-4.1"]

    for model in models:
        print(f"\n{'='*70}")
        print(f"Model: {model}")
        print(f"{'='*70}")

        # Test 1: Very long misspelled document
        print(f"\n--- Very Long Document ({len(VERY_LONG_CLEAN)} chars clean, {len(VERY_LONG_MISSPELLED)} chars misspelled) ---")
        for version, text in [("clean", VERY_LONG_CLEAN), ("misspelled", VERY_LONG_MISSPELLED)]:
            prompt = PROMPT.format(text=text)
            output = call_model(prompt, model=model, max_tokens=16384)
            metrics = compute_metrics(text, output)

            result = {"experiment": "very_long_doc", "version": version, "model": model,
                      "stimulus_len": len(text), **metrics}

            if version == "misspelled":
                corr, pres, oth = find_corrections(VERY_LONG_MISSPELLED, output, VERY_LONG_CLEAN)
                result["n_corrected"] = corr
                result["n_preserved"] = pres
                result["n_other"] = oth
                result["correction_rate"] = corr / max(corr + pres + oth, 1)

            all_results.append(result)
            extra = f" corr_rate={result.get('correction_rate', 'N/A')}" if version == "misspelled" else ""
            print(f"  [{version}] exact={metrics['exact_match']}, norm_edit={metrics['normalized_edit_distance']:.4f}, "
                  f"char_acc={metrics['char_accuracy']:.4f}{extra}")

        # Test 2: Zero-width character preservation
        print(f"\n--- Zero-Width Character Tests ---")
        for text, description in ZERO_WIDTH_TESTS:
            prompt = PROMPT.format(text=text)
            output = call_model(prompt, model=model)
            metrics = compute_metrics(text, output)
            result = {"experiment": "zero_width", "description": description, "model": model,
                      "stimulus": repr(text), "output": repr((output or "").strip()), **metrics}
            all_results.append(result)
            print(f"  [{description}] exact={metrics['exact_match']}, edit={metrics['edit_distance']}")

        # Test 3: Homoglyph preservation
        print(f"\n--- Homoglyph Tests ---")
        for text, description in HOMOGLYPH_TESTS:
            prompt = PROMPT.format(text=text)
            output = call_model(prompt, model=model)
            metrics = compute_metrics(text, output)
            # Check if the homoglyph was preserved or normalized
            output_stripped = (output or "").strip()
            result = {"experiment": "homoglyph", "description": description, "model": model,
                      "stimulus": text, "output": output_stripped, **metrics}
            all_results.append(result)
            preserved = "preserved" if metrics["exact_match"] else "changed"
            print(f"  [{description}] {preserved}: '{text}' → '{output_stripped}' (edit={metrics['edit_distance']})")

        # Test 4: Famous glitch tokens (single word)
        print(f"\n--- Famous Glitch Tokens ---")
        glitch_tokens = ["SolidGoldMagikarp", "TheNitromeFan", "cloneembedaliased",
                         "rawdownloadcloneembedreportprint", " petertodd",
                         "PsychExpandoExceptionObjectSyntax", "exaboralivedire",
                         " guiActiveUn", "StreamerBot", "InstoreAndOnline"]
        for tok in glitch_tokens:
            prompt = PROMPT.format(text=tok)
            output = call_model(prompt, model=model)
            metrics = compute_metrics(tok, output)
            result = {"experiment": "glitch_token", "model": model, "stimulus": tok,
                      "output": (output or "").strip(), **metrics}
            all_results.append(result)
            status = "✓" if metrics["exact_match"] else "✗"
            if not metrics["exact_match"]:
                print(f"  {status} '{tok}' → '{(output or '').strip()[:60]}' (edit={metrics['edit_distance']})")
            else:
                print(f"  {status} '{tok}'")

        # Test 5: Near-miss common phrases
        print(f"\n--- Near-Miss Famous Phrases ---")
        for text in NEAR_MISS_TESTS:
            prompt = PROMPT.format(text=text)
            output = call_model(prompt, model=model)
            metrics = compute_metrics(text, output)
            result = {"experiment": "near_miss", "model": model, "stimulus": text,
                      "output": (output or "").strip(), **metrics}
            all_results.append(result)
            status = "✓" if metrics["exact_match"] else "✗"
            if not metrics["exact_match"]:
                print(f"  {status} '{text}' → '{(output or '').strip()}'")
            else:
                print(f"  {status} '{text}'")

    # Save
    out_path = os.path.join(BASE, "results/raw_results_v3.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)
    print(f"\nSaved {len(all_results)} results to {out_path}")
    print(f"Finished: {datetime.now().isoformat()}")


if __name__ == "__main__":
    main()
