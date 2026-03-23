"""
Comprehensive analysis and visualization of all tongue twister experiments.
"""

import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
from scipy import stats

BASE = "/workspaces/llm-tongue-twisters-claude"
PLOT_DIR = os.path.join(BASE, "results/plots")
os.makedirs(PLOT_DIR, exist_ok=True)

plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("Set2")


def load_all_results():
    """Load results from all experiment versions."""
    all_data = {}
    for version in ["raw_results.json", "raw_results_v2.json", "raw_results_v3.json",
                     "raw_results_v4.json", "raw_results_v5.json", "raw_results_v6.json"]:
        path = os.path.join(BASE, f"results/{version}")
        if os.path.exists(path):
            with open(path) as f:
                all_data[version] = json.load(f)
    return all_data


def analyze_v1_token_reproduction(data):
    """Analyze token reproduction from v1 (direct template only)."""
    results = [r for r in data if r.get("template") == "direct"]

    categories = {}
    for r in results:
        cat = r["category"]
        if cat not in categories:
            categories[cat] = {"exact": 0, "total": 0, "edit_dists": []}
        categories[cat]["total"] += 1
        if r["exact_match"]:
            categories[cat]["exact"] += 1
        categories[cat]["edit_dists"].append(r["normalized_edit_distance"])

    print("\n=== V1: Token Reproduction (direct template, GPT-4.1) ===")
    for cat, stats_dict in sorted(categories.items()):
        rate = stats_dict["exact"] / max(stats_dict["total"], 1)
        avg_edit = np.mean(stats_dict["edit_dists"])
        print(f"  {cat:25s}: {stats_dict['exact']}/{stats_dict['total']} exact ({rate:.1%}), "
              f"avg_norm_edit={avg_edit:.4f}")

    return categories


def analyze_v3_cross_model(data):
    """Analyze cross-model comparison from v3."""
    print("\n=== V3: Cross-Model Comparison ===")

    # Very long document
    for r in data:
        if r["experiment"] == "very_long_doc":
            print(f"  {r['model']:20s} | {r['version']:10s} | exact={r['exact_match']}, "
                  f"edit={r['edit_distance']}")

    # Glitch tokens by model
    print("\n  Glitch Token Results:")
    for model in ["gpt-4o-mini", "gpt-4.1-mini", "gpt-4.1"]:
        model_glitch = [r for r in data if r["experiment"] == "glitch_token" and r["model"] == model]
        if model_glitch:
            exact = sum(1 for r in model_glitch if r["exact_match"])
            print(f"    {model:20s}: {exact}/{len(model_glitch)} exact")


def analyze_v5_length_scaling(data):
    """Analyze how reproduction quality scales with length."""
    print("\n=== V5: Length Scaling ===")

    for exp_type in ["length_clean", "length_misspelled", "random_alphanum", "hex_string"]:
        exp_data = [r for r in data if r["experiment"] == exp_type]
        if not exp_data:
            continue
        print(f"\n  {exp_type}:")
        for model in ["gpt-4o-mini", "gpt-4.1-mini", "gpt-4.1"]:
            model_data = sorted([r for r in exp_data if r["model"] == model],
                                key=lambda x: x.get("target_len", x.get("length", 0)))
            for r in model_data:
                length = r.get("target_len", r.get("length", "?"))
                print(f"    {model:20s} | {length:>6} chars | exact={r['exact_match']}, "
                      f"edit={r['edit_distance']}")


def analyze_v6_extreme(data):
    """Analyze extreme length tests."""
    print("\n=== V6: Extreme Length Tests ===")

    for exp_type in ["extreme_length_clean", "extreme_length_repetitive",
                     "extreme_length_misspelled", "long_random"]:
        exp_data = [r for r in data if r["experiment"] == exp_type]
        if not exp_data:
            continue
        print(f"\n  {exp_type}:")
        for model in ["gpt-4o-mini", "gpt-4.1-mini", "gpt-4.1"]:
            model_data = sorted([r for r in exp_data if r["model"] == model],
                                key=lambda x: x.get("target_len", x.get("length", 0)))
            for r in model_data:
                length = r.get("target_len", r.get("length", "?"))
                print(f"    {model:20s} | {length:>6} chars | exact={r['exact_match']}, "
                      f"edit={r['edit_distance']}, len_ratio={r.get('length_ratio', 0):.4f}")


def plot_length_vs_accuracy(all_data):
    """Plot reproduction accuracy vs text length across models."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Gather all length-based experiments
    length_results = []
    for version_data in all_data.values():
        for r in version_data:
            if r.get("experiment") in ["length_clean", "length_misspelled",
                                        "extreme_length_clean", "extreme_length_misspelled",
                                        "extreme_length_repetitive", "long_random",
                                        "random_alphanum"]:
                length = r.get("target_len", r.get("actual_len", r.get("length", 0)))
                if length > 0:
                    length_results.append({
                        "model": r["model"],
                        "length": length,
                        "exact_match": int(r["exact_match"]),
                        "edit_distance": r.get("edit_distance", 0),
                        "norm_edit": r.get("normalized_edit_distance", 0),
                        "experiment": r["experiment"],
                        "text_type": "clean" if "clean" in r.get("experiment", "") or r.get("experiment") in ["random_alphanum", "long_random"] else "misspelled"
                    })

    if not length_results:
        print("No length-based results to plot")
        return

    df = pd.DataFrame(length_results)

    # Plot 1: Exact match rate by length
    ax1 = axes[0]
    for model in df["model"].unique():
        model_df = df[df["model"] == model].sort_values("length")
        ax1.plot(model_df["length"], model_df["exact_match"], 'o-', label=model, markersize=6, alpha=0.7)
    ax1.set_xlabel("Text Length (characters)")
    ax1.set_ylabel("Exact Match (1=yes, 0=no)")
    ax1.set_title("Reproduction Accuracy vs Text Length")
    ax1.legend(fontsize=8)
    ax1.set_xscale('log')

    # Plot 2: Edit distance by length
    ax2 = axes[1]
    for model in df["model"].unique():
        model_df = df[df["model"] == model].sort_values("length")
        ax2.plot(model_df["length"], model_df["edit_distance"], 'o-', label=model, markersize=6, alpha=0.7)
    ax2.set_xlabel("Text Length (characters)")
    ax2.set_ylabel("Edit Distance")
    ax2.set_title("Edit Distance vs Text Length")
    ax2.legend(fontsize=8)
    ax2.set_xscale('log')

    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "length_vs_accuracy.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {PLOT_DIR}/length_vs_accuracy.png")


def plot_category_comparison(all_data):
    """Plot exact match rates by category and model."""
    # Collect from v1 (direct template) and v3
    category_data = []

    if "raw_results.json" in all_data:
        for r in all_data["raw_results.json"]:
            if r.get("template") == "direct":
                category_data.append({
                    "model": r["model"],
                    "category": r["category"],
                    "exact_match": int(r["exact_match"]),
                })

    if "raw_results_v3.json" in all_data:
        for r in all_data["raw_results_v3.json"]:
            if r.get("experiment") in ["glitch_token", "homoglyph", "zero_width"]:
                category_data.append({
                    "model": r["model"],
                    "category": r["experiment"],
                    "exact_match": int(r["exact_match"]),
                })

    if not category_data:
        return

    df = pd.DataFrame(category_data)
    summary = df.groupby(["model", "category"])["exact_match"].mean().reset_index()

    fig, ax = plt.subplots(figsize=(12, 5))
    categories = summary["category"].unique()
    models = summary["model"].unique()
    x = np.arange(len(categories))
    width = 0.25

    for i, model in enumerate(models):
        model_data = summary[summary["model"] == model]
        rates = [model_data[model_data["category"] == c]["exact_match"].values[0]
                 if len(model_data[model_data["category"] == c]) > 0 else 0
                 for c in categories]
        ax.bar(x + i * width, rates, width, label=model, alpha=0.8)

    ax.set_xlabel("Stimulus Category")
    ax.set_ylabel("Exact Match Rate")
    ax.set_title("Reproduction Accuracy by Category and Model")
    ax.set_xticks(x + width)
    ax.set_xticklabels(categories, rotation=30, ha='right')
    ax.legend()
    ax.set_ylim(0, 1.1)

    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "category_comparison.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {PLOT_DIR}/category_comparison.png")


def plot_document_reproduction(all_data):
    """Plot document reproduction results (clean vs misspelled)."""
    doc_data = []

    for version_name, data in all_data.items():
        for r in data:
            if "document" in r.get("category", "") or r.get("experiment") in ["long_document", "very_long_doc"]:
                doc_data.append({
                    "model": r.get("model", "unknown"),
                    "version": "misspelled" if "misspelled" in r.get("category", r.get("version", "")) else "clean",
                    "exact_match": int(r["exact_match"]),
                    "edit_distance": r.get("edit_distance", 0),
                    "doc_name": r.get("doc_name", ""),
                    "length": r.get("stimulus_len", r.get("actual_len", 0)),
                })

    if not doc_data:
        return

    df = pd.DataFrame(doc_data)

    fig, ax = plt.subplots(figsize=(10, 5))
    for model in df["model"].unique():
        for version in ["clean", "misspelled"]:
            subset = df[(df["model"] == model) & (df["version"] == version)]
            if len(subset) > 0:
                style = '-o' if version == "clean" else '--s'
                ax.plot(subset["length"], subset["exact_match"], style,
                        label=f"{model} ({version})", markersize=6, alpha=0.7)

    ax.set_xlabel("Document Length (characters)")
    ax.set_ylabel("Exact Match (1=yes, 0=no)")
    ax.set_title("Document Reproduction: Clean vs Misspelled")
    ax.legend(fontsize=7, ncol=2)

    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "document_reproduction.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {PLOT_DIR}/document_reproduction.png")


def create_summary_table(all_data):
    """Create a comprehensive summary table."""
    print("\n" + "="*80)
    print("COMPREHENSIVE RESULTS SUMMARY")
    print("="*80)

    # Summary by model and experiment type
    summary_rows = []
    for version_name, data in all_data.items():
        for r in data:
            model = r.get("model", "unknown")
            experiment = r.get("experiment", r.get("category", "unknown"))
            exact = r.get("exact_match", False)
            edit = r.get("edit_distance", 0)
            summary_rows.append({
                "source": version_name,
                "model": model,
                "experiment": experiment,
                "exact_match": int(exact),
                "edit_distance": edit if edit >= 0 else None,
            })

    df = pd.DataFrame(summary_rows)

    # Aggregate by model
    print("\n--- Overall Exact Match Rate by Model ---")
    model_summary = df.groupby("model")["exact_match"].agg(["mean", "sum", "count"])
    for model, row in model_summary.iterrows():
        print(f"  {model:25s}: {row['mean']:.1%} ({int(row['sum'])}/{int(row['count'])})")

    # Key finding: failure cases
    print("\n--- All Reproduction Failures ---")
    failures = df[df["exact_match"] == 0]
    if len(failures) == 0:
        print("  No failures detected across all experiments!")
    else:
        for _, r in failures.iterrows():
            print(f"  {r['model']:25s} | {r['experiment']:30s} | edit={r['edit_distance']} | source={r['source']}")

    return df


def main():
    print("Loading all experiment results...")
    all_data = load_all_results()
    print(f"Loaded {len(all_data)} result files:")
    for name, data in all_data.items():
        print(f"  {name}: {len(data)} results")

    # Analyze each version
    if "raw_results.json" in all_data:
        analyze_v1_token_reproduction(all_data["raw_results.json"])

    if "raw_results_v3.json" in all_data:
        analyze_v3_cross_model(all_data["raw_results_v3.json"])

    if "raw_results_v5.json" in all_data:
        analyze_v5_length_scaling(all_data["raw_results_v5.json"])

    if "raw_results_v6.json" in all_data:
        analyze_v6_extreme(all_data["raw_results_v6.json"])

    # Create visualizations
    print("\n--- Creating Visualizations ---")
    plot_length_vs_accuracy(all_data)
    plot_category_comparison(all_data)
    plot_document_reproduction(all_data)

    # Comprehensive summary
    summary_df = create_summary_table(all_data)

    # Save summary
    summary_path = os.path.join(BASE, "results/analysis_summary.json")
    summary_stats = {
        "total_experiments": len(summary_df),
        "total_failures": int(summary_df["exact_match"].eq(0).sum()),
        "failure_rate": float(1 - summary_df["exact_match"].mean()),
        "models_tested": list(summary_df["model"].unique()),
        "experiment_types": list(summary_df["experiment"].unique()),
    }
    with open(summary_path, "w") as f:
        json.dump(summary_stats, f, indent=2)
    print(f"\nSaved analysis summary to {summary_path}")


if __name__ == "__main__":
    main()
