"""
DTI Negative-Strategy Comparison Plot
=====================================

Compares DTI binary-classification metrics for models trained with two
training-time negative sampling strategies:
  - Uniform (random)        -> outputs_dti_evaluation_fixed/
  - Degree-Matched          -> outputs_dti_evaluation_degree_matched/

Both runs use the SAME (degree-matched) evaluation negatives, so the
difference between the two reflects only the training negative strategy.

Generates:
  outputs_dti_negative_strategy_comparison/
    combined_dti_metrics.csv
    per_metric/
      auc_roc_lines.png, auc_pr_lines.png, best_f1_lines.png
      auc_roc_grouped_bars.png, auc_pr_grouped_bars.png, best_f1_grouped_bars.png
    per_model/
      transe_metrics.png, complex_metrics.png, trimodel_metrics.png
    per_dim/
      dim_100_metrics.png, dim_200_metrics.png, dim_300_metrics.png
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(PROJECT_ROOT, "outputs_dti_negative_strategy_comparison")

MODELS = ["TransE", "ComplEx", "TriModel"]
STRATEGIES = ["Uniform", "Degree-Matched"]
METRICS = ["AUC-ROC", "AUC-PR", "Best_F1"]
METRIC_FILE_NAMES = {
    "AUC-ROC": "auc_roc",
    "AUC-PR": "auc_pr",
    "Best_F1": "best_f1",
}

SOURCES = {
    "Uniform": os.path.join(PROJECT_ROOT, "outputs_dti_evaluation_fixed",
                             "dti_metrics_all_models_dims.csv"),
    "Degree-Matched": os.path.join(PROJECT_ROOT, "outputs_dti_evaluation_degree_matched",
                                    "dti_metrics_all_models_dims.csv"),
}

MODEL_COLOR = {"TransE": "#1f77b4", "ComplEx": "#ff7f0e", "TriModel": "#2ca02c"}
STRATEGY_LINESTYLE = {"Uniform": "-", "Degree-Matched": "--"}
STRATEGY_MARKER = {"Uniform": "o", "Degree-Matched": "s"}
STRATEGY_HATCH = {"Uniform": "", "Degree-Matched": "//"}


def load_combined() -> pd.DataFrame:
    frames = []
    for strategy, path in SOURCES.items():
        if not os.path.exists(path):
            print(f"WARNING: missing source CSV for '{strategy}': {path}")
            continue
        df = pd.read_csv(path)
        df["Strategy"] = strategy
        frames.append(df)
    if not frames:
        raise FileNotFoundError("No DTI metrics CSV files found.")
    return pd.concat(frames, ignore_index=True)


def _ensure_dirs():
    for sub in ("per_metric", "per_model", "per_dim"):
        os.makedirs(os.path.join(OUT_DIR, sub), exist_ok=True)


def plot_per_metric_lines(df: pd.DataFrame):
    print("\nPer-metric line plots:")
    for metric in METRICS:
        fig, ax = plt.subplots(figsize=(8, 6))
        for model in MODELS:
            for strategy in STRATEGIES:
                sub = df[(df["Model"] == model) & (df["Strategy"] == strategy)].sort_values("Dimension")
                if sub.empty:
                    continue
                ax.plot(
                    sub["Dimension"], sub[metric],
                    color=MODEL_COLOR[model],
                    linestyle=STRATEGY_LINESTYLE[strategy],
                    marker=STRATEGY_MARKER[strategy],
                    linewidth=2, markersize=8,
                    label=f"{model} ({strategy})",
                )
        ax.set_xticks([100, 200, 300])
        ax.set_xlabel("Embedding dimension")
        ax.set_ylabel(metric)
        ax.set_title(f"DTI {metric}: Uniform vs Degree-Matched training negatives")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=9, ncol=2)
        plt.tight_layout()
        out = os.path.join(OUT_DIR, "per_metric", f"{METRIC_FILE_NAMES[metric]}_lines.png")
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved {out}")


def plot_per_metric_grouped_bars(df: pd.DataFrame):
    print("\nPer-metric grouped-bar plots:")
    dims = [100, 200, 300]
    n_groups = len(dims) * len(MODELS)
    width = 0.38
    for metric in METRICS:
        fig, ax = plt.subplots(figsize=(11, 6))
        x = np.arange(n_groups)
        labels = []
        # build per (dim, model) group with two bars (strategies)
        for i, dim in enumerate(dims):
            for j, model in enumerate(MODELS):
                idx = i * len(MODELS) + j
                labels.append(f"{model}\nd={dim}")
                for k, strategy in enumerate(STRATEGIES):
                    sub = df[(df["Model"] == model) & (df["Dimension"] == dim) & (df["Strategy"] == strategy)]
                    val = float(sub[metric].iloc[0]) if not sub.empty else 0.0
                    offset = (k - 0.5) * width
                    ax.bar(
                        idx + offset, val, width,
                        color=MODEL_COLOR[model],
                        alpha=0.85 if strategy == "Uniform" else 0.95,
                        hatch=STRATEGY_HATCH[strategy],
                        edgecolor="black", linewidth=0.6,
                        label=f"{model} ({strategy})" if (i == 0) else None,
                    )
                    ax.text(idx + offset, val + 0.01, f"{val:.2f}",
                            ha="center", va="bottom", fontsize=7)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_ylabel(metric)
        ax.set_ylim(0, max(1.05, df[metric].max() * 1.15))
        ax.set_title(f"DTI {metric}: Uniform vs Degree-Matched training negatives")
        ax.grid(axis="y", alpha=0.3)
        # de-duplicate legend
        handles, lbls = ax.get_legend_handles_labels()
        seen = {}
        for h, l in zip(handles, lbls):
            if l and l not in seen:
                seen[l] = h
        ax.legend(seen.values(), seen.keys(), fontsize=8, ncol=3, loc="upper left")
        plt.tight_layout()
        out = os.path.join(OUT_DIR, "per_metric", f"{METRIC_FILE_NAMES[metric]}_grouped_bars.png")
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved {out}")


def plot_per_model(df: pd.DataFrame):
    print("\nPer-model plots:")
    dims = [100, 200, 300]
    for model in MODELS:
        fig, axes = plt.subplots(1, len(METRICS), figsize=(5 * len(METRICS), 5), sharex=True)
        for ax, metric in zip(axes, METRICS):
            for strategy in STRATEGIES:
                sub = df[(df["Model"] == model) & (df["Strategy"] == strategy)].sort_values("Dimension")
                if sub.empty:
                    continue
                ax.plot(
                    sub["Dimension"], sub[metric],
                    color=MODEL_COLOR[model],
                    linestyle=STRATEGY_LINESTYLE[strategy],
                    marker=STRATEGY_MARKER[strategy],
                    linewidth=2, markersize=8,
                    label=strategy,
                )
            ax.set_xticks(dims)
            ax.set_xlabel("Embedding dimension")
            ax.set_ylabel(metric)
            ax.set_title(metric)
            ax.grid(alpha=0.3)
            ax.legend(fontsize=9)
        fig.suptitle(f"DTI metrics for {model}: Uniform vs Degree-Matched training negatives", fontsize=13)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        out = os.path.join(OUT_DIR, "per_model", f"{model.lower()}_metrics.png")
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved {out}")


def plot_per_dim(df: pd.DataFrame):
    print("\nPer-dimension plots:")
    dims = [100, 200, 300]
    width = 0.38
    for dim in dims:
        fig, axes = plt.subplots(1, len(METRICS), figsize=(5 * len(METRICS), 5), sharey=False)
        for ax, metric in zip(axes, METRICS):
            x = np.arange(len(MODELS))
            for k, strategy in enumerate(STRATEGIES):
                vals = []
                for model in MODELS:
                    sub = df[(df["Model"] == model) & (df["Dimension"] == dim) & (df["Strategy"] == strategy)]
                    vals.append(float(sub[metric].iloc[0]) if not sub.empty else 0.0)
                offset = (k - 0.5) * width
                bars = ax.bar(
                    x + offset, vals, width,
                    color=[MODEL_COLOR[m] for m in MODELS],
                    alpha=0.85 if strategy == "Uniform" else 0.95,
                    hatch=STRATEGY_HATCH[strategy],
                    edgecolor="black", linewidth=0.6,
                    label=strategy,
                )
                for b, v in zip(bars, vals):
                    ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.2f}",
                            ha="center", va="bottom", fontsize=8)
            ax.set_xticks(x)
            ax.set_xticklabels(MODELS)
            ax.set_ylabel(metric)
            ax.set_title(metric)
            ax.set_ylim(0, max(1.05, df[metric].max() * 1.15))
            ax.grid(axis="y", alpha=0.3)
            ax.legend(fontsize=9)
        fig.suptitle(f"DTI metrics at dim={dim}: Uniform vs Degree-Matched training negatives", fontsize=13)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        out = os.path.join(OUT_DIR, "per_dim", f"dim_{dim}_metrics.png")
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved {out}")


def main():
    _ensure_dirs()
    df = load_combined()

    keep_cols = ["Model", "Dimension", "Strategy", "AUC-ROC", "AUC-PR", "Best_F1",
                 "Precision@Best_F1", "Recall@Best_F1", "Best_Threshold",
                 "n_positives", "n_negatives", "neg_sampling"]
    keep_cols = [c for c in keep_cols if c in df.columns]
    combined_csv = os.path.join(OUT_DIR, "combined_dti_metrics.csv")
    df[keep_cols].sort_values(["Model", "Dimension", "Strategy"]).to_csv(combined_csv, index=False)
    print(f"Combined metrics written to: {combined_csv}")

    plot_per_metric_lines(df)
    plot_per_metric_grouped_bars(df)
    plot_per_model(df)
    plot_per_dim(df)

    print(f"\nAll plots saved under: {OUT_DIR}/")


if __name__ == "__main__":
    main()
