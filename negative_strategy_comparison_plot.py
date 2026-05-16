"""
Negative Sampling Strategy Comparison Plots
===========================================

Compare link-prediction evaluation metrics across:
    - Models   : TransE, ComplEx, TriModel
    - Strategies: Uniform (random corruption) vs Degree-Matched negatives
    - Dimensions: 100, 200, 300

Inputs (must already exist):
    outputs_transe/evaluation_summary.csv
    outputs_complex/evaluation_summary.csv
    outputs_trimodel/evaluation_summary.csv
    outputs_transe_degree_matched/evaluation_summary.csv
    outputs_complex_degree_matched/evaluation_summary.csv
    outputs_trimodel_degree_matched/evaluation_summary.csv

Outputs:
    outputs_negative_strategy_comparison/
        combined_metrics.csv
        per_metric/
            <metric>_lines.png         # Lines: dim on x, MRR/Hits on y, 2 lines per model
            <metric>_grouped_bars.png  # Grouped bars per dim & model, 2 bars per model
        per_model/
            <model>_metrics.png        # Metrics for one model: naive vs degree-matched
        per_dim/
            dim_<d>_metrics.png        # All models at this dim: naive vs degree-matched
"""

import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.ioff()


# ----------------------------
# Configuration
# ----------------------------
EMBEDDING_DIMS = [100, 200, 300]
METRICS = ["MRR", "Hits@1", "Hits@3", "Hits@10"]

MODELS = ["TransE", "ComplEx", "TriModel"]
STRATEGIES = ["Uniform", "Degree-Matched"]

# (model, strategy) -> evaluation_summary.csv path
SOURCES: Dict[Tuple[str, str], str] = {
    ("TransE", "Uniform"):          "outputs_transe/evaluation_summary.csv",
    ("TransE", "Degree-Matched"):   "outputs_transe_degree_matched/evaluation_summary.csv",
    ("ComplEx", "Uniform"):         "outputs_complex/evaluation_summary.csv",
    ("ComplEx", "Degree-Matched"):  "outputs_complex_degree_matched/evaluation_summary.csv",
    ("TriModel", "Uniform"):        "outputs_trimodel/evaluation_summary.csv",
    ("TriModel", "Degree-Matched"): "outputs_trimodel_degree_matched/evaluation_summary.csv",
}

OUTPUT_DIR = "outputs_negative_strategy_comparison"
PER_METRIC_DIR = os.path.join(OUTPUT_DIR, "per_metric")
PER_MODEL_DIR = os.path.join(OUTPUT_DIR, "per_model")
PER_DIM_DIR = os.path.join(OUTPUT_DIR, "per_dim")

MODEL_COLORS = {
    "TransE":   "#1f77b4",
    "ComplEx":  "#ff7f0e",
    "TriModel": "#2ca02c",
}
STRATEGY_HATCH = {"Uniform": "", "Degree-Matched": "//"}
STRATEGY_LINESTYLE = {"Uniform": "-", "Degree-Matched": "--"}
STRATEGY_MARKER = {"Uniform": "o", "Degree-Matched": "s"}


# ----------------------------
# Data loading
# ----------------------------
def load_combined() -> pd.DataFrame:
    """Load all CSVs and combine into a long-format DataFrame."""
    rows = []
    for (model, strategy), path in SOURCES.items():
        if not os.path.exists(path):
            print(f"WARNING: missing file, skipping {model}/{strategy}: {path}")
            continue
        df = pd.read_csv(path)
        if "dimension" in df.columns and "Dimension" not in df.columns:
            df = df.rename(columns={"dimension": "Dimension"})
        for _, r in df.iterrows():
            row = {"Model": model, "Strategy": strategy, "Dimension": int(r["Dimension"])}
            for m in METRICS:
                if m in df.columns:
                    row[m] = float(r[m])
            rows.append(row)

    combined = pd.DataFrame(rows)
    if combined.empty:
        raise RuntimeError("No evaluation_summary.csv files were found.")
    combined = combined.sort_values(["Model", "Strategy", "Dimension"]).reset_index(drop=True)
    return combined


# ----------------------------
# Plot helpers
# ----------------------------
def _ensure_dirs():
    for d in (OUTPUT_DIR, PER_METRIC_DIR, PER_MODEL_DIR, PER_DIM_DIR):
        os.makedirs(d, exist_ok=True)


def _annotate_bars(ax, bars, fmt="{:.3f}"):
    for b in bars:
        h = b.get_height()
        ax.text(
            b.get_x() + b.get_width() / 2.0,
            h,
            fmt.format(h),
            ha="center", va="bottom",
            fontsize=8, color="black",
        )


# ----------------------------
# Per-metric line plots: dim on x, one line per (model, strategy)
# ----------------------------
def plot_per_metric_lines(df: pd.DataFrame):
    for metric in METRICS:
        fig, ax = plt.subplots(figsize=(8, 5.5))
        for model in MODELS:
            color = MODEL_COLORS[model]
            for strategy in STRATEGIES:
                sub = df[(df["Model"] == model) & (df["Strategy"] == strategy)]
                if sub.empty or metric not in sub.columns:
                    continue
                sub = sub.sort_values("Dimension")
                ax.plot(
                    sub["Dimension"], sub[metric],
                    color=color,
                    linestyle=STRATEGY_LINESTYLE[strategy],
                    marker=STRATEGY_MARKER[strategy],
                    linewidth=2, markersize=8,
                    label=f"{model} ({strategy})",
                )

        ax.set_xticks(EMBEDDING_DIMS)
        ax.set_xlabel("Embedding Dimension", fontsize=11)
        ax.set_ylabel(metric, fontsize=11)
        ax.set_title(
            f"{metric}: Uniform vs Degree-Matched Negative Sampling",
            fontsize=13, fontweight="bold",
        )
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=9, ncol=2)
        plt.tight_layout()
        out = os.path.join(PER_METRIC_DIR, f"{metric.replace('@', '').lower()}_lines.png")
        plt.savefig(out, dpi=150)
        plt.close()
        print(f"  Saved {out}")


# ----------------------------
# Per-metric grouped bars: groups by (model, dim), 2 bars per group
# ----------------------------
def plot_per_metric_grouped_bars(df: pd.DataFrame):
    for metric in METRICS:
        fig, ax = plt.subplots(figsize=(11, 5.5))

        # x positions: groups by model, sub-groups by dim
        n_models = len(MODELS)
        n_dims = len(EMBEDDING_DIMS)
        bar_width = 0.35
        group_gap = 1.2

        x_centers = []
        x_labels = []
        for mi, model in enumerate(MODELS):
            for di, dim in enumerate(EMBEDDING_DIMS):
                center = mi * (n_dims * group_gap + 1.5) + di * group_gap
                x_centers.append(center)
                x_labels.append(f"{model}\nd={dim}")

        # Plot two strategy bars at each center
        for j, strategy in enumerate(STRATEGIES):
            heights = []
            for mi, model in enumerate(MODELS):
                for di, dim in enumerate(EMBEDDING_DIMS):
                    sub = df[
                        (df["Model"] == model)
                        & (df["Strategy"] == strategy)
                        & (df["Dimension"] == dim)
                    ]
                    val = float(sub[metric].iloc[0]) if (not sub.empty and metric in sub.columns) else np.nan
                    heights.append(val)

            offset = (j - 0.5) * bar_width
            xs = [c + offset for c in x_centers]
            colors = []
            for mi, _ in enumerate(MODELS):
                colors.extend([MODEL_COLORS[MODELS[mi]]] * n_dims)
            bars = ax.bar(
                xs, heights,
                width=bar_width,
                color=colors,
                edgecolor="black",
                hatch=STRATEGY_HATCH[strategy],
                alpha=0.85 if strategy == "Uniform" else 0.65,
                label=strategy,
            )
            _annotate_bars(ax, bars)

        ax.set_xticks(x_centers)
        ax.set_xticklabels(x_labels, fontsize=9)
        ax.set_ylabel(metric, fontsize=11)
        ax.set_title(
            f"{metric}: Uniform vs Degree-Matched Negatives (per model & dimension)",
            fontsize=12, fontweight="bold",
        )
        ax.grid(True, axis="y", alpha=0.3)
        # Build a clean legend (color = model, hatch = strategy)
        from matplotlib.patches import Patch
        legend_handles = [
            Patch(facecolor=MODEL_COLORS[m], edgecolor="black", label=m) for m in MODELS
        ] + [
            Patch(facecolor="white", edgecolor="black", hatch=STRATEGY_HATCH[s], label=s)
            for s in STRATEGIES
        ]
        ax.legend(handles=legend_handles, loc="upper left", fontsize=9, ncol=2)
        plt.tight_layout()
        out = os.path.join(PER_METRIC_DIR, f"{metric.replace('@', '').lower()}_grouped_bars.png")
        plt.savefig(out, dpi=150)
        plt.close()
        print(f"  Saved {out}")


# ----------------------------
# Per-model figure: 4 subplots (one per metric), x = dim, two lines (strategies)
# ----------------------------
def plot_per_model(df: pd.DataFrame):
    for model in MODELS:
        sub_model = df[df["Model"] == model]
        if sub_model.empty:
            continue

        fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True)
        axes = axes.flatten()
        color = MODEL_COLORS[model]

        for k, metric in enumerate(METRICS):
            ax = axes[k]
            for strategy in STRATEGIES:
                s = sub_model[sub_model["Strategy"] == strategy].sort_values("Dimension")
                if s.empty or metric not in s.columns:
                    continue
                ax.plot(
                    s["Dimension"], s[metric],
                    color=color,
                    linestyle=STRATEGY_LINESTYLE[strategy],
                    marker=STRATEGY_MARKER[strategy],
                    linewidth=2, markersize=8,
                    label=strategy,
                )
                # Value labels
                for x, y in zip(s["Dimension"], s[metric]):
                    ax.annotate(f"{y:.3f}", xy=(x, y), xytext=(0, 6),
                                textcoords="offset points", ha="center",
                                fontsize=8, color=color)
            ax.set_xticks(EMBEDDING_DIMS)
            ax.set_title(metric, fontsize=11, fontweight="bold")
            ax.set_xlabel("Dimension", fontsize=10)
            ax.set_ylabel(metric, fontsize=10)
            ax.grid(True, alpha=0.3)
            ax.legend(loc="best", fontsize=9)

        fig.suptitle(
            f"{model}: Uniform vs Degree-Matched Negative Sampling",
            fontsize=13, fontweight="bold",
        )
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        out = os.path.join(PER_MODEL_DIR, f"{model.lower()}_metrics.png")
        plt.savefig(out, dpi=150)
        plt.close()
        print(f"  Saved {out}")


# ----------------------------
# Per-dim figure: 4 subplots (metrics), each: model on x, 2 bars per model
# ----------------------------
def plot_per_dim(df: pd.DataFrame):
    for dim in EMBEDDING_DIMS:
        sub_dim = df[df["Dimension"] == dim]
        if sub_dim.empty:
            continue

        fig, axes = plt.subplots(2, 2, figsize=(11, 8))
        axes = axes.flatten()
        x = np.arange(len(MODELS))
        bar_width = 0.36

        for k, metric in enumerate(METRICS):
            ax = axes[k]
            for j, strategy in enumerate(STRATEGIES):
                heights = []
                for model in MODELS:
                    s = sub_dim[(sub_dim["Model"] == model) & (sub_dim["Strategy"] == strategy)]
                    val = float(s[metric].iloc[0]) if (not s.empty and metric in s.columns) else np.nan
                    heights.append(val)
                offset = (j - 0.5) * bar_width
                colors = [MODEL_COLORS[m] for m in MODELS]
                bars = ax.bar(
                    x + offset, heights, width=bar_width,
                    color=colors, edgecolor="black",
                    hatch=STRATEGY_HATCH[strategy],
                    alpha=0.85 if strategy == "Uniform" else 0.65,
                    label=strategy,
                )
                _annotate_bars(ax, bars)
            ax.set_xticks(x)
            ax.set_xticklabels(MODELS)
            ax.set_title(metric, fontsize=11, fontweight="bold")
            ax.set_ylabel(metric, fontsize=10)
            ax.grid(True, axis="y", alpha=0.3)

            from matplotlib.patches import Patch
            legend_handles = [
                Patch(facecolor="white", edgecolor="black", hatch=STRATEGY_HATCH[s], label=s)
                for s in STRATEGIES
            ]
            ax.legend(handles=legend_handles, loc="best", fontsize=9)

        fig.suptitle(
            f"Dimension {dim}: Uniform vs Degree-Matched Negative Sampling",
            fontsize=13, fontweight="bold",
        )
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        out = os.path.join(PER_DIM_DIR, f"dim_{dim}_metrics.png")
        plt.savefig(out, dpi=150)
        plt.close()
        print(f"  Saved {out}")


# ----------------------------
# Main
# ----------------------------
def main():
    _ensure_dirs()
    df = load_combined()

    combined_path = os.path.join(OUTPUT_DIR, "combined_metrics.csv")
    df.to_csv(combined_path, index=False)
    print(f"\nCombined metrics written to: {combined_path}\n")

    print("Per-metric line plots:")
    plot_per_metric_lines(df)

    print("\nPer-metric grouped-bar plots:")
    plot_per_metric_grouped_bars(df)

    print("\nPer-model plots:")
    plot_per_model(df)

    print("\nPer-dimension plots:")
    plot_per_dim(df)

    print(f"\nAll plots saved under: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
