"""
Link Prediction - Unified Cross-Model Comparison Plots
======================================================

Generate comparison plots for link prediction metrics across models and dimensions.

Inputs:
- outputs_transe/evaluation_summary.csv
- outputs_complex/evaluation_summary.csv
- outputs_trimodel/evaluation_summary.csv

Outputs:
- Per-dimension model comparison bar charts
- Per-metric dimension progression line charts
"""

import os
from typing import Dict, List

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.ioff()


EMBEDDING_DIMS = [100, 200, 300]

MODELS = {
    "TransE": "outputs_transe/evaluation_summary.csv",
    "ComplEx": "outputs_complex/evaluation_summary.csv",
    "TriModel": "outputs_trimodel/evaluation_summary.csv",
}

METRICS = ["MRR", "Hits@1", "Hits@3", "Hits@10"]
OPTIONAL_METRICS = ["Mean Rank"]

OUTPUT_DIR = "outputs_link_prediction_comparison"
COMPARISON_DIR = os.path.join(OUTPUT_DIR, "comparison_plots")

MODEL_COLORS = {
    "TransE": "#1f77b4",
    "ComplEx": "#ff7f0e",
    "TriModel": "#2ca02c",
}
MODEL_MARKERS = {
    "TransE": "s",
    "ComplEx": "o",
    "TriModel": "^",
}


def load_all_results() -> pd.DataFrame:
    """Load and merge all model-level evaluation summary CSV files."""
    rows: List[pd.DataFrame] = []

    for model_name, csv_path in MODELS.items():
        if not os.path.exists(csv_path):
            print(f"Warning: Missing file for {model_name}: {csv_path}")
            continue

        df = pd.read_csv(csv_path)
        if df.empty:
            print(f"Warning: Empty file for {model_name}: {csv_path}")
            continue

        # Normalize expected dimension column name.
        if "dimension" in df.columns and "Dimension" not in df.columns:
            df = df.rename(columns={"dimension": "Dimension"})

        required = ["Dimension"] + METRICS
        missing_required = [c for c in required if c not in df.columns]
        if missing_required:
            print(f"Warning: {model_name} missing required columns: {missing_required}")
            continue

        df["Model"] = model_name
        rows.append(df)

    if not rows:
        return pd.DataFrame()

    all_df = pd.concat(rows, ignore_index=True)
    all_df["Dimension"] = all_df["Dimension"].astype(int)

    # Keep only configured dimensions and sort for stable plotting.
    all_df = all_df[all_df["Dimension"].isin(EMBEDDING_DIMS)].copy()
    all_df = all_df.sort_values(["Dimension", "Model"]).reset_index(drop=True)
    return all_df


def plot_model_comparison_per_dimension(df: pd.DataFrame, dim: int):
    """Bar chart for all models at one dimension across core metrics."""
    dim_df = df[df["Dimension"] == dim].copy()
    if dim_df.empty:
        print(f"Warning: No results for Dimension={dim}")
        return

    model_names = [m for m in MODELS.keys() if m in set(dim_df["Model"])]
    if not model_names:
        return

    x = np.arange(len(model_names))
    width = 0.18

    fig, ax = plt.subplots(figsize=(11, 7))

    for i, metric in enumerate(METRICS):
        values = []
        for model_name in model_names:
            match = dim_df[dim_df["Model"] == model_name]
            values.append(float(match.iloc[0][metric]))

        bars = ax.bar(
            x + (i - 1.5) * width,
            values,
            width,
            label=metric,
            alpha=0.85,
        )

        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.008,
                f"{val:.3f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    ax.set_xlabel("Model", fontsize=12, fontweight="bold")
    ax.set_ylabel("Score", fontsize=12, fontweight="bold")
    ax.set_title(f"Link Prediction Metric Comparison (Dimension {dim})", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, fontsize=11)
    ax.set_ylim(0, 1.0)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.legend(fontsize=10, loc="upper left", framealpha=0.95)

    plt.tight_layout()
    out_path = os.path.join(COMPARISON_DIR, f"01_metrics_comparison_dim_{dim}.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def plot_metric_progression(df: pd.DataFrame, metric: str, file_prefix: str):
    """Line chart showing each model's metric progression across dimensions."""
    if metric not in df.columns:
        print(f"Skipping progression for missing metric: {metric}")
        return

    fig, ax = plt.subplots(figsize=(9, 6))

    for model_name in MODELS.keys():
        mdf = df[df["Model"] == model_name].sort_values("Dimension")
        if mdf.empty:
            continue

        ax.plot(
            mdf["Dimension"],
            mdf[metric],
            label=model_name,
            marker=MODEL_MARKERS.get(model_name, "o"),
            markersize=8,
            linewidth=2.4,
            color=MODEL_COLORS.get(model_name, "#000000"),
            alpha=0.9,
        )

        for _, row in mdf.iterrows():
            ax.text(
                int(row["Dimension"]),
                float(row[metric]) + 0.008,
                f"{float(row[metric]):.3f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    ax.set_xlabel("Embedding Dimension", fontsize=12, fontweight="bold")
    ax.set_ylabel(metric, fontsize=12, fontweight="bold")
    ax.set_title(f"{metric} Progression Across Dimensions", fontsize=14, fontweight="bold")
    ax.set_xticks(EMBEDDING_DIMS)

    # Mean Rank is unbounded and lower-is-better, other metrics are in [0, 1].
    if metric != "Mean Rank":
        ax.set_ylim(0, 1.0)

    ax.grid(alpha=0.3, linestyle="--")
    ax.legend(fontsize=10, loc="best", framealpha=0.95)

    plt.tight_layout()
    out_path = os.path.join(COMPARISON_DIR, f"{file_prefix}_{metric.lower().replace('@', 'at').replace(' ', '_')}.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def plot_overall_grid(df: pd.DataFrame):
    """Create one figure with all core metrics as subplots."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.ravel()

    for ax, metric in zip(axes, METRICS):
        for model_name in MODELS.keys():
            mdf = df[df["Model"] == model_name].sort_values("Dimension")
            if mdf.empty:
                continue

            ax.plot(
                mdf["Dimension"],
                mdf[metric],
                label=model_name,
                marker=MODEL_MARKERS.get(model_name, "o"),
                markersize=7,
                linewidth=2.2,
                color=MODEL_COLORS.get(model_name, "#000000"),
            )

        ax.set_title(metric, fontsize=12, fontweight="bold")
        ax.set_xlabel("Dimension")
        ax.set_ylabel("Score")
        ax.set_xticks(EMBEDDING_DIMS)
        ax.set_ylim(0, 1.0)
        ax.grid(alpha=0.3, linestyle="--")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.suptitle(
        "Link Prediction: Model Progression Across Dimensions",
        fontsize=15,
        fontweight="bold",
        y=0.99,
    )
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=len(labels),
        framealpha=0.95,
        bbox_to_anchor=(0.5, 0.0),
    )

    # Reserve top space for suptitle and bottom space for the figure-level legend
    # so they no longer overlap subplot content or each other.
    plt.tight_layout(rect=[0, 0.06, 1, 0.95])
    out_path = os.path.join(COMPARISON_DIR, "03_overall_progression_grid.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def save_merged_summary(df: pd.DataFrame):
    """Save merged data used for plotting."""
    out_csv = os.path.join(OUTPUT_DIR, "link_prediction_metrics_all_models_dims.csv")
    df.to_csv(out_csv, index=False)
    print(f"Saved: {out_csv}")


def main():
    print("=" * 70)
    print("LINK PREDICTION - UNIFIED COMPARISON PLOTS")
    print("=" * 70)

    os.makedirs(COMPARISON_DIR, exist_ok=True)

    df = load_all_results()
    if df.empty:
        print("Error: No valid evaluation_summary.csv files found.")
        return

    print("Loaded results:")
    print(df[["Model", "Dimension", "MRR", "Hits@1", "Hits@3", "Hits@10"]].to_string(index=False))

    save_merged_summary(df)

    # Per-dimension comparison charts
    for dim in EMBEDDING_DIMS:
        plot_model_comparison_per_dimension(df, dim)

    # Per-metric progression charts
    for metric in METRICS:
        plot_metric_progression(df, metric, file_prefix="02_progression")

    # Optional progression for mean rank if available in all or some models.
    if "Mean Rank" in df.columns and df["Mean Rank"].notna().any():
        plot_metric_progression(df[df["Mean Rank"].notna()], "Mean Rank", file_prefix="02_progression")

    # Combined grid for quick comparison
    plot_overall_grid(df)

    print("\n" + "=" * 70)
    print("DONE: LINK PREDICTION COMPARISON PLOTS GENERATED")
    print("=" * 70)
    print(f"Output directory: {COMPARISON_DIR}")


if __name__ == "__main__":
    main()
