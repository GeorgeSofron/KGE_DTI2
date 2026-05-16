"""
Create a model-centric graph showing metric differences across embedding dimensions.
"""

import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

INPUT_CSV = "outputs_link_prediction_comparison/link_prediction_metrics_all_models_dims.csv"
OUT_DIR = "outputs_link_prediction_comparison/comparison_plots"
OUT_FILE = "05_model_dimension_differences.png"

METRICS = ["MRR", "Hits@1", "Hits@3", "Hits@10"]
MODELS = ["TransE", "ComplEx", "TriModel"]
COLORS = {
    "MRR": "#1f77b4",
    "Hits@1": "#ff7f0e",
    "Hits@3": "#2ca02c",
    "Hits@10": "#d62728",
}


def main():
    if not os.path.exists(INPUT_CSV):
        raise FileNotFoundError(f"Missing input CSV: {INPUT_CSV}")

    df = pd.read_csv(INPUT_CSV)
    if "Dimension" not in df.columns and "dimension" in df.columns:
        df = df.rename(columns={"dimension": "Dimension"})

    df["Dimension"] = df["Dimension"].astype(int)
    dims = [100, 200, 300]

    os.makedirs(OUT_DIR, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    fig.suptitle("Link Prediction: Metric Differences Across Dimensions (Per Model)", fontsize=15, fontweight="bold")

    for ax, model in zip(axes, MODELS):
        mdf = df[df["Model"] == model].sort_values("Dimension")
        if mdf.empty:
            ax.set_title(f"{model} (no data)")
            ax.set_xticks(dims)
            ax.grid(alpha=0.3, linestyle="--")
            continue

        for metric in METRICS:
            if metric not in mdf.columns:
                continue
            ax.plot(
                mdf["Dimension"],
                mdf[metric],
                marker="o",
                linewidth=2.2,
                markersize=7,
                label=metric,
                color=COLORS[metric],
            )

        # Annotate MRR delta as compact summary on each subplot.
        if "MRR" in mdf.columns and set(dims).issubset(set(mdf["Dimension"])):
            m100 = float(mdf[mdf["Dimension"] == 100]["MRR"].iloc[0])
            m300 = float(mdf[mdf["Dimension"] == 300]["MRR"].iloc[0])
            delta = m300 - m100
            ax.text(
                0.02,
                0.04,
                f"ΔMRR (300-100): {delta:+.3f}",
                transform=ax.transAxes,
                fontsize=10,
                bbox=dict(facecolor="white", alpha=0.75, edgecolor="#999999"),
            )

        ax.set_title(model, fontsize=12, fontweight="bold")
        ax.set_xlabel("Embedding Dimension")
        ax.set_xticks(dims)
        ax.set_ylim(0.0, 1.0)
        ax.grid(alpha=0.3, linestyle="--")

    axes[0].set_ylabel("Metric Value")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, framealpha=0.95)

    plt.tight_layout(rect=[0, 0, 1, 0.90])
    out_path = os.path.join(OUT_DIR, OUT_FILE)
    plt.savefig(out_path, dpi=170, bbox_inches="tight")
    plt.close()

    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
