"""
DTI Evaluation - Unified Cross-Model Comparison Plots
======================================================

Generate comparison plots showing all models for each dimension:
- ROC curves (all models @ same dimension)
- PR curves (all models @ same dimension)
- AUC comparison (all models @ same dimension)

Reads from individual model results and creates aggregated visualizations.
"""

import os
import sys
import numpy as np
import pandas as pd
from typing import Dict, Tuple

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.ioff()

from sklearn.metrics import (
    precision_recall_curve,
    roc_curve,
)


# ----------------------------
# Configuration
# ----------------------------

EMBEDDING_DIMS = [100, 200, 300]
MODELS = {
    'TransE': ('outputs_transe', 'transe'),
    'ComplEx': ('outputs_complex', 'complex'),
    'TriModel': ('outputs_trimodel', 'trimodel'),
}

OUTPUT_BASE = 'outputs_dti_evaluation_fixed'
COMPARISON_DIR = os.path.join(OUTPUT_BASE, 'comparison_plots')


# ----------------------------
# Helper functions
# ----------------------------

def load_model_results(model_name: str, dim: int) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Load evaluation results (y_true, y_scores, metrics) for a model at a dimension.

    Uses score arrays saved by dti_evaluation.py so comparison curves exactly
    match standalone per-model curves.
    """
    model_output_dir = os.path.join(OUTPUT_BASE, model_name)
    dim_output_dir = os.path.join(model_output_dir, f"dim_{dim}")
    
    # Read metrics CSV
    metrics_csv = os.path.join(dim_output_dir, f"{model_name.lower()}_metrics.csv")
    if not os.path.exists(metrics_csv):
        print(f"  ⚠️ Metrics file not found: {metrics_csv}")
        return None, None, None
    
    df = pd.read_csv(metrics_csv)
    if df.empty:
        return None, None, None
    
    metrics = df.iloc[0].to_dict()
    
    # Load exact y_true / y_scores used in standalone plots.
    scores_npz = os.path.join(dim_output_dir, f"{model_name.lower()}_scores.npz")
    if not os.path.exists(scores_npz):
        print(f"  ⚠️ Score file not found: {scores_npz}")
        print("     Re-run DTI/dti_evaluation.py to generate score files.")
        return None, None, None

    data = np.load(scores_npz)
    y_true = data["y_true"]
    y_scores = data["y_scores"]
    
    return y_true, y_scores, metrics


def plot_roc_curves_per_dimension(dim: int, all_results: Dict):
    """
    Plot ROC curves for all models at a given dimension.
    
    all_results: {model_name: (y_true, y_scores, metrics)}
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    colors = {'TransE': '#1f77b4', 'ComplEx': '#ff7f0e', 'TriModel': '#2ca02c'}
    
    for model_name, (y_true, y_scores, metrics) in all_results.items():
        if y_true is None or y_scores is None:
            print(f"    ⚠️ Skipping {model_name}: no results")
            continue
        
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        auc = metrics['AUC-ROC']
        
        color = colors.get(model_name, '#000000')
        ax.plot(fpr, tpr, 
                label=f'{model_name} (AUC={auc:.4f})',
                linewidth=2.5,
                color=color,
                marker='o' if model_name == 'ComplEx' else ('s' if model_name == 'TransE' else '^'),
                markersize=3,
                markevery=10,
                alpha=0.8)
    
    # Random baseline
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1.5, label='Random', alpha=0.5)
    
    ax.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
    ax.set_title(f'ROC Curves - DTI Prediction (Dimension {dim})', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=11, framealpha=0.95)
    ax.grid(alpha=0.3, linestyle='--')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    
    plt.tight_layout()
    save_path = os.path.join(COMPARISON_DIR, f"01_roc_curves_dim_{dim}.png")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    ✅ Saved: {save_path}")


def plot_pr_curves_per_dimension(dim: int, all_results: Dict):
    """
    Plot PR (Precision-Recall) curves for all models at a given dimension.
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    colors = {'TransE': '#1f77b4', 'ComplEx': '#ff7f0e', 'TriModel': '#2ca02c'}
    
    # Track baseline for display
    baseline = None
    
    for model_name, (y_true, y_scores, metrics) in all_results.items():
        if y_true is None or y_scores is None:
            continue
        
        precision, recall, _ = precision_recall_curve(y_true, y_scores)
        ap = metrics['AUC-PR']
        
        color = colors.get(model_name, '#000000')
        ax.plot(recall, precision,
                label=f'{model_name} (AP={ap:.4f})',
                linewidth=2.5,
                color=color,
                marker='o' if model_name == 'ComplEx' else ('s' if model_name == 'TransE' else '^'),
                markersize=3,
                markevery=10,
                alpha=0.8)
        
        if baseline is None:
            baseline = y_true.mean()
    
    # Baseline
    if baseline is not None:
        ax.axhline(y=baseline, color='k', linestyle='--', linewidth=1.5,
                   label=f'Random baseline={baseline:.4f}', alpha=0.5)
    
    ax.set_xlabel('Recall', fontsize=12, fontweight='bold')
    ax.set_ylabel('Precision', fontsize=12, fontweight='bold')
    ax.set_title(f'Precision-Recall Curves - DTI Prediction (Dimension {dim})', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=11, framealpha=0.95)
    ax.grid(alpha=0.3, linestyle='--')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    
    plt.tight_layout()
    save_path = os.path.join(COMPARISON_DIR, f"02_pr_curves_dim_{dim}.png")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    ✅ Saved: {save_path}")


def plot_auc_comparison_per_dimension(dim: int, all_results: Dict):
    """
    Bar plot comparing AUC-ROC and AUC-PR for all models at a dimension.
    """
    model_names = [m for m, (y_true, _, _) in all_results.items() if y_true is not None]
    
    if not model_names:
        print(f"    ⚠️ No results for dimension {dim}")
        return
    
    auc_rocs = [all_results[m][2]['AUC-ROC'] for m in model_names]
    auc_prs = [all_results[m][2]['AUC-PR'] for m in model_names]
    best_f1s = [all_results[m][2]['Best_F1'] for m in model_names]
    
    x = np.arange(len(model_names))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(11, 7))
    
    bars1 = ax.bar(x - width, auc_rocs, width, label='AUC-ROC', color='#1f77b4', alpha=0.8)
    bars2 = ax.bar(x, auc_prs, width, label='AUC-PR', color='#ff7f0e', alpha=0.8)
    bars3 = ax.bar(x + width, best_f1s, width, label='Best F1', color='#2ca02c', alpha=0.8)
    
    # Add value labels on bars
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{height:.3f}',
                   ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax.set_xlabel('Model', fontsize=12, fontweight='bold')
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title(f'Model Comparison - AUC Metrics (Dimension {dim})', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, fontsize=11)
    ax.set_ylim([0, 1.0])
    ax.legend(fontsize=11, loc='upper left', framealpha=0.95)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    save_path = os.path.join(COMPARISON_DIR, f"03_auc_comparison_dim_{dim}.png")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    ✅ Saved: {save_path}")


def plot_dimension_progression():
    """
    Plot how each model performs across dimensions.
    """
    # Load all results
    results_df = []
    for model_name in MODELS.keys():
        for dim in EMBEDDING_DIMS:
            model_output_dir = os.path.join(OUTPUT_BASE, model_name)
            dim_output_dir = os.path.join(model_output_dir, f"dim_{dim}")
            metrics_csv = os.path.join(dim_output_dir, f"{model_name.lower()}_metrics.csv")
            
            if os.path.exists(metrics_csv):
                df = pd.read_csv(metrics_csv)
                if not df.empty:
                    results_df.append(df.iloc[0].to_dict())
    
    if not results_df:
        print("  ⚠️ No results to plot")
        return
    
    df = pd.DataFrame(results_df)
    
    # Plot AUC-ROC progression
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    colors = {'TransE': '#1f77b4', 'ComplEx': '#ff7f0e', 'TriModel': '#2ca02c'}
    markers = {'TransE': 's', 'ComplEx': 'o', 'TriModel': '^'}
    
    metrics_to_plot = [
        ('AUC-ROC', 'AUC-ROC Score'),
        ('AUC-PR', 'AUC-PR Score'),
        ('Best_F1', 'Best F1 Score')
    ]
    
    for ax, (metric, title) in zip(axes, metrics_to_plot):
        for model_name in MODELS.keys():
            model_data = df[df['Model'] == model_name].sort_values('Dimension')
            ax.plot(model_data['Dimension'], model_data[metric],
                   label=model_name,
                   marker=markers[model_name],
                   markersize=10,
                   linewidth=2.5,
                   color=colors[model_name],
                   alpha=0.8)
        
        ax.set_xlabel('Embedding Dimension', fontsize=11, fontweight='bold')
        ax.set_ylabel(title, fontsize=11, fontweight='bold')
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xticks(EMBEDDING_DIMS)
        ax.grid(alpha=0.3, linestyle='--')
        ax.legend(fontsize=10, loc='best', framealpha=0.95)
        ax.set_ylim([0, 1.0])
    
    plt.tight_layout()
    save_path = os.path.join(COMPARISON_DIR, "04_model_progression_across_dimensions.png")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✅ Saved: {save_path}")


def main():
    print("=" * 70)
    print("DTI EVALUATION - UNIFIED COMPARISON PLOTS")
    print("=" * 70)
    print()
    
    os.makedirs(COMPARISON_DIR, exist_ok=True)
    
    # Generate plots per dimension
    for dim in EMBEDDING_DIMS:
        print(f"Generating plots for Dimension {dim}...")
        
        # Load results for all models at this dimension
        all_results = {}
        for model_name in MODELS.keys():
            y_true, y_scores, metrics = load_model_results(model_name, dim)
            all_results[model_name] = (y_true, y_scores, metrics)
        
        # Generate comparison plots
        plot_roc_curves_per_dimension(dim, all_results)
        plot_pr_curves_per_dimension(dim, all_results)
        plot_auc_comparison_per_dimension(dim, all_results)
        print()
    
    # Generate progression plot
    print("Generating model progression plot across dimensions...")
    plot_dimension_progression()
    
    print("\n" + "=" * 70)
    print("✅ ALL COMPARISON PLOTS GENERATED")
    print("=" * 70)
    print(f"\nPlots saved in: {COMPARISON_DIR}/")
    print()
    print("Generated files:")
    print("  - 01_roc_curves_dim_100.png         (ROC curves for dimension 100)")
    print("  - 01_roc_curves_dim_200.png         (ROC curves for dimension 200)")
    print("  - 01_roc_curves_dim_300.png         (ROC curves for dimension 300)")
    print("  - 02_pr_curves_dim_100.png          (PR curves for dimension 100)")
    print("  - 02_pr_curves_dim_200.png          (PR curves for dimension 200)")
    print("  - 02_pr_curves_dim_300.png          (PR curves for dimension 300)")
    print("  - 03_auc_comparison_dim_100.png     (AUC comparison for dimension 100)")
    print("  - 03_auc_comparison_dim_200.png     (AUC comparison for dimension 200)")
    print("  - 03_auc_comparison_dim_300.png     (AUC comparison for dimension 300)")
    print("  - 04_model_progression_across_dimensions.png  (Performance across all dimensions)")
    print()


if __name__ == "__main__":
    main()
