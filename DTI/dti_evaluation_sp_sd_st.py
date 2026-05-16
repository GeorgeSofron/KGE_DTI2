"""
DTI Evaluation - Sp / Sd / St stratified evaluation
===================================================

Implements the test-set partitioning scheme described by Pahikkala et al.
("Toward more realistic drug-target interaction predictions"):

    Sp : test DTIs where BOTH the drug AND the target already appear in
         training DTIs (warm-start, easiest case, largest group).
    Sd : test DTIs whose DRUG does NOT appear in any training DTI
         (cold drug).
    St : test DTIs whose TARGET does NOT appear in any training DTI
         (cold target).

(Optionally also Sdt: both unseen - reported here for completeness.)

This script:
- Loads the same train/valid/test triples used by the existing pipeline.
- Buckets every test DTI positive into Sp / Sd / St (and Sdt).
- Generates per-drug negatives ONCE per group (so positives and negatives
  share the same "coldness" property within the group), with the same
  negative-sampling semantics as ``dti_evaluation.py`` (random or
  degree-matched).
- Scores positives + negatives with each trained model (TransE, ComplEx,
  TriModel) at each embedding dimension.
- Computes AUC-ROC and AUC-PR (plus Best F1) for EACH group, mirroring
  the protocol from the cited literature.
- Saves per-group / per-model / per-dimension CSVs, a master CSV
  ``dti_metrics_sp_sd_st.csv`` and a master text report.

Nothing in the existing scripts is modified - this is a new, additive
analysis built on top of the helpers already defined in
``DTI/dti_evaluation.py``.

Usage:
    python -m DTI.dti_evaluation_sp_sd_st
    # or, from repo root:
    python DTI/dti_evaluation_sp_sd_st.py
"""

import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Tuple, Set, Optional

import numpy as np
import pandas as pd

# Make repo root importable so we can reuse helpers from dti_evaluation.py
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

# Reuse all existing infrastructure (loaders, scorers, metrics, plots, ...)
from DTI.dti_evaluation import (  # type: ignore
    EMBEDDING_DIMS,
    MODELS,
    load_triples,
    get_dti_pairs,
    build_positive_set,
    get_all_entities_by_type,
    load_transe_model,
    load_complex_model,
    load_trimodel_model,
    score_pairs,
    compute_target_degree,
    generate_negatives_per_drug,
    compute_metrics,
    plot_roc_curves,
    plot_pr_curves,
    plot_metrics_comparison,
)


# ----------------------------
# Config
# ----------------------------

GROUPS = ("Sp", "Sd", "St", "Sdt")  # Sdt = both drug & target unseen


@dataclass
class SpSdStConfig:
    dti_relation: str = "DRUG_TARGET"
    device: str = "cpu"
    neg_ratio: int = 10
    neg_sampling: str = "degree_matched"   # "random" or "degree_matched"
    seed: int = 42
    batch_size: int = 4096
    output_dir: str = "outputs_dti_evaluation_sp_sd_st"


# ----------------------------
# Group partitioning
# ----------------------------

def partition_test_positives_by_group(
    test_pos_pairs: List[Tuple[str, str]],
    train_pos_pairs: List[Tuple[str, str]],
) -> Dict[str, List[Tuple[str, str]]]:
    """
    Bucket each test (drug, target) DTI into Sp / Sd / St / Sdt based on
    whether its drug and/or target appear in TRAINING DTIs.

        Sp  : drug in train-DTIs  AND target in train-DTIs
        Sd  : drug NOT in train-DTIs, target IN train-DTIs
        St  : drug IN train-DTIs,    target NOT in train-DTIs
        Sdt : drug NOT in train-DTIs AND target NOT in train-DTIs
    """
    train_drugs = {d for (d, _) in train_pos_pairs}
    train_targets = {t for (_, t) in train_pos_pairs}

    groups: Dict[str, List[Tuple[str, str]]] = {g: [] for g in GROUPS}
    for (d, t) in test_pos_pairs:
        d_known = d in train_drugs
        t_known = t in train_targets
        if d_known and t_known:
            groups["Sp"].append((d, t))
        elif (not d_known) and t_known:
            groups["Sd"].append((d, t))
        elif d_known and (not t_known):
            groups["St"].append((d, t))
        else:
            groups["Sdt"].append((d, t))
    return groups


# ----------------------------
# Per-model evaluation for one group
# ----------------------------

def evaluate_group(
    model, model_type: str, model_name: str,
    group_name: str,
    pos_pairs: List[Tuple[str, str]],
    neg_pairs: List[Tuple[str, str]],
    entity2id: Dict[str, int],
    relation2id: Dict[str, int],
    cfg: SpSdStConfig,
) -> Optional[Dict]:
    if cfg.dti_relation not in relation2id:
        print(f"      ⚠️ {model_name}/{group_name}: relation "
              f"{cfg.dti_relation} not in relation2id")
        return None
    if not pos_pairs:
        print(f"      ⚠️ {model_name}/{group_name}: no positives in group")
        return None

    rel_id = relation2id[cfg.dti_relation]

    pos_valid = [(d, p) for (d, p) in pos_pairs if d in entity2id and p in entity2id]
    neg_valid = [(d, p) for (d, p) in neg_pairs if d in entity2id and p in entity2id]
    dropped_pos = len(pos_pairs) - len(pos_valid)
    dropped_neg = len(neg_pairs) - len(neg_valid)

    if not pos_valid or not neg_valid:
        print(f"      ⚠️ {model_name}/{group_name}: empty after OOV filter "
              f"(pos={len(pos_valid)}, neg={len(neg_valid)})")
        return None

    pos_scores = score_pairs(model, model_type, pos_valid, rel_id,
                             entity2id, cfg.device, batch_size=cfg.batch_size)
    neg_scores = score_pairs(model, model_type, neg_valid, rel_id,
                             entity2id, cfg.device, batch_size=cfg.batch_size)

    if len(pos_scores) == 0 or len(neg_scores) == 0:
        return None

    y_scores = np.concatenate([pos_scores, neg_scores])
    y_true = np.concatenate([np.ones(len(pos_scores)),
                             np.zeros(len(neg_scores))])

    metrics = compute_metrics(y_true, y_scores)
    metrics["n_positives"] = int(len(pos_scores))
    metrics["n_negatives"] = int(len(neg_scores))
    metrics["dropped_positives_oov"] = int(dropped_pos)
    metrics["dropped_negatives_oov"] = int(dropped_neg)
    return {"metrics": metrics, "y_true": y_true, "y_scores": y_scores}


# ----------------------------
# Main
# ----------------------------

def main():
    cfg = SpSdStConfig()
    rng = np.random.default_rng(cfg.seed)

    print("=" * 72)
    print("DTI EVALUATION - Sp / Sd / St STRATIFIED")
    print("=" * 72)
    print(f"Dimensions: {EMBEDDING_DIMS}")
    print(f"neg_ratio={cfg.neg_ratio}  neg_sampling={cfg.neg_sampling}  "
          f"seed={cfg.seed}")
    print()

    # ---- Load data (same fallbacks as dti_evaluation.py) ----
    data_dirs = ['data/trimodel', 'data/transe', 'data/complex', 'data']
    train_df = valid_df = test_df = None
    data_dir_used = None
    for d in data_dirs:
        try:
            train_df = load_triples(f"{d}/train.txt")
            test_df = load_triples(f"{d}/test.txt")
            try:
                valid_df = load_triples(f"{d}/valid.txt")
            except Exception:
                valid_df = pd.DataFrame(columns=["source", "relation", "target"])
            data_dir_used = d
            print(f"✅ Loaded data from: {d}")
            break
        except Exception:
            continue
    if train_df is None:
        print("❌ Could not load train/test data from expected folders.")
        return

    # ---- Build entity / positive sets ----
    all_df = pd.concat([train_df, valid_df, test_df], ignore_index=True)
    drugs, proteins = get_all_entities_by_type(all_df, cfg.dti_relation)
    print(f"Drugs={len(drugs)} Proteins={len(proteins)} (from all splits)")

    all_positive_pairs = build_positive_set(train_df, valid_df, test_df,
                                            cfg.dti_relation)
    train_pos_pairs = get_dti_pairs(train_df, cfg.dti_relation)
    test_pos_pairs = get_dti_pairs(test_df, cfg.dti_relation)
    print(f"Train DTIs : {len(train_pos_pairs)}")
    print(f"Test DTIs  : {len(test_pos_pairs)}")
    print(f"All known DTIs: {len(all_positive_pairs)}")

    # ---- Partition test set into Sp / Sd / St / Sdt ----
    grouped_pos = partition_test_positives_by_group(test_pos_pairs,
                                                    train_pos_pairs)
    print("\nTest-set partition (positives):")
    for g in GROUPS:
        print(f"  {g:3s}: {len(grouped_pos[g]):6d}")
    print()

    # ---- Per-group fixed negatives (sampled once, reused across models) ----
    p_deg = compute_target_degree(train_pos_pairs)
    proteins_universe = sorted(list(proteins))

    grouped_neg: Dict[str, List[Tuple[str, str]]] = {}
    for g in GROUPS:
        if not grouped_pos[g]:
            grouped_neg[g] = []
            continue
        # Use an independent RNG per group so adding/removing a group
        # doesn't shift the negatives sampled for the others.
        g_rng = np.random.default_rng(cfg.seed + abs(hash(g)) % (2**31))
        negs = generate_negatives_per_drug(
            pos_pairs=grouped_pos[g],
            all_pos_set=all_positive_pairs,
            proteins_universe=proteins_universe,
            neg_ratio=cfg.neg_ratio,
            rng=g_rng,
            mode=cfg.neg_sampling,
            p_deg=p_deg if cfg.neg_sampling == "degree_matched" else None,
        )
        grouped_neg[g] = negs
        print(f"  Negatives for {g}: {len(negs)} (target = "
              f"{len(grouped_pos[g]) * cfg.neg_ratio})")
    print()

    # ---- Output dir ----
    os.makedirs(cfg.output_dir, exist_ok=True)
    all_rows: List[Dict] = []

    # ---- Loop over models / dimensions / groups ----
    for model_name in ('TransE', 'ComplEx', 'TriModel'):
        print("=" * 72)
        print(f"MODEL: {model_name}")
        print("=" * 72)

        model_dir = os.path.join(cfg.output_dir, model_name)
        os.makedirs(model_dir, exist_ok=True)

        for dim in EMBEDDING_DIMS:
            print(f"\n  ---- Dimension {dim} ----")
            dim_dir = os.path.join(model_dir, f"dim_{dim}")
            os.makedirs(dim_dir, exist_ok=True)
            os.makedirs(os.path.join(dim_dir, "figures"), exist_ok=True)

            if model_name == 'TransE':
                ckpt_path = f'{MODELS["TransE"]}/dim_{dim}/transe_model.pt'
                model_type = 'transe'
                model_data_dir = 'data/transe'
                loader = load_transe_model
            elif model_name == 'ComplEx':
                ckpt_path = f'{MODELS["ComplEx"]}/dim_{dim}/complex_model.pt'
                model_type = 'complex'
                model_data_dir = 'data/complex'
                loader = load_complex_model
            elif model_name == 'TriModel':
                ckpt_path = f'{MODELS["TriModel"]}/dim_{dim}/trimodel_model.pt'
                model_type = 'trimodel'
                model_data_dir = 'data/trimodel'
                loader = load_trimodel_model
            else:
                raise ValueError(model_name)

            if not os.path.exists(ckpt_path):
                print(f"    ⚠️ Checkpoint not found: {ckpt_path}")
                continue

            print(f"    Loading {model_name} (dim={dim})...")
            model, entity2id, relation2id = loader(ckpt_path, cfg.device)

            # Optional model-specific test split (kept consistent with
            # dti_evaluation.py). If present and different, re-partition.
            try:
                mtest_df = load_triples(f"{model_data_dir}/test.txt")
                mtest_pos = get_dti_pairs(mtest_df, cfg.dti_relation)
                if set(mtest_pos) != set(test_pos_pairs):
                    print(f"    Note: model-specific test split differs "
                          f"({len(mtest_pos)} vs {len(test_pos_pairs)}); "
                          f"re-partitioning for this model.")
                    grouped_pos_m = partition_test_positives_by_group(
                        mtest_pos, train_pos_pairs)
                    grouped_neg_m = {}
                    for g in GROUPS:
                        if not grouped_pos_m[g]:
                            grouped_neg_m[g] = []
                            continue
                        g_rng = np.random.default_rng(
                            cfg.seed + abs(hash((model_name, dim, g))) % (2**31))
                        grouped_neg_m[g] = generate_negatives_per_drug(
                            pos_pairs=grouped_pos_m[g],
                            all_pos_set=all_positive_pairs,
                            proteins_universe=proteins_universe,
                            neg_ratio=cfg.neg_ratio,
                            rng=g_rng,
                            mode=cfg.neg_sampling,
                            p_deg=p_deg if cfg.neg_sampling == "degree_matched" else None,
                        )
                else:
                    grouped_pos_m = grouped_pos
                    grouped_neg_m = grouped_neg
            except Exception:
                grouped_pos_m = grouped_pos
                grouped_neg_m = grouped_neg

            # Aggregate plot inputs across groups for this (model, dim)
            results_for_plot: Dict[str, Dict] = {}

            for g in GROUPS:
                pos_pairs = grouped_pos_m[g]
                neg_pairs = grouped_neg_m[g]
                print(f"    [{g}] positives={len(pos_pairs)}  "
                      f"negatives={len(neg_pairs)}")

                out = evaluate_group(
                    model=model, model_type=model_type,
                    model_name=model_name, group_name=g,
                    pos_pairs=pos_pairs, neg_pairs=neg_pairs,
                    entity2id=entity2id, relation2id=relation2id,
                    cfg=cfg,
                )
                if out is None:
                    continue

                m = out["metrics"]
                print(f"        AUC-ROC={m['AUC-ROC']:.4f}  "
                      f"AUC-PR={m['AUC-PR']:.4f}  "
                      f"BestF1={m['Best_F1']:.4f}  "
                      f"(P/N={m['n_positives']}/{m['n_negatives']})")

                row = {
                    "Model": model_name,
                    "Dimension": dim,
                    "Group": g,
                    "AUC-ROC": m["AUC-ROC"],
                    "AUC-PR": m["AUC-PR"],
                    "Best_F1": m["Best_F1"],
                    "Best_Threshold": m["Best_Threshold"],
                    "Precision@Best_F1": m["Precision@Best_F1"],
                    "Recall@Best_F1": m["Recall@Best_F1"],
                    "n_positives": m["n_positives"],
                    "n_negatives": m["n_negatives"],
                    "dropped_positives_oov": m["dropped_positives_oov"],
                    "dropped_negatives_oov": m["dropped_negatives_oov"],
                    "neg_ratio": cfg.neg_ratio,
                    "neg_sampling": cfg.neg_sampling,
                    "seed": cfg.seed,
                }
                all_rows.append(row)

                # Save per-group raw scores
                np.savez(
                    os.path.join(dim_dir, f"{model_name.lower()}_{g}_scores.npz"),
                    y_true=out["y_true"], y_scores=out["y_scores"],
                )
                results_for_plot[g] = out

            # Per-(model, dim) CSV across groups
            dim_rows = [r for r in all_rows
                        if r["Model"] == model_name and r["Dimension"] == dim]
            if dim_rows:
                pd.DataFrame(dim_rows).to_csv(
                    os.path.join(dim_dir,
                                 f"{model_name.lower()}_metrics_sp_sd_st.csv"),
                    index=False,
                )

            # Plots: each group as a "model" curve in the comparison figure
            if results_for_plot:
                try:
                    plot_roc_curves(
                        results_for_plot,
                        os.path.join(dim_dir, "figures", "roc_by_group.png"),
                    )
                    plot_pr_curves(
                        results_for_plot,
                        os.path.join(dim_dir, "figures", "pr_by_group.png"),
                    )
                    plot_metrics_comparison(
                        results_for_plot,
                        os.path.join(dim_dir, "figures", "auc_by_group.png"),
                    )
                except Exception as e:
                    print(f"    ⚠️ Plot error: {e}")

    # ---- Master outputs ----
    print("\n" + "=" * 72)
    print("AGGREGATED RESULTS (Sp / Sd / St)")
    print("=" * 72)

    if not all_rows:
        print("No results collected.")
        return

    master_df = pd.DataFrame(all_rows)
    master_csv = os.path.join(cfg.output_dir, "dti_metrics_sp_sd_st.csv")
    master_df.to_csv(master_csv, index=False)
    print(f"✅ Saved: {master_csv}")

    # Pivot-style summary printout
    summary = master_df.sort_values(["Model", "Dimension", "Group"])[
        ["Model", "Dimension", "Group",
         "n_positives", "n_negatives", "AUC-ROC", "AUC-PR", "Best_F1"]
    ]
    print("\n" + summary.to_string(index=False))

    master_report = os.path.join(cfg.output_dir,
                                 "dti_evaluation_sp_sd_st_report.txt")
    with open(master_report, "w") as f:
        f.write("DTI EVALUATION - Sp / Sd / St STRATIFIED\n")
        f.write("=" * 72 + "\n")
        f.write(f"Models     : {list(MODELS.keys())}\n")
        f.write(f"Dimensions : {EMBEDDING_DIMS}\n")
        f.write(f"Relation   : {cfg.dti_relation}\n")
        f.write(f"neg_ratio  : {cfg.neg_ratio}\n")
        f.write(f"neg_sampl. : {cfg.neg_sampling}\n")
        f.write(f"seed       : {cfg.seed}\n")
        f.write(f"data dir   : {data_dir_used}\n\n")
        f.write("Group definitions (w.r.t. TRAINING DTIs):\n")
        f.write("  Sp : drug known     AND target known\n")
        f.write("  Sd : drug UNKNOWN   AND target known   (cold drug)\n")
        f.write("  St : drug known     AND target UNKNOWN (cold target)\n")
        f.write("  Sdt: drug UNKNOWN   AND target UNKNOWN (both cold)\n\n")
        f.write("Group sizes (positives in test set):\n")
        for g in GROUPS:
            f.write(f"  {g:3s}: {len(grouped_pos[g])}\n")
        f.write("\nResults:\n")
        f.write("-" * 72 + "\n")
        f.write(summary.to_string(index=False))
        f.write("\n")
    print(f"✅ Saved: {master_report}")

    print("\n" + "=" * 72)
    print("✅ Sp / Sd / St EVALUATION COMPLETE")
    print("=" * 72)


if __name__ == "__main__":
    main()
