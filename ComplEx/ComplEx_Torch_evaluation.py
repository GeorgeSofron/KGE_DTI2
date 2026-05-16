"""
ComplEx Evaluation Script
=========================
Evaluate trained ComplEx model on link prediction task.
Uses filtered ranking protocol with MRR, Hits@1, Hits@3, Hits@10.
"""

import os
import sys

# Add parent directory to path for utils import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

from utils.model import ComplEx  # Shared model definition


# ----------------------------
# Load trained model
# ----------------------------
def load_model(checkpoint_path, device="cpu"):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    model = ComplEx(
        num_entities=ckpt["num_entities"],
        num_relations=ckpt["num_relations"],
        dim=ckpt["embedding_dim"],
        reg_weight=ckpt.get("reg_weight", 0.01),
    ).to(device)

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    return model, ckpt["entity2id"], ckpt["relation2id"]


# ----------------------------
# Load triples
# ----------------------------
def load_triples(path):
    return pd.read_csv(
        path,
        sep="\t",
        header=None,
        names=["source", "relation", "target"]
    )


def triples_to_id_set(df, entity2id, relation2id):
    return set(
        zip(
            df["source"].map(entity2id),
            df["relation"].map(relation2id),
            df["target"].map(entity2id),
        )
    )


def encode_triples(df, entity2id, relation2id):
    """Encode triples DataFrame to tensor of IDs."""
    h = df["source"].map(entity2id).to_numpy()
    r = df["relation"].map(relation2id).to_numpy()
    t = df["target"].map(entity2id).to_numpy()
    triples = np.stack([h, r, t], axis=1)
    return torch.tensor(triples, dtype=torch.long)


# ----------------------------
# Ranking functions (filtered) - Optimized with vectorization
# ----------------------------
@torch.no_grad()
def compute_ranks_batch(model, test_triples, all_true, num_entities, device, batch_size=256):
    """
    Compute filtered ranks for a batch of test triples.
    
    For ComplEx, higher scores are better (unlike TransE where lower is better).
    
    Args:
        model: Trained ComplEx model
        test_triples: Tensor of shape (N, 3) with [head, relation, tail] IDs
        all_true: Set of all true triples for filtering
        num_entities: Total number of entities
        device: torch device
        batch_size: Number of triples to evaluate at once
        
    Returns:
        Tuple of (head_ranks, tail_ranks) arrays
    """
    model.eval()
    n_test = test_triples.size(0)
    
    all_tail_ranks = []
    all_head_ranks = []
    
    for start in tqdm(range(0, n_test, batch_size), desc="Evaluating batches"):
        batch = test_triples[start:start + batch_size].to(device)
        batch_h = batch[:, 0]
        batch_r = batch[:, 1]
        batch_t = batch[:, 2]
        
        # --- Tail prediction: score all entities as potential tails ---
        tail_scores = model.score_all_tails(batch_h, batch_r)  # (batch, num_entities)
        tail_scores = tail_scores.cpu().numpy()
        
        # --- Head prediction: score all entities as potential heads ---
        head_scores = model.score_all_heads(batch_r, batch_t)  # (batch, num_entities)
        head_scores = head_scores.cpu().numpy()
        
        # Apply filtering and compute ranks
        for i in range(batch.size(0)):
            h, r, t = batch[i].cpu().tolist()
            
            # Filter tail scores (higher is better for ComplEx)
            tail_sc = tail_scores[i].copy()
            for e in range(num_entities):
                if (h, r, e) in all_true and e != t:
                    tail_sc[e] = -np.inf  # Filter out other true triples
            # Rank: how many entities have HIGHER score than the correct tail
            tail_rank = int((tail_sc > tail_sc[t]).sum() + 1)
            all_tail_ranks.append(tail_rank)
            
            # Filter head scores
            head_sc = head_scores[i].copy()
            for e in range(num_entities):
                if (e, r, t) in all_true and e != h:
                    head_sc[e] = -np.inf
            head_rank = int((head_sc > head_sc[h]).sum() + 1)
            all_head_ranks.append(head_rank)
    
    return np.array(all_head_ranks), np.array(all_tail_ranks)


# ----------------------------
# Evaluation
# ----------------------------
def evaluate(model, test_triples, all_true, num_entities, device, batch_size=256):
    """
    Evaluate model on test triples using batched computation.
    
    Returns:
        Dictionary with MRR, Hits@1, Hits@3, Hits@10
    """
    head_ranks, tail_ranks = compute_ranks_batch(
        model, test_triples, all_true, num_entities, device, batch_size
    )
    
    # Combine head and tail ranks
    all_ranks = np.concatenate([head_ranks, tail_ranks])
    
    return {
        "MRR": float(np.mean(1.0 / all_ranks)),
        "Hits@1": float((all_ranks <= 1).mean()),
        "Hits@3": float((all_ranks <= 3).mean()),
        "Hits@10": float((all_ranks <= 10).mean()),
    }


def save_evaluation_results(metrics: dict, output_path: str):
    """Save evaluation results to file."""
    with open(output_path, "w") as f:
        f.write("ComplEx Evaluation Results\n")
        f.write("=" * 40 + "\n\n")
        for name, value in metrics.items():
            f.write(f"{name}: {value:.4f}\n")
    print(f"Results saved to: {output_path}")


# ----------------------------
# Main
# ----------------------------
if __name__ == "__main__":
    DEVICE = "cpu"  # change to "cuda" if available
    BATCH_SIZE = 256
    EMBEDDING_DIMS = [100, 200, 300]

    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR = os.path.join(PROJECT_ROOT, "data", "complex")
    OUTPUT_ROOT = os.path.join(PROJECT_ROOT, "outputs_complex")

    train_df = load_triples(os.path.join(DATA_DIR, "train.txt"))
    test_df = load_triples(os.path.join(DATA_DIR, "test.txt"))
    valid_path = os.path.join(DATA_DIR, "valid.txt")
    valid_df = load_triples(valid_path) if os.path.exists(valid_path) else None

    summary_rows = []

    for dim in EMBEDDING_DIMS:
        model_dir = os.path.join(OUTPUT_ROOT, f"dim_{dim}")
        model_path = os.path.join(model_dir, "complex_model.pt")
        output_path = os.path.join(model_dir, "Evaluation.txt")

        if not os.path.exists(model_path):
            print(f"Skipping dimension {dim}: checkpoint not found at {model_path}")
            continue

        print(f"\n{'=' * 60}")
        print(f"Evaluating ComplEx with embedding dimension {dim}")
        print(f"Loading model from {model_path}")
        print(f"{'=' * 60}")

        model, entity2id, relation2id = load_model(model_path, DEVICE)
        num_entities = len(entity2id)
        print(f"Loaded ComplEx model: {num_entities} entities, {len(relation2id)} relations")

        test_df_filtered = test_df[
            (test_df["source"].isin(entity2id.keys())) &
            (test_df["relation"].isin(relation2id.keys())) &
            (test_df["target"].isin(entity2id.keys()))
        ].reset_index(drop=True)

        print(f"Filtered test set: {len(test_df_filtered)} triples")

        test_triples = encode_triples(test_df_filtered, entity2id, relation2id)

        all_true = triples_to_id_set(train_df, entity2id, relation2id)
        all_true |= triples_to_id_set(test_df_filtered, entity2id, relation2id)

        if valid_df is not None:
            valid_df_filtered = valid_df[
                (valid_df["source"].isin(entity2id.keys())) &
                (valid_df["relation"].isin(relation2id.keys())) &
                (valid_df["target"].isin(entity2id.keys()))
            ]
            all_true |= triples_to_id_set(valid_df_filtered, entity2id, relation2id)
            print(f"Added {len(valid_df_filtered)} validation triples to filter set")

        print(f"Total true triples for filtering: {len(all_true)}")

        print("\nEvaluating ComplEx model...")
        metrics = evaluate(model, test_triples, all_true, num_entities, DEVICE, BATCH_SIZE)

        print("\n" + "=" * 40)
        print(f"ComplEx Evaluation Results (dim={dim})")
        print("=" * 40)
        for name, value in metrics.items():
            print(f"{name}: {value:.4f}")
        print("=" * 40)

        save_evaluation_results(metrics, output_path)
        summary_rows.append({"dimension": dim, **metrics})

    if summary_rows:
        summary_path = os.path.join(OUTPUT_ROOT, "evaluation_summary.csv")
        pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
        print(f"Summary saved to: {summary_path}")
