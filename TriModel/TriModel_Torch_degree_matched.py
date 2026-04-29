"""
TriModel: Tri-vector Embeddings for Knowledge Graph Completion
===============================================================
Degree-matched negative sampling variant.

This script trains TriModel using degree-matched negative samples: when
corrupting a positive triple, the replacement entity is drawn from the same
(or a neighbouring) degree bin as the original head/tail entity. This produces
"harder", less trivially distinguishable negatives compared to uniform random
sampling and gives a more realistic picture of model performance on entities
that are similar in connectivity to the true target.

Reference:
    Based on libkge implementation by Sameh Kamaleldin
    https://github.com/samehkamaleldin/libkge
"""

import os
import sys
import time
from datetime import datetime

# Add parent directory to path for utils import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split

from utils.model import TriModel, compute_kge_loss, pairwise_logistic_loss, bce_loss


# ----------------------------
# Tee logger: duplicate stdout to a log file
# ----------------------------
class TeeLogger:
    """Write to both the original stream and a log file."""

    def __init__(self, stream, log_file):
        self.stream = stream
        self.log_file = log_file

    def write(self, data):
        self.stream.write(data)
        self.log_file.write(data)
        self.log_file.flush()

    def flush(self):
        self.stream.flush()
        self.log_file.flush()


def _format_duration(seconds: float) -> str:
    """Format a duration in seconds as HH:MM:SS."""
    seconds = int(round(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


# ----------------------------
# Reproducibility
# ----------------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ----------------------------
# Data loading
# ----------------------------
def load_triples(path: str) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        sep="\t",
        header=None,
        names=["source", "relation", "target"]
    )
    return df


def make_mappings(df: pd.DataFrame):
    entities = pd.Index(pd.concat([df["source"], df["target"]], ignore_index=True).unique())
    relations = pd.Index(df["relation"].unique())

    entity2id = {e: i for i, e in enumerate(entities)}
    relation2id = {r: i for i, r in enumerate(relations)}
    return entity2id, relation2id


def encode_triples(df: pd.DataFrame, entity2id, relation2id) -> torch.Tensor:
    h = df["source"].map(entity2id).to_numpy()
    r = df["relation"].map(relation2id).to_numpy()
    t = df["target"].map(entity2id).to_numpy()
    triples = np.stack([h, r, t], axis=1)
    return torch.tensor(triples, dtype=torch.long)


def save_splits(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    valid_df: pd.DataFrame = None,
    data_dir: str = "data"
):
    """Save train/valid/test splits to disk for reproducibility."""
    os.makedirs(data_dir, exist_ok=True)

    train_path = os.path.join(data_dir, "train.txt")
    test_path = os.path.join(data_dir, "test.txt")

    train_df.to_csv(train_path, sep="\t", header=False, index=False)
    test_df.to_csv(test_path, sep="\t", header=False, index=False)

    print(f"\nData splits saved:")
    print(f" - {train_path} ({len(train_df)} triples)")

    if valid_df is not None:
        valid_path = os.path.join(data_dir, "valid.txt")
        valid_df.to_csv(valid_path, sep="\t", header=False, index=False)
        print(f" - {valid_path} ({len(valid_df)} triples)")

    print(f" - {test_path} ({len(test_df)} triples)")


# ----------------------------
# Negative sampling (degree-matched)
# ----------------------------
def build_true_triples_set(triples: torch.Tensor) -> set:
    """Convert tensor of triples to a set of tuples for O(1) lookup."""
    return set(map(tuple, triples.cpu().numpy().tolist()))


def compute_entity_degrees(triples: torch.Tensor, num_entities: int) -> np.ndarray:
    """
    Count how many times each entity appears (as head or tail) in the training set.
    """
    arr = triples.cpu().numpy()
    deg = np.zeros(num_entities, dtype=np.int64)
    np.add.at(deg, arr[:, 0], 1)
    np.add.at(deg, arr[:, 2], 1)
    return deg


def build_degree_bins(degrees: np.ndarray, bin_size: int = 5):
    """
    Group entity ids into bins based on their degree (degree // bin_size).
    """
    bin_of_entity = degrees // bin_size
    bins = {}
    for e_id, b in enumerate(bin_of_entity):
        bins.setdefault(int(b), []).append(e_id)
    bins = {b: np.asarray(ids, dtype=np.int64) for b, ids in bins.items()}
    return bin_of_entity, bins


def _candidate_pool(
    pos_bin: int,
    bins: dict,
    all_entities: np.ndarray,
    min_pool: int = 20,
) -> np.ndarray:
    """
    Build a candidate pool for a given positive bin, expanding to neighbouring bins
    if it is too small, falling back to all entities as a last resort.
    """
    pool = bins.get(pos_bin, np.empty(0, dtype=np.int64))
    if pool.size >= min_pool:
        return pool

    pieces = [pool]
    for offset in (-1, 1, -2, 2, -3, 3):
        pieces.append(bins.get(pos_bin + offset, np.empty(0, dtype=np.int64)))
        if sum(p.size for p in pieces) >= min_pool:
            return np.concatenate(pieces)

    combined = np.concatenate(pieces)
    if combined.size >= min_pool:
        return combined
    return all_entities


def corrupt_triples(
    pos_triples: torch.Tensor,
    num_entities: int,
    true_triples: set = None,
    bin_of_entity: np.ndarray = None,
    bins: dict = None,
    all_entities: np.ndarray = None,
    max_attempts: int = 10,
) -> torch.Tensor:
    """
    Create one degree-matched negative triple per positive triple by corrupting
    head OR tail with an entity drawn from the same (or a neighbouring) degree bin.
    Falls back to uniform random sampling if no degree information is supplied.
    """
    neg = pos_triples.clone().cpu()
    batch_size = neg.size(0)
    use_degree = (
        bin_of_entity is not None and bins is not None and all_entities is not None
    )

    mask = torch.rand(batch_size) < 0.5  # True => corrupt head, else tail

    for i in range(batch_size):
        h, r, t = neg[i].tolist()
        corrupt_head = bool(mask[i].item())
        target_entity = h if corrupt_head else t

        if use_degree:
            pos_bin = int(bin_of_entity[target_entity])
            pool = _candidate_pool(pos_bin, bins, all_entities)
        else:
            pool = None

        random_entity = None
        for _ in range(max_attempts):
            if pool is not None:
                random_entity = int(pool[np.random.randint(pool.size)])
            else:
                random_entity = int(np.random.randint(num_entities))

            if corrupt_head:
                candidate = (random_entity, r, t)
            else:
                candidate = (h, r, random_entity)

            if true_triples is None or candidate not in true_triples:
                break

        if corrupt_head:
            neg[i, 0] = random_entity
        else:
            neg[i, 2] = random_entity

    return neg


# ----------------------------
# Training loop
# ----------------------------
@torch.no_grad()
def compute_validation_loss(
    model: TriModel,
    valid_triples: torch.Tensor,
    num_entities: int,
    true_triples: set,
    loss_type: str,
    device: str,
    bin_of_entity: np.ndarray = None,
    bins: dict = None,
    all_entities: np.ndarray = None,
) -> float:
    """Compute average loss on validation set with degree-matched negatives."""
    model.eval()
    valid_triples = valid_triples.to(device)
    neg = corrupt_triples(
        valid_triples,
        num_entities,
        true_triples,
        bin_of_entity=bin_of_entity,
        bins=bins,
        all_entities=all_entities,
    ).to(device)

    pos_scores = model(valid_triples)
    neg_scores = model(neg)

    loss = compute_kge_loss(pos_scores, neg_scores, loss_type=loss_type)
    return loss.item()


def train_trimodel(
    train_triples: torch.Tensor,
    num_entities: int,
    num_relations: int,
    valid_triples: torch.Tensor = None,
    dim: int = 100,
    reg_weight: float = 0.01,
    lr: float = 1e-3,
    batch_size: int = 1024,
    epochs: int = 100,
    device: str = "cpu",
    filter_negatives: bool = True,
    early_stopping_patience: int = 10,
    lr_scheduler_patience: int = 5,
    lr_scheduler_factor: float = 0.5,
    negative_samples: int = 1,
    loss_type: str = "bce",
    degree_bin_size: int = 5,
):
    """
    Train TriModel with degree-matched negative sampling.

    Args:
        train_triples: Training triples tensor
        num_entities: Number of unique entities
        num_relations: Number of unique relations
        valid_triples: Validation triples for early stopping
        dim: Embedding dimension (for each of 3 components)
        reg_weight: L3 regularization weight
        lr: Learning rate
        batch_size: Training batch size
        epochs: Maximum epochs
        device: 'cpu' or 'cuda'
        filter_negatives: Whether to filter false negatives
        early_stopping_patience: Epochs without improvement before stopping
        lr_scheduler_patience: Epochs before reducing LR
        lr_scheduler_factor: Factor to reduce LR by
        negative_samples: Number of negative samples per positive
        loss_type: Loss function type (see compute_kge_loss for options)
        degree_bin_size: Width of the degree bins used for negative sampling

    Returns:
        Trained model and loss history
    """
    model = TriModel(num_entities, num_relations, dim=dim, reg_weight=reg_weight).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Degree-matched negative sampling: build entity degrees / bins from training data
    entity_degrees = compute_entity_degrees(train_triples, num_entities)
    bin_of_entity, bins = build_degree_bins(entity_degrees, bin_size=degree_bin_size)
    all_entities = np.arange(num_entities, dtype=np.int64)
    print(
        f"Degree-matched sampling: bin_size={degree_bin_size}, num_bins={len(bins)}, "
        f"max_degree={int(entity_degrees.max())}, mean_degree={entity_degrees.mean():.2f}"
    )

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        patience=lr_scheduler_patience,
        factor=lr_scheduler_factor
    )

    # Build set of true triples for filtering
    true_triples = build_true_triples_set(train_triples) if filter_negatives else None

    # Early stopping
    best_valid_loss = float('inf')
    best_model_state = None
    patience_counter = 0
    use_early_stopping = valid_triples is not None and early_stopping_patience > 0

    losses_per_epoch = []
    valid_losses_per_epoch = []

    train_triples = train_triples.to(device)
    n = train_triples.size(0)

    print(f"\nTraining TriModel (degree-matched negatives) with loss_type='{loss_type}'")
    print(f"Entities: {num_entities}, Relations: {num_relations}, Dim: {dim}")
    print(f"Regularization weight: {reg_weight}, Negative samples: {negative_samples}\n")

    epoch_times = []
    train_start = time.perf_counter()

    for epoch in range(1, epochs + 1):
        epoch_start = time.perf_counter()
        model.train()
        perm = torch.randperm(n, device=device)
        epoch_loss = 0.0
        num_batches = 0

        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            pos = train_triples[idx]

            # Generate multiple negative samples (degree-matched)
            neg_list = []
            for _ in range(negative_samples):
                neg_list.append(
                    corrupt_triples(
                        pos,
                        num_entities,
                        true_triples,
                        bin_of_entity=bin_of_entity,
                        bins=bins,
                        all_entities=all_entities,
                    ).to(device)
                )
            neg = torch.cat(neg_list, dim=0)

            # Compute scores
            pos_scores = model(pos)
            neg_scores = model(neg)

            # Repeat positive scores for multiple negatives if needed
            if negative_samples > 1:
                pos_scores_expanded = pos_scores.repeat(negative_samples)
            else:
                pos_scores_expanded = pos_scores

            # Compute loss using unified loss function
            base_loss = compute_kge_loss(pos_scores_expanded, neg_scores, loss_type=loss_type)

            # Regularization (L3 norm as in original TriModel)
            reg_loss = model.regularization(pos)

            loss = base_loss + reg_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

        avg_loss = epoch_loss / max(1, num_batches)
        losses_per_epoch.append(avg_loss)

        # Validation and early stopping
        if use_early_stopping:
            valid_loss = compute_validation_loss(
                model, valid_triples, num_entities, true_triples, loss_type, device,
                bin_of_entity=bin_of_entity, bins=bins, all_entities=all_entities,
            )
            valid_losses_per_epoch.append(valid_loss)

            scheduler.step(valid_loss)
            current_lr = optimizer.param_groups[0]['lr']

            epoch_time = time.perf_counter() - epoch_start
            epoch_times.append(epoch_time)
            print(
                f"Epoch {epoch:03d}/{epochs} | train_loss={avg_loss:.6f} | "
                f"valid_loss={valid_loss:.6f} | lr={current_lr:.2e} | "
                f"epoch_time={epoch_time:.2f}s"
            )

            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                best_model_state = model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    print(f"\nEarly stopping at epoch {epoch} (patience={early_stopping_patience})")
                    print(f"Best validation loss: {best_valid_loss:.6f}")
                    break
        else:
            epoch_time = time.perf_counter() - epoch_start
            epoch_times.append(epoch_time)
            print(
                f"Epoch {epoch:03d}/{epochs} | loss={avg_loss:.6f} | "
                f"epoch_time={epoch_time:.2f}s"
            )

    total_train_time = time.perf_counter() - train_start
    avg_epoch_time = float(np.mean(epoch_times)) if epoch_times else 0.0
    print(
        f"\nTraining finished: epochs_run={len(epoch_times)} | "
        f"total_time={_format_duration(total_train_time)} ({total_train_time:.2f}s) | "
        f"avg_epoch_time={avg_epoch_time:.2f}s"
    )

    # Restore best model
    if use_early_stopping and best_model_state is not None:
        model.load_state_dict(best_model_state)
        print("Restored best model from early stopping.")

    return model, losses_per_epoch, valid_losses_per_epoch, epoch_times, total_train_time


# ----------------------------
# Save outputs
# ----------------------------
def save_outputs(
    output_dir, train_losses, valid_losses, entity2id, relation2id, model: TriModel,
    epoch_times=None, total_train_time=None,
):
    os.makedirs(output_dir, exist_ok=True)

    # Loss CSV + plot
    loss_data = {
        "epoch": np.arange(1, len(train_losses) + 1),
        "train_loss": train_losses
    }
    if valid_losses:
        loss_data["valid_loss"] = valid_losses
    if epoch_times is not None and len(epoch_times) == len(train_losses):
        loss_data["epoch_time_seconds"] = epoch_times
    loss_df = pd.DataFrame(loss_data)
    loss_df.to_csv(os.path.join(output_dir, "training_loss.csv"), index=False)

    # Timing summary
    if epoch_times is not None:
        timing_path = os.path.join(output_dir, "timing.txt")
        avg_epoch = float(np.mean(epoch_times)) if epoch_times else 0.0
        with open(timing_path, "w") as f:
            f.write("TriModel (Degree-Matched Negatives) Training Timing\n")
            f.write("=" * 50 + "\n")
            f.write(f"Epochs run        : {len(epoch_times)}\n")
            if total_train_time is not None:
                f.write(
                    f"Total train time  : {_format_duration(total_train_time)} "
                    f"({total_train_time:.2f} s)\n"
                )
            f.write(f"Average epoch time: {avg_epoch:.2f} s\n")
            if epoch_times:
                f.write(f"Min epoch time    : {min(epoch_times):.2f} s\n")
                f.write(f"Max epoch time    : {max(epoch_times):.2f} s\n")

    fig, ax = plt.subplots(figsize=(10, 6))
    epochs = loss_df["epoch"].values
    train_arr = loss_df["train_loss"].values

    # Plot training loss
    ax.plot(epochs, train_arr, 'b-', linewidth=2, label='Training Loss')

    # Plot validation loss if available
    if valid_losses:
        valid_arr = loss_df["valid_loss"].values
        ax.plot(epochs, valid_arr, 'r-', linewidth=2, label='Validation Loss')

    # Annotate key points with epoch numbers
    # Training: First point
    ax.annotate(f'Epoch {epochs[0]}: {train_arr[0]:.4f}',
                xy=(epochs[0], train_arr[0]),
                xytext=(epochs[0] + 3, train_arr[0] - 0.03),
                fontsize=9, color='blue')

    # Training: Last point
    ax.annotate(f'Epoch {epochs[-1]}: {train_arr[-1]:.4f}',
                xy=(epochs[-1], train_arr[-1]),
                xytext=(epochs[-1] - 15, train_arr[-1] + 0.04),
                fontsize=9, color='blue')

    if valid_losses:
        # Validation: Minimum point (best model)
        min_idx = np.argmin(valid_arr)
        ax.annotate(f'Best: Epoch {epochs[min_idx]}, {valid_arr[min_idx]:.4f}',
                    xy=(epochs[min_idx], valid_arr[min_idx]),
                    xytext=(epochs[min_idx] - 10, valid_arr[min_idx] + 0.05),
                    fontsize=9, color='green', fontweight='bold',
                    arrowprops=dict(arrowstyle='->', color='green', lw=1.5))
        ax.scatter([epochs[min_idx]], [valid_arr[min_idx]], color='green', s=60, zorder=5,
                   label=f'Best Valid (Epoch {epochs[min_idx]})')

    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("Loss", fontsize=11)
    ax.set_title("TriModel (Degree-Matched Negatives) Training & Validation Loss", fontsize=13, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "training_loss.png"), dpi=150)
    plt.close()

    # Mappings
    pd.DataFrame({"entity": list(entity2id.keys()), "id": list(entity2id.values())}).to_csv(
        os.path.join(output_dir, "entity2id.csv"), index=False
    )
    pd.DataFrame({"relation": list(relation2id.keys()), "id": list(relation2id.values())}).to_csv(
        os.path.join(output_dir, "relation2id.csv"), index=False
    )

    # Embeddings (concatenate all 3 components)
    id2entity = {v: k for k, v in entity2id.items()}
    id2relation = {v: k for k, v in relation2id.items()}

    ent_v1 = model.ent_v1.weight.detach().cpu().numpy()
    ent_v2 = model.ent_v2.weight.detach().cpu().numpy()
    ent_v3 = model.ent_v3.weight.detach().cpu().numpy()
    ent_emb = np.concatenate([ent_v1, ent_v2, ent_v3], axis=1)  # (num_entities, 3*dim)

    rel_v1 = model.rel_v1.weight.detach().cpu().numpy()
    rel_v2 = model.rel_v2.weight.detach().cpu().numpy()
    rel_v3 = model.rel_v3.weight.detach().cpu().numpy()
    rel_emb = np.concatenate([rel_v1, rel_v2, rel_v3], axis=1)  # (num_relations, 3*dim)

    ent_df = pd.DataFrame(ent_emb, index=[id2entity[i] for i in range(ent_emb.shape[0])])
    rel_df = pd.DataFrame(rel_emb, index=[id2relation[i] for i in range(rel_emb.shape[0])])

    ent_df.to_csv(os.path.join(output_dir, "entity_embeddings.csv"))
    rel_df.to_csv(os.path.join(output_dir, "relation_embeddings.csv"))

    print(f"\nSaved to: {output_dir}/")
    print(" - training_loss.csv")
    print(" - training_loss.png")
    print(" - entity_embeddings.csv")
    print(" - relation_embeddings.csv")
    print(" - entity2id.csv")
    print(" - relation2id.csv")


def save_model(model, entity2id, relation2id, output_dir="outputs"):
    os.makedirs(output_dir, exist_ok=True)

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "num_entities": model.num_entities,
        "num_relations": model.num_relations,
        "embedding_dim": model.dim,
        "reg_weight": model.reg_weight,
        "entity2id": entity2id,
        "relation2id": relation2id,
        "model_type": "TriModel",
    }

    torch.save(checkpoint, os.path.join(output_dir, "trimodel_model.pt"))
    print(f"Model saved to {output_dir}/trimodel_model.pt")


# ----------------------------
# Main
# ----------------------------
if __name__ == "__main__":
    set_seed(42)

    embedding_dims = [100, 200, 300]

    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR = os.path.join(PROJECT_ROOT, "data", "trimodel")
    OUTPUT_ROOT = os.path.join(PROJECT_ROOT, "outputs_trimodel_degree_matched")

    # Set up logging: tee stdout/stderr to a run log file alongside the outputs
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    LOG_PATH = os.path.join(
        PROJECT_ROOT,
        f"outputs_trimodel_degree_matched_run.log",
    )
    _log_file = open(LOG_PATH, "w", encoding="utf-8")
    sys.stdout = TeeLogger(sys.__stdout__, _log_file)
    sys.stderr = TeeLogger(sys.__stderr__, _log_file)
    print(f"Logging run output to: {LOG_PATH}")
    print(f"Run started at: {datetime.now().isoformat(timespec='seconds')}")
    overall_start = time.perf_counter()

    # Reuse the existing splits produced by TriModel_Torch.py for a fair comparison
    train_path = os.path.join(DATA_DIR, "train.txt")
    valid_path = os.path.join(DATA_DIR, "valid.txt")
    test_path = os.path.join(DATA_DIR, "test.txt")
    for p in (train_path, valid_path, test_path):
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"Expected split file not found: {p}. Run TriModel_Torch.py first to generate splits."
            )

    train_df = load_triples(train_path)
    valid_df = load_triples(valid_path)
    test_df = load_triples(test_path)

    print(f"Loaded existing splits from: {DATA_DIR}")
    print(f"Train triples: {len(train_df)}")
    print(f"Valid triples: {len(valid_df)}")
    print(f"Test triples:  {len(test_df)}")

    entity2id, relation2id = make_mappings(train_df)
    train_triples = encode_triples(train_df, entity2id, relation2id)

    # Encode validation triples (filter to known entities/relations)
    valid_df_filtered = valid_df[
        (valid_df["source"].isin(entity2id.keys())) &
        (valid_df["relation"].isin(relation2id.keys())) &
        (valid_df["target"].isin(entity2id.keys()))
    ].reset_index(drop=True)
    valid_triples = encode_triples(valid_df_filtered, entity2id, relation2id)
    print(f"Validation triples (filtered): {len(valid_triples)}")

    device = "cpu"  # change to "cuda" if you have GPU
    for dim in embedding_dims:
        set_seed(42)
        output_dir = os.path.join(OUTPUT_ROOT, f"dim_{dim}")
        print(f"\n{'=' * 60}")
        print(f"Training TriModel (degree-matched negatives) with embedding dimension {dim}")
        print(f"Outputs will be saved to: {output_dir}")
        print(f"{'=' * 60}")

        model, train_losses, valid_losses, epoch_times, total_train_time = train_trimodel(
            train_triples=train_triples,
            num_entities=len(entity2id),
            num_relations=len(relation2id),
            valid_triples=valid_triples,
            dim=dim,
            reg_weight=0.01,
            lr=1e-3,
            batch_size=1024,
            epochs=100,
            device=device,
            early_stopping_patience=10,
            negative_samples=5,
            loss_type="bce",  # Options: 'pairwise_logistic', 'bce', 'pairwise_hinge', etc.
        )

        save_outputs(
            output_dir, train_losses, valid_losses, entity2id, relation2id, model,
            epoch_times=epoch_times, total_train_time=total_train_time,
        )
        save_model(model, entity2id, relation2id, output_dir=output_dir)

    overall_elapsed = time.perf_counter() - overall_start
    print(
        f"\nAll dimensions complete. Wall-clock time: "
        f"{_format_duration(overall_elapsed)} ({overall_elapsed:.2f}s)"
    )
    print(f"Run finished at: {datetime.now().isoformat(timespec='seconds')}")
    _log_file.close()
