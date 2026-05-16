"""
ComplEx: Complex Embeddings for Simple Link Prediction
=======================================================
Implementation of ComplEx model for knowledge graph embedding.

ComplEx uses complex-valued embeddings to model both symmetric and 
antisymmetric relations, which TransE struggles with.

Reference:
    Trouillon et al. "Complex Embeddings for Simple Link Prediction" (ICML 2016)
"""

import os
import sys

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

from utils.model import ComplEx  # Shared model definition
from utils.negative_sampling import (
    build_true_triples_set,
    compute_bern_probs,
    corrupt_triples,
)


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
# Negative sampling: see utils.negative_sampling (vectorized, supports bern).
# ----------------------------


# ----------------------------
# Training loop
# ----------------------------
@torch.no_grad()
@torch.no_grad()
def compute_validation_loss(
    model: ComplEx,
    valid_triples: torch.Tensor,
    num_entities: int,
    true_triples: set,
    device: str,
    neg_mode: str = "unif",
    bern_probs: torch.Tensor = None,
) -> float:
    """Compute average loss on validation set."""
    model.eval()
    valid_triples = valid_triples.to(device)
    neg = corrupt_triples(
        valid_triples, num_entities, true_triples,
        mode=neg_mode, bern_probs=bern_probs,
    ).to(device)
    
    pos_scores = model(valid_triples)
    neg_scores = model(neg)
    
    # Binary cross-entropy with logits
    pos_labels = torch.ones_like(pos_scores)
    neg_labels = torch.zeros_like(neg_scores)
    
    scores = torch.cat([pos_scores, neg_scores])
    labels = torch.cat([pos_labels, neg_labels])
    
    loss = F.binary_cross_entropy_with_logits(scores, labels)
    return loss.item()


def train_complex(
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
    neg_mode: str = "unif",
):
    """
    Train ComplEx model.
    
    Args:
        train_triples: Training triples tensor
        num_entities: Number of unique entities
        num_relations: Number of unique relations
        valid_triples: Validation triples for early stopping
        dim: Embedding dimension (for each of real/imaginary parts)
        reg_weight: L2 regularization weight
        lr: Learning rate
        batch_size: Training batch size
        epochs: Maximum epochs
        device: 'cpu' or 'cuda'
        filter_negatives: Whether to filter false negatives
        early_stopping_patience: Epochs without improvement before stopping
        lr_scheduler_patience: Epochs before reducing LR
        lr_scheduler_factor: Factor to reduce LR by
        negative_samples: Number of negative samples per positive
        
    Returns:
        Trained model and loss history
    """
    model = ComplEx(num_entities, num_relations, dim=dim, reg_weight=reg_weight).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        patience=lr_scheduler_patience, 
        factor=lr_scheduler_factor
    )

    # Build set of true triples for filtering
    true_triples = build_true_triples_set(train_triples) if filter_negatives else None

    bern_probs = (
        compute_bern_probs(train_triples, num_relations)
        if neg_mode == "bern" else None
    )
    
    # Early stopping
    best_valid_loss = float('inf')
    best_model_state = None
    patience_counter = 0
    use_early_stopping = valid_triples is not None and early_stopping_patience > 0
    
    losses_per_epoch = []
    valid_losses_per_epoch = []

    train_triples = train_triples.to(device)
    n = train_triples.size(0)

    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(n, device=device)
        epoch_loss = 0.0
        num_batches = 0

        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            pos = train_triples[idx]
            
            # Generate multiple negative samples
            neg_list = []
            for _ in range(negative_samples):
                neg_list.append(
                    corrupt_triples(
                        pos, num_entities, true_triples,
                        mode=neg_mode, bern_probs=bern_probs,
                    ).to(device)
                )
            neg = torch.cat(neg_list, dim=0)

            # Compute scores
            pos_scores = model(pos)
            neg_scores = model(neg)
            
            # Repeat positive labels for multiple negatives
            pos_labels = torch.ones_like(pos_scores)
            neg_labels = torch.zeros_like(neg_scores)
            
            scores = torch.cat([pos_scores, neg_scores])
            labels = torch.cat([pos_labels, neg_labels])
            
            # Binary cross-entropy loss
            bce_loss = F.binary_cross_entropy_with_logits(scores, labels)
            
            # Regularization
            reg_loss = model.regularization(pos)
            
            loss = bce_loss + reg_loss

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
                model, valid_triples, num_entities, true_triples, device,
                neg_mode=neg_mode, bern_probs=bern_probs,
            )
            valid_losses_per_epoch.append(valid_loss)
            
            scheduler.step(valid_loss)
            current_lr = optimizer.param_groups[0]['lr']
            
            print(f"Epoch {epoch:03d}/{epochs} | train_loss={avg_loss:.6f} | valid_loss={valid_loss:.6f} | lr={current_lr:.2e}")
            
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
            print(f"Epoch {epoch:03d}/{epochs} | loss={avg_loss:.6f}")

    # Restore best model
    if use_early_stopping and best_model_state is not None:
        model.load_state_dict(best_model_state)
        print("Restored best model from early stopping.")

    return model, losses_per_epoch, valid_losses_per_epoch


# ----------------------------
# Save outputs
# ----------------------------
def save_outputs(output_dir, train_losses, valid_losses, entity2id, relation2id, model: ComplEx):
    os.makedirs(output_dir, exist_ok=True)

    # Loss CSV + plot
    loss_data = {
        "epoch": np.arange(1, len(train_losses) + 1), 
        "train_loss": train_losses
    }
    if valid_losses:
        loss_data["valid_loss"] = valid_losses
    loss_df = pd.DataFrame(loss_data)
    loss_df.to_csv(os.path.join(output_dir, "training_loss.csv"), index=False)

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
    ax.set_title("ComplEx Training & Validation Loss", fontsize=13, fontweight='bold')
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

    # Embeddings (concatenate real and imaginary parts)
    id2entity = {v: k for k, v in entity2id.items()}
    id2relation = {v: k for k, v in relation2id.items()}

    ent_re = model.ent_re.weight.detach().cpu().numpy()
    ent_im = model.ent_im.weight.detach().cpu().numpy()
    ent_emb = np.concatenate([ent_re, ent_im], axis=1)  # (num_entities, 2*dim)
    
    rel_re = model.rel_re.weight.detach().cpu().numpy()
    rel_im = model.rel_im.weight.detach().cpu().numpy()
    rel_emb = np.concatenate([rel_re, rel_im], axis=1)  # (num_relations, 2*dim)

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
        "model_type": "ComplEx",
    }

    torch.save(checkpoint, os.path.join(output_dir, "complex_model.pt"))
    print("Model saved to outputs/complex_model.pt")


# ----------------------------
# Main
# ----------------------------
if __name__ == "__main__":
    set_seed(42)

    embedding_dims = [100, 200, 300]

    DATA_PATH = "drugbank_facts.txt"
    DATA_DIR = "data/complex"
    OUTPUT_ROOT = "outputs_complex"  # Separate output directory

    df = load_triples(DATA_PATH)
    print("Dataset preview:")
    print(df.head())
    print(f"\nTotal triples: {len(df)}")

    # 3-way split: 70% train, 10% validation, 20% test
    train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42)
    valid_df, test_df = train_test_split(temp_df, test_size=0.67, random_state=42)
    
    print(f"Train triples: {len(train_df)}")
    print(f"Valid triples: {len(valid_df)}")
    print(f"Test triples:  {len(test_df)}")
    
    # Save splits (uses same data dir for consistency with TransE)
    save_splits(train_df, test_df, valid_df, data_dir=DATA_DIR)

    entity2id, relation2id = make_mappings(train_df)
    train_triples = encode_triples(train_df, entity2id, relation2id)
    
    # Encode validation triples
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
        print(f"Training ComplEx with embedding dimension {dim}")
        print(f"Outputs will be saved to: {output_dir}")
        print(f"{'=' * 60}")

        model, train_losses, valid_losses = train_complex(
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
            negative_samples=5,  # More negatives can help ComplEx
        )

        save_outputs(output_dir, train_losses, valid_losses, entity2id, relation2id, model)
        save_model(model, entity2id, relation2id, output_dir=output_dir)
