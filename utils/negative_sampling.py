"""
Negative sampling for KG embeddings
====================================
Shared, vectorized negative sampler used by TransE / ComplEx / TriModel
trainers (uniform-random variants).

Strategies:
    - "unif": flip head/tail with probability 0.5  (Bordes et al., 2013)
    - "bern": flip head/tail with relation-conditional probability
              p(corrupt head | r) = tph[r] / (tph[r] + hpt[r])
              (Wang et al., 2014 -- the "bernoulli" trick).
              Reduces false negatives for 1-N / N-1 relations.

Differences vs the previous per-trainer implementation:
    - Fully vectorized (one tensor of random entities per batch, not a
      Python for-loop over the batch).
    - When filtering is enabled the resampling loop only re-draws the
      colliding rows; no silent fallback to a known-true triple.
    - Optional "bern" mode for cardinality-aware corruption.
"""

from __future__ import annotations

import warnings
from typing import Optional

import torch


# ---------------------------------------------------------------------------
# True-triple set (shared helper)
# ---------------------------------------------------------------------------
def build_true_triples_set(triples: torch.Tensor) -> set:
    """Convert a (N, 3) tensor of triples to a set of (h, r, t) tuples."""
    return set(map(tuple, triples.detach().cpu().numpy().tolist()))


# ---------------------------------------------------------------------------
# Bernoulli head/tail-corruption probabilities
# ---------------------------------------------------------------------------
def compute_bern_probs(
    train_triples: torch.Tensor,
    num_relations: int,
) -> torch.Tensor:
    """
    Compute per-relation P(corrupt head) for the bernoulli sampling trick.

    Returns a 1-D float tensor of length ``num_relations``. Relations not
    seen in ``train_triples`` default to 0.5.
    """
    triples = train_triples.detach().cpu().numpy()
    probs = torch.full((num_relations,), 0.5, dtype=torch.float32)

    # For each relation r:
    #   tph ≈ #triples(r) / #distinct_heads(r)   (tails per head)
    #   hpt ≈ #triples(r) / #distinct_tails(r)   (heads per tail)
    #   p(corrupt head) = tph / (tph + hpt)
    import numpy as np

    rels = triples[:, 1]
    for r in np.unique(rels):
        mask = rels == r
        n = int(mask.sum())
        n_heads = int(np.unique(triples[mask, 0]).size)
        n_tails = int(np.unique(triples[mask, 2]).size)
        if n_heads == 0 or n_tails == 0:
            continue
        tph = n / n_heads
        hpt = n / n_tails
        denom = tph + hpt
        if denom > 0:
            probs[int(r)] = float(tph / denom)
    return probs


# ---------------------------------------------------------------------------
# Vectorized corruption
# ---------------------------------------------------------------------------
def corrupt_triples(
    pos_triples: torch.Tensor,
    num_entities: int,
    true_triples: Optional[set] = None,
    max_attempts: int = 10,
    *,
    mode: str = "unif",
    bern_probs: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Generate one negative triple per positive by corrupting head OR tail.

    Args:
        pos_triples: (B, 3) tensor of positive triples (long).
        num_entities: total entity count -- random entity id range [0, N).
        true_triples: optional set of (h, r, t) tuples to avoid. If None,
            no filtering is performed.
        max_attempts: maximum resample rounds for colliding rows.
        mode: "unif" (default, p=0.5) or "bern" (uses ``bern_probs``).
        bern_probs: required when ``mode="bern"``; tensor of shape
            (num_relations,) giving P(corrupt head | r).

    Returns:
        (B, 3) tensor of corrupted (negative) triples on CPU.
    """
    if mode not in ("unif", "bern"):
        raise ValueError(f"Unknown sampling mode: {mode!r}")
    if mode == "bern" and bern_probs is None:
        raise ValueError("mode='bern' requires bern_probs to be provided")

    neg = pos_triples.detach().cpu().clone()
    B = neg.size(0)
    if B == 0:
        return neg

    # 1. Decide head-vs-tail corruption per row.
    if mode == "unif":
        replace_head = torch.rand(B) < 0.5
    else:
        rels = neg[:, 1].long()
        p_head = bern_probs.detach().cpu()[rels]
        replace_head = torch.rand(B) < p_head

    # 2. Sample random entity ids in one shot.
    rand_ents = torch.randint(low=0, high=num_entities, size=(B,))

    # 3. Place them at column 0 or 2.
    cols = torch.where(replace_head, torch.zeros(B, dtype=torch.long),
                       torch.full((B,), 2, dtype=torch.long))
    rows = torch.arange(B)
    neg[rows, cols] = rand_ents

    # 4. Resample only colliding rows (against true_triples or self-identity).
    if true_triples is not None:
        for _ in range(max_attempts):
            # Self-identity collision (random entity == original head/tail).
            orig_ents = pos_triples.detach().cpu()[rows, cols]
            self_collide = neg[rows, cols] == orig_ents

            # Membership collision against known true triples.
            triple_list = neg.numpy().tolist()
            true_collide = torch.tensor(
                [tuple(t) in true_triples for t in triple_list],
                dtype=torch.bool,
            )

            collide = self_collide | true_collide
            n_bad = int(collide.sum())
            if n_bad == 0:
                break
            new_ents = torch.randint(low=0, high=num_entities, size=(n_bad,))
            bad_rows = rows[collide]
            bad_cols = cols[collide]
            neg[bad_rows, bad_cols] = new_ents
        else:
            # Any remaining collisions are extremely rare; warn rather than
            # silently emit a known-true triple as a "negative".
            n_bad = int(collide.sum())
            if n_bad:
                warnings.warn(
                    f"corrupt_triples: {n_bad}/{B} negatives still collide "
                    f"after {max_attempts} attempts; keeping them.",
                    RuntimeWarning,
                    stacklevel=2,
                )

    return neg
