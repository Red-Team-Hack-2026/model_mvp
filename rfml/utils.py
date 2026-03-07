from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch


def seed_everything(seed: int = 7):
    import random, os
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def stratified_split(indices: List[int], labels: List[int], val_frac: float = 0.15, seed: int = 7):
    """Simple stratified split by label."""
    rng = np.random.default_rng(seed)
    labels = np.asarray(labels)
    indices = np.asarray(indices)
    train_idx = []
    val_idx = []
    for y in np.unique(labels):
        idx_y = indices[labels == y]
        rng.shuffle(idx_y)
        n_val = max(1, int(round(len(idx_y) * val_frac)))
        val_idx.extend(idx_y[:n_val].tolist())
        train_idx.extend(idx_y[n_val:].tolist())
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    return train_idx, val_idx


@torch.no_grad()
def accuracy(logits: torch.Tensor, y: torch.Tensor) -> float:
    pred = torch.argmax(logits, dim=-1)
    return (pred == y).float().mean().item()


@torch.no_grad()
def compute_class_centroids(embeddings: np.ndarray, labels: np.ndarray) -> Dict[int, np.ndarray]:
    """Compute L2-normalized centroid per class in embedding space."""
    centroids = {}
    for c in np.unique(labels):
        e = embeddings[labels == c]
        mu = e.mean(axis=0)
        mu = mu / (np.linalg.norm(mu) + 1e-12)
        centroids[int(c)] = mu.astype(np.float32)
    return centroids


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / ((np.linalg.norm(a) + 1e-12) * (np.linalg.norm(b) + 1e-12)))
