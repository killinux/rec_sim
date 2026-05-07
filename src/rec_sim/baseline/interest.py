"""Build user interest profiles and compute interest matching scores."""
from __future__ import annotations
import numpy as np
import pandas as pd
from ast import literal_eval


def build_category_map(items: pd.DataFrame) -> dict[int, list[int]]:
    """Map video_id -> list of category IDs."""
    result = {}
    for _, row in items.iterrows():
        vid = int(row["video_id"])
        feat = row["feat"]
        if isinstance(feat, str):
            feat = literal_eval(feat)
        result[vid] = list(feat) if isinstance(feat, (list, tuple)) else [int(feat)]
    return result


def get_all_categories(cat_map: dict[int, list[int]]) -> list[int]:
    """Get sorted list of all unique category IDs."""
    all_cats = set()
    for cats in cat_map.values():
        all_cats.update(cats)
    return sorted(all_cats)


def build_user_interest_vectors(
    interactions: pd.DataFrame,
    cat_map: dict[int, list[int]],
    all_cats: list[int],
) -> dict[int, np.ndarray]:
    """Build per-user interest vector weighted by watch_ratio.

    Each user gets a vector of length len(all_cats), where entry i is the
    total watch_ratio-weighted exposure to category all_cats[i], normalized to sum=1.
    """
    cat_index = {c: i for i, c in enumerate(all_cats)}
    n_cats = len(all_cats)
    user_vectors = {}

    for user_id, group in interactions.groupby("user_id"):
        vec = np.zeros(n_cats)
        for _, row in group.iterrows():
            item_id = int(row["item_id"])
            wr = float(row["watch_ratio"])
            cats = cat_map.get(item_id, [])
            for c in cats:
                idx = cat_index.get(c)
                if idx is not None:
                    vec[idx] += wr
        total = vec.sum()
        if total > 0:
            vec = vec / total
        user_vectors[int(user_id)] = vec

    return user_vectors


def build_archetype_interest_vectors(
    user_vectors: dict[int, np.ndarray],
    user_ids: np.ndarray,
    labels: np.ndarray,
) -> dict[int, np.ndarray]:
    """Average interest vectors per archetype cluster."""
    archetype_vectors = {}
    for arch_id in sorted(set(labels)):
        mask = labels == arch_id
        arch_user_ids = user_ids[mask]
        vecs = [user_vectors[uid] for uid in arch_user_ids if uid in user_vectors]
        if vecs:
            archetype_vectors[int(arch_id)] = np.mean(vecs, axis=0)
    return archetype_vectors


def build_item_vector(item_id: int, cat_map: dict[int, list[int]], all_cats: list[int]) -> np.ndarray:
    """Build a one-hot-ish category vector for a single item."""
    cat_index = {c: i for i, c in enumerate(all_cats)}
    vec = np.zeros(len(all_cats))
    for c in cat_map.get(item_id, []):
        idx = cat_index.get(c)
        if idx is not None:
            vec[idx] = 1.0
    total = vec.sum()
    if total > 0:
        vec = vec / total
    return vec


def compute_interest_match(user_vec: np.ndarray, item_vec: np.ndarray) -> float:
    """Cosine similarity between user interest vector and item category vector."""
    dot = np.dot(user_vec, item_vec)
    norm_u = np.linalg.norm(user_vec)
    norm_i = np.linalg.norm(item_vec)
    if norm_u < 1e-10 or norm_i < 1e-10:
        return 0.0
    return float(np.clip(dot / (norm_u * norm_i), 0.0, 1.0))
