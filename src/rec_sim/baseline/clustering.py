"""Extract user features and cluster into archetypes."""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


def extract_user_features(interactions: pd.DataFrame, items: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    user_stats = interactions.groupby("user_id").agg(
        avg_watch_ratio=("watch_ratio", "mean"),
        n_interactions=("item_id", "count"),
        watch_ratio_std=("watch_ratio", "std"),
        avg_duration_ms=("duration_ms", "mean"),
        unique_items=("item_id", "nunique"),
    ).fillna(0)

    if "feat" in items.columns:
        item_cats = items.set_index("video_id")["feat"]
        top_cats = _build_category_features(interactions, item_cats, top_n=10)
        user_stats = user_stats.join(top_cats, how="left").fillna(0)

    feature_names = list(user_stats.columns)
    scaler = StandardScaler()
    features = scaler.fit_transform(user_stats.values)
    return features, feature_names


def _build_category_features(interactions: pd.DataFrame, item_cats: pd.Series, top_n: int) -> pd.DataFrame:
    merged = interactions[["user_id", "item_id"]].copy()
    merged["cats"] = merged["item_id"].map(item_cats)
    merged = merged.dropna(subset=["cats"])

    all_cats = merged["cats"].explode()
    top = all_cats.value_counts().head(top_n).index.tolist()

    rows = []
    for user_id, group in merged.groupby("user_id"):
        user_cats = group["cats"].explode()
        total = len(user_cats)
        cat_ratios = {f"cat_{c}": (user_cats == c).sum() / max(total, 1) for c in top}
        cat_ratios["user_id"] = user_id
        rows.append(cat_ratios)

    return pd.DataFrame(rows).set_index("user_id")


def cluster_users(features: np.ndarray, n_clusters: int = 50, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    km = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
    labels = km.fit_predict(features)
    return labels, km.cluster_centers_
