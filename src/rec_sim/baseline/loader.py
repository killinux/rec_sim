"""Load KuaiRec datasets into a unified interaction DataFrame."""
import pandas as pd
from rec_sim.config import KUAIREC_DIR


def load_kuairec(use_small=True) -> pd.DataFrame:
    fname = "small_matrix.csv" if use_small else "big_matrix.csv"
    path = KUAIREC_DIR / fname
    raw = pd.read_csv(path)
    df = pd.DataFrame({
        "user_id": raw["user_id"],
        "item_id": raw["video_id"],
        "watch_ratio": raw["watch_ratio"].clip(lower=0),
        "duration_ms": (raw["play_duration"] * 1000).astype(int),
        "video_duration_ms": (raw["video_duration"] * 1000).astype(int),
    })
    return df


def load_kuairec_users() -> pd.DataFrame:
    path = KUAIREC_DIR / "user_features.csv"
    return pd.read_csv(path)


def load_kuairec_items() -> pd.DataFrame:
    path = KUAIREC_DIR / "item_categories.csv"
    return pd.read_csv(path)
