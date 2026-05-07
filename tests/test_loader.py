import pytest
import pandas as pd
from rec_sim.baseline.loader import load_kuairec


def test_load_kuairec_returns_interactions():
    df = load_kuairec()
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0
    required_cols = ["user_id", "item_id", "watch_ratio", "duration_ms", "video_duration_ms"]
    for col in required_cols:
        assert col in df.columns, f"Missing column: {col}"


def test_load_kuairec_watch_ratio_range():
    df = load_kuairec()
    assert df["watch_ratio"].min() >= 0
    assert df["watch_ratio"].median() > 0


def test_load_kuairec_no_null_ids():
    df = load_kuairec()
    assert df["user_id"].notna().all()
    assert df["item_id"].notna().all()
