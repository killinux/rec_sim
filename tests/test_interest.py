from __future__ import annotations
import pytest
import numpy as np
import pandas as pd
from rec_sim.baseline.interest import (
    build_category_map, get_all_categories, build_user_interest_vectors,
    build_item_vector, compute_interest_match,
)


def test_build_category_map():
    items = pd.DataFrame({"video_id": [0, 1, 2], "feat": ["[8]", "[27, 9]", "[8, 27]"]})
    cat_map = build_category_map(items)
    assert cat_map[0] == [8]
    assert cat_map[1] == [27, 9]
    assert cat_map[2] == [8, 27]


def test_get_all_categories():
    cat_map = {0: [8], 1: [27, 9], 2: [8, 27]}
    all_cats = get_all_categories(cat_map)
    assert all_cats == [8, 9, 27]


def test_build_user_interest_vectors():
    interactions = pd.DataFrame({
        "user_id": [1, 1, 1, 2, 2],
        "item_id": [0, 1, 2, 0, 1],
        "watch_ratio": [1.0, 0.5, 0.8, 0.3, 1.0],
    })
    cat_map = {0: [8], 1: [27, 9], 2: [8, 27]}
    all_cats = [8, 9, 27]
    vecs = build_user_interest_vectors(interactions, cat_map, all_cats)
    assert len(vecs) == 2
    assert abs(sum(vecs[1]) - 1.0) < 1e-6
    assert abs(sum(vecs[2]) - 1.0) < 1e-6


def test_compute_interest_match_identical():
    vec = np.array([0.5, 0.3, 0.2])
    assert compute_interest_match(vec, vec) == pytest.approx(1.0, abs=0.01)


def test_compute_interest_match_orthogonal():
    u = np.array([1.0, 0.0, 0.0])
    v = np.array([0.0, 1.0, 0.0])
    assert compute_interest_match(u, v) == pytest.approx(0.0, abs=0.01)


def test_compute_interest_match_partial():
    u = np.array([0.7, 0.3, 0.0])
    v = np.array([0.5, 0.5, 0.0])
    score = compute_interest_match(u, v)
    assert 0 < score < 1
