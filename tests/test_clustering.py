import pytest
import numpy as np
from rec_sim.baseline.loader import load_kuairec, load_kuairec_items
from rec_sim.baseline.clustering import extract_user_features, cluster_users


@pytest.fixture(scope="module")
def interactions():
    return load_kuairec()


@pytest.fixture(scope="module")
def items():
    return load_kuairec_items()


def test_extract_user_features_shape(interactions, items):
    features, feature_names = extract_user_features(interactions, items)
    n_users = interactions["user_id"].nunique()
    assert features.shape[0] == n_users
    assert features.shape[1] == len(feature_names)
    assert features.shape[1] >= 5


def test_cluster_users_returns_labels(interactions, items):
    features, feature_names = extract_user_features(interactions, items)
    labels, centers = cluster_users(features, n_clusters=20)
    assert len(labels) == features.shape[0]
    assert len(set(labels)) == 20
    assert centers.shape == (20, features.shape[1])


def test_cluster_users_all_assigned(interactions, items):
    features, _ = extract_user_features(interactions, items)
    labels, _ = cluster_users(features, n_clusters=10)
    assert all(0 <= l < 10 for l in labels)
