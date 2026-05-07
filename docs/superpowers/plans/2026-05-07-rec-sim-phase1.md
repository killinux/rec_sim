# RecSim Phase 1: End-to-End Minimum Viable Loop

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the minimum end-to-end simulation loop: load KuaiRec data, extract user archetypes, generate persona skeletons via LHS, run Layer 0+1 parametric decisions, compute fidelity metrics against real distributions.

**Architecture:** Python package `rec_sim` with 5 modules mapping to design spec layers. KuaiRec small_matrix as primary data source (fully-observed, ~1411 users x 3327 items). No LLM calls in Phase 1 — pure parametric simulation to establish the statistical baseline.

**Tech Stack:** Python 3.11+, pandas, numpy, scipy, scikit-learn, pytest. No external API dependencies in Phase 1.

**Dev workflow:** Code written at `C:\Users\Administrator\Desktop\macwork\rec_sim\`, synced to Mac at `/Users/bytedance/Desktop/hehe/research/rec_sim/`, run and tested on Mac via HTTP remote (`localhost:8900`).

**Data paths (Mac):**
- KuaiRec: `/Users/bytedance/Desktop/hehe/datasets/KuaiRec/data/`
- MovieLens-25M: `/Users/bytedance/Desktop/hehe/datasets/ml-25m/`

---

## File Structure

```
rec_sim/
├── docs/                          # (existing)
├── src/
│   └── rec_sim/
│       ├── __init__.py
│       ├── config.py              # paths, constants, default params
│       ├── baseline/
│       │   ├── __init__.py
│       │   ├── loader.py          # load KuaiRec CSVs into unified schema
│       │   ├── clustering.py      # user feature extraction + archetype clustering
│       │   └── distribution.py    # extract & serialize joint distributions per archetype
│       ├── fidelity/
│       │   ├── __init__.py
│       │   └── metrics.py         # KL, Wasserstein, correlation, composite F
│       ├── persona/
│       │   ├── __init__.py
│       │   └── skeleton.py        # LHS placement + skeleton parameter sampling
│       ├── interaction/
│       │   ├── __init__.py
│       │   ├── infra.py           # infrastructure state model (quality, latency, stall)
│       │   ├── context.py         # session context model (session type, time, network)
│       │   ├── layer0.py          # experience decision: stall/quality -> skip/exit
│       │   ├── layer1.py          # content decision: interest matching -> watch/like/etc
│       │   └── engine.py          # orchestrates L0+L1 per video, manages agent state
│       └── runner.py              # simulation loop: N agents x M sessions
├── tests/
│   ├── __init__.py
│   ├── test_loader.py
│   ├── test_clustering.py
│   ├── test_distribution.py
│   ├── test_metrics.py
│   ├── test_skeleton.py
│   ├── test_infra.py
│   ├── test_layer0.py
│   ├── test_layer1.py
│   ├── test_engine.py
│   └── test_runner.py
├── sync.sh                        # rsync code to Mac
├── pyproject.toml
└── README.md                      # (not created — existing docs suffice)
```

---

## Task 0: Project Scaffold + Sync

**Files:**
- Create: `pyproject.toml`
- Create: `src/rec_sim/__init__.py`
- Create: `src/rec_sim/config.py`
- Create: `sync.sh`
- Create: `tests/__init__.py`

- [ ] **Step 1: Create pyproject.toml**

```toml
[project]
name = "rec-sim"
version = "0.1.0"
requires-python = ">=3.10"
dependencies = [
    "pandas>=2.0",
    "numpy>=1.24",
    "scipy>=1.10",
    "scikit-learn>=1.3",
]

[project.optional-dependencies]
dev = ["pytest>=7.0"]

[tool.pytest.ini_options]
testpaths = ["tests"]
```

- [ ] **Step 2: Create config.py**

```python
from pathlib import Path

DATA_ROOT = Path("/Users/bytedance/Desktop/hehe/datasets")
KUAIREC_DIR = DATA_ROOT / "KuaiRec" / "data"
MOVIELENS_DIR = DATA_ROOT / "ml-25m"

N_ARCHETYPES = 50
N_AGENTS = 1000
RANDOM_SEED = 42
```

- [ ] **Step 3: Create src/rec_sim/__init__.py**

```python
"""RecSim: Large-scale user simulation for recommendation systems."""
```

- [ ] **Step 4: Create sync.sh**

```bash
#!/bin/bash
# Sync rec_sim code to Mac for testing
# Usage: bash sync.sh <MAC_IP> [PORT]
#
# Since Mac connects via reverse tunnel to localhost:8900,
# this script pushes code by encoding it in commands.
# For larger syncs, use scp/rsync if direct SSH is available.

SRC_DIR="$(cd "$(dirname "$0")" && pwd)"
REMOTE_DIR="/Users/bytedance/Desktop/hehe/research/rec_sim"
SERVER="http://localhost:8900"

echo "Syncing $SRC_DIR -> Mac:$REMOTE_DIR via command server..."

# Create remote directory structure
curl -s -X POST "$SERVER/cmd" \
  -H "Content-Type: application/json" \
  -d "{\"cmd\": \"mkdir -p $REMOTE_DIR/src/rec_sim/baseline $REMOTE_DIR/src/rec_sim/fidelity $REMOTE_DIR/src/rec_sim/persona $REMOTE_DIR/src/rec_sim/interaction $REMOTE_DIR/tests\"}"

echo "Directory structure created. Use individual file push commands for each file."
echo "For bulk sync, use: scp -r src/ tests/ pyproject.toml <mac>:$REMOTE_DIR/"
```

- [ ] **Step 5: Create tests/__init__.py**

Empty file.

- [ ] **Step 6: Sync scaffold to Mac and verify**

Push the directory structure to Mac via remote command, then install the package in dev mode:

```
Mac command: cd /Users/bytedance/Desktop/hehe/research/rec_sim && pip install -e ".[dev]"
```

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml src/ tests/ sync.sh
git commit -m "scaffold: project structure and config"
```

---

## Task 1: KuaiRec Data Loader

**Files:**
- Create: `src/rec_sim/baseline/__init__.py`
- Create: `src/rec_sim/baseline/loader.py`
- Create: `tests/test_loader.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_loader.py
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/bytedance/Desktop/hehe/research/rec_sim && python -m pytest tests/test_loader.py -v`
Expected: FAIL with `ModuleNotFoundError` or `ImportError`

- [ ] **Step 3: Write loader implementation**

```python
# src/rec_sim/baseline/loader.py
"""Load KuaiRec datasets into a unified interaction DataFrame."""
import pandas as pd
from rec_sim.config import KUAIREC_DIR


def load_kuairec(use_small=True) -> pd.DataFrame:
    """Load KuaiRec interaction matrix.

    Args:
        use_small: If True, load small_matrix (1411 users, fully observed).
                   If False, load big_matrix (7176 users, sparse).
    """
    fname = "small_matrix.csv" if use_small else "big_matrix.csv"
    path = KUAIREC_DIR / fname
    raw = pd.read_csv(path)

    df = pd.DataFrame({
        "user_id": raw["user_id"],
        "item_id": raw["video_id"],
        "watch_ratio": raw["watch_ratio"].clip(lower=0),
        "duration_ms": (raw["play_duration"] * 1000).astype(int),
        "video_duration_ms": (raw["video_duration"] * 1000).astype(int),
        "timestamp": pd.to_datetime(raw["date"] if "date" in raw.columns else 0, errors="coerce"),
    })
    return df


def load_kuairec_users() -> pd.DataFrame:
    """Load KuaiRec user features."""
    path = KUAIREC_DIR / "user_features.csv"
    return pd.read_csv(path)


def load_kuairec_items() -> pd.DataFrame:
    """Load KuaiRec item categories."""
    path = KUAIREC_DIR / "item_categories.csv"
    return pd.read_csv(path)
```

- [ ] **Step 4: Create baseline/__init__.py**

```python
"""Baseline distribution extraction from real datasets."""
```

- [ ] **Step 5: Run test to verify it passes**

Run: `cd /Users/bytedance/Desktop/hehe/research/rec_sim && python -m pytest tests/test_loader.py -v`
Expected: PASS (3 tests)

- [ ] **Step 6: Commit**

```bash
git add src/rec_sim/baseline/ tests/test_loader.py
git commit -m "feat: KuaiRec data loader with unified schema"
```

---

## Task 2: User Clustering into Archetypes

**Files:**
- Create: `src/rec_sim/baseline/clustering.py`
- Create: `tests/test_clustering.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_clustering.py
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_clustering.py -v`
Expected: FAIL with ImportError

- [ ] **Step 3: Write clustering implementation**

```python
# src/rec_sim/baseline/clustering.py
"""Extract user features and cluster into archetypes."""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


def extract_user_features(interactions: pd.DataFrame, items: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """Build a per-user feature matrix from interaction data.

    Features extracted:
    - avg_watch_ratio: mean completion rate
    - n_interactions: total videos watched
    - watch_ratio_std: variance in completion (consistency)
    - avg_duration_ms: avg play duration
    - category_entropy: diversity of content consumed
    """
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
    """Build per-user category consumption ratios for the top-N categories."""
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
    """Cluster user feature matrix into archetypes using KMeans."""
    km = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
    labels = km.fit_predict(features)
    return labels, km.cluster_centers_
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_clustering.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add src/rec_sim/baseline/clustering.py tests/test_clustering.py
git commit -m "feat: user feature extraction and archetype clustering"
```

---

## Task 3: Distribution Extraction per Archetype

**Files:**
- Create: `src/rec_sim/baseline/distribution.py`
- Create: `tests/test_distribution.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_distribution.py
import pytest
import json
import numpy as np
from rec_sim.baseline.distribution import ArchetypeDistribution, extract_archetype_distributions


def test_archetype_distribution_from_data():
    watch_ratios = np.array([0.3, 0.5, 0.7, 0.4, 0.6, 0.8, 0.2, 0.9, 0.5, 0.6])
    durations = np.array([5000, 10000, 15000, 8000, 12000, 20000, 3000, 25000, 11000, 14000])
    dist = ArchetypeDistribution.from_data(
        archetype_id=0,
        watch_ratios=watch_ratios,
        durations=durations,
        n_users=10,
    )
    assert dist.archetype_id == 0
    assert dist.n_users == 10
    assert 0 < dist.watch_ratio_mean < 1
    assert dist.duration_mean > 0


def test_archetype_distribution_sample():
    watch_ratios = np.random.beta(2, 3, size=100)
    durations = np.random.lognormal(9, 0.5, size=100)
    dist = ArchetypeDistribution.from_data(0, watch_ratios, durations, 100)

    samples = dist.sample_watch_ratios(50, seed=42)
    assert len(samples) == 50
    assert all(0 <= s <= 1 for s in samples)


def test_archetype_distribution_serializable():
    watch_ratios = np.random.beta(2, 3, size=50)
    durations = np.random.lognormal(9, 0.5, size=50)
    dist = ArchetypeDistribution.from_data(0, watch_ratios, durations, 50)

    d = dist.to_dict()
    text = json.dumps(d)
    restored = ArchetypeDistribution.from_dict(json.loads(text))
    assert abs(restored.watch_ratio_mean - dist.watch_ratio_mean) < 1e-6
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_distribution.py -v`
Expected: FAIL with ImportError

- [ ] **Step 3: Write distribution implementation**

```python
# src/rec_sim/baseline/distribution.py
"""Per-archetype distribution extraction and serialization."""
from dataclasses import dataclass, asdict
import numpy as np
from scipy import stats


@dataclass
class ArchetypeDistribution:
    archetype_id: int
    n_users: int
    proportion: float
    watch_ratio_mean: float
    watch_ratio_std: float
    watch_ratio_beta_a: float
    watch_ratio_beta_b: float
    duration_mean: float
    duration_std: float
    duration_log_mu: float
    duration_log_sigma: float

    @classmethod
    def from_data(cls, archetype_id: int, watch_ratios: np.ndarray,
                  durations: np.ndarray, n_users: int, total_users: int = 0):
        wr = np.clip(watch_ratios, 1e-6, 1 - 1e-6)
        wr_mean, wr_std = float(wr.mean()), float(wr.std()) or 0.01

        a, b, _, _ = stats.beta.fit(wr, floc=0, fscale=1)

        dur = durations[durations > 0]
        d_mean, d_std = float(dur.mean()), float(dur.std()) or 1.0
        log_dur = np.log(dur + 1)
        log_mu, log_sigma = float(log_dur.mean()), float(log_dur.std()) or 0.1

        return cls(
            archetype_id=archetype_id,
            n_users=n_users,
            proportion=n_users / total_users if total_users > 0 else 0.0,
            watch_ratio_mean=wr_mean,
            watch_ratio_std=wr_std,
            watch_ratio_beta_a=float(a),
            watch_ratio_beta_b=float(b),
            duration_mean=d_mean,
            duration_std=d_std,
            duration_log_mu=log_mu,
            duration_log_sigma=log_sigma,
        )

    def sample_watch_ratios(self, n: int, seed: int = None) -> np.ndarray:
        rng = np.random.default_rng(seed)
        samples = rng.beta(self.watch_ratio_beta_a, self.watch_ratio_beta_b, size=n)
        return np.clip(samples, 0, 1)

    def sample_durations(self, n: int, seed: int = None) -> np.ndarray:
        rng = np.random.default_rng(seed)
        return rng.lognormal(self.duration_log_mu, self.duration_log_sigma, size=n)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ArchetypeDistribution":
        return cls(**d)


def extract_archetype_distributions(
    interactions, labels: np.ndarray, user_ids: np.ndarray
) -> list[ArchetypeDistribution]:
    """Extract distributions for each archetype cluster."""
    user_label_map = dict(zip(user_ids, labels))
    interactions = interactions.copy()
    interactions["archetype"] = interactions["user_id"].map(user_label_map)
    interactions = interactions.dropna(subset=["archetype"])
    interactions["archetype"] = interactions["archetype"].astype(int)

    total_users = len(user_ids)
    distributions = []

    for arch_id in sorted(interactions["archetype"].unique()):
        group = interactions[interactions["archetype"] == arch_id]
        n_users_in_arch = int((labels == arch_id).sum())

        dist = ArchetypeDistribution.from_data(
            archetype_id=arch_id,
            watch_ratios=group["watch_ratio"].values,
            durations=group["duration_ms"].values,
            n_users=n_users_in_arch,
            total_users=total_users,
        )
        distributions.append(dist)

    return distributions
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_distribution.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add src/rec_sim/baseline/distribution.py tests/test_distribution.py
git commit -m "feat: archetype distribution extraction with Beta/LogNormal fits"
```

---

## Task 4: Fidelity Metrics

**Files:**
- Create: `src/rec_sim/fidelity/__init__.py`
- Create: `src/rec_sim/fidelity/metrics.py`
- Create: `tests/test_metrics.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_metrics.py
import pytest
import numpy as np
from rec_sim.fidelity.metrics import (
    kl_divergence,
    js_divergence,
    wasserstein_1d,
    correlation_matrix_distance,
    composite_fidelity,
)


def test_kl_identical_distributions():
    p = np.array([0.2, 0.3, 0.5])
    assert kl_divergence(p, p) == pytest.approx(0.0, abs=1e-10)


def test_kl_different_distributions():
    p = np.array([0.5, 0.3, 0.2])
    q = np.array([0.2, 0.3, 0.5])
    assert kl_divergence(p, q) > 0


def test_js_symmetric():
    p = np.array([0.5, 0.3, 0.2])
    q = np.array([0.2, 0.3, 0.5])
    assert js_divergence(p, q) == pytest.approx(js_divergence(q, p), abs=1e-10)


def test_js_bounded():
    p = np.array([0.5, 0.3, 0.2])
    q = np.array([0.2, 0.3, 0.5])
    assert 0 <= js_divergence(p, q) <= 1


def test_wasserstein_identical():
    a = np.array([1.0, 2.0, 3.0])
    assert wasserstein_1d(a, a) == pytest.approx(0.0, abs=1e-10)


def test_wasserstein_shifted():
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([2.0, 3.0, 4.0])
    assert wasserstein_1d(a, b) == pytest.approx(1.0, abs=0.1)


def test_correlation_distance_identical():
    m = np.array([[1, 0.5], [0.5, 1]])
    assert correlation_matrix_distance(m, m) == pytest.approx(0.0, abs=1e-10)


def test_composite_fidelity_perfect():
    metrics = {"kl": 0.0, "js": 0.0, "wass": 0.0}
    max_vals = {"kl": 1.0, "js": 1.0, "wass": 10.0}
    f = composite_fidelity(metrics, max_vals)
    assert f == pytest.approx(1.0)


def test_composite_fidelity_half():
    metrics = {"kl": 0.5, "js": 0.5}
    max_vals = {"kl": 1.0, "js": 1.0}
    f = composite_fidelity(metrics, max_vals)
    assert f == pytest.approx(0.5)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_metrics.py -v`
Expected: FAIL with ImportError

- [ ] **Step 3: Write metrics implementation**

```python
# src/rec_sim/fidelity/metrics.py
"""Fidelity metrics for comparing real vs simulated distributions."""
import numpy as np
from scipy import stats as sp_stats


def kl_divergence(p: np.ndarray, q: np.ndarray, epsilon: float = 1e-10) -> float:
    p = np.asarray(p, dtype=float) + epsilon
    q = np.asarray(q, dtype=float) + epsilon
    p = p / p.sum()
    q = q / q.sum()
    return float(np.sum(p * np.log(p / q)))


def js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)
    return float(0.5 * kl_divergence(p, m) + 0.5 * kl_divergence(q, m))


def wasserstein_1d(a: np.ndarray, b: np.ndarray) -> float:
    return float(sp_stats.wasserstein_distance(a, b))


def correlation_matrix_distance(real: np.ndarray, sim: np.ndarray) -> float:
    return float(np.linalg.norm(real - sim, "fro"))


def relative_error(real_val: float, sim_val: float) -> float:
    if abs(real_val) < 1e-10:
        return abs(sim_val)
    return abs(real_val - sim_val) / abs(real_val)


def composite_fidelity(
    metrics: dict[str, float],
    max_acceptable: dict[str, float],
    weights: dict[str, float] | None = None,
) -> float:
    if weights is None:
        weights = {k: 1.0 for k in metrics}
    total_w = sum(weights[k] for k in metrics)

    f = 0.0
    for k, val in metrics.items():
        w = weights.get(k, 1.0) / total_w
        capped = min(val / max_acceptable[k], 1.0)
        f += w * (1.0 - capped)
    return float(f)
```

- [ ] **Step 4: Create fidelity/__init__.py**

```python
"""Fidelity measurement framework."""
```

- [ ] **Step 5: Run test to verify it passes**

Run: `python -m pytest tests/test_metrics.py -v`
Expected: PASS (10 tests)

- [ ] **Step 6: Commit**

```bash
git add src/rec_sim/fidelity/ tests/test_metrics.py
git commit -m "feat: fidelity metrics (KL, JS, Wasserstein, composite F)"
```

---

## Task 5: Persona Skeleton Generation (LHS)

**Files:**
- Create: `src/rec_sim/persona/__init__.py`
- Create: `src/rec_sim/persona/skeleton.py`
- Create: `tests/test_skeleton.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_skeleton.py
import pytest
import numpy as np
from rec_sim.baseline.distribution import ArchetypeDistribution
from rec_sim.persona.skeleton import generate_skeletons, PersonaSkeleton


def _make_dist(arch_id, proportion, wr_mean):
    return ArchetypeDistribution(
        archetype_id=arch_id, n_users=100, proportion=proportion,
        watch_ratio_mean=wr_mean, watch_ratio_std=0.15,
        watch_ratio_beta_a=2.0, watch_ratio_beta_b=3.0,
        duration_mean=10000, duration_std=3000,
        duration_log_mu=9.2, duration_log_sigma=0.5,
    )


def test_generate_skeletons_count():
    dists = [_make_dist(0, 0.6, 0.5), _make_dist(1, 0.4, 0.7)]
    skeletons = generate_skeletons(dists, n_agents=100, seed=42)
    assert len(skeletons) == 100


def test_generate_skeletons_proportional():
    dists = [_make_dist(0, 0.7, 0.5), _make_dist(1, 0.3, 0.7)]
    skeletons = generate_skeletons(dists, n_agents=1000, seed=42)
    arch0_count = sum(1 for s in skeletons if s.archetype_id == 0)
    assert 650 < arch0_count < 750


def test_skeleton_has_required_fields():
    dists = [_make_dist(0, 1.0, 0.5)]
    skeletons = generate_skeletons(dists, n_agents=10, seed=42)
    s = skeletons[0]
    assert isinstance(s, PersonaSkeleton)
    assert 0 <= s.watch_ratio_baseline <= 1
    assert s.duration_baseline > 0
    assert s.stall_tolerance > 0
    assert 0 <= s.quality_sensitivity <= 1


def test_skeletons_have_variance():
    dists = [_make_dist(0, 1.0, 0.5)]
    skeletons = generate_skeletons(dists, n_agents=50, seed=42)
    baselines = [s.watch_ratio_baseline for s in skeletons]
    assert np.std(baselines) > 0.01
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_skeleton.py -v`
Expected: FAIL with ImportError

- [ ] **Step 3: Write skeleton implementation**

```python
# src/rec_sim/persona/skeleton.py
"""Generate persona skeletons using Latin Hypercube Sampling."""
from dataclasses import dataclass
import numpy as np
from scipy.stats import qmc
from rec_sim.baseline.distribution import ArchetypeDistribution


@dataclass
class PersonaSkeleton:
    agent_id: int
    archetype_id: int
    watch_ratio_baseline: float
    duration_baseline: float
    stall_tolerance: float
    quality_sensitivity: float
    fatigue_rate: float
    first_session_patience: int

    def to_dict(self) -> dict:
        return self.__dict__.copy()


def generate_skeletons(
    distributions: list[ArchetypeDistribution],
    n_agents: int = 1000,
    seed: int = 42,
) -> list[PersonaSkeleton]:
    rng = np.random.default_rng(seed)
    total_prop = sum(d.proportion for d in distributions) or 1.0
    allocations = _allocate_agents(distributions, n_agents, total_prop)

    sampler = qmc.LatinHypercube(d=4, seed=seed)
    samples = sampler.random(n=n_agents)

    skeletons = []
    agent_id = 0
    for dist, count in zip(distributions, allocations):
        for i in range(count):
            if agent_id >= n_agents:
                break
            s = samples[agent_id]

            wr = dist.sample_watch_ratios(1, seed=seed + agent_id)[0]
            dur = dist.sample_durations(1, seed=seed + agent_id)[0]

            skeleton = PersonaSkeleton(
                agent_id=agent_id,
                archetype_id=dist.archetype_id,
                watch_ratio_baseline=float(wr),
                duration_baseline=float(dur),
                stall_tolerance=float(qmc.scale(s[0:1].reshape(1, -1), [500], [5000])[0, 0]),
                quality_sensitivity=float(s[1]),
                fatigue_rate=float(qmc.scale(s[2:3].reshape(1, -1), [0.01], [0.15])[0, 0]),
                first_session_patience=int(qmc.scale(s[3:4].reshape(1, -1), [2], [8])[0, 0]),
            )
            skeletons.append(skeleton)
            agent_id += 1

    return skeletons


def _allocate_agents(distributions, n_agents, total_prop):
    raw = [d.proportion / total_prop * n_agents for d in distributions]
    floors = [int(r) for r in raw]
    remainders = [r - f for r, f in zip(raw, floors)]

    deficit = n_agents - sum(floors)
    top_indices = sorted(range(len(remainders)), key=lambda i: -remainders[i])
    for i in range(deficit):
        floors[top_indices[i]] += 1

    return floors
```

- [ ] **Step 4: Create persona/__init__.py**

```python
"""Persona generation and management."""
```

- [ ] **Step 5: Run test to verify it passes**

Run: `python -m pytest tests/test_skeleton.py -v`
Expected: PASS (4 tests)

- [ ] **Step 6: Commit**

```bash
git add src/rec_sim/persona/ tests/test_skeleton.py
git commit -m "feat: LHS-based persona skeleton generation"
```

---

## Task 6: Infrastructure & Context Models

**Files:**
- Create: `src/rec_sim/interaction/__init__.py`
- Create: `src/rec_sim/interaction/infra.py`
- Create: `src/rec_sim/interaction/context.py`
- Create: `tests/test_infra.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_infra.py
import pytest
from rec_sim.interaction.infra import InfraState, sample_infra_state
from rec_sim.interaction.context import SessionContext, sample_session_context


def test_infra_state_fields():
    state = InfraState(quality="720p", bitrate_kbps=2400, codec="h265",
                       first_frame_ms=800, stall_count=0, stall_duration_ms=0)
    assert state.quality == "720p"
    assert state.bitrate_kbps == 2400


def test_sample_infra_state():
    state = sample_infra_state(network="4g", device_tier="mid", seed=42)
    assert state.quality in ("360p", "480p", "720p", "1080p")
    assert state.first_frame_ms > 0
    assert state.stall_count >= 0


def test_session_context_fields():
    ctx = SessionContext(session_type="first_visit", time_slot="evening",
                         network="wifi", step_index=0, fatigue=0.0)
    assert ctx.session_type == "first_visit"
    assert ctx.fatigue == 0.0


def test_sample_session_context():
    ctx = sample_session_context(session_type="normal", step_index=5, seed=42)
    assert ctx.time_slot in ("morning", "noon", "afternoon", "evening", "night")
    assert ctx.fatigue >= 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_infra.py -v`
Expected: FAIL with ImportError

- [ ] **Step 3: Write infra.py**

```python
# src/rec_sim/interaction/infra.py
"""Infrastructure state model for video delivery simulation."""
from dataclasses import dataclass
import numpy as np

QUALITY_LEVELS = ("360p", "480p", "720p", "1080p")
CODECS = ("h264", "h265", "av1")

NETWORK_PROFILES = {
    "2g":   {"quality_dist": [0.6, 0.3, 0.1, 0.0], "stall_rate": 0.3, "first_frame_base": 3000},
    "3g":   {"quality_dist": [0.2, 0.4, 0.3, 0.1], "stall_rate": 0.15, "first_frame_base": 1500},
    "4g":   {"quality_dist": [0.05, 0.15, 0.5, 0.3], "stall_rate": 0.05, "first_frame_base": 800},
    "wifi": {"quality_dist": [0.0, 0.05, 0.35, 0.6], "stall_rate": 0.02, "first_frame_base": 400},
}

DEVICE_DECODE_PENALTY = {"low": 1.5, "mid": 1.0, "high": 0.8}


@dataclass
class InfraState:
    quality: str
    bitrate_kbps: int
    codec: str
    first_frame_ms: int
    stall_count: int
    stall_duration_ms: int


BITRATE_MAP = {"360p": 600, "480p": 1200, "720p": 2400, "1080p": 4800}


def sample_infra_state(network: str = "4g", device_tier: str = "mid", seed: int = None) -> InfraState:
    rng = np.random.default_rng(seed)
    profile = NETWORK_PROFILES.get(network, NETWORK_PROFILES["4g"])

    quality = rng.choice(QUALITY_LEVELS, p=profile["quality_dist"])
    codec = rng.choice(CODECS, p=[0.4, 0.45, 0.15])
    bitrate = BITRATE_MAP[quality]

    device_mult = DEVICE_DECODE_PENALTY.get(device_tier, 1.0)
    first_frame = int(profile["first_frame_base"] * device_mult * rng.uniform(0.7, 1.3))

    stall_count = rng.poisson(profile["stall_rate"] * 5)
    stall_duration = int(stall_count * rng.uniform(300, 2000)) if stall_count > 0 else 0

    return InfraState(
        quality=quality, bitrate_kbps=bitrate, codec=codec,
        first_frame_ms=first_frame, stall_count=stall_count,
        stall_duration_ms=stall_duration,
    )
```

- [ ] **Step 4: Write context.py**

```python
# src/rec_sim/interaction/context.py
"""Session context model."""
from dataclasses import dataclass
import numpy as np

TIME_SLOTS = ("morning", "noon", "afternoon", "evening", "night")
TIME_SLOT_WEIGHTS = [0.1, 0.15, 0.15, 0.35, 0.25]
SESSION_TYPES = ("first_visit", "normal", "return_user")


@dataclass
class SessionContext:
    session_type: str
    time_slot: str
    network: str
    step_index: int
    fatigue: float


def sample_session_context(
    session_type: str = "normal", step_index: int = 0, seed: int = None
) -> SessionContext:
    rng = np.random.default_rng(seed)
    time_slot = rng.choice(TIME_SLOTS, p=TIME_SLOT_WEIGHTS)
    network = rng.choice(["2g", "3g", "4g", "wifi"], p=[0.02, 0.08, 0.45, 0.45])
    fatigue = 1.0 - np.exp(-0.05 * step_index)

    return SessionContext(
        session_type=session_type, time_slot=time_slot,
        network=network, step_index=step_index, fatigue=fatigue,
    )
```

- [ ] **Step 5: Create interaction/__init__.py**

```python
"""Interaction layer: decision engine and environment models."""
```

- [ ] **Step 6: Run test to verify it passes**

Run: `python -m pytest tests/test_infra.py -v`
Expected: PASS (4 tests)

- [ ] **Step 7: Commit**

```bash
git add src/rec_sim/interaction/ tests/test_infra.py
git commit -m "feat: infrastructure state and session context models"
```

---

## Task 7: Layer 0 — Experience Decision

**Files:**
- Create: `src/rec_sim/interaction/layer0.py`
- Create: `tests/test_layer0.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_layer0.py
import pytest
import numpy as np
from rec_sim.interaction.layer0 import experience_decision, ExperienceResult
from rec_sim.interaction.infra import InfraState
from rec_sim.persona.skeleton import PersonaSkeleton


def _make_skeleton(**overrides):
    defaults = dict(agent_id=0, archetype_id=0, watch_ratio_baseline=0.6,
                    duration_baseline=10000, stall_tolerance=1500,
                    quality_sensitivity=0.5, fatigue_rate=0.05, first_session_patience=5)
    defaults.update(overrides)
    return PersonaSkeleton(**defaults)


def _make_infra(**overrides):
    defaults = dict(quality="720p", bitrate_kbps=2400, codec="h265",
                    first_frame_ms=500, stall_count=0, stall_duration_ms=0)
    defaults.update(overrides)
    return InfraState(**defaults)


def test_no_stall_no_penalty():
    result = experience_decision(_make_skeleton(), _make_infra(), is_first_visit=False, seed=42)
    assert isinstance(result, ExperienceResult)
    assert result.watch_pct_factor >= 0.9
    assert not result.force_skip


def test_heavy_stall_causes_skip():
    skeleton = _make_skeleton(stall_tolerance=500)
    infra = _make_infra(stall_count=5, stall_duration_ms=8000)
    result = experience_decision(skeleton, infra, is_first_visit=False, seed=42)
    assert result.watch_pct_factor < 0.5 or result.force_skip


def test_first_visit_lower_tolerance():
    skeleton = _make_skeleton(stall_tolerance=1500)
    infra = _make_infra(first_frame_ms=3000, stall_count=1, stall_duration_ms=2000)
    result_first = experience_decision(skeleton, infra, is_first_visit=True, seed=42)
    result_normal = experience_decision(skeleton, infra, is_first_visit=False, seed=42)
    assert result_first.watch_pct_factor <= result_normal.watch_pct_factor


def test_low_quality_reduces_factor():
    skeleton = _make_skeleton(quality_sensitivity=0.8)
    infra_good = _make_infra(quality="1080p")
    infra_bad = _make_infra(quality="360p")
    r_good = experience_decision(skeleton, infra_good, is_first_visit=False, seed=42)
    r_bad = experience_decision(skeleton, infra_bad, is_first_visit=False, seed=42)
    assert r_bad.watch_pct_factor < r_good.watch_pct_factor
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_layer0.py -v`
Expected: FAIL with ImportError

- [ ] **Step 3: Write layer0 implementation**

```python
# src/rec_sim/interaction/layer0.py
"""Layer 0: Experience decision — infrastructure quality impact on user behavior."""
from dataclasses import dataclass
import numpy as np
from rec_sim.interaction.infra import InfraState
from rec_sim.persona.skeleton import PersonaSkeleton

QUALITY_SCORE = {"360p": 0.3, "480p": 0.55, "720p": 0.8, "1080p": 1.0}


@dataclass
class ExperienceResult:
    watch_pct_factor: float
    force_skip: bool
    force_exit: bool
    detail: dict


def experience_decision(
    skeleton: PersonaSkeleton,
    infra: InfraState,
    is_first_visit: bool = False,
    seed: int = None,
) -> ExperienceResult:
    rng = np.random.default_rng(seed)
    tolerance_mult = 0.5 if is_first_visit else 1.0
    effective_tolerance = skeleton.stall_tolerance * tolerance_mult

    stall_penalty = 0.0
    if infra.stall_duration_ms > 0:
        ratio = infra.stall_duration_ms / max(effective_tolerance, 1)
        stall_penalty = 1.0 - np.exp(-0.7 * ratio)

    quality_score = QUALITY_SCORE.get(infra.quality, 0.5)
    quality_penalty = skeleton.quality_sensitivity * (1.0 - quality_score) * 0.4

    first_frame_penalty = 0.0
    threshold = 1000 if is_first_visit else 2000
    if infra.first_frame_ms > threshold:
        over = (infra.first_frame_ms - threshold) / threshold
        first_frame_penalty = min(over * 0.3, 0.5)

    total_penalty = min(stall_penalty + quality_penalty + first_frame_penalty, 1.0)
    noise = rng.normal(0, 0.03)
    watch_pct_factor = float(np.clip(1.0 - total_penalty + noise, 0.05, 1.0))

    force_skip = rng.random() < (stall_penalty * 0.8 + first_frame_penalty * 0.5)
    force_exit = (is_first_visit and total_penalty > 0.7 and rng.random() < 0.5)

    return ExperienceResult(
        watch_pct_factor=watch_pct_factor,
        force_skip=force_skip,
        force_exit=force_exit,
        detail={"stall_penalty": stall_penalty, "quality_penalty": quality_penalty,
                "first_frame_penalty": first_frame_penalty},
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_layer0.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add src/rec_sim/interaction/layer0.py tests/test_layer0.py
git commit -m "feat: Layer 0 experience decision with stall/quality/first-frame modeling"
```

---

## Task 8: Layer 1 — Content Decision

**Files:**
- Create: `src/rec_sim/interaction/layer1.py`
- Create: `tests/test_layer1.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_layer1.py
import pytest
import numpy as np
from rec_sim.interaction.layer1 import content_decision, ContentResult
from rec_sim.persona.skeleton import PersonaSkeleton


def _make_skeleton(**overrides):
    defaults = dict(agent_id=0, archetype_id=0, watch_ratio_baseline=0.6,
                    duration_baseline=10000, stall_tolerance=1500,
                    quality_sensitivity=0.5, fatigue_rate=0.05, first_session_patience=5)
    defaults.update(overrides)
    return PersonaSkeleton(**defaults)


def test_content_decision_returns_result():
    result = content_decision(
        skeleton=_make_skeleton(),
        interest_match=0.8,
        l0_factor=1.0,
        fatigue=0.0,
        seed=42,
    )
    assert isinstance(result, ContentResult)
    assert 0 <= result.watch_pct <= 1
    assert isinstance(result.liked, bool)
    assert isinstance(result.commented, bool)
    assert isinstance(result.shared, bool)


def test_high_interest_higher_completion():
    sk = _make_skeleton(watch_ratio_baseline=0.5)
    r_high = content_decision(sk, interest_match=0.9, l0_factor=1.0, fatigue=0.0, seed=42)
    r_low = content_decision(sk, interest_match=0.1, l0_factor=1.0, fatigue=0.0, seed=42)
    assert r_high.watch_pct > r_low.watch_pct


def test_l0_factor_reduces_completion():
    sk = _make_skeleton(watch_ratio_baseline=0.7)
    r_full = content_decision(sk, interest_match=0.8, l0_factor=1.0, fatigue=0.0, seed=42)
    r_degraded = content_decision(sk, interest_match=0.8, l0_factor=0.5, fatigue=0.0, seed=42)
    assert r_degraded.watch_pct < r_full.watch_pct


def test_fatigue_reduces_completion():
    sk = _make_skeleton()
    r_fresh = content_decision(sk, interest_match=0.7, l0_factor=1.0, fatigue=0.0, seed=42)
    r_tired = content_decision(sk, interest_match=0.7, l0_factor=1.0, fatigue=0.8, seed=42)
    assert r_tired.watch_pct < r_fresh.watch_pct
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_layer1.py -v`
Expected: FAIL with ImportError

- [ ] **Step 3: Write layer1 implementation**

```python
# src/rec_sim/interaction/layer1.py
"""Layer 1: Content decision — interest matching and engagement."""
from dataclasses import dataclass
import numpy as np
from rec_sim.persona.skeleton import PersonaSkeleton


@dataclass
class ContentResult:
    watch_pct: float
    liked: bool
    commented: bool
    shared: bool


def content_decision(
    skeleton: PersonaSkeleton,
    interest_match: float,
    l0_factor: float = 1.0,
    fatigue: float = 0.0,
    seed: int = None,
) -> ContentResult:
    rng = np.random.default_rng(seed)

    base = skeleton.watch_ratio_baseline
    interest_boost = (interest_match - 0.5) * 0.4
    fatigue_penalty = fatigue * 0.3
    noise = rng.normal(0, 0.08)

    raw_pct = base + interest_boost - fatigue_penalty + noise
    watch_pct = float(np.clip(raw_pct * l0_factor, 0.0, 1.0))

    engagement_base = watch_pct * interest_match
    liked = bool(rng.random() < engagement_base * 0.25)
    commented = bool(rng.random() < engagement_base * 0.03)
    shared = bool(rng.random() < engagement_base * 0.02)

    return ContentResult(watch_pct=watch_pct, liked=liked, commented=commented, shared=shared)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_layer1.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add src/rec_sim/interaction/layer1.py tests/test_layer1.py
git commit -m "feat: Layer 1 content decision with interest matching and fatigue"
```

---

## Task 9: Decision Engine (L0+L1 Orchestration)

**Files:**
- Create: `src/rec_sim/interaction/engine.py`
- Create: `tests/test_engine.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_engine.py
import pytest
from rec_sim.interaction.engine import DecisionEngine, VideoItem, StepResult
from rec_sim.interaction.infra import InfraState
from rec_sim.interaction.context import SessionContext
from rec_sim.persona.skeleton import PersonaSkeleton


def _make_skeleton(**overrides):
    defaults = dict(agent_id=0, archetype_id=0, watch_ratio_baseline=0.6,
                    duration_baseline=10000, stall_tolerance=1500,
                    quality_sensitivity=0.5, fatigue_rate=0.05, first_session_patience=5)
    defaults.update(overrides)
    return PersonaSkeleton(**defaults)


def test_engine_step_returns_result():
    engine = DecisionEngine(seed=42)
    skeleton = _make_skeleton()
    video = VideoItem(video_id="v1", category="food", duration_ms=15000, interest_match=0.7)
    infra = InfraState("720p", 2400, "h265", 500, 0, 0)
    ctx = SessionContext("normal", "evening", "wifi", 0, 0.0)

    result = engine.step(skeleton, video, infra, ctx)
    assert isinstance(result, StepResult)
    assert result.action in ("watch", "skip", "exit_app")
    assert 0 <= result.watch_pct <= 1
    assert result.decision_layer in (0, 1)
    assert result.fidelity_tag in ("rule", "parametric")


def test_engine_exit_on_heavy_stall_first_visit():
    engine = DecisionEngine(seed=0)
    skeleton = _make_skeleton(stall_tolerance=500)
    video = VideoItem("v1", "tech", 10000, 0.5)
    infra = InfraState("360p", 600, "h264", 5000, 8, 15000)
    ctx = SessionContext("first_visit", "morning", "3g", 0, 0.0)

    results = [engine.step(skeleton, video, infra, ctx) for _ in range(20)]
    exit_count = sum(1 for r in results if r.action == "exit_app")
    assert exit_count > 0


def test_engine_step_result_log_format():
    engine = DecisionEngine(seed=42)
    skeleton = _make_skeleton()
    video = VideoItem("v1", "food", 15000, 0.8)
    infra = InfraState("1080p", 4800, "h265", 300, 0, 0)
    ctx = SessionContext("normal", "evening", "wifi", 0, 0.0)

    result = engine.step(skeleton, video, infra, ctx)
    log = result.to_log(session_id="s1")
    assert log["session_id"] == "s1"
    assert log["agent_id"] == 0
    assert "infra_state" in log
    assert "fidelity_tag" in log
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_engine.py -v`
Expected: FAIL with ImportError

- [ ] **Step 3: Write engine implementation**

```python
# src/rec_sim/interaction/engine.py
"""Decision engine: orchestrates Layer 0 + Layer 1 per video."""
from dataclasses import dataclass
import numpy as np
from rec_sim.persona.skeleton import PersonaSkeleton
from rec_sim.interaction.infra import InfraState
from rec_sim.interaction.context import SessionContext
from rec_sim.interaction.layer0 import experience_decision
from rec_sim.interaction.layer1 import content_decision


@dataclass
class VideoItem:
    video_id: str
    category: str
    duration_ms: int
    interest_match: float


@dataclass
class StepResult:
    action: str
    watch_pct: float
    liked: bool
    commented: bool
    shared: bool
    decision_layer: int
    fidelity_tag: str
    l0_factor: float
    l1_base_pct: float
    agent_id: int

    def to_log(self, session_id: str) -> dict:
        return {
            "session_id": session_id,
            "agent_id": self.agent_id,
            "action": self.action,
            "watch_pct": self.watch_pct,
            "liked": self.liked,
            "commented": self.commented,
            "shared": self.shared,
            "decision_layer": self.decision_layer,
            "fidelity_tag": self.fidelity_tag,
            "l0_factor": self.l0_factor,
            "l1_base_pct": self.l1_base_pct,
            "infra_state": {},
        }


class DecisionEngine:
    def __init__(self, seed: int = 42):
        self._seed_base = seed
        self._call_count = 0

    def step(
        self,
        skeleton: PersonaSkeleton,
        video: VideoItem,
        infra: InfraState,
        ctx: SessionContext,
    ) -> StepResult:
        seed = self._seed_base + self._call_count
        self._call_count += 1

        is_first = ctx.session_type == "first_visit"
        l0 = experience_decision(skeleton, infra, is_first_visit=is_first, seed=seed)

        if l0.force_exit:
            return StepResult(
                action="exit_app", watch_pct=0.0, liked=False, commented=False,
                shared=False, decision_layer=0, fidelity_tag="rule",
                l0_factor=l0.watch_pct_factor, l1_base_pct=0.0, agent_id=skeleton.agent_id,
            )

        if l0.force_skip:
            return StepResult(
                action="skip", watch_pct=0.0, liked=False, commented=False,
                shared=False, decision_layer=0, fidelity_tag="rule",
                l0_factor=l0.watch_pct_factor, l1_base_pct=0.0, agent_id=skeleton.agent_id,
            )

        l1 = content_decision(
            skeleton, interest_match=video.interest_match,
            l0_factor=l0.watch_pct_factor, fatigue=ctx.fatigue, seed=seed + 1000,
        )

        action = "watch" if l1.watch_pct > 0.05 else "skip"

        return StepResult(
            action=action, watch_pct=l1.watch_pct,
            liked=l1.liked, commented=l1.commented, shared=l1.shared,
            decision_layer=1, fidelity_tag="parametric",
            l0_factor=l0.watch_pct_factor, l1_base_pct=l1.watch_pct / max(l0.watch_pct_factor, 0.01),
            agent_id=skeleton.agent_id,
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_engine.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add src/rec_sim/interaction/engine.py tests/test_engine.py
git commit -m "feat: decision engine orchestrating Layer 0 + Layer 1"
```

---

## Task 10: Simulation Runner

**Files:**
- Create: `src/rec_sim/runner.py`
- Create: `tests/test_runner.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_runner.py
import pytest
from rec_sim.runner import run_simulation, SimulationConfig, SimulationResult
from rec_sim.persona.skeleton import PersonaSkeleton
from rec_sim.baseline.distribution import ArchetypeDistribution


def _make_dist():
    return ArchetypeDistribution(
        archetype_id=0, n_users=100, proportion=1.0,
        watch_ratio_mean=0.5, watch_ratio_std=0.15,
        watch_ratio_beta_a=2.0, watch_ratio_beta_b=3.0,
        duration_mean=10000, duration_std=3000,
        duration_log_mu=9.2, duration_log_sigma=0.5,
    )


def test_run_simulation_basic():
    config = SimulationConfig(n_agents=10, videos_per_session=5, seed=42)
    result = run_simulation(config, distributions=[_make_dist()])
    assert isinstance(result, SimulationResult)
    assert len(result.logs) == 10 * 5
    assert result.summary["n_agents"] == 10
    assert 0 < result.summary["avg_watch_pct"] < 1


def test_run_simulation_logs_have_required_fields():
    config = SimulationConfig(n_agents=5, videos_per_session=3, seed=42)
    result = run_simulation(config, distributions=[_make_dist()])
    log = result.logs[0]
    assert "session_id" in log
    assert "agent_id" in log
    assert "action" in log
    assert "fidelity_tag" in log


def test_run_simulation_fidelity():
    config = SimulationConfig(n_agents=50, videos_per_session=10, seed=42)
    result = run_simulation(config, distributions=[_make_dist()])
    assert "fidelity" in result.summary
    assert "F_overall" in result.summary["fidelity"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_runner.py -v`
Expected: FAIL with ImportError

- [ ] **Step 3: Write runner implementation**

```python
# src/rec_sim/runner.py
"""Simulation runner: N agents x M videos per session."""
from dataclasses import dataclass, field
import numpy as np
from rec_sim.persona.skeleton import generate_skeletons, PersonaSkeleton
from rec_sim.baseline.distribution import ArchetypeDistribution
from rec_sim.interaction.engine import DecisionEngine, VideoItem, StepResult
from rec_sim.interaction.infra import sample_infra_state
from rec_sim.interaction.context import sample_session_context
from rec_sim.fidelity.metrics import relative_error


@dataclass
class SimulationConfig:
    n_agents: int = 100
    videos_per_session: int = 30
    seed: int = 42


@dataclass
class SimulationResult:
    logs: list[dict] = field(default_factory=list)
    summary: dict = field(default_factory=dict)


def run_simulation(
    config: SimulationConfig,
    distributions: list[ArchetypeDistribution],
) -> SimulationResult:
    rng = np.random.default_rng(config.seed)
    skeletons = generate_skeletons(distributions, config.n_agents, config.seed)
    engine = DecisionEngine(seed=config.seed)

    logs = []
    categories = ["food", "travel", "tech", "beauty", "comedy", "sports", "music", "education"]

    for skeleton in skeletons:
        session_id = f"s_{skeleton.agent_id}"
        session_type = rng.choice(["first_visit", "normal", "return_user"], p=[0.1, 0.8, 0.1])

        for step in range(config.videos_per_session):
            cat = rng.choice(categories)
            interest = float(rng.beta(2, 2))
            video = VideoItem(
                video_id=f"v_{rng.integers(0, 100000)}",
                category=cat,
                duration_ms=int(rng.lognormal(9.5, 0.6)),
                interest_match=interest,
            )
            ctx = sample_session_context(session_type=session_type, step_index=step, seed=config.seed + step)
            infra = sample_infra_state(network=ctx.network, seed=config.seed + skeleton.agent_id + step)

            result = engine.step(skeleton, video, infra, ctx)
            log = result.to_log(session_id=session_id)
            log["step_index"] = step
            log["video_id"] = video.video_id
            log["category"] = cat
            log["context"] = {"session_type": session_type, "time_slot": ctx.time_slot,
                              "network": ctx.network, "fatigue": ctx.fatigue}
            logs.append(log)

            if result.action == "exit_app":
                break

    watch_pcts = [l["watch_pct"] for l in logs if l["action"] == "watch"]
    avg_wr = float(np.mean(watch_pcts)) if watch_pcts else 0.0
    target_wr = float(np.mean([d.watch_ratio_mean for d in distributions]))

    summary = {
        "n_agents": config.n_agents,
        "total_steps": len(logs),
        "avg_watch_pct": avg_wr,
        "exit_rate": sum(1 for l in logs if l["action"] == "exit_app") / max(len(logs), 1),
        "skip_rate": sum(1 for l in logs if l["action"] == "skip") / max(len(logs), 1),
        "fidelity": {
            "F_overall": 1.0 - relative_error(target_wr, avg_wr),
            "target_watch_ratio": target_wr,
            "actual_watch_ratio": avg_wr,
        },
    }

    return SimulationResult(logs=logs, summary=summary)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_runner.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add src/rec_sim/runner.py tests/test_runner.py
git commit -m "feat: simulation runner with end-to-end loop and fidelity reporting"
```

---

## Task 11: End-to-End Integration Test on Real Data

**Files:**
- Create: `tests/test_e2e.py`

- [ ] **Step 1: Write integration test**

```python
# tests/test_e2e.py
"""End-to-end: load KuaiRec -> cluster -> generate personas -> simulate -> check fidelity."""
import pytest
from rec_sim.baseline.loader import load_kuairec, load_kuairec_items
from rec_sim.baseline.clustering import extract_user_features, cluster_users
from rec_sim.baseline.distribution import extract_archetype_distributions
from rec_sim.runner import run_simulation, SimulationConfig


@pytest.fixture(scope="module")
def archetype_distributions():
    interactions = load_kuairec(use_small=True)
    items = load_kuairec_items()
    features, _ = extract_user_features(interactions, items)
    labels, _ = cluster_users(features, n_clusters=20)
    user_ids = interactions.groupby("user_id").first().index.values
    dists = extract_archetype_distributions(interactions, labels, user_ids)
    return dists


def test_e2e_pipeline(archetype_distributions):
    config = SimulationConfig(n_agents=100, videos_per_session=10, seed=42)
    result = run_simulation(config, archetype_distributions)

    assert result.summary["n_agents"] == 100
    assert result.summary["total_steps"] > 0
    assert 0 < result.summary["avg_watch_pct"] < 1
    print(f"\n=== E2E Results ===")
    print(f"Agents: {result.summary['n_agents']}")
    print(f"Total steps: {result.summary['total_steps']}")
    print(f"Avg watch %: {result.summary['avg_watch_pct']:.3f}")
    print(f"Exit rate: {result.summary['exit_rate']:.3f}")
    print(f"Skip rate: {result.summary['skip_rate']:.3f}")
    print(f"Fidelity F: {result.summary['fidelity']['F_overall']:.3f}")
    print(f"  target WR: {result.summary['fidelity']['target_watch_ratio']:.3f}")
    print(f"  actual WR: {result.summary['fidelity']['actual_watch_ratio']:.3f}")
```

- [ ] **Step 2: Run on Mac with real KuaiRec data**

Run: `cd /Users/bytedance/Desktop/hehe/research/rec_sim && python -m pytest tests/test_e2e.py -v -s`
Expected: PASS with printed fidelity metrics

- [ ] **Step 3: Commit**

```bash
git add tests/test_e2e.py
git commit -m "test: end-to-end integration test with real KuaiRec data"
```

---

## Verification

After all tasks complete:

1. Run full test suite on Mac: `python -m pytest tests/ -v`
2. Check fidelity output from `test_e2e.py` — `F_overall` should be > 0.5 (Phase 1 baseline; calibration loop in Phase 2 will improve this)
3. Verify all commits are clean: `git log --oneline`

## Phase 2 Preview (not in this plan)

- Layer 2 LLM integration (Claude API calls for complex decisions)
- Calibration loop (three nested loops)
- Support Points + column generation for persona optimization
- Vine Copula extrapolation layer
- Evaluation dashboard
