"""M3 cost predictor — dataset, feature builder, stratified split.

Reads deployment_warmup parquets (DEPLOY rows only), builds one-hot
categorical features + numeric features, extracts 6-dim cost targets,
and provides a stratified 70/15/15 split by (action_type × mission_phase
× family_present_flag) with fallback to (action_type × mission_phase)
if any cell has fewer than 10 samples.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import polars as pl

# ---------------------------------------------------------------------------
# Constants — locked at module load; changing these invalidates saved models.
# ---------------------------------------------------------------------------

ACTION_TYPES: tuple[str, ...] = ("decoy_telemetry", "honey_beacon")
TARGET_IFACES: tuple[str, ...] = ("decoy_mavlink_channel", "honey_beacon_channel")
MISSION_PHASES: tuple[str, ...] = ("cruise", "takeoff")

# 12 configured attack families + "none" sentinel for idle periods.
# Order is locked; new families must be appended, never inserted.
ATTACK_FAMILIES: tuple[str, ...] = (
    "none",
    "BruteForce",
    "DDoS",
    "Deauthentication",
    "DoS",
    "FakeLanding",
    "ICMP_Flooding",
    "Jamming",
    "MITM",
    "Reconnaissance",
    "Replay",
    "UDP_Flooding",
    "TCP_Flooding",
)

COST_DIMS: tuple[str, ...] = ("cpu", "ram", "lat", "bw", "energy", "mission_delay")

# One-hot dims + 7 numeric (gen_confidence + 6 generator-declared pred_*).
# The 6 pred_* features let M3 correct the generator's self-estimate rather
# than replace it (Path A: residual correction). Without them M3 cannot
# distinguish e.g. safe_beacon (bw=48) from high_cost_beacon (bw=800) since
# both share (action_type, target_iface, family, phase).
# tick_norm was removed because it encodes episode position, which has no
# runtime analog at deployment.
PRED_COLS: tuple[str, ...] = tuple(f"pred_{d}" for d in COST_DIMS)
N_FEATURES: int = (
    len(ACTION_TYPES)
    + len(TARGET_IFACES)
    + len(MISSION_PHASES)
    + len(ATTACK_FAMILIES)
    + 1
    + len(PRED_COLS)
)
N_TARGETS: int = len(COST_DIMS)

# ---------------------------------------------------------------------------
# Aux stats — mean/std of numeric features + targets, computed on train split.
# ---------------------------------------------------------------------------


@dataclass
class CostStats:
    """Normalization stats computed on the training split only."""

    feature_means: list[float] = field(default_factory=list)
    feature_stds: list[float] = field(default_factory=list)
    target_means: list[float] = field(default_factory=list)
    target_stds: list[float] = field(default_factory=list)

    def to_json(self, path: Path) -> None:
        path.write_text(
            json.dumps(
                {
                    "feature_means": self.feature_means,
                    "feature_stds": self.feature_stds,
                    "target_means": self.target_means,
                    "target_stds": self.target_stds,
                },
                indent=2,
            )
        )

    @classmethod
    def from_json(cls, path: Path) -> "CostStats":
        d = json.loads(path.read_text())
        return cls(**d)


# ---------------------------------------------------------------------------
# Loading + filtering
# ---------------------------------------------------------------------------


def load_training_rows(warmup_dir: Path) -> pl.DataFrame:
    """Concatenate episode parquets and return rows with observed cost.

    Includes both DEPLOY rows and L3-rejected rows (which now carry
    counterfactual observed cost from the simulator effector). L1/L2
    rejects are excluded because their actions are forbidden by
    construction and were never executed. Breaks the shield's
    selection bias on the training distribution (Jin & Ren, JRSS-B 2024).
    """
    files = sorted(warmup_dir.glob("episode_*.parquet"))
    if not files:
        raise FileNotFoundError(f"no episode parquets under {warmup_dir}")
    df = pl.concat([pl.read_parquet(f) for f in files])
    train = df.filter(pl.col("obs_bw").is_not_null())
    if train.height == 0:
        raise ValueError("no rows with observed cost in warmup corpus")
    return train


# Backwards-compatible alias — existing imports keep working during transition.
load_deploy_rows = load_training_rows


# ---------------------------------------------------------------------------
# Feature builder — vectorized, no per-row Python loops.
# ---------------------------------------------------------------------------


def _one_hot(values: pl.Series, categories: tuple[str, ...]) -> np.ndarray:
    """Return (n, len(categories)) float32 one-hot matrix.

    Unknown values (should not happen with locked categories) become all-zero
    rows — this is defensive; the caller should assert coverage before use.
    """
    idx_map = {c: i for i, c in enumerate(categories)}
    n = values.len()
    out = np.zeros((n, len(categories)), dtype=np.float32)
    vals = values.to_list()
    for i, v in enumerate(vals):
        j = idx_map.get(v)
        if j is not None:
            out[i, j] = 1.0
    return out


def build_features(df: pl.DataFrame) -> np.ndarray:
    """Build (n, N_FEATURES) float32 feature matrix.

    Column order:
        [ACTION_TYPES one-hot]      len 2
        [TARGET_IFACES one-hot]     len 2
        [MISSION_PHASES one-hot]    len 2
        [ATTACK_FAMILIES one-hot]   len 13 (with "none" sentinel)
        [gen_confidence]            len 1
        [pred_cpu..pred_mission_delay]  len 6  (generator self-estimate)
    Total: 26
    """
    # Replace null primary_attack_family with "none" sentinel.
    fam = df["primary_attack_family"].fill_null("none")

    action_oh = _one_hot(df["action_type"], ACTION_TYPES)
    iface_oh = _one_hot(df["target_iface"], TARGET_IFACES)
    phase_oh = _one_hot(df["mission_phase"], MISSION_PHASES)
    family_oh = _one_hot(fam, ATTACK_FAMILIES)

    gen_conf = df["gen_confidence"].to_numpy().astype(np.float32).reshape(-1, 1)

    pred_block = np.stack(
        [df[c].to_numpy().astype(np.float32) for c in PRED_COLS],
        axis=1,
    )

    feats = np.concatenate(
        [action_oh, iface_oh, phase_oh, family_oh, gen_conf, pred_block],
        axis=1,
    ).astype(np.float32, copy=False)
    assert (
        feats.shape[1] == N_FEATURES
    ), f"expected {N_FEATURES} cols, got {feats.shape[1]}"
    assert feats.dtype == np.float32, f"expected float32, got {feats.dtype}"
    return feats


def build_targets(df: pl.DataFrame) -> np.ndarray:
    """Build (n, 6) float32 target matrix in COST_DIMS order."""
    cols = [f"obs_{d}" for d in COST_DIMS]
    arr = np.stack([df[c].to_numpy().astype(np.float32) for c in cols], axis=1)
    if np.isnan(arr).any():
        n_bad = int(np.isnan(arr).any(axis=1).sum())
        raise ValueError(f"{n_bad} DEPLOY rows have NaN in obs_* — data corruption")
    return arr


# ---------------------------------------------------------------------------
# Stratified split
# ---------------------------------------------------------------------------


def stratified_split(
    df: pl.DataFrame,
    seed: int = 42,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """70/15/15 stratified split by (action_type, mission_phase, family_present).

    Falls back to (action_type, mission_phase) if any stratum has <10 samples.
    Returns row-index arrays (train_idx, val_idx, test_idx) into df.
    """
    rng = np.random.default_rng(seed)
    n = df.height

    family_present = df["primary_attack_family"].is_not_null().to_numpy()
    action = df["action_type"].to_numpy()
    phase = df["mission_phase"].to_numpy()

    # Primary strata.
    strata = np.array(
        [f"{a}|{p}|{int(f)}" for a, p, f in zip(action, phase, family_present)]
    )
    unique, counts = np.unique(strata, return_counts=True)
    if counts.min() < 10:
        # Fallback: drop family_present dim.
        strata = np.array([f"{a}|{p}" for a, p in zip(action, phase)])

    train_idx: list[int] = []
    val_idx: list[int] = []
    test_idx: list[int] = []
    for s in np.unique(strata):
        idx = np.where(strata == s)[0]
        rng.shuffle(idx)
        n_s = len(idx)
        n_tr = int(round(n_s * train_frac))
        n_va = int(round(n_s * val_frac))
        train_idx.extend(idx[:n_tr].tolist())
        val_idx.extend(idx[n_tr : n_tr + n_va].tolist())
        test_idx.extend(idx[n_tr + n_va :].tolist())

    tr = np.array(sorted(train_idx), dtype=np.int64)
    va = np.array(sorted(val_idx), dtype=np.int64)
    te = np.array(sorted(test_idx), dtype=np.int64)

    # Sanity: disjoint and covering.
    assert len(set(tr) & set(va)) == 0
    assert len(set(tr) & set(te)) == 0
    assert len(set(va) & set(te)) == 0
    assert len(tr) + len(va) + len(te) == n

    return tr, va, te


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------


def compute_stats(
    features: np.ndarray, targets: np.ndarray, train_idx: np.ndarray
) -> CostStats:
    """Compute mean/std of features and targets on the training split.

    LightGBM does not require feature normalization (tree-based), but we compute
    and save stats for downstream consumers (e.g., conformal calibration
    diagnostics, dashboard). Targets are NOT normalized during training —
    LightGBM handles raw scales fine; stats are informational only.
    """
    f_tr = features[train_idx]
    t_tr = targets[train_idx]
    return CostStats(
        feature_means=f_tr.mean(axis=0).tolist(),
        feature_stds=f_tr.std(axis=0).tolist(),
        target_means=t_tr.mean(axis=0).tolist(),
        target_stds=t_tr.std(axis=0).tolist(),
    )
