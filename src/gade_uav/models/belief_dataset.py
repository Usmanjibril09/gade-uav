"""Belief sequence dataset, stratified split, and class-balanced sampler.

Loads windows from data/interim/features/belief_sequences.parquet, remaps
msg_ids via configs/belief_vocab.json (UNK=0), applies z-score normalization
to 6 aux features (heartbeat_gap_ms zero-filled + presence indicator) using
statistics computed on the training split only.

Split: stratified by (canonical_label, origin) with fallback to label-only
when a cell has fewer than MIN_CELL samples, seed=42, ratios 70/15/15.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl
import torch
from torch.utils.data import Dataset, WeightedRandomSampler

# Locked constants (per Phase 2 Track 2 spec)
WINDOW_SIZE = 256
LABELS = (
    "Normal",
    "MAVLink_Heartbeat_Flooding",
    "MAVLink_Ping_Flooding",
    "MAVLink_Request_Flooding",
    "MAVLink_Corruption",
    "MAVLink_Suspension",
)
LABEL_TO_IDX = {lbl: i for i, lbl in enumerate(LABELS)}
IDX_TO_LABEL = {i: lbl for lbl, i in LABEL_TO_IDX.items()}

AUX_COLS = (
    "msg_rate_hz",
    "unique_msg_ids_in_window",
    "dominant_msg_share",
    "entropy_msg_id",
    "heartbeat_gap_ms",
)
# 6th feature is the heartbeat presence indicator, computed on the fly.
N_AUX = len(AUX_COLS) + 1  # = 6

MIN_CELL = 10  # below this, (label,origin) cell degrades to label-only strat
SPLIT_SEED = 42
DEFAULT_RATIOS = (0.70, 0.15, 0.15)


@dataclass(frozen=True)
class AuxStats:
    """Per-feature z-score stats. Order matches AUX_COLS + presence indicator."""

    means: tuple[float, ...]
    stds: tuple[float, ...]

    def to_dict(self) -> dict:
        return {"means": list(self.means), "stds": list(self.stds)}

    @classmethod
    def from_dict(cls, d: dict) -> "AuxStats":
        return cls(means=tuple(d["means"]), stds=tuple(d["stds"]))


def _load_vocab(path: Path) -> dict[int, int]:
    """Load msg_id -> dense_index map. UNK is index 0."""
    data = json.loads(path.read_text())
    # JSON keys are strings; convert numeric ones back to int, keep 'UNK'.
    out: dict[int, int] = {}
    for k, v in data["msg_id_to_idx"].items():
        if k == "UNK":
            continue
        out[int(k)] = int(v)
    return out


def _remap_sequence(msg_ids: list[int], vocab: dict[int, int]) -> np.ndarray:
    """Map raw msg_ids to dense indices; unseen -> UNK (0). Fixed length WINDOW_SIZE."""
    out = np.zeros(WINDOW_SIZE, dtype=np.int64)
    for i, mid in enumerate(msg_ids[:WINDOW_SIZE]):
        out[i] = vocab.get(int(mid), 0)
    return out


def _build_aux_row(row: dict) -> np.ndarray:
    """Extract 6 aux features from a parquet row, zero-filling heartbeat_gap NaN."""
    hb = row["heartbeat_gap_ms"]
    hb_present = 0.0 if (hb is None or np.isnan(hb)) else 1.0
    hb_filled = 0.0 if hb_present == 0.0 else float(hb)
    return np.array(
        [
            float(row["msg_rate_hz"]),
            float(row["unique_msg_ids_in_window"]),
            float(row["dominant_msg_share"]),
            float(row["entropy_msg_id"]),
            hb_filled,
            hb_present,
        ],
        dtype=np.float32,
    )


def stratified_split(
    df: pl.DataFrame,
    ratios: tuple[float, float, float] = DEFAULT_RATIOS,
    seed: int = SPLIT_SEED,
    min_cell: int = MIN_CELL,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (train_idx, val_idx, test_idx) with 70/15/15 stratification.

    Stratifies by (canonical_label, origin) when the cell has >= min_cell
    samples; otherwise falls back to label-only stratification for that cell.
    """
    assert abs(sum(ratios) - 1.0) < 1e-6, "ratios must sum to 1"
    rng = np.random.default_rng(seed)
    n = df.height

    # Group indices by (label, origin) then by label for fallback.
    cell_indices: dict[tuple[str, str], list[int]] = defaultdict(list)
    label_indices: dict[str, list[int]] = defaultdict(list)
    labels = df["canonical_label"].to_list()
    origins = df["origin"].to_list()
    for i in range(n):
        cell_indices[(labels[i], origins[i])].append(i)
        label_indices[labels[i]].append(i)

    train, val, test = [], [], []
    # Track which indices are handled by fallback to avoid double-assignment.
    handled: set[int] = set()

    # First pass: cells with enough samples get (label, origin) stratification.
    for cell, idxs in cell_indices.items():
        if len(idxs) < min_cell:
            continue
        idxs = list(idxs)
        rng.shuffle(idxs)
        n_train = int(round(len(idxs) * ratios[0]))
        n_val = int(round(len(idxs) * ratios[1]))
        train.extend(idxs[:n_train])
        val.extend(idxs[n_train : n_train + n_val])
        test.extend(idxs[n_train + n_val :])
        handled.update(idxs)

    # Second pass: fallback to label-only strat for the small cells.
    for lbl, idxs in label_indices.items():
        remaining = [i for i in idxs if i not in handled]
        if not remaining:
            continue
        rng.shuffle(remaining)
        n_train = int(round(len(remaining) * ratios[0]))
        n_val = int(round(len(remaining) * ratios[1]))
        train.extend(remaining[:n_train])
        val.extend(remaining[n_train : n_train + n_val])
        test.extend(remaining[n_train + n_val :])

    return np.array(sorted(train)), np.array(sorted(val)), np.array(sorted(test))


def compute_aux_stats(df: pl.DataFrame, indices: np.ndarray) -> AuxStats:
    """Compute per-feature mean/std from training rows only."""
    sub = df[indices.tolist()] if len(indices) < df.height else df
    rows = sub.to_dicts()
    aux = np.stack([_build_aux_row(r) for r in rows])  # (n_train, 6)
    means = aux.mean(axis=0)
    stds = aux.std(axis=0)
    # Guard against zero-variance features (presence indicator can be all 1s in some folds)
    stds = np.where(stds < 1e-8, 1.0, stds)
    return AuxStats(means=tuple(means.tolist()), stds=tuple(stds.tolist()))


class BeliefSequenceDataset(Dataset):
    """Belief-window dataset. Emits (msg_idx_seq, aux_vec, label_idx, origin_str)."""

    def __init__(
        self,
        parquet_path: Path,
        vocab_path: Path,
        indices: np.ndarray,
        aux_stats: AuxStats,
    ) -> None:
        self._df = pl.read_parquet(parquet_path)
        self._vocab = _load_vocab(vocab_path)
        self._indices = np.asarray(indices, dtype=np.int64)
        self._stats = aux_stats
        self._means = np.array(aux_stats.means, dtype=np.float32)
        self._stds = np.array(aux_stats.stds, dtype=np.float32)

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor, int, str]:
        row = self._df.row(int(self._indices[i]), named=True)
        seq = _remap_sequence(row["msg_ids"], self._vocab)
        aux_raw = _build_aux_row(row)
        aux = (aux_raw - self._means) / self._stds
        label = LABEL_TO_IDX[row["canonical_label"]]
        origin = row["origin"]
        return (
            torch.from_numpy(seq),
            torch.from_numpy(aux),
            label,
            origin,
        )

    def label_counts(self) -> np.ndarray:
        labels = self._df["canonical_label"].to_list()
        counts = np.zeros(len(LABELS), dtype=np.int64)
        for i in self._indices:
            counts[LABEL_TO_IDX[labels[int(i)]]] += 1
        return counts


def make_weighted_sampler(
    labels: np.ndarray, num_samples: int | None = None
) -> WeightedRandomSampler:
    """Sampler where per-sample weight = 1 / count(class). Balances mini-batches."""
    counts = np.bincount(labels, minlength=len(LABELS))
    counts = np.where(counts == 0, 1, counts)  # avoid div-by-zero for absent classes
    per_class_w = 1.0 / counts
    weights = per_class_w[labels]
    if num_samples is None:
        num_samples = len(labels)
    return WeightedRandomSampler(
        weights=torch.as_tensor(weights, dtype=torch.double),
        num_samples=num_samples,
        replacement=True,
    )


def compute_class_weights(labels: np.ndarray) -> torch.Tensor:
    """w_c = N_total / (n_classes * N_c). Used in CrossEntropyLoss(weight=)."""
    counts = np.bincount(labels, minlength=len(LABELS)).astype(np.float64)
    counts = np.where(counts == 0, 1.0, counts)
    n_total = counts.sum()
    w = n_total / (len(LABELS) * counts)
    return torch.as_tensor(w, dtype=torch.float32)


__all__ = [
    "WINDOW_SIZE",
    "LABELS",
    "LABEL_TO_IDX",
    "IDX_TO_LABEL",
    "AUX_COLS",
    "N_AUX",
    "MIN_CELL",
    "SPLIT_SEED",
    "AuxStats",
    "BeliefSequenceDataset",
    "stratified_split",
    "compute_aux_stats",
    "make_weighted_sampler",
    "compute_class_weights",
]
