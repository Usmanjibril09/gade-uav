"""Figure 3: M3 cost predictor accuracy.

(a) Predicted vs observed scatter for bw (R^2=0.998).
(b) Same for cpu.
(c) Same for lat.
(d) Bar chart of MAE and R^2 across all six cost dims (dual axis).

Uses frozen predictions and observed costs from the original M3 test split.
"""

from __future__ import annotations

import sys
from pathlib import Path as _Path

sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl
from sklearn.model_selection import train_test_split

from gade_uav.models.cost_dataset import (
    COST_DIMS,
    build_features,
    build_targets,
)

from scripts.plotting.style import (
    OKABE_ITO,
    apply_style,
    label_panel,
    new_2x2,
    save_fig,
)

REPO = Path(__file__).resolve().parents[2]


def load_test_predictions(seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """Rebuild the training-time test split, load boosters, return (y_true, y_pred)."""
    archive = REPO / "artifacts" / "cost_predictor" / "scatter_inputs.npz"
    if archive.is_file():
        with np.load(archive, allow_pickle=False) as records:
            return records["observed"], records["predicted"]
    warmup_dir = REPO / "data" / "interim" / "deployment_warmup"
    files = sorted(warmup_dir.glob("episode_*.parquet"))
    df = pl.concat([pl.read_parquet(f) for f in files])
    df = df.filter(pl.col("obs_bw").is_not_null())

    X = build_features(df)
    y = build_targets(df)

    # Same split as cost_train: 70/15/15 stratified by action_type
    strat = df["action_type"].to_numpy()
    idx = np.arange(len(X))
    tr_idx, tmp_idx = train_test_split(
        idx, test_size=0.30, random_state=seed, stratify=strat
    )
    val_idx, te_idx = train_test_split(
        tmp_idx, test_size=0.50, random_state=seed, stratify=strat[tmp_idx]
    )

    X_te = X[te_idx]
    y_te = y[te_idx]

    art = REPO / "artifacts" / "cost_predictor"
    y_pred = np.zeros_like(y_te)
    for i, dim in enumerate(COST_DIMS):
        booster = lgb.Booster(model_file=str(art / f"booster_{dim}.txt"))
        y_pred[:, i] = booster.predict(X_te)
    return y_te, y_pred


def scatter_panel(ax, y_true, y_pred, dim: str, label: str, colour: str) -> None:
    # Downsample for legibility (270k pts too dense)
    rng = np.random.default_rng(0)
    n = len(y_true)
    idx = rng.choice(n, size=min(5000, n), replace=False)
    yt, yp = y_true[idx], y_pred[idx]

    ax.scatter(yt, yp, s=8, alpha=0.25, color=colour, edgecolor="none", rasterized=True)

    lo = min(yt.min(), yp.min())
    hi = max(yt.max(), yp.max())
    ax.plot([lo, hi], [lo, hi], "k--", lw=1.2, alpha=0.7, label="Identity")

    r2 = 1 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - y_true.mean()) ** 2)
    mae = np.mean(np.abs(y_true - y_pred))
    ax.text(
        0.05,
        0.92,
        f"$R^2 = {r2:.3f}$\n$\\mathrm{{MAE}} = {mae:.3g}$",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=17,
        bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="0.7", lw=0.6),
    )

    ax.set_xlabel(f"Observed {label}")
    ax.set_ylabel(f"Predicted {label}")
    ax.legend(loc="lower right", fontsize=15)


def panel_d_summary(ax, report: dict) -> None:
    dims = list(COST_DIMS)
    r2 = [report["test_metrics"][d]["r2"] for d in dims]
    mae = [report["test_metrics"][d]["mae"] for d in dims]

    x = np.arange(len(dims))
    ax2 = ax.twinx()
    ax.bar(
        x,
        r2,
        width=0.55,
        color=OKABE_ITO[0],
        alpha=0.85,
        edgecolor="white",
        lw=0.8,
        label="$R^2$",
    )
    ax2.plot(x, mae, "D-", color=OKABE_ITO[1], lw=1.8, ms=8, label="MAE")

    ax.set_xticks(x)
    ax.set_xticklabels(dims, rotation=30, ha="right")
    ax.set_ylabel("$R^2$", color=OKABE_ITO[0])
    ax.tick_params(axis="y", labelcolor=OKABE_ITO[0])
    ax.set_ylim(0, 1.05)
    ax.axhline(1.0, color="k", ls=":", lw=0.6, alpha=0.5)

    ax2.set_ylabel("MAE (native units)", color=OKABE_ITO[1])
    ax2.tick_params(axis="y", labelcolor=OKABE_ITO[1])
    ax2.set_yscale("log")
    ax2.grid(False)

    ax.set_title(r"$R^2$ (bars) and MAE (line)")
    label_panel(ax, "d")


def main() -> None:
    apply_style(paper=True)
    y_true, y_pred = load_test_predictions()
    report = json.loads(
        (REPO / "artifacts" / "cost_predictor" / "training_report.json").read_text()
    )

    fig, axes = new_2x2(figsize=(12.0, 9.0))
    scatter_panel(
        axes[0, 0],
        y_true[:, COST_DIMS.index("bw")],
        y_pred[:, COST_DIMS.index("bw")],
        "bw",
        "bandwidth (B/decision)",
        OKABE_ITO[0],
    )
    label_panel(axes[0, 0], "a")

    scatter_panel(
        axes[0, 1],
        y_true[:, COST_DIMS.index("cpu")],
        y_pred[:, COST_DIMS.index("cpu")],
        "cpu",
        "CPU (fraction)",
        OKABE_ITO[2],
    )
    label_panel(axes[0, 1], "b")

    scatter_panel(
        axes[1, 0],
        y_true[:, COST_DIMS.index("lat")],
        y_pred[:, COST_DIMS.index("lat")],
        "lat",
        "latency (ms)",
        OKABE_ITO[3],
    )
    label_panel(axes[1, 0], "c")

    panel_d_summary(axes[1, 1], report)

    save_fig(fig, "fig03_m3_accuracy")


if __name__ == "__main__":
    main()
