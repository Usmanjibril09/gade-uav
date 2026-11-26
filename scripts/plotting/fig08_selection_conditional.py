"""Figure 8: Selection-conditional coverage (Jin & Ren style).

Rows = attack family, columns = cost dimension. Each cell = corrected empirical
one-sided coverage of the M3 conformal upper bound on the evaluation corpus for
that (family, dimension) subpopulation. Target = 0.90.

(a) Coverage(family x dim) using corrected training-time quantiles.
(b) Same using corrected recalibrated quantiles.
(c) Coverage gain (b - a).
(d) Worst 10 (family, dim) cells after corrected recalibration.
"""
from __future__ import annotations

import sys
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))

import json
from pathlib import Path

import numpy as np
import polars as pl
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

from gade_uav.models.cost_dataset import build_features
from gade_uav.shield.cost_estimator import M3CostEstimator

from scripts.plotting.style import (
    OKABE_ITO, apply_style, label_panel, new_2x2, save_fig,
)

REPO = Path(__file__).resolve().parents[2]
COST_DIMS = ("cpu", "ram", "lat", "bw", "energy", "mission_delay")

CORRECTED_DIR = (
    REPO
    / "artifacts"
    / "shield_eval_full"
    / "conformal"
)


def load_coverage() -> tuple[pl.DataFrame, pl.DataFrame]:
    """Load corrected one-sided per-family coverage artifacts."""
    fixed_path = CORRECTED_DIR / "fixed_q_per_family.parquet"
    recal_path = CORRECTED_DIR / "recalibrated_per_family.parquet"

    if not fixed_path.exists():
        raise FileNotFoundError(fixed_path)
    if not recal_path.exists():
        raise FileNotFoundError(recal_path)

    fixed = pl.read_parquet(fixed_path)
    recal = pl.read_parquet(recal_path)

    required = [
        "family",
        "n",
    ] + [
        f"coverage_{dim}"
        for dim in COST_DIMS
    ]

    for name, df in (
        ("fixed_q_per_family", fixed),
        ("recalibrated_per_family", recal),
    ):
        missing = [
            c
            for c in required
            if c not in df.columns
        ]

        if missing:
            raise RuntimeError(
                f"{name} missing columns: {missing}\n"
                f"Available: {df.columns}"
            )

    return fixed, recal


def coverage_matrix(
    df: pl.DataFrame,
    min_n: int = 200,
) -> tuple[np.ndarray, list, list]:
    """Read corrected coverage per (family, dim) cell."""
    filtered = (
        df
        .filter(pl.col("n") >= min_n)
        .sort("family")
    )

    fams = filtered["family"].to_list()
    dims = list(COST_DIMS)

    mat = np.full(
        (len(fams), len(dims)),
        np.nan,
    )

    for i, row in enumerate(
        filtered.iter_rows(named=True)
    ):
        for j, dim in enumerate(dims):
            value = row[f"coverage_{dim}"]

            if value is not None:
                mat[i, j] = float(value)

    return mat, fams, dims


DISPLAY_FAMILY = {
    "Deauthentication": "Deauth.",
    "Reconnaissance": "Recon.",
    "ICMP_Flooding": "ICMP flood",
    "UDP_Flooding": "UDP flood",
}


def _heatmap_cov(ax, mat, fams, dims, title, letter, vmin=0.60, vmax=1.00):
    cmap = LinearSegmentedColormap.from_list(
        "cov", ["#B2182B", "#F7F7F7", "#2166AC"], N=256)
    im = ax.imshow(mat, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat[i, j]
            if np.isnan(v):
                ax.text(j, i, "-", ha="center", va="center", fontsize=13,
                        color="0.5")
            else:
                colour = "white" if abs(v - 0.80) > 0.12 else "black"
                ax.text(j, i, f"{v:.3f}", ha="center", va="center",
                        fontsize=12, color=colour)
    ax.set_xticks(range(len(dims)))
    ax.set_xticklabels([dim.replace("mission_delay", "delay") for dim in dims],
                       rotation=30, ha="right", fontsize=15)
    ax.set_yticks(range(len(fams)))
    ax.set_yticklabels([DISPLAY_FAMILY.get(fam, fam) for fam in fams], fontsize=15)
    ax.tick_params(axis="both", length=0)
    ax.set_title(title)
    cbar = ax.figure.colorbar(im, ax=ax, shrink=0.75, pad=0.02)
    cbar.set_label("Coverage", fontsize=15)
    label_panel(ax, letter, x=-0.10, y=1.08)


def _heatmap_diff(ax, mat_a, mat_b, fams, dims, letter):
    diff = mat_b - mat_a
    cmap = LinearSegmentedColormap.from_list(
        "diff", ["#B2182B", "#F7F7F7", "#2166AC"], N=256)
    vmax = np.nanmax(np.abs(diff)) if np.isfinite(diff).any() else 0.2
    vmax = max(vmax, 0.05)
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    im = ax.imshow(diff, cmap=cmap, norm=norm, aspect="auto")
    for i in range(diff.shape[0]):
        for j in range(diff.shape[1]):
            v = diff[i, j]
            if np.isnan(v):
                ax.text(j, i, "-", ha="center", va="center", fontsize=13,
                        color="0.5")
                continue
            colour = "white" if abs(v) > vmax * 0.6 else "black"
            sign = "+" if v >= 0 else ""
            ax.text(j, i, f"{sign}{v:.3f}", ha="center", va="center",
                    fontsize=12, color=colour)
    ax.set_xticks(range(len(dims)))
    ax.set_xticklabels([dim.replace("mission_delay", "delay") for dim in dims],
                       rotation=30, ha="right", fontsize=15)
    ax.set_yticks(range(len(fams)))
    ax.set_yticklabels([DISPLAY_FAMILY.get(fam, fam) for fam in fams], fontsize=15)
    ax.tick_params(axis="both", length=0)
    ax.set_title("Coverage change")
    cbar = ax.figure.colorbar(im, ax=ax, shrink=0.75, pad=0.02)
    cbar.set_label(r"$\Delta$Coverage", fontsize=15)
    label_panel(ax, letter, x=-0.10, y=1.08)


def panel_d_worst_cells(ax, recal_df: pl.DataFrame) -> None:
    """After corrected recalibration, which (family, dim) cells have lowest coverage?"""
    rows = []

    for row in recal_df.iter_rows(named=True):
        fam = row["family"]
        n = int(row["n"])

        if n < 200:
            continue

        for dim in COST_DIMS:
            cov = row[f"coverage_{dim}"]

            if cov is None:
                continue

            rows.append(
                {
                    "dim": dim,
                    "family": fam,
                    "cov": float(cov),
                    "n": n,
                }
            )

    all_cells = pl.DataFrame(rows).sort("cov")
    worst = all_cells.head(10)

    labels = [
        f"{r['dim'].replace('mission_delay', 'delay')} $\\mid$ "
        f"{DISPLAY_FAMILY.get(r['family'], r['family'])}  ($n$={r['n']:,})"
        for r in worst.iter_rows(named=True)
    ]
    covs = worst["cov"].to_numpy()
    y = np.arange(len(labels))
    colours = [OKABE_ITO[1] if c < 0.85 else OKABE_ITO[6] for c in covs]
    ax.barh(y, covs, color=colours, edgecolor="white", lw=0.6, alpha=0.9)
    ax.axvline(0.90, color="k", ls="--", lw=1.2, alpha=0.7,
               label="Target = 0.90")
    for i, c in enumerate(covs):
        ax.text(0.985, i, f"{c:.3f}", ha="right", va="center", fontsize=13)

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=13)
    ax.set_xlim(0.6, 1.0)
    ax.set_xlabel("Coverage (recalibrated)")
    ax.invert_yaxis()
    ax.set_title("Lowest coverage (target 0.90)", fontsize=17)
    label_panel(ax, "d", x=-0.10, y=1.08)


def main() -> None:
    apply_style(paper=True)

    print("[fig08] loading corrected one-sided conformal coverage...")
    fixed_df, recal_df = load_coverage()

    mat_train, fams, dims = coverage_matrix(fixed_df)
    mat_recal, fams_recal, dims_recal = coverage_matrix(recal_df)

    if fams != fams_recal:
        raise RuntimeError(
            "Family ordering differs between corrected fixed/recalibrated artifacts."
        )

    if dims != dims_recal:
        raise RuntimeError(
            "Dimension ordering differs between corrected fixed/recalibrated artifacts."
        )

    fig, axes = new_2x2(figsize=(12.4, 9.3))
    _heatmap_cov(axes[0, 0], mat_train, fams, dims,
                 "Training quantile", "a")
    _heatmap_cov(axes[0, 1], mat_recal, fams, dims,
                 "Recalibrated quantile", "b")
    _heatmap_diff(axes[1, 0], mat_train, mat_recal, fams, dims, "c")
    panel_d_worst_cells(axes[1, 1], recal_df)
    save_fig(fig, "fig08_selection_conditional")


if __name__ == "__main__":
    main()
