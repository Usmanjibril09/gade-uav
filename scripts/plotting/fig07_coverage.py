"""Figure 7: Corrected one-sided coverage results.

(a) Marginal coverage per dimension: three thin bars per dimension
    (training held-out, post-split with training q, post-split with recalibrated q).
(b) Per-family bw coverage heatmap: training q vs recalibrated q.
(c) Per-family lat coverage heatmap: training q vs recalibrated q.
(d) Quantile ratio per dimension: recalibrated q / training q.

This figure uses ONLY the corrected one-sided conformal artifacts under
artifacts/shield_eval_full/conformal/.
"""

from __future__ import annotations

import sys
from pathlib import Path as _Path

sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))

import json
from pathlib import Path

import numpy as np
import polars as pl
from matplotlib.colors import LinearSegmentedColormap

from scripts.plotting.style import (
    OKABE_ITO,
    apply_style,
    label_panel,
    new_2x2,
    save_fig,
)

REPO = Path(__file__).resolve().parents[2]
CORR = REPO / "artifacts" / "shield_eval_full" / "conformal"

COST_DIMS = ("cpu", "ram", "lat", "bw", "energy", "mission_delay")
TARGET = 0.90


def _coverage_cmap():
    return LinearSegmentedColormap.from_list(
        "cov",
        ["#B2182B", "#F7F7F7", "#2166AC"],
        N=256,
    )


def _family_col(df: pl.DataFrame) -> str:
    for c in ("family", "primary_attack_family", "attack_family"):
        if c in df.columns:
            return c
    raise ValueError(f"Could not locate family column. Columns: {df.columns}")


def _annotate_bars(ax, bars, fmt="{:.3f}", dy=0.008, fontsize=14):
    for bar in bars:
        h = bar.get_height()
        if np.isnan(h):
            continue
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            h + dy,
            fmt.format(h),
            ha="center",
            va="bottom",
            fontsize=fontsize,
        )


def panel_a_marginal(ax, summary: dict) -> None:
    dims = list(COST_DIMS)

    train_cov = [
        float(summary["training_test_coverage_from_saved_report"][d]) for d in dims
    ]
    post_train_q = [
        float(summary["post_split_using_training_q_overall_coverage"][d]) for d in dims
    ]
    post_recal_q = [
        float(summary["post_split_using_recalibrated_q_overall_coverage"][d])
        for d in dims
    ]

    x = np.arange(len(dims))
    w = 0.18  # thin bars

    bars1 = ax.bar(
        x - w,
        train_cov,
        width=w,
        color=OKABE_ITO[0],
        label="Training held-out",
        edgecolor="white",
        lw=0.7,
        alpha=0.92,
    )
    bars2 = ax.bar(
        x,
        post_train_q,
        width=w,
        color=OKABE_ITO[1],
        label="Post-split, fixed q",
        edgecolor="white",
        lw=0.7,
        alpha=0.92,
    )
    bars3 = ax.bar(
        x + w,
        post_recal_q,
        width=w,
        color=OKABE_ITO[2],
        label="Post-split, recal. q",
        edgecolor="white",
        lw=0.7,
        alpha=0.92,
    )

    # Stagger labels vertically because all three coverage values
    # are tightly clustered near 0.90.
    _annotate_bars(ax, bars1, fmt="{:.3f}", dy=0.008, fontsize=13)
    _annotate_bars(ax, bars2, fmt="{:.3f}", dy=0.028, fontsize=13)
    _annotate_bars(ax, bars3, fmt="{:.3f}", dy=0.048, fontsize=13)

    ax.axhline(
        TARGET,
        color="k",
        ls="--",
        lw=1.1,
        alpha=0.75,
        label=f"Target = {TARGET:.2f}",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(dims, rotation=25, ha="right")
    ax.set_ylabel("Coverage")
    ax.set_ylim(0.55, 1.045)
    ax.set_title("Marginal coverage")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(loc="lower right", fontsize=14, ncols=1, framealpha=0.95)
    label_panel(ax, "a")


def _heatmap_panel(
    ax,
    fixed_df: pl.DataFrame,
    recal_df: pl.DataFrame,
    dim: str,
    letter: str,
) -> None:
    fam_col_fixed = _family_col(fixed_df)
    fam_col_recal = _family_col(recal_df)

    cov_col = f"coverage_{dim}"
    if cov_col not in fixed_df.columns or cov_col not in recal_df.columns:
        raise ValueError(
            f"Expected column '{cov_col}' in both per-family tables. "
            f"fixed columns={fixed_df.columns}, recal columns={recal_df.columns}"
        )

    fixed_map = {
        r[fam_col_fixed]: float(r[cov_col])
        for r in fixed_df.select([fam_col_fixed, cov_col]).iter_rows(named=True)
        if r[fam_col_fixed] is not None
    }
    recal_map = {
        r[fam_col_recal]: float(r[cov_col])
        for r in recal_df.select([fam_col_recal, cov_col]).iter_rows(named=True)
        if r[fam_col_recal] is not None
    }

    families = sorted(set(fixed_map) | set(recal_map))
    mat = np.array(
        [
            [fixed_map.get(f, np.nan), recal_map.get(f, np.nan)]
            for f in families
        ],
        dtype=float,
    )

    cmap = _coverage_cmap()
    im = ax.imshow(mat, cmap=cmap, vmin=0.70, vmax=1.00, aspect="auto")

    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat[i, j]
            if np.isnan(v):
                continue
            colour = "white" if (v <= 0.80 or v >= 0.96) else "black"
            ax.text(
                j,
                i,
                f"{v:.3f}",
                ha="center",
                va="center",
                fontsize=15,
                color=colour,
            )

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Training q", "Recal q"], fontsize=16)
    ax.set_yticks(range(len(families)))
    ax.set_yticklabels(families, fontsize=14)
    ax.set_title(f"{dim} coverage by attack family")

    cbar = ax.figure.colorbar(im, ax=ax, shrink=0.82, pad=0.02)
    cbar.set_label("Coverage", fontsize=16)

    label_panel(ax, letter)


def panel_d_quantile_ratio(ax, summary: dict) -> None:
    dims = list(COST_DIMS)
    ratios = [
        float(summary["recalibrated_to_training_quantile_ratio"][d]) for d in dims
    ]

    x = np.arange(len(dims))
    colours = [OKABE_ITO[1] if r > 1.20 else OKABE_ITO[0] for r in ratios]

    bars = ax.bar(
        x,
        ratios,
        width=0.38,  # thin bars
        color=colours,
        edgecolor="white",
        lw=0.7,
        alpha=0.92,
    )

    for bar, r in zip(bars, ratios):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            r + 0.03,
            f"{r:.2f}×",
            ha="center",
            va="bottom",
            fontsize=14,
        )

    ax.axhline(
        1.0,
        color="k",
        ls="--",
        lw=1.0,
        alpha=0.7,
        label="No shift (ratio = 1)",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(dims, rotation=25, ha="right")
    ax.set_ylabel("Recalibrated q / training q")
    ax.set_ylim(0, max(ratios) * 1.22)
    ax.set_title("Conformal quantile ratio (no shift = 1)", fontsize=17)
    ax.grid(True, axis="y", alpha=0.25)
    label_panel(ax, "d")


def main() -> None:
    apply_style(paper=True)

    summary = json.loads((CORR / "summary.json").read_text())

    fixed_per_family = pl.read_parquet(CORR / "fixed_q_per_family.parquet")
    recal_per_family = pl.read_parquet(CORR / "recalibrated_per_family.parquet")

    fig, axes = new_2x2(figsize=(12.4, 9.3))

    panel_a_marginal(axes[0, 0], summary)
    _heatmap_panel(axes[0, 1], fixed_per_family, recal_per_family, "bw", "b")
    _heatmap_panel(axes[1, 0], fixed_per_family, recal_per_family, "lat", "c")
    panel_d_quantile_ratio(axes[1, 1], summary)

    save_fig(fig, "fig07_coverage")


if __name__ == "__main__":
    main()