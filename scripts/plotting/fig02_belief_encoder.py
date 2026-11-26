"""Figure 2: Belief encoder performance.

(a) Confusion matrix on test set (test_all).
(b) Per-class F1 bar chart with test_all vs test_real_only comparison.
(c) LOPO per-fold macro-F1 and held-class F1 (scatter with means).
(d) Validation macro-F1 learning curves per LOPO fold.
"""

from __future__ import annotations

import sys
from pathlib import Path as _Path

sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))

import json
from pathlib import Path

import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from scripts.plotting.style import (
    OKABE_ITO,
    apply_style,
    label_panel,
    new_2x2,
    save_fig,
)

REPO = Path(__file__).resolve().parents[2]


def panel_a_confusion(ax, report: dict) -> None:
    cm = np.array(report["test_all"]["confusion_matrix"], dtype=float)
    labels = report["test_all"]["labels_order"]
    cm_norm = cm / np.clip(cm.sum(axis=1, keepdims=True), 1, None)

    cmap = LinearSegmentedColormap.from_list("blues", ["#F7FBFF", OKABE_ITO[0]])
    im = ax.imshow(cm_norm, cmap=cmap, vmin=0, vmax=1, aspect="equal")

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            v = cm_norm[i, j]
            colour = "white" if v > 0.55 else "black"
            ax.text(
                j, i, f"{v:.2f}", ha="center", va="center", fontsize=16, color=colour
            )

    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    short = [l.replace("MAVLink_", "").replace("_Flooding", "_Fl") for l in labels]
    ax.set_xticklabels(short, rotation=30, ha="right")
    ax.set_yticklabels(short)
    ax.tick_params(axis="both", length=0)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    cbar = ax.figure.colorbar(im, ax=ax, shrink=0.75, pad=0.02)
    cbar.set_label("Fraction of true class", fontsize=15)
    ax.set_title("Confusion matrix")
    label_panel(ax, "a")


def panel_b_per_class_f1(ax, report: dict) -> None:
    labels = report["test_all"]["labels_order"]
    f1_all = [report["test_all"]["per_class_f1"][l] for l in labels]
    f1_real = [report["test_real_only"]["per_class_f1"][l] for l in labels]

    x = np.arange(len(labels))
    w = 0.38
    ax.bar(
        x - w / 2,
        f1_all,
        w,
        color=OKABE_ITO[0],
        label="All test",
        edgecolor="white",
        lw=0.6,
    )
    ax.bar(
        x + w / 2,
        f1_real,
        w,
        color=OKABE_ITO[1],
        label="Real only",
        edgecolor="white",
        lw=0.6,
    )

    short = [l.replace("MAVLink_", "").replace("_Flooding", "_Fl") for l in labels]
    ax.set_xticks(x)
    ax.set_xticklabels(short, rotation=30, ha="right")
    ax.set_ylabel("F1 score")
    ax.set_ylim(0, 1.05)
    ax.axhline(1.0, color="k", ls=":", lw=0.6, alpha=0.5)
    ax.set_title("Per-class F1")
    ax.legend(loc="lower left")
    label_panel(ax, "b")


def panel_c_lopo(ax, summary: dict) -> None:
    folds = list(summary["test_macro_f1_per_fold"].keys())
    macro = np.array([summary["test_macro_f1_per_fold"][f] for f in folds])
    held = np.array([summary["held_class_f1_per_fold"][f] for f in folds])
    x = np.arange(len(folds))

    ax.scatter(
        x - 0.08,
        macro,
        s=90,
        color=OKABE_ITO[0],
        zorder=3,
        label="Test macro-F1",
        edgecolor="white",
        lw=1.0,
    )
    ax.scatter(
        x + 0.08,
        held,
        s=90,
        color=OKABE_ITO[3],
        zorder=3,
        marker="D",
        label="Held-class F1",
        edgecolor="white",
        lw=1.0,
    )

    ax.axhline(
        summary["test_macro_f1_mean"], color=OKABE_ITO[0], ls="--", lw=1.2, alpha=0.7
    )
    ax.axhline(
        summary["held_class_f1_mean"], color=OKABE_ITO[3], ls="--", lw=1.2, alpha=0.7
    )
    ax.fill_between(
        [-0.5, len(folds) - 0.5],
        summary["test_macro_f1_mean"] - summary["test_macro_f1_std"],
        summary["test_macro_f1_mean"] + summary["test_macro_f1_std"],
        color=OKABE_ITO[0],
        alpha=0.12,
    )
    ax.fill_between(
        [-0.5, len(folds) - 0.5],
        summary["held_class_f1_mean"] - summary["held_class_f1_std"],
        summary["held_class_f1_mean"] + summary["held_class_f1_std"],
        color=OKABE_ITO[3],
        alpha=0.12,
    )

    short = [f.replace("MAVLink_", "").replace("_Flooding", "_Fl") for f in folds]
    ax.set_xticks(x)
    ax.set_xticklabels(short, rotation=30, ha="right")
    ax.set_xlim(-0.5, len(folds) - 0.5)
    ax.set_ylim(-0.05, 1.05)
    ax.set_ylabel("F1 score")
    ax.set_title("Leave-one-PCAP-out F1")

    ax.text(
        0.02,
        0.96,
        (
            f"Mean macro-F1 = {summary['test_macro_f1_mean']:.3f}\n"
            f"Mean held-class F1 = {summary['held_class_f1_mean']:.3f}"
        ),
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=14,
        bbox=dict(
            boxstyle="round,pad=0.30",
            facecolor="white",
            edgecolor="0.70",
            alpha=0.92,
        ),
    )

    ax.legend(loc="center right")
    label_panel(ax, "c")


def panel_d_learning_curves(ax, lopo_dir: Path) -> None:
    fold_files = sorted(lopo_dir.glob("fold_*_report.json"))
    for i, fp in enumerate(fold_files):
        report = json.loads(fp.read_text())
        hist = report["history"]
        epochs = [h["epoch"] for h in hist]
        f1 = [h["val_macro_f1"] for h in hist]
        label = report["held_class"].replace("MAVLink_", "").replace("_Flooding", "_Fl")
        ax.plot(
            epochs,
            f1,
            "o-",
            color=OKABE_ITO[i % len(OKABE_ITO)],
            label=label,
            lw=1.6,
            ms=5,
            alpha=0.9,
        )

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation macro-F1")
    ax.set_ylim(0, 1.05)
    ax.set_title("LOPO validation F1")
    ax.legend(loc="lower right", fontsize=16)
    label_panel(ax, "d")


def main() -> None:
    apply_style(paper=True)
    report = json.loads(
        (REPO / "artifacts" / "belief_encoder" / "test_report.json").read_text()
    )
    summary = json.loads(
        (REPO / "artifacts" / "lopo_belief" / "summary.json").read_text()
    )
    lopo_dir = REPO / "artifacts" / "lopo_belief"

    fig, axes = new_2x2(figsize=(12.0, 9.0))
    panel_a_confusion(axes[0, 0], report)
    panel_b_per_class_f1(axes[0, 1], report)
    panel_c_lopo(axes[1, 0], summary)
    panel_d_learning_curves(axes[1, 1], lopo_dir)

    save_fig(fig, "fig02_belief_encoder")


if __name__ == "__main__":
    main()
