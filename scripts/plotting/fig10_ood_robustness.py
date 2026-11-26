"""Figure 10: Resource-profile shift robustness.

(a) Admission rate under nominal and out-of-support resource profiles.
(b) Attack Mitigation Rate (AMR) under the same conditions.

The figure compares the generator-based and M3-based shield pipelines.
"""

from __future__ import annotations

import sys
from pathlib import Path as _Path

sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))

import numpy as np
import matplotlib.pyplot as plt

from scripts.plotting.style import (
    OKABE_ITO,
    apply_style,
    label_panel,
    save_fig,
)


def main() -> None:
    apply_style(paper=True)

    print("[fig10] plotting resource-profile shift robustness...")

    methods = ["Generator", "M3"]

    # ---------------------------------------------------------
    # Locked nominal and OOD results
    # ---------------------------------------------------------
    nominal_admission = np.array([66.67, 66.67])
    ood_admission     = np.array([51.56,  9.72])

    nominal_amr       = np.array([0.290, 0.290])
    nominal_amr_std   = np.array([0.046, 0.046])
    ood_amr           = np.array([0.218, 0.032])
    ood_amr_std       = np.array([0.027, 0.022])

    x = np.arange(len(methods))
    width = 0.25

    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.8))

    nominal_color = OKABE_ITO[2]  # bluish green
    ood_color     = OKABE_ITO[5]  # reddish purple

    # =========================================================
    # (a) Admission rate
    # =========================================================
    ax = axes[0]

    bars_nom = ax.bar(
        x - width / 2, nominal_admission, width,
        label="Nominal", color=nominal_color,
        edgecolor="white", lw=0.6, alpha=0.9,
    )
    bars_ood = ax.bar(
        x + width / 2, ood_admission, width,
        label="Resource shift", color=ood_color,
        edgecolor="white", lw=0.6, alpha=0.9,
    )

    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.set_ylabel("Admission rate (%)")
    ax.set_ylim(0, 95)
    ax.set_title("Admission rate")
    ax.legend(loc="upper center", frameon=False, ncol=2)
    ax.grid(axis="y", alpha=0.3, ls="--", lw=0.6)
    ax.set_axisbelow(True)

    for bar in list(bars_nom) + list(bars_ood):
        v = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            v + 1.2, f"{v:.2f}",
            ha="center", va="bottom", fontsize=16,
        )

    label_panel(ax, "a")

    # =========================================================
    # (b) Attack Mitigation Rate
    # =========================================================
    ax = axes[1]

    bars_nom = ax.bar(
        x - width / 2, nominal_amr, width,
        yerr=nominal_amr_std, capsize=3,
        label="Nominal", color=nominal_color,
        edgecolor="white", lw=0.6, alpha=0.9,
        error_kw=dict(lw=1.0, ecolor="black"),
    )
    bars_ood = ax.bar(
        x + width / 2, ood_amr, width,
        yerr=ood_amr_std, capsize=3,
        label="Resource shift", color=ood_color,
        edgecolor="white", lw=0.6, alpha=0.9,
        error_kw=dict(lw=1.0, ecolor="black"),
    )

    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.set_ylabel("Attack Mitigation Rate")
    ax.set_ylim(0, 0.50)
    ax.set_title("Attack mitigation")
    ax.legend(loc="upper center", frameon=False, ncol=2)
    ax.grid(axis="y", alpha=0.3, ls="--", lw=0.6)
    ax.set_axisbelow(True)

    for bar, err in zip(list(bars_nom) + list(bars_ood),
                       list(nominal_amr_std) + list(ood_amr_std)):
        v = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            v + err + 0.015, f"{v:.3f}",
            ha="center", va="bottom", fontsize=16,
        )

    label_panel(ax, "b")

    fig.tight_layout()
    save_fig(fig, "fig10_ood_robustness")


if __name__ == "__main__":
    main()
