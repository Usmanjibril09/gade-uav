"""Plot the audited five-seed matched mission safety replay."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from matplotlib.ticker import StrMethodFormatter

from scripts.plotting.style import apply_style, label_panel, save_fig

REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO / "artifacts/mission_safety_compare/reference/episodes.csv"
ARMS = ("generator_shield", "m3_shield", "simulator_bypass")
LABELS = ("Declared", "M3", "Bypass")
COLORS = ("#0072B2", "#009E73", "#D55E00")


def main() -> None:
    apply_style(paper=True)
    results = pl.read_csv(RESULTS)
    groups = [results.filter(pl.col("arm") == arm).sort("seed") for arm in ARMS]
    assert (
        groups[0]["seed"].to_list()
        == groups[1]["seed"].to_list()
        == groups[2]["seed"].to_list()
    )
    assert (
        groups[0]["candidates"].to_list()
        == groups[1]["candidates"].to_list()
        == groups[2]["candidates"].to_list()
    )
    x = np.arange(len(ARMS))
    fig, axes = plt.subplots(2, 2, figsize=(11.8, 6.1), layout="constrained")

    for ax, letter, metric, title in zip(
        axes.flat,
        "abc",
        ("admitted", "deterministic_violations", "budget_overruns"),
        ("Admissions", "Schema violations", "Budget overruns"),
    ):
        values = [group[metric].sum() for group in groups]
        bars = ax.bar(x, values, width=0.54, color=COLORS, zorder=3)
        ax.bar_label(
            bars, labels=[f"{value:,}" for value in values], padding=4, fontsize=16
        )
        ax.set_ylim(0, max(values) * 1.30)
        ax.yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
        ax.set_title(title)
        label_panel(ax, letter, x=0.0, y=1.01)

    ax = axes[1, 1]
    mitigation = (
        np.array([group["modeled_mitigation"].to_list() for group in groups]) * 100
    )
    for seed_values in mitigation.T:
        ax.plot(x, seed_values, "o-", color="#657281", alpha=0.65, markersize=5)
    ax.scatter(
        x,
        mitigation.mean(axis=1),
        marker="D",
        s=75,
        color=COLORS,
        edgecolor="black",
        linewidth=0.7,
        zorder=4,
    )
    ax.set_ylim(-0.35, mitigation.max() * 1.10)
    ax.set_ylabel("Mitigation (%)")
    ax.set_title("Modeled mitigation")
    label_panel(ax, "d", x=0.0, y=1.01)

    for ax in axes.flat:
        ax.set_xticks(x, LABELS)
        ax.set_xlim(-0.52, 2.52)
        ax.grid(False, axis="x")
        ax.set_axisbelow(True)
        ax.spines[["right", "top"]].set_visible(False)
    save_fig(fig, "fig11_mission_safety")


if __name__ == "__main__":
    main()
