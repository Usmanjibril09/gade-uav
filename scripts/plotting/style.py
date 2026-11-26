"""Shared plotting style for all paper figures.

Q1-journal defaults: Times New Roman, colourblind-safe palette, constrained
layout, PDF+PNG output at 300 dpi. Import `apply_style()` at the top of every
figure script; call `save_fig(fig, name)` to write both formats.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

# Okabe-Ito colourblind-safe palette (8 colours, universally distinguishable).
OKABE_ITO = [
    "#0072B2",  # blue
    "#D55E00",  # vermillion
    "#009E73",  # green
    "#CC79A7",  # reddish purple
    "#F0E442",  # yellow
    "#56B4E9",  # sky blue
    "#E69F00",  # orange
    "#000000",  # black
]

FIG_DIR = Path("artifacts/figures")


def apply_style(*, paper: bool = False) -> None:
    """Set global rcParams. Call once at the start of every figure script."""
    mpl.rcParams.update(
        {
            # Fonts
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 15,
            "axes.titlesize": 16,
            "axes.titleweight": "bold",
            "axes.labelsize": 16,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "legend.fontsize": 14,
            "figure.titlesize": 17,
            # Lines and markers
            "lines.linewidth": 1.8,
            "lines.markersize": 6,
            "axes.linewidth": 1.0,
            # Ticks
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.major.size": 4,
            "ytick.major.size": 4,
            "xtick.minor.size": 2,
            "ytick.minor.size": 2,
            "xtick.top": True,
            "ytick.right": True,
            # Grid
            "axes.grid": True,
            "grid.linestyle": ":",
            "grid.linewidth": 0.6,
            "grid.alpha": 0.5,
            # Legend
            "legend.frameon": True,
            "legend.framealpha": 0.9,
            "legend.edgecolor": "0.8",
            # Save
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.05,
            "pdf.fonttype": 42,  # TrueType, editable in Illustrator
            "ps.fonttype": 42,
        }
    )

    if paper:
        mpl.rcParams.update({
            "font.size": 19,
            "axes.titlesize": 21,
            "axes.labelsize": 21,
            "xtick.labelsize": 18,
            "ytick.labelsize": 18,
            "legend.fontsize": 17,
            "axes.linewidth": 1.15,
            "xtick.top": False,
            "ytick.right": False,
            "grid.alpha": 0.30,
            "grid.color": "#8795A4",
        })


def label_panel(
    ax, letter: str, x: float = -0.08, y: float = 1.04, fontsize: int = 17
) -> None:
    """Add bold (a)/(b)/(c)/(d) label above the axis (outside the plot area).

    Positioning above-left avoids collisions with data (heatmap cells, legends,
    scatter points). Override x/y if a specific panel needs re-placement.
    """
    ax.text(
        x,
        y,
        f"({letter})",
        transform=ax.transAxes,
        fontsize=fontsize,
        fontweight="bold",
        va="bottom",
        ha="left",
    )


def save_fig(fig, name: str, subdir: str | None = None) -> None:
    """Save figure as both PDF (vector) and PNG (300 dpi)."""
    out = FIG_DIR / subdir if subdir else FIG_DIR
    out.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        path = out / f"{name}.{ext}"
        fig.savefig(path)
        print(f"  wrote {path}")
    plt.close(fig)


def new_2x2(figsize: tuple[float, float] = (11.0, 8.5)):
    """Create a 2x2 subplot grid with constrained layout."""
    fig, axes = plt.subplots(2, 2, figsize=figsize, constrained_layout=True)
    return fig, axes
