"""
plotting.py
===========

A single visual language for every figure in the repository.

The rules are deliberately narrow: one serif family, one accent colour, muted
grids behind the data, no chartjunk, and a fixed colour per trader type so that
"green" always means adapters whether the reader is looking at a time series, a
phase plot or a bar chart. Nothing here computes anything -- all quantities
arrive pre-computed from :mod:`bh1998.analysis`.

Author: Kamal El Halfaoui
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

__all__ = [
    "use_style",
    "ACCENT",
    "INK",
    "MUTED",
    "save",
    "label_panel",
    "legend_below",
    "shade_regimes",
]

# --- palette --------------------------------------------------------------

ACCENT = "#1b3a6b"   # navy, used for the primary series
INK = "#1c1c1c"      # text
MUTED = "#8a8a8a"    # secondary lines, annotations
HILITE = "#c0392b"   # emphasis / warning
GRIDC = "#d9d9d9"


def use_style() -> None:
    """Install the repository-wide matplotlib style."""
    mpl.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 200,
            "savefig.bbox": "tight",
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "font.family": "serif",
            "font.serif": ["DejaVu Serif", "Liberation Serif", "STIXGeneral"],
            "mathtext.fontset": "stix",
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
            "axes.titlepad": 8,
            "axes.labelsize": 10,
            "axes.edgecolor": "#4a4a4a",
            "axes.linewidth": 0.8,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": GRIDC,
            "grid.linewidth": 0.6,
            "grid.alpha": 0.9,
            "legend.frameon": False,
            "legend.fontsize": 9,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "lines.linewidth": 1.3,
            "lines.solid_capstyle": "round",
            "text.color": INK,
            "axes.labelcolor": INK,
            "xtick.color": "#4a4a4a",
            "ytick.color": "#4a4a4a",
        }
    )


# --- helpers --------------------------------------------------------------


def save(fig, path: str | Path, close: bool = True) -> Path:
    """Write a figure to ``path``, creating parent directories as needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    if close:
        plt.close(fig)
    return path


def label_panel(ax, text: str, dx: float = -0.06, dy: float = 1.04) -> None:
    """Put a bold panel label such as ``(a)`` outside the top-left corner."""
    ax.text(
        dx, dy, text, transform=ax.transAxes, fontweight="bold",
        fontsize=10, va="bottom", ha="left", color=INK,
    )


def legend_below(fig, handles, labels, ncol: int = 6, y: float = -0.01) -> None:
    """A single shared legend under the whole figure."""
    fig.legend(
        handles, labels, loc="lower center", ncol=ncol,
        bbox_to_anchor=(0.5, y), frameon=False,
    )


def shade_regimes(ax, bands: Sequence[tuple[float, float, str, str]]) -> None:
    """Shade parameter bands on a bifurcation-style axis.

    Parameters
    ----------
    bands : sequence of ``(start, stop, colour, label)``
    """
    for start, stop, colour, label in bands:
        ax.axvspan(start, stop, color=colour, alpha=0.13, lw=0, zorder=0)
        ax.text(
            0.5 * (start + stop), 0.965, label, transform=ax.get_xaxis_transform(),
            ha="center", va="top", fontsize=8.5, color="#555555",
        )
