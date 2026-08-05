"""Shared bootstrap for the figure scripts: import path, style, output folders."""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
FIGDIR = ROOT / "figures"
TABDIR = ROOT / "results"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

FIGDIR.mkdir(exist_ok=True)
TABDIR.mkdir(exist_ok=True)

# Explosive calibrations are caught and flagged inside the kernel; the
# intermediate overflow warnings they raise are expected and only add noise.
warnings.filterwarnings("ignore", category=RuntimeWarning)

from bh1998.plotting import use_style  # noqa: E402

use_style()


def announce(name: str) -> None:
    print(f"  -> {name}")
