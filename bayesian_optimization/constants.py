"""Path constants."""

from __future__ import annotations

from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent.resolve()
DATA_DIR = ROOT_DIR / "data"
PLOTS_DIR = ROOT_DIR / "plots"
RESULTS_DIR = ROOT_DIR / "results"