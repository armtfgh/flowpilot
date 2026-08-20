"""Canonical paths for ablation source files and generated artifacts."""

from __future__ import annotations

import os
from pathlib import Path


CODE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = CODE_ROOT.parent
BENCHMARKS_ROOT = CODE_ROOT / "benchmarks"


def _results_root() -> Path:
    configured = os.getenv("FLOWPILOT_ABLATION_RESULTS_DIR")
    if configured:
        return Path(configured).expanduser().resolve()
    return PROJECT_ROOT / "ablation_results"


RESULTS_ROOT = _results_root()
RUNS_ROOT = RESULTS_ROOT / "runs"
STUDIES_ROOT = RESULTS_ROOT / "studies"
FIGURES_ROOT = RESULTS_ROOT / "figures"
TABLES_ROOT = RESULTS_ROOT / "tables"
REPORTS_ROOT = RESULTS_ROOT / "reports"
EXPERT_SCORING_ROOT = RESULTS_ROOT / "expert_scoring"
PACKAGES_ROOT = RESULTS_ROOT / "packages"


_LEGACY_RESULT_DIRS = {
    "runs": RUNS_ROOT,
    "studies": STUDIES_ROOT,
    "figures": FIGURES_ROOT,
    "tables": TABLES_ROOT,
    "reports": REPORTS_ROOT,
    "expert_scoring": EXPERT_SCORING_ROOT,
}


def resolve_artifact_path(value: str | Path) -> Path:
    """Resolve a saved path, rebasing paths from the pre-split layout."""

    candidate = Path(value).expanduser()
    if candidate.exists():
        return candidate.resolve()

    normalized = candidate.as_posix()
    for legacy_name, current_root in _LEGACY_RESULT_DIRS.items():
        marker = f"ablation_test/{legacy_name}/"
        if marker in normalized:
            relative = normalized.split(marker, 1)[1]
            return current_root / relative
    return candidate
