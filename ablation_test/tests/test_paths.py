from pathlib import Path

from ablation_test.src.paths import (
    BENCHMARKS_ROOT,
    CODE_ROOT,
    RESULTS_ROOT,
    RUNS_ROOT,
    resolve_artifact_path,
)


def test_code_and_results_roots_are_separate():
    assert CODE_ROOT.name == "ablation_test"
    assert BENCHMARKS_ROOT.parent == CODE_ROOT
    assert RESULTS_ROOT != CODE_ROOT
    assert CODE_ROOT not in RESULTS_ROOT.parents
    assert RUNS_ROOT == RESULTS_ROOT / "runs"


def test_default_results_root_is_git_ignored_sibling():
    assert RESULTS_ROOT == Path(__file__).resolve().parents[2] / "ablation_results"


def test_legacy_run_path_is_rebased_after_results_split():
    legacy = "/old/checkout/ablation_test/runs/example/cell/result.json"
    assert resolve_artifact_path(legacy) == RUNS_ROOT / "example/cell/result.json"
