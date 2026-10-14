import json
from pathlib import Path

from ablation_test.scripts import build_all_models_three_repeat_summary
from ablation_test.scripts import build_manuscript_visualization_package
from ablation_test.scripts import build_manuscript_visualization_revisions


ROOT = Path(__file__).resolve().parents[2]
EXPECTED_MODELS = {
    "Qwen3.6-27B",
    "Qwen3.8-27B",
    "GPT-4o",
    "Claude Sonnet 4.6",
    "Claude Opus 4.6",
}


def test_current_benchmark_config_has_only_publication_models():
    config = json.loads(
        (ROOT / "ablation_test/configs/benchmark.json").read_text(encoding="utf-8")
    )
    configured = {
        bundle["model"] for bundle in config["model_bundles"].values()
    }
    assert configured == {
        "/models/Qwen3.6-27B",
        "/models/Qwen3.8-27B",
        "gpt-4o",
        "claude-sonnet-4-6",
        "claude-opus-4-6",
    }


def test_publication_builders_share_the_same_model_scope():
    assert set(build_all_models_three_repeat_summary.MODEL_ORDER) == EXPECTED_MODELS
    assert set(build_manuscript_visualization_package.MODEL_ORDER) == EXPECTED_MODELS
    assert set(build_manuscript_visualization_revisions.MODEL_ORDER) == EXPECTED_MODELS

