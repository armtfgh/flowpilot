import csv
import json

import pytest

from ablation_test.scripts.build_qwen_gpt_architecture_comparison import (
    validate_cross_model_identity,
)


def _write_package(root, digest: str = "same-hash") -> None:
    (root / "frozen").mkdir(parents=True)
    (root / "tables").mkdir()
    (root / "frozen" / "scoring_spec.json").write_text(
        '{"version": 1}\n', encoding="utf-8"
    )
    (root / "summary.json").write_text(
        json.dumps(
            {
                "input_identity_passed": True,
                "cell_count": 60,
                "paired_feasible_count": 15,
            }
        ),
        encoding="utf-8",
    )
    with (root / "tables" / "cell_level_scores.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "architecture",
                "scenario_id",
                "repeat",
                "design_input_sha256",
            ),
        )
        writer.writeheader()
        for architecture in ("one_shot", "flowpilot"):
            writer.writerow(
                {
                    "architecture": architecture,
                    "scenario_id": "case_feasible",
                    "repeat": 1,
                    "design_input_sha256": digest,
                }
            )


def test_cross_model_identity_accepts_matched_packages(tmp_path) -> None:
    qwen = tmp_path / "qwen"
    gpt = tmp_path / "gpt"
    _write_package(qwen)
    _write_package(gpt)

    audit = validate_cross_model_identity(qwen, gpt)

    assert audit["passed"]
    assert audit["checks"]["cross_model_input_hashes_equal"]


def test_cross_model_identity_rejects_different_inputs(tmp_path) -> None:
    qwen = tmp_path / "qwen"
    gpt = tmp_path / "gpt"
    _write_package(qwen, "qwen-hash")
    _write_package(gpt, "gpt-hash")

    with pytest.raises(ValueError, match="cross_model_input_hashes_equal"):
        validate_cross_model_identity(qwen, gpt)
