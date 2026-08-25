"""Utilities for extending a frozen one-round campaign without rerunning repeat 1."""

from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def _cell_key(row: dict[str, Any]) -> tuple[str, str, str]:
    return row["case_label"], row["condition_id"], row["repeat_id"]


def freeze_wrapper_sources(target: Path, *sources: Path) -> None:
    destination = target.resolve() / "frozen/source_code"
    destination.mkdir(parents=True, exist_ok=True)
    for source in sources:
        source = source.resolve()
        shutil.copy2(source, destination / source.name)


def bootstrap_repeat_one(source: Path, target: Path) -> dict[str, Any]:
    """Copy an accepted repeat 1 into a new three-repeat campaign with provenance."""

    source = source.resolve()
    target = target.resolve()
    marker = target / "frozen/reused_repeat1_provenance.json"
    if marker.is_file():
        return _read_json(marker)

    source_plan_path = source / "frozen/execution_plan.json"
    target_plan_path = target / "frozen/execution_plan.json"
    if not source_plan_path.is_file() or not target_plan_path.is_file():
        raise RuntimeError("Both campaigns must be initialized before repeat-1 reuse")

    source_rows = {
        _cell_key(row): row
        for row in _read_json(source_plan_path)["cells"]
        if row["repeat_id"] == "repeat_01"
    }
    target_rows = {
        _cell_key(row): row
        for row in _read_json(target_plan_path)["cells"]
        if row["repeat_id"] == "repeat_01"
    }
    if set(source_rows) != set(target_rows):
        raise RuntimeError("Repeat-1 source and target cells do not match")

    immutable_fields = (
        "case_id",
        "generator_family",
        "generator_provider_family",
        "architecture_key",
        "architecture",
        "variant",
        "model_id",
        "design_input_sha256",
        "seed",
    )
    for key in source_rows:
        for field in immutable_fields:
            if source_rows[key].get(field) != target_rows[key].get(field):
                raise RuntimeError(f"Repeat-1 mismatch for {key}: {field}")

    copied_generation: list[str] = []
    for case_label, condition_id, repeat_id in sorted(source_rows):
        case_dir = case_label.lower().replace(" ", "_")
        source_run = source / "generation" / case_dir / condition_id / repeat_id
        target_run = target / "generation" / case_dir / condition_id / repeat_id
        if not (source_run / "result.json").is_file():
            raise RuntimeError(f"Missing accepted repeat-1 result: {source_run}")
        if not target_run.exists():
            target_run.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(source_run, target_run)
        copied_generation.append(str(target_run.relative_to(target)))

    copied_judgments = 0
    source_judgments = source / "judgments"
    target_judgments = target / "judgments"
    if source_judgments.is_dir():
        for judge_dir in source_judgments.iterdir():
            if not judge_dir.is_dir():
                continue
            for candidate_dir in judge_dir.iterdir():
                if not candidate_dir.is_dir():
                    continue
                destination = target_judgments / judge_dir.name / candidate_dir.name
                if not destination.exists():
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copytree(candidate_dir, destination)
                copied_judgments += 1

    source_files = sorted(path for path in source.rglob("*") if path.is_file())
    provenance = {
        "schema_version": "flowpilot_repeat_reuse_v1.0",
        "created_at": datetime.now().astimezone().isoformat(),
        "source_campaign": str(source),
        "source_campaign_manifest_sha256": _sha256(source / "frozen/campaign_manifest.json"),
        "source_execution_plan_sha256": _sha256(source_plan_path),
        "source_file_count": len(source_files),
        "source_tree_digest": hashlib.sha256(
            "\n".join(
                f"{path.relative_to(source)} {_sha256(path)}" for path in source_files
            ).encode("utf-8")
        ).hexdigest(),
        "reused_repeat_id": "repeat_01",
        "copied_generation_cells": copied_generation,
        "copied_judgment_directories": copied_judgments,
        "reuse_policy": (
            "Accepted repeat 1 was copied before observing repeats 2 and 3. "
            "Only missing frozen cells may be generated in this campaign."
        ),
    }
    _write_json(marker, provenance)
    return provenance
