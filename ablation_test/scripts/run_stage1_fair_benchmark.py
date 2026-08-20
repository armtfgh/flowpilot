from __future__ import annotations

import csv
import argparse
import hashlib
import json
import logging
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from ablation_test.src.cases import ROOT, load_cases_from_path, select_cases
from ablation_test.src.providers import endpoint_health
from ablation_test.src.paths import BENCHMARKS_ROOT, STUDIES_ROOT
from ablation_test.src.runner import execute_cell, write_checksums
from ablation_test.src.stage1_oracle import (
    score_scenario,
    validate_oracle_witness,
    write_tree_checksums,
)


DEFAULT_STUDY_DIR = BENCHMARKS_ROOT / "fair_architecture_benchmark_v1_20260730"
DEFAULT_OUTPUT_DIR = STUDIES_ROOT / "fair_architecture_benchmark_v1_20260730"
BASE_CONFIG_PATH = ROOT / "configs" / "benchmark.json"
ORACLE_PATH = ROOT / "src" / "stage1_oracle.py"
LEGACY_SCORER_PATH = ROOT / "src" / "metrics.py"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def _package_versions() -> dict[str, str]:
    output: dict[str, str] = {}
    for name in ("anthropic", "openai", "pydantic", "numpy", "pandas"):
        try:
            output[name] = version(name)
        except PackageNotFoundError:
            output[name] = "not_installed"
    return output


def _git_metadata() -> dict[str, Any]:
    project_root = ROOT.parent

    def run(*args: str) -> str:
        result = subprocess.run(
            ["git", *args],
            cwd=project_root,
            capture_output=True,
            text=True,
            check=False,
        )
        return result.stdout.strip()

    return {
        "commit": run("rev-parse", "HEAD"),
        "branch": run("branch", "--show-current"),
        "status_porcelain": run("status", "--porcelain"),
        "diff_stat": run("diff", "--stat"),
    }


def _environment() -> dict[str, Any]:
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "cwd": os.getcwd(),
        "packages": _package_versions(),
        "credentials": {
            "ANTHROPIC_API_KEY_set": bool(os.getenv("ANTHROPIC_API_KEY")),
            "OPENAI_API_KEY_set": bool(os.getenv("OPENAI_API_KEY")),
        },
    }


def _finish_reasons(run_dir: Path) -> list[str]:
    path = run_dir / "llm_events.jsonl"
    if not path.exists():
        return []
    reasons: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        reason = event.get("finish_reason") or event.get("stop_reason")
        if reason:
            reasons.append(str(reason))
    return reasons


def _artifact_audit(run_dir: Path, variant: str) -> dict[str, Any]:
    common = {
        "metadata.json",
        "input_public.json",
        "input_inventory.json",
        "result.json",
        "metrics.json",
        "oracle_metrics.json",
        "run_summary.json",
        "llm_events.jsonl",
        "checksums.sha256",
    }
    variant_files = (
        {"stage_events.jsonl", "prepared_context.json"}
        if variant == "full"
        else {"prompt.json", "raw_response.json"}
    )
    required = sorted(common | variant_files)
    missing = [name for name in required if not (run_dir / name).is_file()]
    return {
        "required": required,
        "missing": missing,
        "complete": not missing,
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _safe_attempt_dir(study_dir: Path) -> Path:
    for attempt in range(1, 100):
        candidate = study_dir / f"stage1_smoke_run_20260730_attempt{attempt:02d}"
        if not candidate.exists():
            candidate.mkdir(parents=True)
            return candidate
    raise RuntimeError("No available Stage 1 attempt directory")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Stage 1 fair-benchmark gate.")
    parser.add_argument("--study-dir", type=Path, default=DEFAULT_STUDY_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    study_dir = args.study_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = study_dir / "benchmark_config.json"
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    base_config = json.loads(BASE_CONFIG_PATH.read_text(encoding="utf-8"))
    scenario_path = study_dir / config["scenario_file"]
    all_cases = load_cases_from_path(scenario_path)
    smoke_cases = select_cases(all_cases, config["smoke_scenario_ids"])
    bundles = base_config["model_bundles"]
    run_root = _safe_attempt_dir(output_dir)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        handlers=[
            logging.FileHandler(run_root / "run.log", encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    log = logging.getLogger("flowpilot.stage1")

    frozen_dir = run_root / "frozen_sources"
    frozen_dir.mkdir()
    for source in (config_path, scenario_path, ORACLE_PATH, LEGACY_SCORER_PATH):
        shutil.copy2(source, frozen_dir / source.name)

    selected_bundle_ids = sorted(
        {condition["bundle"] for condition in config["conditions"].values()}
    )
    health = {
        bundle_id: endpoint_health(bundles[bundle_id])
        for bundle_id in selected_bundle_ids
    }
    _write_json(run_root / "endpoint_health.json", health)
    _write_json(run_root / "environment.json", _environment())
    _write_json(run_root / "git_state.json", _git_metadata())

    witnesses = [validate_oracle_witness(case) for case in all_cases]
    _write_json(run_root / "oracle_witness_validation.json", witnesses)

    cells: list[dict[str, Any]] = []
    for condition_id, condition in config["conditions"].items():
        bundle_name = condition["bundle"]
        for case in smoke_cases:
            cells.append(
                {
                    "condition_id": condition_id,
                    "condition_label": condition["label"],
                    "variant": condition["variant"],
                    "bundle_name": bundle_name,
                    "bundle": bundles[bundle_name],
                    "case": case,
                    "repeat": 1,
                    "seed": config["random_seed"] + 1,
                    "run_dir": (
                        run_root
                        / "cells"
                        / condition_id
                        / case.scenario_id
                        / "repeat_01"
                    ),
                }
            )

    execution_plan = {
        "schema_version": config["schema_version"],
        "study_id": config["study_id"],
        "stage": 1,
        "cell_count": len(cells),
        "conditions": config["conditions"],
        "model_bundles": {
            bundle_id: bundles[bundle_id] for bundle_id in selected_bundle_ids
        },
        "cases": [case.manifest_payload() for case in smoke_cases],
        "all_scenario_ids": [case.scenario_id for case in all_cases],
        "random_seed": config["random_seed"],
        "temperature": config["temperature"],
        "candidate_budget": config["candidate_budget"],
        "stage_1_gate": config["stage_1_gate"],
        "frozen_files": {
            "benchmark_config.json": _sha256(config_path),
            "protocol_scenarios.json": _sha256(scenario_path),
            "stage1_oracle.py": _sha256(ORACLE_PATH),
            "metrics.py": _sha256(LEGACY_SCORER_PATH),
        },
    }
    _write_json(run_root / "execution_plan.json", execution_plan)

    rows: list[dict[str, Any]] = []
    for index, cell in enumerate(cells, start=1):
        case = cell["case"]
        log.info(
            "cell %d/%d condition=%s scenario=%s",
            index,
            len(cells),
            cell["condition_id"],
            case.scenario_id,
        )
        summary = execute_cell(
            case=case,
            variant=cell["variant"],
            bundle_name=cell["bundle_name"],
            bundle=cell["bundle"],
            run_dir=cell["run_dir"],
            candidate_budget=config["candidate_budget"],
            temperature=config["temperature"],
            seed=cell["seed"],
        )
        result_path = cell["run_dir"] / "result.json"
        oracle_metrics: dict[str, Any] = {}
        if result_path.exists():
            result = json.loads(result_path.read_text(encoding="utf-8"))
            oracle_metrics = score_scenario(case, result)
            _write_json(cell["run_dir"] / "oracle_metrics.json", oracle_metrics)
            write_checksums(cell["run_dir"])

        reasons = _finish_reasons(cell["run_dir"])
        audit = _artifact_audit(cell["run_dir"], cell["variant"])
        _write_json(cell["run_dir"] / "artifact_audit.json", audit)
        write_checksums(cell["run_dir"])
        rows.append(
            {
                "condition_id": cell["condition_id"],
                "condition_label": cell["condition_label"],
                "variant": cell["variant"],
                "bundle": cell["bundle_name"],
                "provider": cell["bundle"]["provider"],
                "model": cell["bundle"]["model"],
                "scenario_id": case.scenario_id,
                "pair_id": case.pair_id,
                "scenario_kind": case.scenario_kind,
                "repeat": 1,
                "status": summary.get("status", "unknown"),
                "design_input_sha256": case.design_input_sha256,
                "expected_disposition": case.expected_disposition,
                "reported_disposition": oracle_metrics.get("reported_disposition", ""),
                "disposition_correct": oracle_metrics.get("disposition_correct", False),
                "critical_engineering_pass": oracle_metrics.get(
                    "critical_engineering_pass", False
                ),
                "critical_violation_count": oracle_metrics.get(
                    "critical_violation_count", ""
                ),
                "finish_reasons": "|".join(reasons),
                "artifact_complete": audit["complete"],
                "runtime_s": summary.get(
                    "runtime_total_s", summary.get("runtime_s", 0)
                ),
                "llm_call_count": summary.get("llm_call_count", 0),
                "total_tokens": (summary.get("token_totals") or {}).get(
                    "total_tokens", 0
                ),
                "run_dir": str(cell["run_dir"].resolve()),
            }
        )
        _write_csv(run_root / "run_manifest.csv", rows)

    input_hash_groups: dict[str, list[str]] = {}
    for row in rows:
        input_hash_groups.setdefault(row["scenario_id"], []).append(
            row["design_input_sha256"]
        )
    identical_inputs = all(
        len(set(hashes)) == 1 and len(hashes) == len(config["conditions"])
        for hashes in input_hash_groups.values()
    )
    truncation_markers = {"length", "max_tokens", "max_output_tokens"}
    systemic_truncation: dict[str, bool] = {}
    for condition_id in config["conditions"]:
        condition_rows = [row for row in rows if row["condition_id"] == condition_id]
        systemic_truncation[condition_id] = bool(condition_rows) and all(
            any(
                marker in row["finish_reasons"].lower()
                for marker in truncation_markers
            )
            for row in condition_rows
        )

    endpoint_pass = all(
        item.get("reachable") and item.get("model_advertised")
        for item in health.values()
    )
    recorded_pass = len(rows) == config["stage_1_gate"]["required_recorded_cells"]
    status_pass = all(row["status"] == "completed" for row in rows)
    artifacts_pass = all(row["artifact_complete"] for row in rows)
    witness_pass_rate = sum(item["passed"] for item in witnesses) / len(witnesses)
    gate_checks = {
        "providers_and_models_available": endpoint_pass,
        "all_cells_recorded": recorded_pass,
        "all_cells_completed_without_adapter_error": status_pass,
        "no_systemic_truncation": not any(systemic_truncation.values()),
        "complete_artifact_sets": artifacts_pass,
        "identical_input_hashes_per_scenario": identical_inputs,
        "oracle_witness_pass_rate_100pct": witness_pass_rate == 1.0,
    }
    gate = {
        "study_id": config["study_id"],
        "stage": 1,
        "passed": all(gate_checks.values()),
        "checks": gate_checks,
        "endpoint_health": health,
        "systemic_truncation": systemic_truncation,
        "oracle_witness_pass_rate": witness_pass_rate,
        "model_accuracy_not_a_gate": True,
        "observed_model_disposition_accuracy": (
            sum(bool(row["disposition_correct"]) for row in rows) / len(rows)
        ),
        "next_stage": "STAGE_2_CLEARED" if all(gate_checks.values()) else "HOLD",
    }
    _write_json(run_root / "gate_results.json", gate)

    report = f"""# Stage 1 Gate Report

Study: `{config["study_id"]}`

Run directory: `{run_root.name}`

## Decision

**{gate["next_stage"]}**

Stage 1 gate passed: `{gate["passed"]}`

## Gate checks

""" + "\n".join(
        f"- {name}: `{'PASS' if passed else 'FAIL'}`"
        for name, passed in gate_checks.items()
    ) + f"""

## Smoke observations

- Planned and recorded cells: `{len(cells)}`
- Oracle witness pass rate: `{witness_pass_rate:.1%}`
- Observed disposition accuracy: `{gate["observed_model_disposition_accuracy"]:.1%}`
- Model accuracy was not used as a stage gate.

## Scope

This smoke run uses one feasible/infeasible photochemical pair across four
conditions. It validates adapters, frozen inputs, deterministic scoring, and
artifact capture. It does not estimate comparative model performance.
"""
    (run_root / "STAGE1_REPORT.md").write_text(report, encoding="utf-8")
    write_tree_checksums(run_root)
    print(json.dumps(gate, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
