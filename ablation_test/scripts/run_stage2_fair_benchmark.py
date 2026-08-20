from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import os
import platform
import shutil
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from ablation_test.src.cases import ROOT, load_cases_from_path, select_cases
from ablation_test.src.providers import endpoint_health
from ablation_test.src.paths import (
    BENCHMARKS_ROOT,
    STUDIES_ROOT,
    resolve_artifact_path,
)
from ablation_test.src.runner import execute_cell, write_checksums
from ablation_test.src.stage1_oracle import score_scenario, write_tree_checksums


DEFAULT_STUDY_DIR = BENCHMARKS_ROOT / "fair_architecture_benchmark_v1_1_20260730"
DEFAULT_OUTPUT_DIR = STUDIES_ROOT / "fair_architecture_benchmark_v1_1_20260730"
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


def _append_jsonl(path: Path, payload: Any) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


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


def _read_rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    rows = sorted(
        rows,
        key=lambda row: (
            row["condition_id"],
            row["scenario_id"],
            int(row["repeat"]),
        ),
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _cell_key(row: dict[str, Any]) -> tuple[str, str, int]:
    return (
        str(row["condition_id"]),
        str(row["scenario_id"]),
        int(row["repeat"]),
    )


def _finish_reasons(run_dir: Path) -> list[str]:
    path = run_dir / "llm_events.jsonl"
    if not path.is_file():
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


def _artifact_audit(run_dir: Path, variant: str, status: str) -> dict[str, Any]:
    if status != "completed":
        required = {"metadata.json", "error.json", "run_summary.json", "checksums.sha256"}
    else:
        required = {
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
        required |= (
            {"stage_events.jsonl", "prepared_context.json"}
            if variant == "full"
            else {"prompt.json", "raw_response.json"}
        )
    missing = sorted(name for name in required if not (run_dir / name).is_file())
    return {
        "required": sorted(required),
        "missing": missing,
        "complete": not missing,
    }


def _run_cell(payload: dict[str, Any]) -> dict[str, Any]:
    case = payload["case"]
    summary = execute_cell(
        case=case,
        variant=payload["variant"],
        bundle_name=payload["bundle_name"],
        bundle=payload["bundle"],
        run_dir=payload["run_dir"],
        candidate_budget=payload["candidate_budget"],
        temperature=payload["temperature"],
        seed=payload["seed"],
    )
    status = str(summary.get("status", "unknown"))
    result_path = payload["run_dir"] / "result.json"
    oracle: dict[str, Any] = {}
    if result_path.is_file():
        result = json.loads(result_path.read_text(encoding="utf-8"))
        oracle = score_scenario(case, result)
        _write_json(payload["run_dir"] / "oracle_metrics.json", oracle)
        write_checksums(payload["run_dir"])

    audit = _artifact_audit(payload["run_dir"], payload["variant"], status)
    _write_json(payload["run_dir"] / "artifact_audit.json", audit)
    write_checksums(payload["run_dir"])
    reasons = _finish_reasons(payload["run_dir"])
    return {
        "condition_id": payload["condition_id"],
        "condition_label": payload["condition_label"],
        "variant": payload["variant"],
        "bundle": payload["bundle_name"],
        "provider": payload["bundle"]["provider"],
        "model": payload["bundle"]["model"],
        "scenario_id": case.scenario_id,
        "pair_id": case.pair_id,
        "scenario_kind": case.scenario_kind,
        "repeat": payload["repeat"],
        "seed": payload["seed"],
        "status": status,
        "design_input_sha256": case.design_input_sha256,
        "expected_disposition": case.expected_disposition,
        "reported_disposition": oracle.get("reported_disposition", ""),
        "disposition_correct": oracle.get("disposition_correct", ""),
        "critical_engineering_pass": oracle.get("critical_engineering_pass", ""),
        "critical_violation_count": oracle.get("critical_violation_count", ""),
        "finish_reasons": "|".join(reasons),
        "artifact_complete": audit["complete"],
        "runtime_s": summary.get("runtime_total_s", summary.get("runtime_s", 0)),
        "llm_call_count": summary.get("llm_call_count", 0),
        "total_tokens": (summary.get("token_totals") or {}).get("total_tokens", 0),
        "run_dir": str(payload["run_dir"].resolve()),
        "error": summary.get("error", ""),
    }


def _build_input_audit(
    run_root: Path,
    rows: list[dict[str, Any]],
    all_condition_ids: list[str],
) -> dict[str, Any]:
    completed = [row for row in rows if row["status"] == "completed"]
    records: list[dict[str, Any]] = []
    for scenario_id in sorted({row["scenario_id"] for row in completed}):
        scenario_rows = [row for row in completed if row["scenario_id"] == scenario_id]
        public_hashes: list[str] = []
        inventory_hashes: list[str] = []
        scenario_records: list[dict[str, Any]] = []
        for row in scenario_rows:
            run_dir = resolve_artifact_path(row["run_dir"])
            public_path = run_dir / "input_public.json"
            inventory_path = run_dir / "input_inventory.json"
            public_hash = _sha256(public_path)
            inventory_hash = _sha256(inventory_path)
            public_hashes.append(public_hash)
            inventory_hashes.append(inventory_hash)
            scenario_records.append(
                {
                    "condition_id": row["condition_id"],
                    "scenario_id": scenario_id,
                    "repeat": int(row["repeat"]),
                    "input_public_sha256": public_hash,
                    "input_inventory_sha256": inventory_hash,
                }
            )
        public_identical = len(set(public_hashes)) == 1
        inventory_identical = len(set(inventory_hashes)) == 1
        for record in scenario_records:
            record["public_artifact_identical"] = public_identical
            record["inventory_artifact_identical"] = inventory_identical
        records.extend(scenario_records)
    represented = sorted({row["condition_id"] for row in rows})
    return {
        "schema_version": "flowpilot_stage2_input_audit_v1.0",
        "passed_for_recorded_cells": all(
            row["public_artifact_identical"] and row["inventory_artifact_identical"]
            for row in records
        ),
        "represented_conditions": represented,
        "missing_conditions": sorted(set(all_condition_ids) - set(represented)),
        "records": records,
    }


def _summary(
    config: dict[str, Any],
    rows: list[dict[str, Any]],
    health_history: Path,
) -> dict[str, Any]:
    planned = int(config["completion_gate"]["planned_cells"])
    status_counts: dict[str, int] = {}
    condition_counts: dict[str, dict[str, int]] = {}
    for row in rows:
        status = row["status"]
        status_counts[status] = status_counts.get(status, 0) + 1
        bucket = condition_counts.setdefault(row["condition_id"], {})
        bucket[status] = bucket.get(status, 0) + 1
    complete = len(rows) == planned and all(
        str(row["artifact_complete"]).lower() == "true"
        for row in rows
    )
    return {
        "schema_version": config["schema_version"],
        "study_id": config["study_id"],
        "stage": 2,
        "planned_cells": planned,
        "accounted_cells": len(rows),
        "remaining_cells": planned - len(rows),
        "status_counts": status_counts,
        "condition_status_counts": condition_counts,
        "complete": complete,
        "status": "COMPLETE" if complete else "PARTIAL_RESUMABLE",
        "health_history": str(health_history.resolve()),
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the fair matched Stage 2 benchmark.")
    parser.add_argument("--study-dir", type=Path, default=DEFAULT_STUDY_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--condition", action="append", dest="conditions")
    parser.add_argument("--scenario", action="append", dest="scenarios")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument(
        "--output-name",
        default="stage2_matched_run_20260730",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    study_dir = args.study_dir.resolve()
    output_dir = args.output_dir.resolve()
    config_path = study_dir / "stage2_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    base_config = json.loads(BASE_CONFIG_PATH.read_text(encoding="utf-8"))
    scenario_path = study_dir / config["scenario_file"]
    configured_cases = select_cases(
        load_cases_from_path(scenario_path),
        config["scenario_ids"],
    )
    requested_scenarios = args.scenarios or config["scenario_ids"]
    unknown_scenarios = sorted(
        set(requested_scenarios) - {case.scenario_id for case in configured_cases}
    )
    if unknown_scenarios:
        raise KeyError(f"Unknown scenarios: {unknown_scenarios}")
    cases = [
        case for case in configured_cases
        if case.scenario_id in set(requested_scenarios)
    ]
    all_condition_ids = list(config["conditions"])
    condition_ids = args.conditions or all_condition_ids
    unknown = sorted(set(condition_ids) - set(all_condition_ids))
    if unknown:
        raise KeyError(f"Unknown conditions: {unknown}")

    run_root = output_dir / args.output_name
    run_root.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        handlers=[
            logging.FileHandler(run_root / "run.log", encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    log = logging.getLogger("flowpilot.stage2")
    bundles = base_config["model_bundles"]

    frozen_dir = run_root / "frozen_sources"
    frozen_dir.mkdir(exist_ok=True)
    for source in (
        config_path,
        scenario_path,
        ORACLE_PATH,
        LEGACY_SCORER_PATH,
        Path(__file__),
        ROOT / "src" / "runner.py",
        ROOT / "src" / "cases.py",
        ROOT / "src" / "providers.py",
    ):
        destination = frozen_dir / source.name
        if not destination.exists():
            shutil.copy2(source, destination)

    environment_path = run_root / "environment.json"
    git_path = run_root / "git_state.json"
    if not environment_path.exists():
        _write_json(environment_path, _environment())
    if not git_path.exists():
        _write_json(git_path, _git_metadata())

    selected_bundle_ids = sorted(
        {config["conditions"][condition_id]["bundle"] for condition_id in condition_ids}
    )
    health = {
        bundle_id: endpoint_health(bundles[bundle_id])
        for bundle_id in selected_bundle_ids
    }
    health_history = run_root / "endpoint_health_history.jsonl"
    _append_jsonl(
        health_history,
        {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "selected_conditions": condition_ids,
            "selected_scenarios": requested_scenarios,
            "health": health,
        },
    )

    execution_plan_path = run_root / "execution_plan.json"
    if not execution_plan_path.exists():
        execution_plan = {
            "schema_version": config["schema_version"],
            "study_id": config["study_id"],
            "stage": 2,
            "planned_cell_count": (
                len(condition_ids) * len(cases) * int(config["repeats"])
            ),
            "selected_conditions": condition_ids,
            "selected_scenarios": requested_scenarios,
            "conditions": config["conditions"],
            "model_bundles": {
                condition["bundle"]: bundles[condition["bundle"]]
                for condition in config["conditions"].values()
            },
            "cases": [case.manifest_payload() for case in cases],
            "repeats": config["repeats"],
            "random_seed": config["random_seed"],
            "temperature": config["temperature"],
            "candidate_budget": config["candidate_budget"],
            "primary_comparisons": config["primary_comparisons"],
            "completion_gate": config["completion_gate"],
            "frozen_files": {
                "stage2_config.json": _sha256(config_path),
                "protocol_scenarios.json": _sha256(scenario_path),
                "stage1_oracle.py": _sha256(ORACLE_PATH),
                "metrics.py": _sha256(LEGACY_SCORER_PATH),
            },
        }
        _write_json(execution_plan_path, execution_plan)

    manifest_path = run_root / "run_manifest.csv"
    existing_rows = _read_rows(manifest_path)
    by_key = {_cell_key(row): row for row in existing_rows}
    payloads: list[dict[str, Any]] = []
    for condition_id in condition_ids:
        condition = config["conditions"][condition_id]
        bundle_name = condition["bundle"]
        for case in cases:
            for repeat in range(1, int(config["repeats"]) + 1):
                key = (condition_id, case.scenario_id, repeat)
                existing = by_key.get(key)
                if existing:
                    if not args.resume:
                        continue
                    if existing["status"] == "completed":
                        continue
                    if not args.retry_failed:
                        continue
                payloads.append(
                    {
                        "condition_id": condition_id,
                        "condition_label": condition["label"],
                        "variant": condition["variant"],
                        "bundle_name": bundle_name,
                        "bundle": bundles[bundle_name],
                        "case": case,
                        "repeat": repeat,
                        "candidate_budget": config["candidate_budget"],
                        "temperature": config["temperature"],
                        "seed": config["random_seed"] + repeat,
                        "run_dir": (
                            run_root
                            / "cells"
                            / condition_id
                            / case.scenario_id
                            / f"repeat_{repeat:02d}"
                        ),
                    }
                )

    log.info(
        "Stage 2 invocation: selected=%s pending=%d workers=%d",
        condition_ids,
        len(payloads),
        args.workers,
    )
    if args.workers <= 1:
        for index, payload in enumerate(payloads, start=1):
            log.info(
                "cell %d/%d condition=%s scenario=%s repeat=%d",
                index,
                len(payloads),
                payload["condition_id"],
                payload["case"].scenario_id,
                payload["repeat"],
            )
            row = _run_cell(payload)
            by_key[_cell_key(row)] = row
            _write_rows(manifest_path, list(by_key.values()))
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            future_map = {
                executor.submit(_run_cell, payload): payload for payload in payloads
            }
            for index, future in enumerate(as_completed(future_map), start=1):
                payload = future_map[future]
                try:
                    row = future.result()
                except Exception as exc:
                    row = {
                        "condition_id": payload["condition_id"],
                        "condition_label": payload["condition_label"],
                        "variant": payload["variant"],
                        "bundle": payload["bundle_name"],
                        "provider": payload["bundle"]["provider"],
                        "model": payload["bundle"]["model"],
                        "scenario_id": payload["case"].scenario_id,
                        "pair_id": payload["case"].pair_id,
                        "scenario_kind": payload["case"].scenario_kind,
                        "repeat": payload["repeat"],
                        "seed": payload["seed"],
                        "status": "worker_failed",
                        "design_input_sha256": payload["case"].design_input_sha256,
                        "expected_disposition": payload["case"].expected_disposition,
                        "reported_disposition": "",
                        "disposition_correct": "",
                        "critical_engineering_pass": "",
                        "critical_violation_count": "",
                        "finish_reasons": "",
                        "artifact_complete": False,
                        "runtime_s": 0,
                        "llm_call_count": 0,
                        "total_tokens": 0,
                        "run_dir": str(payload["run_dir"].resolve()),
                        "error": str(exc),
                    }
                by_key[_cell_key(row)] = row
                _write_rows(manifest_path, list(by_key.values()))
                log.info(
                    "completed %d/%d condition=%s scenario=%s repeat=%d status=%s",
                    index,
                    len(payloads),
                    row["condition_id"],
                    row["scenario_id"],
                    int(row["repeat"]),
                    row["status"],
                )

    rows = list(by_key.values())
    input_audit = _build_input_audit(run_root, rows, all_condition_ids)
    _write_json(run_root / "input_artifact_audit.json", input_audit)
    summary = _summary(config, rows, health_history)
    _write_json(run_root / "stage2_status.json", summary)
    write_tree_checksums(run_root)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
