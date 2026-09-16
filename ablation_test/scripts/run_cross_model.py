from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import os
import platform
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from ablation_test.src.cases import ROOT, AblationCase, load_cases, select_cases
from ablation_test.src.providers import endpoint_health
from ablation_test.src.paths import RUNS_ROOT
from ablation_test.src.runner import execute_cell


CONFIG_PATH = ROOT / "configs" / "cross_model_benchmark.json"
BASE_BENCHMARK_PATH = ROOT / "configs" / "benchmark.json"
SCORER_PATH = ROOT / "src" / "metrics.py"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _package_versions() -> dict[str, str]:
    packages: dict[str, str] = {}
    for name in (
        "anthropic",
        "openai",
        "pydantic",
        "chromadb",
        "numpy",
        "pandas",
        "matplotlib",
        "seaborn",
    ):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = "not_installed"
    return packages


def _git_metadata(project_root: Path) -> dict[str, Any]:
    def run(*args: str) -> str:
        completed = subprocess.run(
            ["git", *args],
            cwd=project_root,
            text=True,
            capture_output=True,
            check=False,
        )
        return completed.stdout.strip()

    return {
        "commit": run("rev-parse", "HEAD"),
        "branch": run("branch", "--show-current"),
        "status_porcelain": run("status", "--porcelain"),
        "diff_stat": run("diff", "--stat"),
    }


def _environment_manifest() -> dict[str, Any]:
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


def _run_cell(payload: dict[str, Any]) -> dict[str, Any]:
    summary = execute_cell(
        case=payload["case"],
        variant=payload["variant"],
        bundle_name=payload["bundle_name"],
        bundle=payload["bundle"],
        run_dir=payload["run_dir"],
        candidate_budget=payload["candidate_budget"],
        temperature=payload["temperature"],
        seed=payload["seed"],
    )
    return {
        "condition_id": payload["condition_id"],
        "condition_label": payload["condition_label"],
        "variant": payload["variant"],
        "bundle_name": payload["bundle_name"],
        "provider": payload["bundle"]["provider"],
        "model": payload["bundle"]["model"],
        "case_id": payload["case"].case_id,
        "category": payload["case"].category,
        "repeat": payload["repeat"],
        "status": summary.get("status", "unknown"),
        "runtime_s": summary.get("runtime_total_s", summary.get("runtime_s", 0)),
        "llm_call_count": summary.get("llm_call_count", 0),
        "total_tokens": (summary.get("token_totals") or {}).get("total_tokens", 0),
        "run_dir": str(payload["run_dir"].resolve()),
        "error": summary.get("error", ""),
    }


def _write_manifest(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _completed_row(payload: dict[str, Any]) -> dict[str, Any] | None:
    summary_path = payload["run_dir"] / "run_summary.json"
    if not summary_path.exists():
        return None
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("status") != "completed":
        return None
    return {
        "condition_id": payload["condition_id"],
        "condition_label": payload["condition_label"],
        "variant": payload["variant"],
        "bundle_name": payload["bundle_name"],
        "provider": payload["bundle"]["provider"],
        "model": payload["bundle"]["model"],
        "case_id": payload["case"].case_id,
        "category": payload["case"].category,
        "repeat": payload["repeat"],
        "status": "completed",
        "runtime_s": summary.get("runtime_total_s", summary.get("runtime_s", 0)),
        "llm_call_count": summary.get("llm_call_count", 0),
        "total_tokens": (summary.get("token_totals") or {}).get("total_tokens", 0),
        "run_dir": str(payload["run_dir"].resolve()),
        "error": "",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the matched Qwen/frontier FlowPilot benchmark."
    )
    parser.add_argument(
        "--profile",
        choices=("smoke", "publication"),
        default="smoke",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=CONFIG_PATH,
        help="Benchmark condition config; defaults to the original cross-model study.",
    )
    parser.add_argument("--condition", action="append", dest="conditions")
    parser.add_argument("--case", action="append", dest="case_ids")
    parser.add_argument("--repeats", type=int)
    parser.add_argument("--candidate-budget", type=int)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--output-id")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    base_config = json.loads(BASE_BENCHMARK_PATH.read_text(encoding="utf-8"))
    profile = config["profiles"][args.profile]
    cases = select_cases(load_cases(), args.case_ids or profile["case_ids"])
    condition_ids = args.conditions or list(config["conditions"])
    unknown = sorted(set(condition_ids) - set(config["conditions"]))
    if unknown:
        raise KeyError(f"Unknown conditions: {unknown}")
    repeats = args.repeats or config["repeats"][args.profile]
    candidate_budget = args.candidate_budget or config["candidate_budget"]
    stamp = args.output_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = RUNS_ROOT / f"cross_model_{args.profile}_{stamp}"
    experiment_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        handlers=[
            logging.FileHandler(experiment_dir / "run.log", encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    log = logging.getLogger("flowpilot.cross_model")

    bundles = base_config["model_bundles"]
    selected_bundle_ids = sorted(
        {config["conditions"][condition_id]["bundle"] for condition_id in condition_ids}
    )
    health = {
        bundle_id: endpoint_health(bundles[bundle_id])
        for bundle_id in selected_bundle_ids
    }
    (experiment_dir / "endpoint_health.json").write_text(
        json.dumps(health, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (experiment_dir / "environment.json").write_text(
        json.dumps(_environment_manifest(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (experiment_dir / "git_state.json").write_text(
        json.dumps(_git_metadata(ROOT.parent), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    cells: list[dict[str, Any]] = []
    for condition_id in condition_ids:
        condition = config["conditions"][condition_id]
        bundle_name = condition["bundle"]
        for case in cases:
            for repeat in range(1, repeats + 1):
                cells.append(
                    {
                        "condition_id": condition_id,
                        "condition_label": condition["label"],
                        "variant": condition["variant"],
                        "bundle_name": bundle_name,
                        "bundle": bundles[bundle_name],
                        "case": case,
                        "repeat": repeat,
                        "candidate_budget": candidate_budget,
                        "temperature": config["temperature"],
                        "seed": config["random_seed"] + repeat,
                        "run_dir": (
                            experiment_dir
                            / "cross_model"
                            / condition_id
                            / case.case_id
                            / f"repeat_{repeat:02d}"
                        ),
                    }
                )

    scorer_hash = _sha256(SCORER_PATH)
    execution_plan = {
        "schema_version": config["schema_version"],
        "config_path": str(config_path),
        "config_sha256": _sha256(config_path),
        "profile": args.profile,
        "candidate_budget": candidate_budget,
        "temperature": config["temperature"],
        "seed": config["random_seed"],
        "repeats": repeats,
        "workers": args.workers,
        "cell_count": len(cells),
        "conditions": {
            condition_id: config["conditions"][condition_id]
            for condition_id in condition_ids
        },
        "primary_comparisons": config["primary_comparisons"],
        "cases": [case.manifest_payload() for case in cases],
        "model_bundles": {
            bundle_id: bundles[bundle_id] for bundle_id in selected_bundle_ids
        },
        "frozen_scorer": {
            "path": str(SCORER_PATH),
            "sha256": scorer_hash,
        },
    }
    (experiment_dir / "execution_plan.json").write_text(
        json.dumps(execution_plan, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    rows: list[dict[str, Any]] = []
    pending: list[dict[str, Any]] = []
    for payload in cells:
        completed = _completed_row(payload) if args.resume else None
        if completed:
            rows.append(completed)
        else:
            pending.append(payload)
    _write_manifest(experiment_dir / "run_manifest.csv", rows)

    log.info(
        "Starting cross-model benchmark: %d cells (%d pending), workers=%d",
        len(cells),
        len(pending),
        args.workers,
    )
    if args.workers <= 1:
        for index, payload in enumerate(pending, start=1):
            log.info(
                "cell %d/%d condition=%s case=%s repeat=%d",
                index,
                len(pending),
                payload["condition_id"],
                payload["case"].case_id,
                payload["repeat"],
            )
            rows.append(_run_cell(payload))
            _write_manifest(experiment_dir / "run_manifest.csv", rows)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            future_map = {
                executor.submit(_run_cell, payload): payload for payload in pending
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
                        "bundle_name": payload["bundle_name"],
                        "provider": payload["bundle"]["provider"],
                        "model": payload["bundle"]["model"],
                        "case_id": payload["case"].case_id,
                        "category": payload["case"].category,
                        "repeat": payload["repeat"],
                        "status": "worker_failed",
                        "runtime_s": 0,
                        "llm_call_count": 0,
                        "total_tokens": 0,
                        "run_dir": str(payload["run_dir"].resolve()),
                        "error": str(exc),
                    }
                rows.append(row)
                _write_manifest(experiment_dir / "run_manifest.csv", rows)
                log.info(
                    "completed %d/%d condition=%s case=%s repeat=%d status=%s",
                    index,
                    len(pending),
                    row["condition_id"],
                    row["case_id"],
                    row["repeat"],
                    row["status"],
                )

    status_counts: dict[str, int] = {}
    for row in rows:
        status = row["status"]
        status_counts[status] = status_counts.get(status, 0) + 1
    summary = {
        "schema_version": config["schema_version"],
        "experiment_dir": str(experiment_dir.resolve()),
        "planned_cells": len(cells),
        "recorded_cells": len(rows),
        "status_counts": status_counts,
        "scorer_sha256": scorer_hash,
    }
    (experiment_dir / "experiment_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
