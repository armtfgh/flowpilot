from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ablation_test.src.cases import ROOT, load_cases, select_cases
from ablation_test.src.providers import endpoint_health
from ablation_test.src.paths import RUNS_ROOT
from ablation_test.src.runner import execute_cell


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the FlowPilot ablation benchmark.")
    parser.add_argument("--profile", choices=("smoke", "pilot", "publication"), default="smoke")
    parser.add_argument("--experiment", choices=("architecture", "portability", "all"), default="all")
    parser.add_argument("--case", action="append", dest="case_ids")
    parser.add_argument("--variant", action="append", dest="variants")
    parser.add_argument("--model", action="append", dest="models")
    parser.add_argument("--repeats", type=int)
    parser.add_argument("--candidate-budget", type=int)
    parser.add_argument("--include-adversarial", action="store_true")
    parser.add_argument("--output-id")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def _git_metadata(project_root: Path) -> dict[str, Any]:
    def run(*args: str) -> str:
        result = subprocess.run(
            ["git", *args],
            cwd=project_root,
            text=True,
            capture_output=True,
            check=False,
        )
        return result.stdout.strip()

    return {
        "commit": run("rev-parse", "HEAD"),
        "branch": run("branch", "--show-current"),
        "status_porcelain": run("status", "--porcelain"),
        "diff_stat": run("diff", "--stat"),
    }


def _environment_manifest() -> dict[str, Any]:
    packages: dict[str, str] = {}
    for name in ("anthropic", "openai", "pydantic", "chromadb", "numpy", "pandas", "matplotlib", "seaborn"):
        try:
            from importlib.metadata import version

            packages[name] = version(name)
        except Exception:
            packages[name] = "not_installed"
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "cwd": os.getcwd(),
        "packages": packages,
        "credentials": {
            "ANTHROPIC_API_KEY_set": bool(os.getenv("ANTHROPIC_API_KEY")),
            "OPENAI_API_KEY_set": bool(os.getenv("OPENAI_API_KEY")),
        },
    }


def _append_manifest(path: Path, row: dict[str, Any]) -> None:
    exists = path.exists()
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def main() -> None:
    args = parse_args()
    config = json.loads((ROOT / "configs" / "benchmark.json").read_text(encoding="utf-8"))
    profile = config["profiles"][args.profile]
    case_ids = args.case_ids or profile["case_ids"]
    all_cases = load_cases(include_adversarial=args.include_adversarial)
    cases = select_cases(all_cases, case_ids)
    variants = args.variants or profile["architecture_variants"]
    portability_models = args.models or profile["model_bundles"]
    architecture_model = args.models[0] if args.models else profile["architecture_model"]
    repeats = args.repeats or config["repeats"][args.profile]
    candidate_budget = args.candidate_budget or config["candidate_budget"]
    stamp = args.output_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = RUNS_ROOT / f"{args.profile}_{stamp}"
    experiment_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        handlers=[
            logging.FileHandler(experiment_dir / "run.log", encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    log = logging.getLogger("flowpilot.ablation.run")

    health = {
        name: endpoint_health(bundle)
        for name, bundle in config["model_bundles"].items()
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
    if args.experiment in ("architecture", "all"):
        for variant in variants:
            for case in cases:
                for repeat in range(1, repeats + 1):
                    cells.append(
                        {
                            "arm": "architecture",
                            "variant": variant,
                            "bundle_name": architecture_model,
                            "case": case,
                            "repeat": repeat,
                        }
                    )
    if args.experiment in ("portability", "all"):
        for bundle_name in portability_models:
            for case in cases:
                for repeat in range(1, repeats + 1):
                    cells.append(
                        {
                            "arm": "portability",
                            "variant": "full",
                            "bundle_name": bundle_name,
                            "case": case,
                            "repeat": repeat,
                        }
                    )

    plan = {
        "schema_version": config["schema_version"],
        "profile": args.profile,
        "experiment": args.experiment,
        "candidate_budget": candidate_budget,
        "temperature": config["temperature"],
        "seed": config["random_seed"],
        "repeats": repeats,
        "cell_count": len(cells),
        "cases": [case.manifest_payload() for case in cases],
        "architecture_variants": variants,
        "architecture_model": architecture_model,
        "portability_models": portability_models,
        "model_bundles": config["model_bundles"],
    }
    (experiment_dir / "execution_plan.json").write_text(
        json.dumps(plan, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    manifest = experiment_dir / "run_manifest.csv"
    completed = 0
    for index, cell in enumerate(cells, start=1):
        case = cell["case"]
        bundle_name = cell["bundle_name"]
        bundle = config["model_bundles"][bundle_name]
        run_dir = (
            experiment_dir
            / cell["arm"]
            / cell["variant"]
            / bundle_name
            / case.case_id
            / f"repeat_{cell['repeat']:02d}"
        )
        if args.resume and (run_dir / "run_summary.json").exists():
            summary = json.loads((run_dir / "run_summary.json").read_text(encoding="utf-8"))
            if summary.get("status") == "completed":
                log.info(
                    "resume skip %s/%s/%s",
                    cell["variant"],
                    bundle_name,
                    case.case_id,
                )
            else:
                log.info(
                    "resume rerun status=%s %s/%s/%s",
                    summary.get("status", "unknown"),
                    cell["variant"],
                    bundle_name,
                    case.case_id,
                )
                summary = execute_cell(
                    case=case,
                    variant=cell["variant"],
                    bundle_name=bundle_name,
                    bundle=bundle,
                    run_dir=run_dir,
                    candidate_budget=candidate_budget,
                    temperature=config["temperature"],
                    seed=config["random_seed"] + cell["repeat"],
                )
        else:
            log.info(
                "cell %d/%d arm=%s variant=%s model=%s case=%s",
                index,
                len(cells),
                cell["arm"],
                cell["variant"],
                bundle_name,
                case.case_id,
            )
            summary = execute_cell(
                case=case,
                variant=cell["variant"],
                bundle_name=bundle_name,
                bundle=bundle,
                run_dir=run_dir,
                candidate_budget=candidate_budget,
                temperature=config["temperature"],
                seed=config["random_seed"] + cell["repeat"],
            )
        row = {
            "arm": cell["arm"],
            "variant": cell["variant"],
            "bundle": bundle_name,
            "provider": bundle["provider"],
            "model": bundle["model"],
            "case_id": case.case_id,
            "category": case.category,
            "repeat": cell["repeat"],
            "status": summary.get("status", "unknown"),
            "runtime_s": summary.get("runtime_total_s", summary.get("runtime_s")),
            "llm_call_count": summary.get("llm_call_count"),
            "total_tokens": (summary.get("token_totals") or {}).get("total_tokens"),
            "run_dir": str(run_dir),
        }
        _append_manifest(manifest, row)
        completed += 1

    summary = {
        "experiment_dir": str(experiment_dir),
        "planned_cells": len(cells),
        "processed_cells": completed,
        "manifest": str(manifest),
    }
    (experiment_dir / "experiment_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
