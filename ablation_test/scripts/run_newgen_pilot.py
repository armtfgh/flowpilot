"""Run the frozen one-case NewGen benchmark pilot with Qwen 27B."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ablation_test.src.cases import load_cases_from_path
from ablation_test.src.providers import endpoint_health
from ablation_test.src.runner import execute_cell, write_checksums
from flora_translate.schemas import LabInventory


BENCHMARK = ROOT / "ablation_test" / "benchmarks" / "newgen_benchmark_v1_pilot_cuaac"
DEFAULT_OUTPUT = ROOT / "ablation_results" / "newgen_benchmark" / "newgen_benchmark_v1_pilot_cuaac_20260813"
SOURCE_PDF = Path("/home/amirreza/SharedFolder/win2fed/mined_papers_2/adsc.200900726.pdf")
BUNDLE = {
    "provider": "ollama",
    "model": "/models/Qwen3.6-27B",
    "base_url": "http://10.13.24.169:8000/v1",
    "upstream_mode": "always",
}
CONDITIONS = (
    ("qwen27b_one_shot", "general_one_shot"),
    ("qwen27b_full_flowpilot", "full"),
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    frozen = output / "frozen"
    runs = output / "runs"
    frozen.mkdir(exist_ok=True)
    runs.mkdir(exist_ok=True)

    case_path = BENCHMARK / "case.json"
    rubric_path = BENCHMARK / "universal_rubric.json"
    oracle_path = BENCHMARK / "hidden_oracle.json"
    case = load_cases_from_path(case_path)[0]
    LabInventory(**case.inventory)

    health = endpoint_health(BUNDLE)
    if not health.get("reachable") or not health.get("model_advertised"):
        raise RuntimeError(f"Qwen endpoint is not ready: {health}")

    for source in (case_path, rubric_path, oracle_path, SOURCE_PDF):
        shutil.copy2(source, frozen / source.name)

    source_record = ROOT / "flora_translate" / "data" / "records" / "adsc.200900726.json"
    source_hashes = {
        str(path): sha256(path)
        for path in (case_path, rubric_path, oracle_path, SOURCE_PDF, source_record)
    }
    write_json(
        frozen / "execution_manifest.json",
        {
            "schema_version": "flowpilot_newgen_execution_manifest_v1.0",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "case_id": case.case_id,
            "model_bundle": BUNDLE,
            "conditions": [
                {"condition_id": condition_id, "variant": variant}
                for condition_id, variant in CONDITIONS
            ],
            "temperature": 0.0,
            "seed": 20260813,
            "candidate_budget": 3,
            "design_input_sha256": case.design_input_sha256,
            "excluded_record_ids": sorted(case.excluded_record_ids),
            "source_hashes": source_hashes,
            "endpoint_health": health,
        },
    )

    summaries = []
    for condition_id, variant in CONDITIONS:
        run_dir = runs / condition_id
        summary_path = run_dir / "run_summary.json"
        if summary_path.is_file() and not args.force:
            previous = read_json(summary_path)
            if previous.get("status") == "completed":
                summaries.append({"condition_id": condition_id, "variant": variant, **previous})
                continue
        summary = execute_cell(
            case=case,
            variant=variant,
            bundle_name="qwen36_27b",
            bundle=BUNDLE,
            run_dir=run_dir,
            candidate_budget=3,
            temperature=0.0,
            seed=20260813,
        )
        summaries.append({"condition_id": condition_id, "variant": variant, **summary})

    write_json(output / "execution_summary.json", {"runs": summaries})
    write_checksums(output)
    print(output)
    print(json.dumps(summaries, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
