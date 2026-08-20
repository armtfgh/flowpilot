"""Rerun the two OpenAI NewGen cells affected by canonicalization defects."""

from __future__ import annotations

import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ablation_test.src.cases import load_cases_from_path
from ablation_test.src.runner import execute_cell, write_checksums


BUNDLE = {
    "provider": "openai",
    "model": "gpt-5.4-2026-03-05",
    "upstream_mode": "always",
}

CELLS = (
    (
        "hydrogenolysis",
        "full",
        "ablation_test/benchmarks/newgen_benchmark_v1_pilot_hydrogenolysis/case.json",
    ),
    (
        "multistep",
        "general_one_shot",
        "ablation_test/benchmarks/newgen_benchmark_v1_pilot_multistep/case.json",
    ),
)


def main() -> None:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output = ROOT / "ablation_results/newgen_benchmark" / f"newgen_openai_repaired_cells_{stamp}"
    output.mkdir(parents=True, exist_ok=False)
    rows = []
    for index, (name, variant, case_path) in enumerate(CELLS, start=1):
        case = load_cases_from_path(ROOT / case_path)[0]
        label = "full_flowpilot" if variant == "full" else "one_shot"
        run_dir = output / f"{name}_gpt54_{label}"
        summary = execute_cell(
            case=case,
            variant=variant,
            bundle_name="gpt54_dated",
            bundle=BUNDLE,
            run_dir=run_dir,
            candidate_budget=1,
            temperature=0.0,
            seed=20260870 + index,
        )
        if summary.get("status") != "completed":
            raise RuntimeError(f"Cell failed: {name}/{variant}: {summary}")
        result = json.loads((run_dir / "result.json").read_text(encoding="utf-8"))
        metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
        if variant == "full":
            for key, target_name in (("png_path", "topology.png"), ("svg_path", "topology.svg")):
                source = Path(str(result.get(key) or ""))
                if source.is_file():
                    shutil.copy2(source, run_dir / target_name)
        events = [
            json.loads(line)
            for line in (run_dir / "llm_events.jsonl").read_text(encoding="utf-8").splitlines()
            if line
        ]
        rows.append(
            {
                "case": name,
                "variant": variant,
                "status": (result.get("final_design") or {}).get("status"),
                "disposition": result.get("recommended_disposition"),
                "schema_valid": result.get("schema_valid"),
                "quality_assurance": metrics.get("quality_assurance_score_v2"),
                "deployment_readiness": metrics.get("deployment_readiness_score_v2"),
                "deterministic_composite": metrics.get("deterministic_composite_score"),
                "gate_reasons": metrics.get("deployment_gate_reasons_v2") or [],
                "llm_calls": len(events),
                "strict_model_only": all(
                    event.get("provider") == "openai"
                    and event.get("model") == BUNDLE["model"]
                    for event in events
                ),
                "run_directory": str(run_dir.resolve()),
            }
        )
        write_checksums(run_dir)
    package = {"model": BUNDLE, "cells": rows}
    (output / "summary.json").write_text(json.dumps(package, indent=2), encoding="utf-8")
    write_checksums(output)
    print(output)
    print(json.dumps(package, indent=2))


if __name__ == "__main__":
    main()
