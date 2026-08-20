"""Run the matched three-case NewGen benchmark with one dated OpenAI model."""

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
from ablation_test.src.providers import endpoint_health
from ablation_test.src.runner import execute_cell, write_checksums


BUNDLE = {
    "provider": "openai",
    "model": "gpt-5.4-2026-03-05",
    "upstream_mode": "always",
}

CASES = {
    "cuaac": "ablation_test/benchmarks/newgen_benchmark_v1_pilot_cuaac/case.json",
    "hydrogenolysis": "ablation_test/benchmarks/newgen_benchmark_v1_pilot_hydrogenolysis/case.json",
    "multistep": "ablation_test/benchmarks/newgen_benchmark_v1_pilot_multistep/case.json",
}


def _events(run_dir: Path) -> list[dict]:
    path = run_dir / "llm_events.jsonl"
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _metric_row(metrics: dict) -> dict:
    return {
        "quality_assurance": metrics.get("quality_assurance_score_v2"),
        "deployment_readiness": metrics.get("deployment_readiness_score_v2"),
        "deterministic_composite": metrics.get("deterministic_composite_score"),
        "gate_reasons": metrics.get("deployment_gate_reasons_v2") or [],
    }


def _copy_topology(result: dict, run_dir: Path) -> dict:
    copied = {}
    for key, name in (("png_path", "topology.png"), ("svg_path", "topology.svg")):
        source = Path(str(result.get(key) or ""))
        if source.is_file():
            target = run_dir / name
            shutil.copy2(source, target)
            copied[key] = str(target.resolve())
    return copied


def main() -> None:
    health = endpoint_health(BUNDLE)
    if not health.get("reachable") or not health.get("model_advertised"):
        raise RuntimeError(f"OpenAI endpoint/model unavailable: {health}")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output = ROOT / "ablation_results/newgen_benchmark" / f"newgen_openai_three_case_{stamp}"
    output.mkdir(parents=True, exist_ok=False)
    (output / "endpoint_health.json").write_text(json.dumps(health, indent=2), encoding="utf-8")

    rows = []
    for case_index, (name, relative_path) in enumerate(CASES.items(), start=1):
        case = load_cases_from_path(ROOT / relative_path)[0]
        case_row = {"case": name, "case_id": case.case_id}
        for variant_index, variant in enumerate(("general_one_shot", "full"), start=1):
            label = "one_shot" if variant == "general_one_shot" else "full_flowpilot"
            run_dir = output / f"{name}_gpt54_{label}"
            summary = execute_cell(
                case=case,
                variant=variant,
                bundle_name="gpt54_dated",
                bundle=BUNDLE,
                run_dir=run_dir,
                candidate_budget=1,
                temperature=0.0,
                seed=20260820 + case_index * 10 + variant_index,
            )
            if summary.get("status") != "completed":
                raise RuntimeError(f"Cell failed: {name}/{variant}: {summary}")
            result = json.loads((run_dir / "result.json").read_text(encoding="utf-8"))
            metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
            events = _events(run_dir)
            non_openai = [
                {"provider": event.get("provider"), "model": event.get("model")}
                for event in events
                if event.get("provider") != BUNDLE["provider"]
                or event.get("model") != BUNDLE["model"]
            ]
            verification = {
                "case_id": case.case_id,
                "variant": variant,
                "run_status": summary.get("status"),
                "llm_call_count": len(events),
                "strict_openai_model_only": not non_openai,
                "non_matching_calls": non_openai,
                "production_pipeline_complete": result.get("production_pipeline_complete"),
                "final_design_status": (result.get("final_design") or {}).get("status"),
                "recommended_disposition": result.get("recommended_disposition"),
                "topology_artifacts": _copy_topology(result, run_dir) if variant == "full" else {},
            }
            (run_dir / "model_verification.json").write_text(
                json.dumps(verification, indent=2), encoding="utf-8"
            )
            write_checksums(run_dir)
            case_row[label] = {
                **_metric_row(metrics),
                "llm_calls": len(events),
                "strict_model_only": not non_openai,
                "final_design_status": verification["final_design_status"],
                "disposition": verification["recommended_disposition"],
                "run_directory": str(run_dir.resolve()),
            }
        rows.append(case_row)

    def mean(path: tuple[str, str]) -> float:
        return round(sum(float(row[path[0]][path[1]]) for row in rows) / len(rows), 4)

    package = {
        "schema_version": "flowpilot_newgen_openai_three_case_v1.0",
        "model": BUNDLE,
        "candidate_budget": 1,
        "temperature": 0.0,
        "cases": rows,
        "aggregate": {
            "full_mean_quality_assurance": mean(("full_flowpilot", "quality_assurance")),
            "one_shot_mean_quality_assurance": mean(("one_shot", "quality_assurance")),
            "full_mean_deployment_readiness": mean(("full_flowpilot", "deployment_readiness")),
            "one_shot_mean_deployment_readiness": mean(("one_shot", "deployment_readiness")),
            "full_mean_deterministic_composite": mean(("full_flowpilot", "deterministic_composite")),
            "one_shot_mean_deterministic_composite": mean(("one_shot", "deterministic_composite")),
        },
    }
    (output / "comparison_summary.json").write_text(json.dumps(package, indent=2), encoding="utf-8")
    write_checksums(output)
    print(output)
    print(json.dumps(package, indent=2))


if __name__ == "__main__":
    main()
