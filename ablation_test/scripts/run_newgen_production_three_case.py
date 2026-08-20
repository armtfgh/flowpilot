"""Run three strict-Qwen NewGen cases through the production FlowPilot path."""

from __future__ import annotations

import json
import sys
import argparse
from copy import deepcopy
from dataclasses import replace
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ablation_test.src.cases import load_cases_from_path
from ablation_test.src.metrics import score_run
from ablation_test.src.providers import endpoint_health
from ablation_test.src.runner import execute_cell, write_checksums


BUNDLE = {
    "provider": "ollama",
    "model": "/models/Qwen3.6-27B",
    "base_url": "http://10.13.24.169:8000/v1",
    "upstream_mode": "always",
}

CASE_CONFIG = {
    "cuaac": {
        "case": "ablation_test/benchmarks/newgen_benchmark_v1_pilot_cuaac/case.json",
        "one_shot": "ablation_results/newgen_benchmark/newgen_benchmark_v1_pilot_cuaac_20260813/runs/qwen27b_one_shot",
    },
    "hydrogenolysis": {
        "case": "ablation_test/benchmarks/newgen_benchmark_v1_pilot_hydrogenolysis/case.json",
        "one_shot": "ablation_results/newgen_benchmark/newgen_benchmark_v1_pilot_hydrogenolysis_20260813/runs/qwen27b_one_shot",
    },
    "multistep": {
        "case": "ablation_test/benchmarks/newgen_benchmark_v1_pilot_multistep/case.json",
        "one_shot": "ablation_results/newgen_benchmark/newgen_benchmark_v1_pilot_multistep_20260813/runs/qwen27b_one_shot",
    },
}

CHEMISTRY_IDENTITIES = {
    "cuaac": {
        "transformation_family": "cuaac",
        "chemist_description": "CuAAC of benzyl azide and phenylacetylene",
        "confirmed": True,
        "source": "frozen_benchmark_chemist_confirmation",
    },
    "hydrogenolysis": {
        "transformation_family": "hydrogenolysis/debenzylation",
        "chemist_description": "Hydrogenolysis/debenzylation of DMAOL to 3-azetidinol",
        "confirmed": True,
        "source": "frozen_benchmark_chemist_confirmation",
    },
    "multistep": {
        "transformation_family": "oxidative amidation",
        "chemist_description": "Two-stage oxidation followed by oxidative amidation",
        "confirmed": True,
        "source": "frozen_benchmark_chemist_confirmation",
    },
}


def _canonical_case(case, name: str):
    inventory = deepcopy(case.inventory)
    accessories = inventory.setdefault("safety_accessories", [])
    accessories.append(
        {
            "equipment_id": f"benchmark_{name}_containment",
            "name": "Declared hazard-compatible shield and secondary containment",
            "type": "safety enclosure",
            "capabilities": ["shield_or_containment"],
        }
    )
    if name == "hydrogenolysis":
        accessories.append(
            {
                "equipment_id": "benchmark_h2_check_valve",
                "name": "Hydrogen-service non-return check valve",
                "type": "check valve",
                "capabilities": ["backflow_prevention"],
            }
        )
    return replace(
        case,
        inventory=inventory,
        chemistry_identity_confirmation=CHEMISTRY_IDENTITIES[name],
    )


def _qwen_verification(result: dict, events: list[dict]) -> dict:
    non_qwen = [
        {"provider": item.get("provider"), "model": item.get("model")}
        for item in events
        if item.get("provider") != "ollama"
        or item.get("model") != BUNDLE["model"]
    ]
    runtime = result.get("pipeline_runtime") or {}
    final = result.get("final_design") or {}
    return {
        "production_pipeline_complete": result.get("production_pipeline_complete"),
        "pipeline_entry_point": runtime.get("pipeline_entry_point"),
        "retrieval_mode": runtime.get("retrieval_mode"),
        "excluded_record_ids": runtime.get("exclude_record_ids") or [],
        "final_design_status": final.get("status"),
        "final_process_graph_present": bool(final.get("process_graph")),
        "recommended_disposition": result.get("recommended_disposition"),
        "llm_call_count": len(events),
        "strict_qwen_only": not non_qwen and runtime.get("retrieval_mode") == "lexical",
        "non_qwen_calls": non_qwen,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cases",
        nargs="+",
        choices=tuple(CASE_CONFIG),
        default=list(CASE_CONFIG),
    )
    parser.add_argument("--label", default="three_case")
    args = parser.parse_args()

    health = endpoint_health(BUNDLE)
    if not health.get("reachable") or not health.get("model_advertised"):
        raise RuntimeError(f"Qwen endpoint is unavailable: {health}")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output = (
        ROOT
        / "ablation_results/newgen_benchmark"
        / f"newgen_production_{args.label}_{stamp}"
    )
    output.mkdir(parents=True, exist_ok=False)
    (output / "endpoint_health.json").write_text(
        json.dumps(health, indent=2), encoding="utf-8"
    )

    rows = []
    selected = ((name, CASE_CONFIG[name]) for name in args.cases)
    for index, (name, config) in enumerate(selected, start=1):
        case = _canonical_case(load_cases_from_path(ROOT / config["case"])[0], name)
        run_dir = output / f"{name}_qwen27b_full_flowpilot"
        summary = execute_cell(
            case=case,
            variant="full",
            bundle_name="qwen36_27b",
            bundle=BUNDLE,
            run_dir=run_dir,
            candidate_budget=1,
            temperature=0.0,
            seed=20260813 + index,
        )
        result = json.loads((run_dir / "result.json").read_text(encoding="utf-8"))
        metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
        events = [
            json.loads(line)
            for line in (run_dir / "llm_events.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        verification = _qwen_verification(result, events)
        verification.update({"case_id": case.case_id, "run_status": summary.get("status")})
        (run_dir / "production_verification.json").write_text(
            json.dumps(verification, indent=2), encoding="utf-8"
        )

        one_shot_dir = ROOT / config["one_shot"]
        one_shot_metrics = None
        if (one_shot_dir / "result.json").is_file():
            one_shot_result = json.loads(
                (one_shot_dir / "result.json").read_text(encoding="utf-8")
            )
            one_shot_metrics = score_run(case, one_shot_result, one_shot_dir)

        rows.append(
            {
                "case": name,
                "case_id": case.case_id,
                "full_final_status": verification["final_design_status"],
                "full_disposition": verification["recommended_disposition"],
                "full_strict_qwen_only": verification["strict_qwen_only"],
                "full_llm_calls": verification["llm_call_count"],
                "full_quality_assurance_score": metrics.get("quality_assurance_score_v2"),
                "full_deployment_readiness_score": metrics.get("deployment_readiness_score_v2"),
                "full_deterministic_composite": metrics.get("deterministic_composite_score"),
                "full_gate_reasons": metrics.get("deployment_gate_reasons_v2") or [],
                "one_shot_quality_assurance_score": (
                    one_shot_metrics.get("quality_assurance_score_v2")
                    if one_shot_metrics else None
                ),
                "one_shot_deployment_readiness_score": (
                    one_shot_metrics.get("deployment_readiness_score_v2")
                    if one_shot_metrics else None
                ),
                "one_shot_deterministic_composite": (
                    one_shot_metrics.get("deterministic_composite_score")
                    if one_shot_metrics else None
                ),
                "one_shot_gate_reasons": (
                    one_shot_metrics.get("deployment_gate_reasons_v2") or []
                    if one_shot_metrics else []
                ),
            }
        )

    package = {
        "schema_version": "flowpilot_newgen_production_three_case_v1.0",
        "model": BUNDLE,
        "candidate_budget": 1,
        "temperature": 0.0,
        "cases": rows,
    }
    (output / "comparison_summary.json").write_text(
        json.dumps(package, indent=2), encoding="utf-8"
    )
    write_checksums(output)
    print(output)
    print(json.dumps(package, indent=2))


if __name__ == "__main__":
    main()
