"""Run one Qwen-only smoke test through the GUI-equivalent production path."""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ablation_test.src.cases import load_cases_from_path
from ablation_test.src.providers import endpoint_health
from ablation_test.src.runner import execute_cell, write_checksums


CASE_PATH = (
    ROOT
    / "ablation_test/benchmarks/newgen_benchmark_v1_pilot_nitration/case.json"
)
BUNDLE = {
    "provider": "ollama",
    "model": "/models/Qwen3.6-27B",
    "base_url": "http://10.13.24.169:8000/v1",
    "upstream_mode": "always",
}


def main() -> None:
    health = endpoint_health(BUNDLE)
    if not health.get("reachable") or not health.get("model_advertised"):
        raise RuntimeError(f"Qwen endpoint is unavailable: {health}")

    case = load_cases_from_path(CASE_PATH)[0]
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output = (
        ROOT
        / "ablation_results/newgen_benchmark"
        / f"newgen_production_path_smoke_nitration_{stamp}"
    )
    run_dir = output / "qwen27b_full_flowpilot"
    output.mkdir(parents=True, exist_ok=False)
    (output / "endpoint_health.json").write_text(
        json.dumps(health, indent=2), encoding="utf-8"
    )

    summary = execute_cell(
        case=case,
        variant="full",
        bundle_name="qwen36_27b",
        bundle=BUNDLE,
        run_dir=run_dir,
        candidate_budget=1,
        temperature=0.0,
        seed=20260813,
    )
    result = json.loads((run_dir / "result.json").read_text(encoding="utf-8"))
    events = [
        json.loads(line)
        for line in (run_dir / "llm_events.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    non_qwen = [
        {"provider": item.get("provider"), "model": item.get("model")}
        for item in events
        if item.get("provider") != "ollama"
        or item.get("model") != BUNDLE["model"]
    ]
    verification = {
        "schema_version": "flowpilot_production_path_smoke_v1.0",
        "case_id": case.case_id,
        "status": summary.get("status"),
        "production_pipeline_complete": result.get("production_pipeline_complete"),
        "pipeline_entry_point": (result.get("pipeline_runtime") or {}).get(
            "pipeline_entry_point"
        ),
        "retrieval_mode": (result.get("pipeline_runtime") or {}).get(
            "retrieval_mode"
        ),
        "remote_embeddings_disabled": (
            (result.get("pipeline_runtime") or {}).get("retrieval_mode")
            == "lexical"
        ),
        "excluded_record_ids": (result.get("pipeline_runtime") or {}).get(
            "exclude_record_ids"
        ),
        "final_design_status": (result.get("final_design") or {}).get("status"),
        "final_process_graph_present": bool(
            (result.get("final_design") or {}).get("process_graph")
        ),
        "recommended_disposition": result.get("recommended_disposition"),
        "llm_call_count": len(events),
        "all_reasoning_model_calls_qwen": not non_qwen,
        "strict_qwen_only": (
            not non_qwen
            and (result.get("pipeline_runtime") or {}).get("retrieval_mode")
            == "lexical"
        ),
        "non_qwen_calls": non_qwen,
        "model": BUNDLE,
    }
    (output / "verification.json").write_text(
        json.dumps(verification, indent=2), encoding="utf-8"
    )
    write_checksums(output)
    print(output)
    print(json.dumps(verification, indent=2))


if __name__ == "__main__":
    main()
