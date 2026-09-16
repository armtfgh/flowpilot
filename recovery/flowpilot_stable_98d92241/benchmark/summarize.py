from __future__ import annotations

import csv
import json
from pathlib import Path


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def summarize_experiment(experiment_dir: Path) -> dict:
    experiment_dir = Path(experiment_dir)
    run_summaries = sorted(experiment_dir.glob("runs/*/budget_*/repeat_*/run_summary.json"))
    llm_event_files = sorted(experiment_dir.glob("runs/*/budget_*/repeat_*/llm_events.jsonl"))

    run_rows: list[dict] = []
    llm_rows: list[dict] = []

    for summary_path in run_summaries:
        summary = _load_json(summary_path)
        run_dir = summary_path.parent
        result_path = run_dir / "result.json"
        result = _load_json(result_path) if result_path.exists() else {}
        metadata = summary.get("metadata", {})
        final_metrics = summary.get("final_metrics", {})
        derived_total_tokens = (
            int(summary.get("token_totals", {}).get("total_tokens", 0) or 0)
            or int(summary.get("token_totals", {}).get("input_tokens", 0) or 0)
            + int(summary.get("token_totals", {}).get("output_tokens", 0) or 0)
        )
        run_rows.append({
            "case_id": metadata.get("case_id"),
            "upstream_bundle": metadata.get("upstream_bundle"),
            "council_bundle": metadata.get("council_bundle"),
            "council_provider": metadata.get("council_provider"),
            "candidate_budget": metadata.get("candidate_budget"),
            "repeat_index": metadata.get("repeat_index"),
            "status": summary.get("status"),
            "runtime_s": summary.get("runtime_s"),
            "llm_call_count": summary.get("llm_call_count"),
            "input_tokens": summary.get("token_totals", {}).get("input_tokens", 0),
            "output_tokens": summary.get("token_totals", {}).get("output_tokens", 0),
            "total_tokens": derived_total_tokens,
            "temperature": metadata.get("temperature"),
            "allow_warning_refinement": metadata.get("allow_warning_refinement"),
            "benchmark_strong_revision_mode": metadata.get("benchmark_strong_revision_mode"),
            "benchmark_branching_revision_mode": metadata.get("benchmark_branching_revision_mode"),
            "benchmark_max_descendants_per_candidate": metadata.get("benchmark_max_descendants_per_candidate"),
            "final_tau_min": final_metrics.get("residence_time_min"),
            "final_flow_rate_mL_min": final_metrics.get("flow_rate_mL_min"),
            "final_tubing_ID_mm": final_metrics.get("tubing_ID_mm"),
            "final_BPR_bar": final_metrics.get("BPR_bar"),
            "final_reactor_volume_mL": final_metrics.get("reactor_volume_mL"),
            "result_path": str(result_path) if result_path.exists() else "",
            "summary_path": str(summary_path),
            "winner_confidence": (((result.get("formatted_result") or {}).get("confidence")) if result else None),
        })

    for event_file in llm_event_files:
        run_dir = event_file.parent
        summary_path = run_dir / "run_summary.json"
        summary = _load_json(summary_path) if summary_path.exists() else {}
        metadata = summary.get("metadata", {})
        with event_file.open("r", encoding="utf-8") as handle:
            for line in handle:
                event = json.loads(line)
                usage = event.get("usage", {}) or {}
                derived_total_tokens = (
                    int(usage.get("total_tokens", 0) or 0)
                    or int(usage.get("input_tokens", 0) or 0)
                    + int(usage.get("output_tokens", 0) or 0)
                )
                llm_rows.append({
                    "case_id": metadata.get("case_id"),
                    "upstream_bundle": metadata.get("upstream_bundle"),
                    "council_bundle": metadata.get("council_bundle"),
                    "council_provider": metadata.get("council_provider"),
                    "candidate_budget": metadata.get("candidate_budget"),
                    "repeat_index": metadata.get("repeat_index"),
                    "timestamp": event.get("timestamp"),
                    "api_name": event.get("api_name"),
                    "provider": event.get("provider"),
                    "model": event.get("model"),
                    "duration_ms": event.get("duration_ms"),
                    "max_tokens": event.get("max_tokens"),
                    "temperature": event.get("temperature"),
                    "seed": event.get("seed"),
                    "tool_calls_requested": event.get("tool_calls_requested"),
                    "input_tokens": usage.get("input_tokens", 0),
                    "output_tokens": usage.get("output_tokens", 0),
                    "prompt_tokens": usage.get("prompt_tokens", 0),
                    "completion_tokens": usage.get("completion_tokens", 0),
                    "total_tokens": derived_total_tokens,
                    "response_chars": event.get("response_chars"),
                })

    runs_csv = experiment_dir / "run_manifest.csv"
    llm_csv = experiment_dir / "llm_manifest.csv"

    if run_rows:
        with runs_csv.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(run_rows[0].keys()))
            writer.writeheader()
            writer.writerows(run_rows)

    if llm_rows:
        with llm_csv.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(llm_rows[0].keys()))
            writer.writeheader()
            writer.writerows(llm_rows)

    summary = {
        "experiment_dir": str(experiment_dir),
        "run_count": len(run_rows),
        "llm_event_count": len(llm_rows),
        "run_manifest_csv": str(runs_csv) if run_rows else "",
        "llm_manifest_csv": str(llm_csv) if llm_rows else "",
    }
    with (experiment_dir / "experiment_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    return summary
