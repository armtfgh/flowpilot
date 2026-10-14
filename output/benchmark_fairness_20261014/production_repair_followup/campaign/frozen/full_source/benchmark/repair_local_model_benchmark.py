from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Repair and summarize a local-model benchmark directory.")
    parser.add_argument("experiment_dir", help="Path to local_model_benchmark_* directory")
    return parser.parse_args()


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    experiment_dir = Path(args.experiment_dir).resolve()

    run_summaries = sorted(experiment_dir.glob("U_*/C_*/runs/*/budget_*/repeat_*/run_summary.json"))
    matrix_rows: list[dict] = []
    run_rows: list[dict] = []
    llm_rows: list[dict] = []

    for summary_path in run_summaries:
        summary = load_json(summary_path)
        run_dir = summary_path.parent
        metadata = summary.get("metadata", {})
        result_path = run_dir / "result.json"
        result = load_json(result_path) if result_path.exists() else {}
        final_metrics = summary.get("final_metrics", {})
        llm_event_path = run_dir / "llm_events.jsonl"
        llm_event_count = 0
        if llm_event_path.exists():
            with llm_event_path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    llm_event_count += 1
                    event = json.loads(line)
                    usage = event.get("usage", {}) or {}
                    total_tokens = int(
                        usage.get("total_tokens", 0) or 0
                    ) or int(usage.get("input_tokens", 0) or 0) + int(usage.get("output_tokens", 0) or 0) or int(
                        usage.get("prompt_tokens", 0) or 0
                    ) + int(
                        usage.get("completion_tokens", 0) or 0
                    )
                    llm_rows.append(
                        {
                            "case_id": metadata.get("case_id"),
                            "upstream_bundle": metadata.get("upstream_bundle"),
                            "council_bundle": metadata.get("council_bundle"),
                            "candidate_budget": metadata.get("candidate_budget"),
                            "repeat_index": metadata.get("repeat_index"),
                            "timestamp": event.get("timestamp"),
                            "api_name": event.get("api_name"),
                            "provider": event.get("provider"),
                            "model": event.get("model"),
                            "duration_ms": event.get("duration_ms"),
                            "max_tokens": event.get("max_tokens"),
                            "temperature": event.get("temperature"),
                            "finish_reason": event.get("finish_reason"),
                            "input_tokens": usage.get("input_tokens", 0),
                            "output_tokens": usage.get("output_tokens", 0),
                            "prompt_tokens": usage.get("prompt_tokens", 0),
                            "completion_tokens": usage.get("completion_tokens", 0),
                            "total_tokens": total_tokens,
                            "response_chars": event.get("response_chars"),
                        }
                    )

        matrix_rows.append(
            {
                "upstream_bundle": metadata.get("upstream_bundle"),
                "council_bundle": metadata.get("council_bundle"),
                "status": summary.get("status"),
                "runtime_s": summary.get("runtime_s", ""),
                "llm_call_count": summary.get("llm_call_count", ""),
                "total_tokens": summary.get("token_totals", {}).get("total_tokens", ""),
                "final_tau_min": final_metrics.get("residence_time_min", ""),
                "final_flow_rate_mL_min": final_metrics.get("flow_rate_mL_min", ""),
                "final_tubing_ID_mm": final_metrics.get("tubing_ID_mm", ""),
                "final_BPR_bar": final_metrics.get("BPR_bar", ""),
                "final_reactor_volume_mL": final_metrics.get("reactor_volume_mL", ""),
                "cell_dir": str(run_dir.parents[3]),
                "run_summary_path": str(summary_path),
                "result_path": str(result_path) if result_path.exists() else "",
                "llm_event_count": llm_event_count,
                "error": summary.get("error", ""),
            }
        )

        run_rows.append(
            {
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
                "prompt_tokens": summary.get("token_totals", {}).get("prompt_tokens", 0),
                "completion_tokens": summary.get("token_totals", {}).get("completion_tokens", 0),
                "total_tokens": summary.get("token_totals", {}).get("total_tokens", 0),
                "temperature": metadata.get("temperature"),
                "allow_warning_refinement": metadata.get("allow_warning_refinement"),
                "benchmark_strong_revision_mode": metadata.get("benchmark_strong_revision_mode"),
                "benchmark_branching_revision_mode": metadata.get("benchmark_branching_revision_mode"),
                "benchmark_max_descendants_per_candidate": metadata.get("benchmark_max_descendants_per_candidate"),
                "benchmark_max_total_revised_candidates": metadata.get("benchmark_max_total_revised_candidates"),
                "final_tau_min": final_metrics.get("residence_time_min"),
                "final_flow_rate_mL_min": final_metrics.get("flow_rate_mL_min"),
                "final_tubing_ID_mm": final_metrics.get("tubing_ID_mm"),
                "final_BPR_bar": final_metrics.get("BPR_bar"),
                "final_reactor_volume_mL": final_metrics.get("reactor_volume_mL"),
                "result_path": str(result_path) if result_path.exists() else "",
                "summary_path": str(summary_path),
                "winner_confidence": ((result.get("formatted_result") or {}).get("confidence")) if result else None,
            }
        )

    matrix_rows.sort(key=lambda r: (r["upstream_bundle"], r["council_bundle"]))
    run_rows.sort(key=lambda r: (r["upstream_bundle"], r["council_bundle"], r["candidate_budget"], r["repeat_index"]))

    if matrix_rows:
        write_csv(experiment_dir / "matrix_manifest.csv", list(matrix_rows[0].keys()), matrix_rows)
    if run_rows:
        write_csv(experiment_dir / "run_manifest.csv", list(run_rows[0].keys()), run_rows)
    if llm_rows:
        write_csv(experiment_dir / "llm_manifest.csv", list(llm_rows[0].keys()), llm_rows)

    summary = {
        "experiment_dir": str(experiment_dir),
        "run_count": len(run_rows),
        "llm_event_count": len(llm_rows),
        "matrix_manifest_csv": str(experiment_dir / "matrix_manifest.csv") if matrix_rows else "",
        "run_manifest_csv": str(experiment_dir / "run_manifest.csv") if run_rows else "",
        "llm_manifest_csv": str(experiment_dir / "llm_manifest.csv") if llm_rows else "",
    }
    with (experiment_dir / "experiment_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
