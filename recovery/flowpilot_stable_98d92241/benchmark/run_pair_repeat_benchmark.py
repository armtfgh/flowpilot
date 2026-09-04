from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmark.pipeline import prepare_case_context, run_council_from_context
from benchmark.recorder import BenchmarkRecorder
from benchmark.run_model_matrix_benchmark import (
    COUNCIL_BUNDLES,
    GEMMA_BASE_URL,
    GEMMA_MODEL,
    UPSTREAM_BUNDLES,
    _case,
    council_bundle,
    upstream_bundle,
)
from benchmark.summarize import summarize_experiment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run repeated full upstream+council benchmark for one upstream/council pair."
    )
    parser.add_argument("--upstream-bundle", default="claude", choices=sorted(UPSTREAM_BUNDLES.keys()))
    parser.add_argument("--council-bundle", default="gpt4o", choices=sorted(COUNCIL_BUNDLES.keys()))
    parser.add_argument("--budget", type=int, default=12)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--objectives", default="balanced")
    parser.add_argument("--allow-warning-refinement", action="store_true", default=True)
    parser.add_argument("--strong-revision-mode", action="store_true", default=True)
    parser.add_argument("--branching-revision-mode", action="store_true", default=True)
    parser.add_argument("--max-descendants-per-candidate", type=int, default=3)
    parser.add_argument("--max-total-revised-candidates", type=int, default=16)
    parser.add_argument("--output-root", default="benchmark/data")
    parser.add_argument("--label", default="radar_pair_repeats")
    return parser.parse_args()


def _manifest_row(
    *,
    upstream_name: str,
    council_name: str,
    repeat_index: int,
    cell_dir: Path,
    run_dir: Path,
) -> dict:
    summary_path = run_dir / "run_summary.json"
    result_path = run_dir / "result.json"
    summary = {}
    if summary_path.exists():
        summary = json.loads(summary_path.read_text())
    final_metrics = summary.get("final_metrics", {}) or {}
    tokens = summary.get("token_totals", {}) or {}
    return {
        "upstream_bundle": upstream_name,
        "council_bundle": council_name,
        "repeat_index": repeat_index,
        "status": summary.get("status", "missing_summary"),
        "runtime_s": summary.get("runtime_s"),
        "llm_call_count": summary.get("llm_call_count"),
        "input_tokens": tokens.get("input_tokens", 0),
        "output_tokens": tokens.get("output_tokens", 0),
        "total_tokens": tokens.get("total_tokens", 0),
        "final_tau_min": final_metrics.get("residence_time_min"),
        "final_flow_rate_mL_min": final_metrics.get("flow_rate_mL_min"),
        "final_tubing_ID_mm": final_metrics.get("tubing_ID_mm"),
        "final_BPR_bar": final_metrics.get("BPR_bar"),
        "final_reactor_volume_mL": final_metrics.get("reactor_volume_mL"),
        "cell_dir": str(cell_dir),
        "run_dir": str(run_dir),
        "summary_path": str(summary_path) if summary_path.exists() else "",
        "result_path": str(result_path) if result_path.exists() else "",
    }


def main() -> None:
    args = parse_args()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = Path(args.output_root) / f"{args.label}_{stamp}"
    upstream_name = args.upstream_bundle
    council_name = args.council_bundle
    case = _case()

    cell_dir = experiment_dir / f"U_{upstream_name}" / f"C_{council_name}"
    cell_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "case": asdict(case),
        "budget": args.budget,
        "repeats": args.repeats,
        "temperature": args.temperature,
        "seed": args.seed,
        "objectives": args.objectives,
        "allow_warning_refinement": args.allow_warning_refinement,
        "strong_revision_mode": args.strong_revision_mode,
        "branching_revision_mode": args.branching_revision_mode,
        "max_descendants_per_candidate": args.max_descendants_per_candidate,
        "max_total_revised_candidates": args.max_total_revised_candidates,
        "upstream_bundle": upstream_name,
        "council_bundle": council_name,
        "upstream_models": UPSTREAM_BUNDLES[upstream_name],
        "council_model": COUNCIL_BUNDLES[council_name],
        "gemma_base_url": GEMMA_BASE_URL,
        "gemma_model": GEMMA_MODEL,
    }
    with (experiment_dir / "benchmark_config.json").open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2, ensure_ascii=False)

    manifest_rows: list[dict] = []
    for repeat_index in range(1, args.repeats + 1):
        repeat_slug = f"repeat_{repeat_index:02d}"
        context_dir = cell_dir / "contexts" / case.case_id / repeat_slug
        run_dir = cell_dir / "runs" / case.case_id / f"budget_{args.budget}" / repeat_slug
        print(json.dumps({
            "event": "repeat_start",
            "repeat_index": repeat_index,
            "context_dir": str(context_dir),
            "run_dir": str(run_dir),
        }))

        with upstream_bundle(upstream_name) as upstream_models:
            prep_recorder = BenchmarkRecorder(
                context_dir,
                {
                    "phase": "prepare_context",
                    "case_id": case.case_id,
                    "case_title": case.title,
                    "repeat_index": repeat_index,
                    "temperature": args.temperature,
                    "protocol": case.protocol,
                    "upstream_bundle": upstream_name,
                    "upstream_models": upstream_models,
                    "gemma_base_url": GEMMA_BASE_URL if upstream_name == "gemma" else None,
                },
            )
            try:
                context = prepare_case_context(case, prep_recorder, temperature=args.temperature)
                prep_recorder.finalize(status="completed")
            except Exception as exc:
                prep_recorder.finalize(status="failed", extra={"error": str(exc)})
                manifest_rows.append(_manifest_row(
                    upstream_name=upstream_name,
                    council_name=council_name,
                    repeat_index=repeat_index,
                    cell_dir=cell_dir,
                    run_dir=run_dir,
                ))
                print(json.dumps({
                    "event": "repeat_failed",
                    "phase": "prepare_context",
                    "repeat_index": repeat_index,
                    "error": str(exc),
                }, ensure_ascii=False))
                continue

        benchmark_claude_compact_mode = council_name == "claude"
        with council_bundle(council_name) as council_cfg:
            recorder = BenchmarkRecorder(
                run_dir,
                {
                    "phase": "council_run",
                    "case_id": case.case_id,
                    "case_title": case.title,
                    "candidate_budget": args.budget,
                    "repeat_index": repeat_index,
                    "temperature": args.temperature,
                    "seed": args.seed,
                    "allow_warning_refinement": args.allow_warning_refinement,
                    "benchmark_strong_revision_mode": args.strong_revision_mode,
                    "benchmark_branching_revision_mode": args.branching_revision_mode,
                    "benchmark_max_descendants_per_candidate": args.max_descendants_per_candidate,
                    "benchmark_max_total_revised_candidates": args.max_total_revised_candidates,
                    "benchmark_claude_compact_mode": benchmark_claude_compact_mode,
                    "objectives": args.objectives,
                    "protocol": case.protocol,
                    "upstream_bundle": upstream_name,
                    "council_bundle": council_name,
                    "council_provider": council_cfg["provider"],
                    "upstream_models": UPSTREAM_BUNDLES[upstream_name],
                    "council_model": council_cfg["model"],
                    "gemma_base_url": GEMMA_BASE_URL if (upstream_name == "gemma" or council_name == "gemma") else None,
                },
            )
            try:
                run_council_from_context(
                    context,
                    recorder,
                    candidate_budget=args.budget,
                    objectives=args.objectives,
                    allow_warning_refinement=args.allow_warning_refinement,
                    temperature=args.temperature,
                    seed=args.seed,
                    benchmark_strict_scoring=True,
                    benchmark_scoring_batch_size=3,
                    benchmark_claude_compact_mode=benchmark_claude_compact_mode,
                    benchmark_strong_revision_mode=args.strong_revision_mode,
                    benchmark_branching_revision_mode=args.branching_revision_mode,
                    benchmark_max_descendants_per_candidate=args.max_descendants_per_candidate,
                    benchmark_max_total_revised_candidates=args.max_total_revised_candidates,
                )
            except Exception as exc:
                print(json.dumps({
                    "event": "repeat_failed",
                    "phase": "council_run",
                    "repeat_index": repeat_index,
                    "error": str(exc),
                    "run_dir": str(run_dir),
                }, ensure_ascii=False))

        manifest_rows.append(_manifest_row(
            upstream_name=upstream_name,
            council_name=council_name,
            repeat_index=repeat_index,
            cell_dir=cell_dir,
            run_dir=run_dir,
        ))
        print(json.dumps(manifest_rows[-1], ensure_ascii=False))

    summarize_experiment(cell_dir)
    manifest_path = experiment_dir / "pair_repeat_manifest.csv"
    if manifest_rows:
        with manifest_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(manifest_rows[0].keys()))
            writer.writeheader()
            writer.writerows(manifest_rows)

    summary = {
        "experiment_dir": str(experiment_dir),
        "cell_dir": str(cell_dir),
        "pair_repeat_manifest_csv": str(manifest_path),
        "completed_repeats": sum(1 for row in manifest_rows if row.get("status") == "completed"),
        "requested_repeats": args.repeats,
    }
    with (experiment_dir / "experiment_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
