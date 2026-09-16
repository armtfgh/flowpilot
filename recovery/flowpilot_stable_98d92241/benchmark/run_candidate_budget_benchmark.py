from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmark.cases import CASES, get_case
from benchmark.pipeline import prepare_case_context, run_council_from_context
from benchmark.recorder import BenchmarkRecorder
from benchmark.summarize import summarize_experiment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run candidate-budget council benchmark.")
    parser.add_argument("--cases", nargs="+", default=["snar"], choices=sorted(CASES.keys()))
    parser.add_argument("--budgets", nargs="+", type=int, default=[1, 6, 12, 24])
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--objectives", default="balanced")
    parser.add_argument("--allow-warning-refinement", action="store_true", default=False)
    parser.add_argument(
        "--output-root",
        default="benchmark/data",
        help="Root folder for benchmark experiments.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = Path(args.output_root) / f"candidate_budget_benchmark_{stamp}"
    experiment_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "cases": args.cases,
        "budgets": args.budgets,
        "repeats": args.repeats,
        "temperature": args.temperature,
        "seed": args.seed,
        "objectives": args.objectives,
        "allow_warning_refinement": args.allow_warning_refinement,
    }
    with (experiment_dir / "benchmark_config.json").open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)

    prepared_contexts = {}
    for case_id in args.cases:
        case = get_case(case_id)
        context_dir = experiment_dir / "contexts" / case_id
        prep_recorder = BenchmarkRecorder(
            context_dir,
            {
                "phase": "prepare_context",
                "case_id": case.case_id,
                "case_title": case.title,
                "temperature": args.temperature,
            },
        )
        prepared_contexts[case_id] = prepare_case_context(case, prep_recorder, temperature=args.temperature)
        prep_recorder.finalize(status="completed")

    for case_id in args.cases:
        context = prepared_contexts[case_id]
        for budget in args.budgets:
            for repeat_index in range(1, args.repeats + 1):
                run_dir = experiment_dir / "runs" / case_id / f"budget_{budget}" / f"repeat_{repeat_index:02d}"
                recorder = BenchmarkRecorder(
                    run_dir,
                    {
                        "phase": "council_run",
                        "case_id": case_id,
                        "candidate_budget": budget,
                        "repeat_index": repeat_index,
                        "temperature": args.temperature,
                        "seed": args.seed,
                        "allow_warning_refinement": args.allow_warning_refinement,
                        "objectives": args.objectives,
                    },
                )
                run_council_from_context(
                    context,
                    recorder,
                    candidate_budget=budget,
                    objectives=args.objectives,
                    allow_warning_refinement=args.allow_warning_refinement,
                    temperature=args.temperature,
                    seed=args.seed,
                )

    summary = summarize_experiment(experiment_dir)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
