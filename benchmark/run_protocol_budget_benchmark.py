from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmark.cases import BenchmarkCase
from benchmark.pipeline import prepare_case_context, run_council_from_context
from benchmark.recorder import BenchmarkRecorder
from benchmark.summarize import summarize_experiment


PROTOCOL = """Phenylacetylene (1a, 0.5 mmol, 1.0 equiv) and ethyl nitroacetate (2, 1.0 mmol, 2.0 equiv) were added directly to TBAB/EG (1:5) DES (1 mL) in an oven-dried 30 mL vial equipped with a magnetic stirring bar; no additional solvent, catalyst, base, or additive was used. The reaction mixture was stirred at 120 °C in an oil bath for 15 min, at which point full conversion (99%) to ethyl 5-phenylisoxazole-3-carboxylate (3a) was achieved via 1,3-dipolar cycloaddition of the in-situ-generated nitrile oxide intermediate with phenylacetylene. After the reaction, the mixture was quenched with water, extracted with dichloromethane, dried over MgSO₄, and the solvent removed under reduced pressure; the isolated NMR yield was 83% (quantified against 1,3,5-trimethoxybenzene as internal standard)."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run repeated candidate-budget benchmark for the DES isoxazole protocol.")
    parser.add_argument("--budgets", nargs="+", type=int, default=[1, 6, 12, 24])
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--objectives", default="balanced")
    parser.add_argument("--allow-warning-refinement", action="store_true", default=True)
    parser.add_argument("--strong-revision-mode", action="store_true")
    parser.add_argument("--branching-revision-mode", action="store_true")
    parser.add_argument("--max-descendants-per-candidate", type=int, default=2)
    parser.add_argument("--output-root", default="benchmark/data")
    return parser.parse_args()


def _case() -> BenchmarkCase:
    return BenchmarkCase(
        case_id="protocol_isoxazole_des_full",
        title="Batch Synthesis of 3,5-Disubstituted Isoxazole in TBAB/EG (1:5) DES",
        protocol=PROTOCOL,
        precedent_level="weak_precedent",
        difficulty="high",
        notes="User-provided full protocol with 15 min batch time, 83% isolated yield, openai council benchmark.",
        tags=("thermal", "cycloaddition", "des", "user_protocol"),
    )


def main() -> None:
    args = parse_args()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = Path(args.output_root) / f"protocol_budget_benchmark_{stamp}"
    experiment_dir.mkdir(parents=True, exist_ok=True)

    case = _case()
    config = {
        "case": asdict(case),
        "budgets": args.budgets,
        "repeats": args.repeats,
        "temperature": args.temperature,
        "seed": args.seed,
        "objectives": args.objectives,
        "allow_warning_refinement": args.allow_warning_refinement,
        "strong_revision_mode": args.strong_revision_mode,
        "branching_revision_mode": args.branching_revision_mode,
        "max_descendants_per_candidate": args.max_descendants_per_candidate,
    }
    with (experiment_dir / "benchmark_config.json").open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2, ensure_ascii=False)

    context_dir = experiment_dir / "contexts" / case.case_id
    prep_recorder = BenchmarkRecorder(
        context_dir,
        {
            "phase": "prepare_context",
            "case_id": case.case_id,
            "case_title": case.title,
            "temperature": args.temperature,
            "protocol": case.protocol,
        },
    )
    context = prepare_case_context(case, prep_recorder, temperature=args.temperature)
    prep_recorder.finalize(status="completed")

    for budget in args.budgets:
        for repeat_index in range(1, args.repeats + 1):
            run_dir = experiment_dir / "runs" / case.case_id / f"budget_{budget}" / f"repeat_{repeat_index:02d}"
            recorder = BenchmarkRecorder(
                run_dir,
                {
                    "phase": "council_run",
                    "case_id": case.case_id,
                    "case_title": case.title,
                    "candidate_budget": budget,
                    "repeat_index": repeat_index,
                    "temperature": args.temperature,
                    "seed": args.seed,
                    "allow_warning_refinement": args.allow_warning_refinement,
                    "benchmark_strong_revision_mode": args.strong_revision_mode,
                    "benchmark_branching_revision_mode": args.branching_revision_mode,
                    "benchmark_max_descendants_per_candidate": args.max_descendants_per_candidate,
                    "objectives": args.objectives,
                    "protocol": case.protocol,
                },
            )
            try:
                run_council_from_context(
                    context,
                    recorder,
                    candidate_budget=budget,
                    objectives=args.objectives,
                    allow_warning_refinement=args.allow_warning_refinement,
                    temperature=args.temperature,
                    seed=args.seed,
                    benchmark_strong_revision_mode=args.strong_revision_mode,
                    benchmark_branching_revision_mode=args.branching_revision_mode,
                    benchmark_max_descendants_per_candidate=args.max_descendants_per_candidate,
                )
            except Exception as exc:
                # Keep batch running; each failed run already writes error.json/run_summary.json
                print(
                    json.dumps(
                        {
                            "budget": budget,
                            "repeat_index": repeat_index,
                            "status": "failed",
                            "error": str(exc),
                            "run_dir": str(run_dir),
                        },
                        ensure_ascii=False,
                    )
                )

    summary = summarize_experiment(experiment_dir)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
