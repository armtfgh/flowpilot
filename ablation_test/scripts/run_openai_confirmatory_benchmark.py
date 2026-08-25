#!/usr/bin/env python3
"""Run the frozen OpenAI-only confirmation after the stationary-catalyst fix."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ablation_test.scripts import run_manuscript_three_model_repeated_benchmark as campaign


DEFAULT_OUTPUT = (
    ROOT
    / "ablation_results/manuscript_benchmark/openai_confirmatory_stationary_fix_20260820"
)
DEFAULT_REPORT = (
    ROOT
    / "deliverables/openai_confirmatory_stationary_fix_20260820"
)


def configure() -> None:
    campaign.MODELS = {"openai": campaign.MODELS["openai"]}
    campaign.JUDGES = ("qwen", "openai", "claude")
    campaign.SEED_NAMESPACE = "openai-confirmatory-stationary-fix-20260820-v1"


def write_acceptance(output: Path) -> None:
    target = output / "frozen/CONFIRMATORY_ACCEPTANCE.md"
    if target.exists():
        return
    target.write_text(
        """# Confirmatory Acceptance Criteria

These criteria were frozen before generation.

1. Run exactly 18 generation cells: three chemistries, two architectures, and three repeats, using GPT-5.4 for both architectures.
2. Obtain exactly 54 valid blinded judgments from Qwen, OpenAI, and Claude; retain every retry and failure.
3. Do not add, replace, or remove repeats in response to observed scores.
4. All nine FlowPilot outputs must close their final contracts or retain any blocked output as a failure.
5. No CuAAC FlowPilot output may serialize the stationary Cu/C catalyst in a pumped stream.
6. The target quality result is a non-negative mean matched FlowPilot-minus-one-shot score across the nine pairs.
7. The sensitivity target is a non-negative matched effect after excluding the OpenAI judge from OpenAI-generated candidates.

Failure of a target triggers defect analysis and code correction, not selective replacement of campaign outcomes. Any subsequent campaign receives a new directory and remains separate.
""",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument(
        "--phase",
        choices=("init", "health", "generate", "packets", "judge", "report", "all"),
        default="all",
    )
    parser.add_argument("--pause-between-calls", type=float, default=1.0)
    parser.add_argument("--skip-health", action="store_true")
    parser.add_argument(
        "--judges",
        nargs="+",
        choices=("qwen", "openai", "claude"),
        default=["qwen", "openai", "claude"],
    )
    parser.add_argument("--judge-shard-index", type=int, default=0)
    parser.add_argument("--judge-shard-count", type=int, default=1)
    args = parser.parse_args()

    configure()
    output = args.output.resolve()
    report = args.report.resolve()
    output.mkdir(parents=True, exist_ok=True)
    cases = campaign.selected_cases()
    campaign.initialize(output, cases)
    write_acceptance(output)
    if args.phase == "init":
        print(output)
        return
    if args.phase in {"health", "generate", "judge", "all"} and not args.skip_health:
        campaign.health_check(output)
    if args.phase in {"generate", "all"}:
        campaign.run_generation(output, cases, ("openai",), args.pause_between_calls)
    if args.phase in {"packets", "all"}:
        campaign.build_packets(output, cases)
    if args.phase in {"judge", "all"}:
        campaign.run_judges(
            output,
            tuple(args.judges),
            args.pause_between_calls,
            shard_index=args.judge_shard_index,
            shard_count=args.judge_shard_count,
        )
    if args.phase in {"report", "all"}:
        campaign.build_repeated_report(output, report)
    print(output)
    print(report)


if __name__ == "__main__":
    main()
