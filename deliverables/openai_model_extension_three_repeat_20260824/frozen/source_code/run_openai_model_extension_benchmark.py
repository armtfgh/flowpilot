#!/usr/bin/env python3
"""Run the frozen NewGen 2.0 extension across three additional GPT models."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ablation_test.scripts import run_manuscript_three_model_repeated_benchmark as campaign
from ablation_test.scripts.repeat_completion import freeze_wrapper_sources


DEFAULT_OUTPUT = (
    ROOT
    / "ablation_results/manuscript_benchmark/openai_model_extension_three_repeat_20260824"
)
DEFAULT_REPORT = ROOT / "deliverables/openai_model_extension_three_repeat_20260824"


def configure() -> None:
    campaign.MODELS = {
        "gpt55": {
            "provider": "openai",
            "model": "gpt-5.5-2026-04-23",
            "upstream_mode": "never",
            "family": "openai",
            "display": "GPT-5.5",
        },
        "gpt54mini": {
            "provider": "openai",
            "model": "gpt-5.4-mini-2026-03-17",
            "upstream_mode": "never",
            "family": "openai",
            "display": "GPT-5.4 mini",
        },
        "gpt52": {
            "provider": "openai",
            "model": "gpt-5.2-2025-12-11",
            "upstream_mode": "never",
            "family": "openai",
            "display": "GPT-5.2",
        },
    }
    campaign.REPEAT_IDS = ("repeat_01", "repeat_02", "repeat_03")
    campaign.JUDGES = ("qwen", "openai", "claude")
    campaign.SEED_NAMESPACE = "openai-model-extension-three-repeat-20260824-v1"
    campaign.MATCHED_SEED_ACROSS_ARCHITECTURES = True


def write_acceptance(output: Path) -> None:
    target = output / "frozen/ACCEPTANCE_CRITERIA.md"
    if target.exists():
        return
    target.write_text(
        """# Frozen Acceptance Criteria

These criteria were frozen before generation.

1. Run exactly 54 generation cells: three frozen chemistries, two architectures, three repeats, and three GPT models.
2. Retain all outcomes, retries, invalid outputs, blocked designs, and failed calls without selective replacement.
3. Use identical frozen case inputs, KHU baseline inventory, NewGen 2.0 rubric, architecture definitions, and candidate budgets.
4. Use matched seeds across one-shot and FlowPilot within each model, chemistry, and repeat.
5. Obtain exactly 162 valid blinded judgments from Qwen, OpenAI, and Claude, retaining judge failures and retries.
6. Report each model separately with matched FlowPilot-minus-one-shot effects and repeat-level standard deviations.
7. Treat this campaign as an extension; do not overwrite or pool it silently with the earlier GPT-5.4 campaign.
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
    parser.add_argument("--pause-between-calls", type=float, default=0.5)
    parser.add_argument("--skip-health", action="store_true")
    parser.add_argument(
        "--families",
        nargs="+",
        choices=("gpt55", "gpt54mini", "gpt52"),
        default=["gpt55", "gpt54mini", "gpt52"],
    )
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
    cases = campaign.selected_cases()
    campaign.initialize(output, cases)
    freeze_wrapper_sources(output, Path(__file__))
    write_acceptance(output)
    if args.phase == "init":
        print(output)
        return
    if args.phase in {"health", "generate", "judge", "all"} and not args.skip_health:
        campaign.health_check(output)
    if args.phase in {"generate", "all"}:
        campaign.run_generation(output, cases, tuple(args.families), args.pause_between_calls)
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
