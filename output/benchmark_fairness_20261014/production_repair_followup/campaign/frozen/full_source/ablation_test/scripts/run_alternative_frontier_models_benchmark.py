#!/usr/bin/env python3
"""Run a frozen NewGen 2.0 comparison with alternative frontier models."""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ablation_test.scripts import run_manuscript_three_model_repeated_benchmark as campaign


DEFAULT_OUTPUT = (
    ROOT
    / "ablation_results/manuscript_benchmark/alternative_frontier_one_round_20260821"
)
DEFAULT_REPORT = ROOT / "deliverables/alternative_frontier_one_round_20260821"


def configure() -> None:
    campaign.MODELS = {
        "gpt4o": {
            "provider": "openai",
            "model": "gpt-4o",
            "upstream_mode": "never",
            "family": "openai",
            "display": "GPT-4o",
        },
        "claude_opus": {
            "provider": "anthropic",
            "model": "claude-opus-4-6",
            "upstream_mode": "never",
            "family": "anthropic",
            "display": "Claude Opus 4.6",
        },
    }
    campaign.REPEAT_IDS = ("repeat_01",)
    campaign.JUDGES = ("qwen", "openai", "claude")
    campaign.SEED_NAMESPACE = "alternative-frontier-one-round-20260821-v1"
    campaign.MATCHED_SEED_ACROSS_ARCHITECTURES = True


def write_frozen_addendum(output: Path) -> None:
    frozen = output / "frozen"
    acceptance = frozen / "ALTERNATIVE_FRONTIER_ACCEPTANCE.md"
    if not acceptance.exists():
        acceptance.write_text(
            """# Alternative Frontier Model Campaign

Frozen before generation:

1. Run exactly 12 outcomes: three fixed held-out chemistries, two generator models, two architectures, and one fresh call per cell.
2. Compare GPT-4o and Claude Opus 4.6 separately; do not pool them into one generator estimate.
3. Use the same authoritative case inputs, strict inventories, hidden source references, and unchanged 14-criterion NewGen 2.0 rubric used by the prior manuscript campaigns.
4. Use the same three judge families: Qwen, OpenAI, and Claude. Generator and architecture labels remain withheld from judges.
5. Retain every generation failure, blocked FlowPilot result, invalid judgment, and retry. Do not replace outcomes according to observed scores.
6. Stop after the frozen 12 outcomes and 36 selected judgments. This is a one-round extension and is not a precision replacement for the three-repeat campaign.
7. Report deterministic validity and critical-error checks beside LLM-judge scores, including generator-family-excluded sensitivity.
""",
            encoding="utf-8",
        )
    wrapper_copy = frozen / "source_code" / Path(__file__).name
    wrapper_copy.parent.mkdir(parents=True, exist_ok=True)
    if not wrapper_copy.exists():
        shutil.copy2(Path(__file__), wrapper_copy)


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
    write_frozen_addendum(output)

    if args.phase == "init":
        print(output)
        return
    if args.phase in {"health", "generate", "judge", "all"} and not args.skip_health:
        campaign.health_check(output)
    if args.phase in {"generate", "all"}:
        campaign.run_generation(
            output,
            cases,
            ("gpt4o", "claude_opus"),
            args.pause_between_calls,
        )
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
