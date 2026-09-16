#!/usr/bin/env python3
"""Complete GPT-4o and Claude Opus 4.6 NewGen 2.0 to three repeats."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ablation_test.scripts import run_manuscript_three_model_repeated_benchmark as campaign
from ablation_test.scripts.repeat_completion import bootstrap_repeat_one, freeze_wrapper_sources


SOURCE = ROOT / "ablation_results/manuscript_benchmark/alternative_frontier_one_round_20260821"
DEFAULT_OUTPUT = ROOT / "ablation_results/manuscript_benchmark/alternative_frontier_three_repeat_20260824"
DEFAULT_REPORT = ROOT / "deliverables/alternative_frontier_three_repeat_20260824"


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
    campaign.REPEAT_IDS = ("repeat_01", "repeat_02", "repeat_03")
    campaign.JUDGES = ("qwen", "openai", "claude")
    campaign.SEED_NAMESPACE = "alternative-frontier-one-round-20260821-v1"
    campaign.MATCHED_SEED_ACROSS_ARCHITECTURES = True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--phase", choices=("init", "health", "generate", "packets", "judge", "report", "all"), default="all")
    parser.add_argument("--pause-between-calls", type=float, default=0.5)
    parser.add_argument("--skip-health", action="store_true")
    parser.add_argument("--families", nargs="+", choices=("gpt4o", "claude_opus"), default=["gpt4o", "claude_opus"])
    parser.add_argument("--judges", nargs="+", choices=("qwen", "openai", "claude"), default=["qwen", "openai", "claude"])
    args = parser.parse_args()

    configure()
    output = args.output.resolve()
    report = args.report.resolve()
    cases = campaign.selected_cases()
    campaign.initialize(output, cases)
    freeze_wrapper_sources(output, Path(__file__), ROOT / "ablation_test/scripts/repeat_completion.py")
    bootstrap_repeat_one(SOURCE, output)
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
        campaign.run_judges(output, tuple(args.judges), args.pause_between_calls)
    if args.phase in {"report", "all"}:
        campaign.build_repeated_report(output, report)
    print(output)
    print(report)


if __name__ == "__main__":
    main()
