#!/usr/bin/env python3
"""Run one frozen matched NewGen 2.0 round with Qwen3.8-27B."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ablation_test.scripts import run_manuscript_three_model_repeated_benchmark as campaign


DEFAULT_OUTPUT = ROOT / "ablation_results/manuscript_benchmark/qwen38_one_round_20260821"
DEFAULT_REPORT = ROOT / "deliverables/qwen38_one_round_20260821"


def configure() -> None:
    campaign.MODELS = {
        "qwen38": {
            "provider": "ollama",
            "model": "/models/Qwen3.8-27B",
            "base_url": "http://10.13.24.104:8000/v1",
            "upstream_mode": "always",
            "family": "qwen",
            "display": "Qwen3.8-27B",
        }
    }
    campaign.REPEAT_IDS = ("repeat_01",)
    campaign.JUDGES = ("qwen", "openai", "claude")
    campaign.SEED_NAMESPACE = "qwen38-one-round-20260821-v1"


def write_acceptance(output: Path) -> None:
    target = output / "frozen/CONFIRMATORY_ACCEPTANCE.md"
    if target.exists():
        return
    target.write_text(
        """# Qwen3.8 One-Round Acceptance Criteria

Frozen before generation:

1. Generate exactly six outcomes: three frozen chemistries, two architectures, and one Qwen3.8-27B call per cell.
2. Use the same authoritative case inventories and unchanged 14-criterion NewGen 2.0 rubric as the final benchmark.
3. Obtain exactly 18 selected valid blinded judgments from the existing Qwen, OpenAI, and Claude judge panel; retain all failed attempts and retries.
4. Do not add or replace outcomes according to observed scores.
5. Retain blocked FlowPilot outcomes as failures; do not substitute another run.
6. Report this one-round campaign separately from the three-repeat final benchmark.
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
        campaign.run_generation(output, cases, ("qwen38",), args.pause_between_calls)
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
