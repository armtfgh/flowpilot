#!/usr/bin/env python3
"""Complete Qwen3.8 NewGen 2.0 to three repeats without rerunning repeat 1."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ablation_test.scripts import run_manuscript_three_model_repeated_benchmark as campaign
from ablation_test.scripts.repeat_completion import bootstrap_repeat_one, freeze_wrapper_sources


SOURCE = ROOT / "ablation_results/manuscript_benchmark/qwen38_one_round_coverage_repair_20260821"
DEFAULT_OUTPUT = ROOT / "ablation_results/manuscript_benchmark/qwen38_three_repeat_20260824"
DEFAULT_REPORT = ROOT / "deliverables/qwen38_three_repeat_20260824"


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
    campaign.REPEAT_IDS = ("repeat_01", "repeat_02", "repeat_03")
    campaign.JUDGES = ("qwen", "openai", "claude")
    campaign.SEED_NAMESPACE = "qwen38-one-round-20260821-v1"
    campaign.MATCHED_SEED_ACROSS_ARCHITECTURES = False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--phase", choices=("init", "health", "generate", "packets", "judge", "report", "all"), default="all")
    parser.add_argument("--pause-between-calls", type=float, default=0.5)
    parser.add_argument("--skip-health", action="store_true")
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
        campaign.run_generation(output, cases, ("qwen38",), args.pause_between_calls)
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
