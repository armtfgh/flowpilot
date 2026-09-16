#!/usr/bin/env python3
"""Run the frozen NewGen 2.0 Qwen3.8 council-module attribution study."""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ablation_test.scripts import run_manuscript_three_model_repeated_benchmark as campaign
from ablation_test.scripts.build_qwen38_module_ablation_report import build_report


DEFAULT_OUTPUT = ROOT / "ablation_results/newgen_2_0_module_attribution/qwen38_v1"
DEFAULT_REPORT = ROOT / "deliverables/newgen_2_0_module_attribution_qwen38_v1"
ALL_CASES = (
    "Hydrogenolysis",
    "Photochemical oxidation",
    "CuAAC",
    "Two-stage amidation",
    "Exothermic dinitration",
)

FULL = {
    "condition_id": "full",
    "enabled_scoring_agents": ["chemistry", "kinetics", "fluidics", "safety"],
    "enable_skeptic": True,
    "enable_preselection_refinement": True,
    "enable_winner_revision": True,
    "enable_chief_llm": True,
    "enable_dfmea": True,
}

CONDITIONS = {
    "one_shot": ("general_one_shot", "One-shot"),
    "no_council": ("no_council", "No council"),
    "full": ("full", "Full FlowPilot"),
    "minus_chemistry": ("full", "Without Chemistry agent"),
    "minus_kinetics": ("full", "Without Kinetics agent"),
    "minus_fluidics": ("full", "Without Fluidics agent"),
    "minus_safety": ("full", "Without Safety agent"),
    "minus_skeptic": ("full", "Without Skeptic audit"),
    "minus_preselection": ("full", "Without preselection refinement"),
    "minus_winner_revision": ("full", "Without winner revision"),
    "deterministic_chief": ("full", "Deterministic selection"),
    "minus_dfmea": ("full", "Without DFMEA"),
    "round_1": ("full", "One-pass council"),
    "budget_1": ("full", "Candidate budget 1"),
    "budget_4": ("full", "Candidate budget 4"),
}


def changed(**updates):
    value = dict(FULL)
    value.update(updates)
    value["condition_id"] = str(updates.get("condition_id") or "ablation")
    return value


EXECUTION_CONFIGS = {
    "full": FULL,
    "minus_chemistry": changed(
        condition_id="minus_chemistry",
        enabled_scoring_agents=["kinetics", "fluidics", "safety"],
    ),
    "minus_kinetics": changed(
        condition_id="minus_kinetics",
        enabled_scoring_agents=["chemistry", "fluidics", "safety"],
    ),
    "minus_fluidics": changed(
        condition_id="minus_fluidics",
        enabled_scoring_agents=["chemistry", "kinetics", "safety"],
    ),
    "minus_safety": changed(
        condition_id="minus_safety",
        enabled_scoring_agents=["chemistry", "kinetics", "fluidics"],
    ),
    "minus_skeptic": changed(condition_id="minus_skeptic", enable_skeptic=False),
    "minus_preselection": changed(
        condition_id="minus_preselection", enable_preselection_refinement=False
    ),
    "minus_winner_revision": changed(
        condition_id="minus_winner_revision", enable_winner_revision=False
    ),
    "deterministic_chief": changed(
        condition_id="deterministic_chief", enable_chief_llm=False
    ),
    "minus_dfmea": changed(condition_id="minus_dfmea", enable_dfmea=False),
    "round_1": changed(
        condition_id="round_1",
        enable_skeptic=False,
        enable_preselection_refinement=False,
        enable_winner_revision=False,
        enable_dfmea=False,
    ),
    "budget_1": changed(condition_id="budget_1"),
    "budget_4": changed(condition_id="budget_4"),
}


def configure(*, cases: tuple[str, ...], repeats: int) -> None:
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
    campaign.SELECTED_CASES = cases
    campaign.REPEAT_IDS = tuple(f"repeat_{index:02d}" for index in range(1, repeats + 1))
    campaign.JUDGES = ("qwen", "openai", "claude")
    campaign.ARCHITECTURES = CONDITIONS
    campaign.EXECUTION_CONFIGS = EXECUTION_CONFIGS
    campaign.CANDIDATE_BUDGETS = {"budget_1": 1, "budget_4": 4}
    campaign.CANDIDATE_BUDGET = 2
    campaign.MATCHED_SEED_ACROSS_ARCHITECTURES = True
    campaign.SEED_NAMESPACE = "newgen-2-module-attribution-qwen38-v1"


def selected_cases():
    by_label = dict(campaign.load_frozen_cases())
    return [(label, by_label[label]) for label in campaign.SELECTED_CASES]


def write_preregistration(output: Path, cases, repeats: int) -> None:
    frozen = output / "frozen"
    shutil.copy2(Path(__file__), frozen / Path(__file__).name)
    (frozen / "MODULE_ATTRIBUTION_PREREGISTRATION.md").write_text(
        "# NewGen 2.0 Qwen3.8 Module Attribution\n\n"
        f"Frozen cases: {', '.join(label for label, _ in cases)}.\n\n"
        f"Frozen conditions: {', '.join(CONDITIONS)}.\n\n"
        f"Repeats: {repeats}. Generator: Qwen3.8-27B for every condition. "
        "The same protocol, objective, held-out-source exclusion, strict case inventory, "
        "temperature, rubric, and judge panel apply to every matched cell. Missing agents "
        "receive no call and no zero-score penalty; active weights are renormalized. "
        "All outcomes, including blocked outcomes, remain in the analysis. No reruns are "
        "added in response to scores.\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument(
        "--profile",
        choices=("pilot", "screen3", "screen", "confirmatory"),
        default="pilot",
    )
    parser.add_argument(
        "--phase",
        choices=("init", "health", "generate", "packets", "judge", "report", "all"),
        default="all",
    )
    parser.add_argument("--pause-between-calls", type=float, default=0.0)
    parser.add_argument("--skip-health", action="store_true")
    parser.add_argument("--judges", nargs="+", choices=("qwen", "openai", "claude"), default=["qwen", "openai", "claude"])
    args = parser.parse_args()

    if args.profile == "pilot":
        cases, repeats = ("CuAAC",), 1
    elif args.profile == "screen3":
        cases, repeats = ALL_CASES[:3], 1
    elif args.profile == "screen":
        cases, repeats = ALL_CASES, 1
    else:
        cases, repeats = ALL_CASES, 3
    configure(cases=cases, repeats=repeats)
    output, report = args.output.resolve(), args.report.resolve()
    output.mkdir(parents=True, exist_ok=True)
    frozen_cases = selected_cases()
    campaign.initialize(output, frozen_cases)
    write_preregistration(output, frozen_cases, repeats)
    if args.phase == "init":
        print(output)
        return
    if args.phase in {"health", "generate", "judge", "all"} and not args.skip_health:
        campaign.health_check(output)
    if args.phase in {"generate", "all"}:
        campaign.run_generation(output, frozen_cases, ("qwen38",), args.pause_between_calls)
    if args.phase in {"packets", "all"}:
        campaign.build_packets(output, frozen_cases)
    if args.phase in {"judge", "all"}:
        campaign.run_judges(output, tuple(args.judges), args.pause_between_calls)
    if args.phase in {"report", "all"}:
        build_report(output, report)
    print(output)
    print(report)


if __name__ == "__main__":
    main()
