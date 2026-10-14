#!/usr/bin/env python3
"""Run the preregistered repeated manuscript benchmark.

The campaign is deliberately separate from the accepted five-case pilot.  It
uses the top three cases selected by the pilot's frozen ranking, regenerates
both architectures for every repeat, and adds Claude as generator and judge.
Every call is checkpointed so an interrupted campaign can be resumed safely.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import statistics
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ablation_test.scripts.build_newgen_2_0_report import aggregate
from ablation_test.scripts.run_manuscript_five_case_benchmark import (
    ARCHITECTURES,
    ORACLES,
    RUBRIC,
    deterministic_sheet,
    load_frozen_cases,
    safe_name,
    sha256_file,
)
from ablation_test.scripts.run_newgen_2_0_benchmark import call_judge
from ablation_test.src.cases import AblationCase
from ablation_test.src.holistic_audit import build_case_context
from ablation_test.src.newgen_2_outcome import (
    build_outcome_packet,
    candidate_id,
    packet_gate,
    read_json,
    stable_seed,
    write_json,
)
from ablation_test.src.providers import endpoint_health
from ablation_test.src.runner import execute_cell


DEFAULT_OUTPUT = ROOT / "ablation_results/manuscript_benchmark/manuscript_three_model_three_repeat_20260819"
DEFAULT_REPORT = ROOT / "deliverables/manuscript_three_model_three_repeat_20260819"
SELECTED_CASES = ("Hydrogenolysis", "Photochemical oxidation", "CuAAC")
REPEAT_IDS = ("repeat_01", "repeat_02", "repeat_03")
JUDGES = ("qwen", "openai", "claude")
MODELS = {
    "qwen": {
        "provider": "ollama",
        "model": "/models/Qwen3.6-27B",
        "base_url": "http://10.13.24.169:8000/v1",
        "upstream_mode": "always",
        "family": "qwen",
        "display": "Qwen3.6-27B",
    },
    "claude": {
        "provider": "anthropic",
        "model": "claude-sonnet-4-6",
        "upstream_mode": "never",
        "family": "anthropic",
        "display": "Claude Sonnet 4.6",
    },
}
TEMPERATURE = 0.2
CANDIDATE_BUDGET = 2
SEED_NAMESPACE = "manuscript-three-model-three-repeat-v1"
EXECUTION_CONFIGS: dict[str, dict[str, Any]] = {}
CANDIDATE_BUDGETS: dict[str, int] = {}
MATCHED_SEED_ACROSS_ARCHITECTURES = False


def append_jsonl(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False) + "\n")


def selected_cases() -> list[tuple[str, AblationCase]]:
    by_label = dict(load_frozen_cases())
    return [(label, by_label[label]) for label in SELECTED_CASES]


def campaign_cells(cases: list[tuple[str, AblationCase]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for label, case in cases:
        for family in MODELS:
            for architecture_key, (variant, architecture) in ARCHITECTURES.items():
                for repeat_id in REPEAT_IDS:
                    rows.append(
                        {
                            "case_label": label,
                            "case_id": case.case_id,
                            "generator_family": family,
                            "generator_provider_family": MODELS[family]["family"],
                            "condition_id": f"{family}_{architecture_key}",
                            "architecture_key": architecture_key,
                            "architecture": architecture,
                            "variant": variant,
                            "repeat_id": repeat_id,
                            "model": MODELS[family]["display"],
                            "model_id": MODELS[family]["model"],
                            "design_input_sha256": case.design_input_sha256,
                            "seed": stable_seed(
                                SEED_NAMESPACE,
                                case.case_id,
                                family,
                                "matched" if MATCHED_SEED_ACROSS_ARCHITECTURES else architecture_key,
                                repeat_id,
                            ),
                            "execution_config": EXECUTION_CONFIGS.get(architecture_key),
                            "candidate_budget": CANDIDATE_BUDGETS.get(
                                architecture_key, CANDIDATE_BUDGET
                            ),
                        }
                    )
    return rows


def run_directory(output: Path, row: dict[str, Any]) -> Path:
    return (
        output
        / "generation"
        / safe_name(row["case_label"])
        / row["condition_id"]
        / row["repeat_id"]
    )


def completed_run(path: Path) -> bool:
    summary = path / "run_summary.json"
    return (
        (path / "result.json").is_file()
        and (path / "metrics.json").is_file()
        and summary.is_file()
        and read_json(summary).get("status") == "completed"
    )


def archive_failed_attempt(run_dir: Path) -> None:
    if not run_dir.exists() or completed_run(run_dir):
        return
    artifacts = [path for path in run_dir.iterdir() if path.name != "attempts"]
    if not artifacts:
        return
    attempts = run_dir / "attempts"
    attempt_id = f"attempt_{len(list(attempts.glob('attempt_*'))) + 1:02d}"
    target = attempts / attempt_id
    target.mkdir(parents=True, exist_ok=False)
    for path in artifacts:
        shutil.move(str(path), str(target / path.name))


def update_progress(output: Path, phase: str, completed: int, total: int, current: str = "") -> None:
    write_json(
        output / "progress.json",
        {
            "phase": phase,
            "completed": completed,
            "total": total,
            "remaining": total - completed,
            "current": current,
            "updated_at": datetime.now().astimezone().isoformat(),
        },
    )


def initialize(output: Path, cases: list[tuple[str, AblationCase]]) -> None:
    frozen = output / "frozen"
    frozen.mkdir(parents=True, exist_ok=True)
    rubric = frozen / "outcome_rubric.json"
    if not rubric.exists():
        shutil.copy2(RUBRIC, rubric)
    cells = campaign_cells(cases)
    manifest = {
        "schema_version": "flowpilot_manuscript_three_model_repeated_v1.0",
        "benchmark_name": f"{len(MODELS)}-model repeated architecture benchmark",
        "created_at": datetime.now().astimezone().isoformat(),
        "case_selection_source": "Frozen manuscript five-case NewGen 2.0 suite",
        "case_selection_rule": (
            "Cases were fixed by the wrapper campaign before generation; no outcome "
            "from this campaign was inspected before selection."
        ),
        "selected_cases": list(SELECTED_CASES),
        "case_count": len(cases),
        "generator_families": list(MODELS),
        "architectures": [value[1] for value in ARCHITECTURES.values()],
        "repeat_count": len(REPEAT_IDS),
        "candidate_count": len(cells),
        "selected_judges": list(JUDGES),
        "judge_count": len(JUDGES),
        "planned_judgments": len(cells) * len(JUDGES),
        "criteria_per_candidate": 14,
        "equal_criterion_weights": True,
        "candidate_budget_flowpilot": CANDIDATE_BUDGET,
        "condition_candidate_budgets": CANDIDATE_BUDGETS,
        "condition_execution_configs": EXECUTION_CONFIGS,
        "matched_seed_across_architectures": MATCHED_SEED_ACROSS_ARCHITECTURES,
        "temperature": TEMPERATURE,
        "independent_call_policy": "Every repeat is a fresh provider call with a distinct recorded stable seed. Anthropic may not honor seed even though calls remain independent.",
        "retry_policy": "Up to three campaign attempts per generation or judgment call; every failed attempt is retained.",
        "stopping_rule": (
            f"Run exactly the frozen {len(cells)} generation cells and "
            f"{len(cells) * len(JUDGES)} judgments; do not add repeats in "
            "response to observed scores."
        ),
        "rubric_frozen_before_generation": True,
        "rubric_sha256": sha256_file(rubric),
        "held_out_source_excluded_from_retrieval": True,
        "architecture_label_withheld_from_judges": True,
        "architecture_blinding_claimed": False,
        "generator_identity_withheld_from_judges": True,
        "models": {
            family: {
                key: value
                for key, value in bundle.items()
                if key not in {"base_url"}
            }
            for family, bundle in MODELS.items()
        },
    }
    manifest_path = frozen / "campaign_manifest.json"
    if manifest_path.exists():
        prior = read_json(manifest_path)
        immutable = (
            "selected_cases",
            "candidate_count",
            "repeat_count",
            "selected_judges",
            "architectures",
            "candidate_budget_flowpilot",
            "condition_candidate_budgets",
            "condition_execution_configs",
            "matched_seed_across_architectures",
            "temperature",
            "rubric_sha256",
            "models",
        )
        for key in immutable:
            if prior.get(key) != manifest.get(key):
                raise RuntimeError(f"Frozen campaign mismatch for {key}; use a new output directory")
    else:
        write_json(manifest_path, manifest)
    write_json(
        frozen / "case_manifest.json",
        {
            "suite_id": "manuscript_three_model_three_repeat_v1",
            "cases": [
                {"case_label": label, **case.manifest_payload()}
                for label, case in cases
            ],
        },
    )
    write_json(frozen / "execution_plan.json", {"cells": cells})
    source_dir = frozen / "source_code"
    source_dir.mkdir(exist_ok=True)
    source_files = (
        Path(__file__),
        ROOT / "ablation_test/scripts/run_manuscript_five_case_benchmark.py",
        ROOT / "ablation_test/scripts/run_newgen_2_0_benchmark.py",
        ROOT / "ablation_test/scripts/build_newgen_2_0_report.py",
        ROOT / "ablation_test/src/runner.py",
        ROOT / "ablation_test/src/newgen_2_outcome.py",
        ROOT / "flora_translate/main.py",
        ROOT / "flora_translate/design_realizer.py",
        ROOT / "flora_translate/final_design_contract.py",
        ROOT / "flora_translate/engine/llm_agents.py",
        ROOT / "flora_translate/engine/council_v4/scoring.py",
        ROOT / "flora_translate/engine/council_v4/chief.py",
        ROOT / "flora_translate/engine/council_v4/execution_config.py",
    )
    hashes: dict[str, str] = {}
    for source in source_files:
        target = source_dir / source.name
        if target.is_file():
            if sha256_file(target) != sha256_file(source):
                raise RuntimeError(
                    f"Frozen source mismatch for {source.name}; use a new output directory"
                )
        else:
            shutil.copy2(source, target)
        hashes[source.name] = sha256_file(target)
    checksum_path = frozen / "source_code_checksums.json"
    if checksum_path.is_file():
        if read_json(checksum_path) != hashes:
            raise RuntimeError(
                "Frozen source checksum manifest changed; use a new output directory"
            )
    else:
        write_json(checksum_path, hashes)
    oracle_dir = frozen / "held_out_oracles"
    oracle_dir.mkdir(exist_ok=True)
    for label, _ in cases:
        target = oracle_dir / f"{safe_name(label)}.json"
        if not target.exists():
            shutil.copy2(ORACLES[label], target)
    model_names = ", ".join(bundle["display"] for bundle in MODELS.values())
    judge_names = ", ".join(name.title() for name in JUDGES)
    (frozen / "PREREGISTRATION.md").write_text(
        "# Frozen Campaign\n\n"
        "This campaign uses three cases selected before generation from the accepted five-case pilot ranking. "
        f"It runs {model_names} across the frozen architecture conditions, with fresh calls per cell. "
        f"{judge_names} independently judge every blinded outcome using the unchanged 14-criterion NewGen 2.0 rubric. "
        f"Exactly {len(cells)} outcomes and {len(cells) * len(JUDGES)} judgments are planned.\n",
        encoding="utf-8",
    )


def health_check(output: Path) -> None:
    status = {name: endpoint_health(bundle) for name, bundle in MODELS.items()}
    write_json(output / "endpoint_health.json", status)
    failed = [
        name
        for name, row in status.items()
        if not row.get("reachable") or not row.get("model_advertised")
    ]
    if failed:
        raise RuntimeError(f"Unavailable benchmark endpoints: {failed}")


def run_generation(
    output: Path,
    cases: list[tuple[str, AblationCase]],
    families: tuple[str, ...],
    pause_s: float,
) -> None:
    case_by_id = {case.case_id: case for _, case in cases}
    cells = [row for row in campaign_cells(cases) if row["generator_family"] in families]
    complete = sum(completed_run(run_directory(output, row)) for row in cells)
    update_progress(output, "generation", complete, len(cells))
    for index, row in enumerate(cells, start=1):
        run_dir = run_directory(output, row)
        if completed_run(run_dir):
            print(f"GEN {index}/{len(cells)} skip {row['case_label']} {row['condition_id']} {row['repeat_id']}", flush=True)
            continue
        last_status: dict[str, Any] = {}
        for attempt in range(1, 4):
            archive_failed_attempt(run_dir)
            append_jsonl(
                output / "campaign_events.jsonl",
                {
                    "event": "generation_started",
                    "cell": row,
                    "campaign_attempt": attempt,
                    "at": datetime.now().astimezone().isoformat(),
                },
            )
            last_status = execute_cell(
                case=case_by_id[row["case_id"]],
                variant=row["variant"],
                bundle_name=row["condition_id"],
                bundle=MODELS[row["generator_family"]],
                run_dir=run_dir,
                candidate_budget=int(row.get("candidate_budget") or CANDIDATE_BUDGET),
                temperature=TEMPERATURE,
                seed=row["seed"],
                execution_config=row.get("execution_config"),
            )
            if completed_run(run_dir):
                break
        complete = sum(completed_run(run_directory(output, item)) for item in cells)
        update_progress(
            output,
            "generation",
            complete,
            len(cells),
            f"{row['case_label']} / {row['condition_id']} / {row['repeat_id']}",
        )
        append_jsonl(
            output / "campaign_events.jsonl",
            {
                "event": "generation_finished",
                "cell": row,
                "status": last_status.get("status"),
                "at": datetime.now().astimezone().isoformat(),
            },
        )
        print(
            f"GEN {index}/{len(cells)} {row['case_label']} {row['condition_id']} "
            f"{row['repeat_id']}: {last_status.get('status')}",
            flush=True,
        )
        if pause_s:
            time.sleep(pause_s)
    failed = [row for row in cells if not completed_run(run_directory(output, row))]
    if failed:
        write_json(output / "generation_failures.json", {"cells": failed})
        raise RuntimeError(f"Generation incomplete for {len(failed)} cells; all attempts retained")


def build_packets(output: Path, cases: list[tuple[str, AblationCase]]) -> None:
    case_by_id = {case.case_id: (label, case) for label, case in cases}
    contexts = {
        case.case_id: build_case_context(
            case_label=label,
            public_input=case.public_payload(),
            inventory=case.inventory,
            oracle=read_json(ORACLES[label]),
        )
        for label, case in cases
    }
    packets = output / "packets"
    packets.mkdir(parents=True, exist_ok=True)
    key_rows: list[dict[str, Any]] = []
    packet_metrics: list[dict[str, Any]] = []
    for row in campaign_cells(cases):
        run_dir = run_directory(output, row)
        label, case = case_by_id[row["case_id"]]
        cid = candidate_id(str(run_dir.relative_to(output)))
        generation_complete = completed_run(run_dir)
        if generation_complete:
            result = read_json(run_dir / "result.json")
            metrics = read_json(run_dir / "metrics.json")
            verification = deterministic_sheet(result, metrics, case)
            result_sha256 = sha256_file(run_dir / "result.json")
            generation_status = "completed"
            generation_error_type = None
        else:
            # A model that exhausts the frozen generation attempts delivered no
            # design. Preserve that reliability failure as a blinded outcome;
            # aborting packet construction would silently remove the worst cell.
            summary_path = run_dir / "run_summary.json"
            error_path = run_dir / "error.json"
            summary = read_json(summary_path) if summary_path.is_file() else {}
            error = read_json(error_path) if error_path.is_file() else {}
            generation_error_type = (
                error.get("type") or summary.get("error_type") or "GenerationError"
            )
            result = {
                "variant": row["variant"],
                "final_design": {"status": "generation_failed"},
                "proposal": {},
                "reported_disposition": "GENERATION_FAILED",
                "disposition_rationale": (
                    "No candidate design was delivered after the frozen generation "
                    "attempts because the response failed the output contract."
                ),
            }
            verification = {
                "generation_status": "failed",
                "delivered_design_available": False,
                "formal_output_contract_valid": False,
                "error_type": generation_error_type,
                "evaluation_instruction": (
                    "Score the absence of a delivered design as a reliability and "
                    "executability failure; do not infer missing parameters."
                ),
            }
            result_sha256 = None
            generation_status = "failed"
        packet = build_outcome_packet(
            candidate=cid,
            case_context=contexts[case.case_id],
            result=result,
            deterministic_verification=verification,
            has_gas=bool(case.expected_features.get("gas_required")),
            is_multistage="multistep" in case.category or "multistage" in case.category,
        )
        errors = packet_gate(packet)
        if errors:
            raise RuntimeError(f"Packet gate failed for {cid}: {errors}")
        write_json(packets / f"{cid}.json", packet)
        key_rows.append(
            {
                "candidate_id": cid,
                "case": label,
                "case_id": case.case_id,
                "generator_model": row["model"],
                "generator_model_id": row["model_id"],
                "generator_family": row["generator_provider_family"],
                "generator_key": row["generator_family"],
                "architecture": row["architecture"],
                "condition_id": row["condition_id"],
                "repeat_id": row["repeat_id"],
                "seed": row["seed"],
                "candidate_budget": row.get("candidate_budget"),
                "execution_config": row.get("execution_config"),
                "run_directory": str(run_dir),
                "result_sha256": result_sha256,
                "generation_status": generation_status,
                "generation_error_type": generation_error_type,
                "design_input_sha256": case.design_input_sha256,
            }
        )
        packet_metrics.append(
            {
                "candidate_id": cid,
                "bytes": (packets / f"{cid}.json").stat().st_size,
                "gate_errors": [],
            }
        )
    write_json(output / "frozen/candidate_key_confidential.json", {"candidates": key_rows})
    write_json(output / "frozen/packet_metrics.json", packet_metrics)
    update_progress(output, "packets", len(key_rows), len(key_rows))


def valid_judgment(output: Path, judge: str, cid: str) -> bool:
    base = output / "judgments" / judge / cid
    statuses = [base / "status.json", *base.glob("attempts/attempt_*/status.json")]
    return any(
        path.is_file() and read_json(path).get("status") == "valid"
        for path in statuses
    )


def run_judges(
    output: Path,
    judges: tuple[str, ...],
    pause_s: float,
    *,
    shard_index: int = 0,
    shard_count: int = 1,
) -> None:
    if shard_count < 1 or not 0 <= shard_index < shard_count:
        raise ValueError("judge shard must satisfy 0 <= shard_index < shard_count")
    candidates = read_json(output / "frozen/candidate_key_confidential.json")["candidates"]
    rubric = read_json(output / "frozen/outcome_rubric.json")
    all_tasks = [(judge, candidate) for judge in judges for candidate in candidates]
    tasks = [
        task
        for index, task in enumerate(all_tasks)
        if index % shard_count == shard_index
    ]
    complete = sum(valid_judgment(output, judge, row["candidate_id"]) for judge, row in tasks)
    update_progress(output, "judging", complete, len(tasks))
    for index, (judge, candidate) in enumerate(tasks, start=1):
        cid = candidate["candidate_id"]
        if valid_judgment(output, judge, cid):
            print(f"JUDGE {index}/{len(tasks)} skip {judge}/{cid}", flush=True)
            continue
        status: dict[str, Any] = {}
        for _ in range(3):
            status = call_judge(output, judge, candidate, rubric)
            if status.get("status") == "valid":
                break
        complete = sum(
            valid_judgment(output, name, row["candidate_id"])
            for name, row in tasks
        )
        update_progress(output, "judging", complete, len(tasks), f"{judge} / {cid}")
        append_jsonl(
            output / "campaign_events.jsonl",
            {
                "event": "judgment_finished",
                "judge": judge,
                "candidate_id": cid,
                "status": status.get("status"),
                "at": datetime.now().astimezone().isoformat(),
            },
        )
        print(f"JUDGE {index}/{len(tasks)} {judge}/{cid}: {status.get('status')}", flush=True)
        if pause_s:
            time.sleep(pause_s)
    incomplete = [
        (judge, row["candidate_id"])
        for judge, row in tasks
        if not valid_judgment(output, judge, row["candidate_id"])
    ]
    if incomplete:
        write_json(output / "judgment_failures.json", {"tasks": incomplete})
        raise RuntimeError(f"Judging incomplete for {len(incomplete)} calls; all attempts retained")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else math.nan


def _t_critical_95(degrees_of_freedom: int) -> float:
    """Two-sided 95% Student-t critical values used by supported campaigns."""
    values = {
        1: 12.706,
        2: 4.303,
        3: 3.182,
        4: 2.776,
        5: 2.571,
        6: 2.447,
        7: 2.365,
        8: 2.306,
        9: 2.262,
        10: 2.228,
        11: 2.201,
        12: 2.179,
        13: 2.160,
        14: 2.145,
        15: 2.131,
        16: 2.120,
        17: 2.110,
        18: 2.101,
        19: 2.093,
        20: 2.086,
        21: 2.080,
        22: 2.074,
        23: 2.069,
        24: 2.064,
        25: 2.060,
        26: 2.056,
        27: 2.052,
        28: 2.048,
        29: 2.045,
        30: 2.042,
    }
    if degrees_of_freedom < 1:
        return 0.0
    return values.get(degrees_of_freedom, 1.96)


def build_repeated_report(output: Path, report: Path) -> None:
    report.mkdir(parents=True, exist_ok=True)
    base_summary = aggregate(output, report)
    key = read_json(output / "frozen/candidate_key_confidential.json")["candidates"]
    key_by_id = {row["candidate_id"]: row for row in key}
    score_rows = list(
        csv.DictReader((report / "tables/candidate_consensus_scores.csv").open(encoding="utf-8"))
    )
    enriched: list[dict[str, Any]] = []
    for row in score_rows:
        meta = key_by_id[row["candidate_id"]]
        enriched.append(
            {
                **row,
                "generator_key": meta["generator_key"],
                "repeat_id": meta["repeat_id"],
                "seed": meta["seed"],
                "result_sha256": meta["result_sha256"],
                "run_directory": meta["run_directory"],
            }
        )
    _write_csv(report / "tables/candidate_scores_with_repeats.csv", enriched)

    paired: list[dict[str, Any]] = []
    for model in sorted({row["model"] for row in enriched}):
        for case in SELECTED_CASES:
            for repeat_id in REPEAT_IDS:
                cells = {
                    row["architecture"]: row
                    for row in enriched
                    if row["model"] == model
                    and row["case"] == case
                    and row["repeat_id"] == repeat_id
                }
                if set(cells) != {"One-shot", "FlowPilot"}:
                    raise RuntimeError(f"Missing matched pair: {model}/{case}/{repeat_id}")
                one = float(cells["One-shot"]["mean_score_0_1"])
                flow = float(cells["FlowPilot"]["mean_score_0_1"])
                one_x = float(cells["One-shot"]["generator_family_excluded_score_0_1"])
                flow_x = float(cells["FlowPilot"]["generator_family_excluded_score_0_1"])
                paired.append(
                    {
                        "model": model,
                        "case": case,
                        "repeat_id": repeat_id,
                        "one_shot_score_0_1": one,
                        "flowpilot_score_0_1": flow,
                        "paired_delta": round(flow - one, 6),
                        "one_shot_excluded_score": one_x,
                        "flowpilot_excluded_score": flow_x,
                        "excluded_paired_delta": round(flow_x - one_x, 6),
                    }
                )
    _write_csv(report / "tables/repeat_level_paired_comparisons.csv", paired)
    # Replace the base aggregator's single-cell compatibility table. Its legacy
    # key does not include repeat_id and is therefore not valid for this study.
    _write_csv(report / "tables/paired_comparisons.csv", paired)

    model_rows: list[dict[str, Any]] = []
    for model in sorted({row["model"] for row in enriched}):
        model_pairs = [row for row in paired if row["model"] == model]
        deltas = [float(row["paired_delta"]) for row in model_pairs]
        excluded = [float(row["excluded_paired_delta"]) for row in model_pairs]
        n = len(deltas)
        delta_sd = statistics.stdev(deltas) if n > 1 else 0.0
        half_width = (
            _t_critical_95(n - 1) * delta_sd / math.sqrt(n)
            if n > 1
            else 0.0
        )
        one = [float(row["mean_score_0_1"]) for row in enriched if row["model"] == model and row["architecture"] == "One-shot"]
        flow = [float(row["mean_score_0_1"]) for row in enriched if row["model"] == model and row["architecture"] == "FlowPilot"]
        model_rows.append(
            {
                "model": model,
                "n_matched_pairs": n,
                "one_shot_mean_0_1": round(_mean(one), 6),
                "flowpilot_mean_0_1": round(_mean(flow), 6),
                "mean_paired_delta": round(_mean(deltas), 6),
                "paired_delta_sd": round(delta_sd, 6),
                "paired_delta_95ci_low": round(_mean(deltas) - half_width, 6),
                "paired_delta_95ci_high": round(_mean(deltas) + half_width, 6),
                "wins": sum(value > 0 for value in deltas),
                "ties": sum(value == 0 for value in deltas),
                "losses": sum(value < 0 for value in deltas),
                "generator_family_excluded_delta": round(_mean(excluded), 6),
            }
        )
    _write_csv(report / "tables/model_architecture_summary.csv", model_rows)

    condition_rows: list[dict[str, Any]] = []
    for model in sorted({row["model"] for row in enriched}):
        for architecture in ("One-shot", "FlowPilot"):
            rows = [
                row
                for row in enriched
                if row["model"] == model and row["architecture"] == architecture
            ]
            values = [float(row["mean_score_0_1"]) for row in rows]
            condition_rows.append(
                {
                    "model": model,
                    "architecture": architecture,
                    "n_outcomes": len(rows),
                    "n_cases": len({row["case"] for row in rows}),
                    "n_repeats_per_case": len(REPEAT_IDS),
                    "mean_score_0_1": round(_mean(values), 6),
                    "outcome_sd": round(statistics.stdev(values), 6),
                    "median_score_0_1": round(statistics.median(values), 6),
                }
            )
    _write_csv(report / "tables/architecture_summary.csv", condition_rows)

    repeatability: list[dict[str, Any]] = []
    for model in sorted({row["model"] for row in enriched}):
        for architecture in ("One-shot", "FlowPilot"):
            for case in SELECTED_CASES:
                rows = [
                    row
                    for row in enriched
                    if row["model"] == model
                    and row["architecture"] == architecture
                    and row["case"] == case
                ]
                values = [float(row["mean_score_0_1"]) for row in rows]
                repeatability.append(
                    {
                        "model": model,
                        "architecture": architecture,
                        "case": case,
                        "n_repeats": len(rows),
                        "mean_score_0_1": round(_mean(values), 6),
                        "repeat_sd": round(statistics.stdev(values), 6) if len(values) > 1 else 0.0,
                        "score_range": round(max(values) - min(values), 6),
                        "exact_result_hash_count": len({row["result_sha256"] for row in rows}),
                        "exact_result_reproducible": len({row["result_sha256"] for row in rows}) == 1,
                    }
                )
    _write_csv(report / "tables/repeatability_summary.csv", repeatability)

    contract_rows: list[dict[str, Any]] = []
    parameter_rows: list[dict[str, Any]] = []
    parameter_fields = (
        "residence_time_min",
        "flow_rate_mL_min",
        "reactor_volume_mL",
        "temperature_C",
        "concentration_M",
        "BPR_bar",
    )
    for model in sorted({row["model"] for row in enriched}):
        for architecture in ("One-shot", "FlowPilot"):
            rows = [
                row
                for row in enriched
                if row["model"] == model and row["architecture"] == architecture
            ]
            statuses: list[str] = []
            schema_valid = 0
            for row in rows:
                result_path = Path(row["run_directory"]) / "result.json"
                if not result_path.is_file():
                    statuses.append("generation_failed")
                    continue
                result = read_json(result_path)
                final = result.get("final_design") or {}
                if architecture == "FlowPilot":
                    statuses.append(str(final.get("status") or "missing"))
                    parameters = final.get("parameters") or result.get("proposal") or {}
                else:
                    statuses.append("assessable" if result.get("proposal") else "missing")
                    parameters = result.get("proposal") or result.get("raw_proposal") or {}
                schema_valid += bool(result.get("schema_valid"))
                for field in parameter_fields:
                    value = parameters.get(field)
                    if isinstance(value, (int, float)) and not isinstance(value, bool):
                        parameter_rows.append(
                            {
                                "model": model,
                                "architecture": architecture,
                                "case": row["case"],
                                "repeat_id": row["repeat_id"],
                                "parameter": field,
                                "value": float(value),
                            }
                        )
            contract_rows.append(
                {
                    "model": model,
                    "architecture": architecture,
                    "n_outcomes": len(rows),
                    "schema_valid": schema_valid,
                    "executable": sum(status == "executable" for status in statuses),
                    "blocked": sum(status == "blocked" for status in statuses),
                    "assessable_one_shot": sum(status == "assessable" for status in statuses),
                    "missing": sum(status == "missing" for status in statuses),
                    "generation_failed": sum(status == "generation_failed" for status in statuses),
                }
            )
    _write_csv(report / "tables/outcome_contract_summary.csv", contract_rows)
    _write_csv(report / "tables/parameter_values_by_repeat.csv", parameter_rows)

    parameter_summary: list[dict[str, Any]] = []
    grouped_parameters: dict[tuple[str, str, str, str], list[float]] = defaultdict(list)
    for row in parameter_rows:
        grouped_parameters[(row["model"], row["architecture"], row["case"], row["parameter"])].append(float(row["value"]))
    for (model, architecture, case, parameter), values in sorted(grouped_parameters.items()):
        value_mean = _mean(values)
        value_sd = statistics.stdev(values) if len(values) > 1 else 0.0
        parameter_summary.append(
            {
                "model": model,
                "architecture": architecture,
                "case": case,
                "parameter": parameter,
                "n_reported": len(values),
                "mean": round(value_mean, 8),
                "sd": round(value_sd, 8),
                "cv_pct": round(100 * value_sd / abs(value_mean), 4) if value_mean else None,
                "minimum": round(min(values), 8),
                "maximum": round(max(values), 8),
            }
        )
    _write_csv(report / "tables/parameter_repeatability.csv", parameter_summary)

    operational_rows: list[dict[str, Any]] = []
    for model in sorted({row["model"] for row in enriched}):
        for architecture in ("One-shot", "FlowPilot"):
            rows = [
                row
                for row in enriched
                if row["model"] == model and row["architecture"] == architecture
            ]
            runtimes: list[float] = []
            calls: list[int] = []
            for row in rows:
                run_summary = read_json(Path(row["run_directory"]) / "run_summary.json")
                runtime = run_summary.get("runtime_total_s", run_summary.get("runtime_s"))
                if isinstance(runtime, (int, float)):
                    runtimes.append(float(runtime))
                if isinstance(run_summary.get("llm_call_count"), int):
                    calls.append(run_summary["llm_call_count"])
            operational_rows.append(
                {
                    "model": model,
                    "architecture": architecture,
                    "n_outcomes": len(rows),
                    "mean_runtime_s": round(_mean(runtimes), 3),
                    "runtime_sd_s": round(statistics.stdev(runtimes), 3) if len(runtimes) > 1 else 0.0,
                    "mean_llm_calls": round(_mean([float(value) for value in calls]), 3) if calls else 1.0,
                    "total_llm_calls": sum(calls) if calls else len(rows),
                }
            )
    _write_csv(report / "tables/operational_summary.csv", operational_rows)

    criterion_rows = list(
        csv.DictReader((report / "tables/criterion_judgments.csv").open(encoding="utf-8"))
    )
    candidate_meta = {
        row["candidate_id"]: (row["model"], row["architecture"])
        for row in enriched
    }
    criterion_groups: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for row in criterion_rows:
        if row["applicability"] != "APPLICABLE":
            continue
        model, architecture = candidate_meta[row["candidate_id"]]
        criterion_groups[(model, architecture, row["criterion_id"])].append(float(row["score_0_4"]) / 4)
    criterion_summary: list[dict[str, Any]] = []
    for model in sorted({row["model"] for row in enriched}):
        criteria = sorted({key[2] for key in criterion_groups if key[0] == model})
        for criterion_id in criteria:
            one = _mean(criterion_groups[(model, "One-shot", criterion_id)])
            flow = _mean(criterion_groups[(model, "FlowPilot", criterion_id)])
            criterion_summary.append(
                {
                    "model": model,
                    "criterion_id": criterion_id,
                    "one_shot_mean_0_1": round(one, 6),
                    "flowpilot_mean_0_1": round(flow, 6),
                    "delta": round(flow - one, 6),
                }
            )
    _write_csv(report / "tables/criterion_architecture_effect.csv", criterion_summary)

    summary = {
        "schema_version": "flowpilot_three_model_repeated_summary_v1.0",
        "candidate_count": len(enriched),
        "judgment_count": base_summary["judgment_count"],
        "case_count": len(SELECTED_CASES),
        "repeat_count": len(REPEAT_IDS),
        "generator_count": len(MODELS),
        "judge_count": len(JUDGES),
        "model_architecture_summary": model_rows,
        "overall_mean_paired_delta": round(_mean([float(row["paired_delta"]) for row in paired]), 6),
        "overall_generator_family_excluded_delta": round(_mean([float(row["excluded_paired_delta"]) for row in paired]), 6),
        "judge_agreement": base_summary["judge_agreement"],
        "outcome_contract_summary": contract_rows,
    }
    write_json(report / "summary.json", summary)
    create_figures(report)
    create_documentation(output, report, summary)
    frozen = report / "frozen"
    frozen.mkdir(exist_ok=True)
    for name in (
        "campaign_manifest.json",
        "case_manifest.json",
        "execution_plan.json",
        "outcome_rubric.json",
        "packet_metrics.json",
        "source_code_checksums.json",
        "PREREGISTRATION.md",
    ):
        source = output / "frozen" / name
        if source.is_file():
            shutil.copy2(source, frozen / name)
    shutil.copytree(output / "frozen/source_code", frozen / "source_code", dirs_exist_ok=True)
    amendments = output / "frozen/amendments"
    if amendments.is_dir():
        shutil.copytree(amendments, frozen / "amendments", dirs_exist_ok=True)
    total_tasks = len(enriched) + int(base_summary["judgment_count"])
    update_progress(output, "complete", total_tasks, total_tasks)


def create_figures(report: Path) -> None:
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    figures = report / "figures"
    figures.mkdir(exist_ok=True)
    candidates = pd.read_csv(report / "tables/candidate_scores_with_repeats.csv")
    pairs = pd.read_csv(report / "tables/repeat_level_paired_comparisons.csv")
    model_summary = pd.read_csv(report / "tables/model_architecture_summary.csv")
    repeats = pd.read_csv(report / "tables/repeatability_summary.csv")
    criteria = pd.read_csv(report / "tables/criterion_architecture_effect.csv")
    models = list(model_summary.model)
    colors = {"One-shot": "#8B939C", "FlowPilot": "#167D8D"}

    fig, ax = plt.subplots(figsize=(9.2, 5.2))
    x = np.arange(len(models)); width = 0.34
    for index, architecture in enumerate(("One-shot", "FlowPilot")):
        means = [candidates[(candidates.model == model) & (candidates.architecture == architecture)].mean_score_0_1.mean() for model in models]
        sds = [candidates[(candidates.model == model) & (candidates.architecture == architecture)].mean_score_0_1.std() for model in models]
        ax.bar(x + (index - .5) * width, means, width, yerr=sds, capsize=4, color=colors[architecture], label=architecture)
    ax.set_xticks(x, models); ax.set_ylim(0, 1); ax.set_ylabel("Consensus score (0-1)")
    outcomes_per_architecture = len(SELECTED_CASES) * len(REPEAT_IDS)
    ax.set_title(
        "Matched outcomes by generator family\n"
        f"(error bars: SD across {outcomes_per_architecture} outcomes)"
    )
    ax.legend(frameon=False); ax.grid(axis="y", alpha=.2)
    fig.tight_layout(); fig.savefig(figures / "fig01_three_model_architecture_scores.png", dpi=300); plt.close(fig)

    fig, axes = plt.subplots(1, len(models), figsize=(max(5.2, 4.2 * len(models)), 4.5), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, model in zip(axes, models):
        frame = pairs[pairs.model == model]
        for _, row in frame.iterrows():
            color = {"Hydrogenolysis": "#4C78A8", "Photochemical oxidation": "#E39C37", "CuAAC": "#7A5195"}[row["case"]]
            ax.plot([0, 1], [row.one_shot_score_0_1, row.flowpilot_score_0_1], color=color, alpha=.65, marker="o", linewidth=1.3)
        ax.set_xticks([0, 1], ["One-shot", "FlowPilot"]); ax.set_title(model); ax.grid(axis="y", alpha=.2)
    from matplotlib.lines import Line2D
    case_colors = {"Hydrogenolysis": "#4C78A8", "Photochemical oxidation": "#E39C37", "CuAAC": "#7A5195"}
    handles = [Line2D([0], [0], color=color, marker="o", linewidth=2, label=case) for case, color in case_colors.items()]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.01))
    axes[0].set_ylabel("Consensus score (0-1)"); axes[0].set_ylim(0, 1)
    fig.suptitle("Every line is one matched case-repeat pair")
    fig.tight_layout(); fig.savefig(figures / "fig02_repeat_level_paired_scores.png", dpi=300); plt.close(fig)

    heat = pairs.groupby(["case", "model"]).paired_delta.mean().unstack().reindex(index=SELECTED_CASES, columns=models)
    fig, ax = plt.subplots(figsize=(9.0, 4.4))
    image = ax.imshow(heat.values, cmap="RdYlGn", vmin=-.25, vmax=.25, aspect="auto")
    ax.set_xticks(range(len(models)), models); ax.set_yticks(range(len(SELECTED_CASES)), SELECTED_CASES)
    for y in range(len(SELECTED_CASES)):
        for x_index in range(len(models)):
            ax.text(x_index, y, f"{heat.iloc[y, x_index]:+.3f}", ha="center", va="center")
    fig.colorbar(image, ax=ax, label="Mean FlowPilot minus one-shot score")
    ax.set_title(
        "Architecture effect"
        + (
            f" averaged across {len(REPEAT_IDS)} repeats"
            if len(REPEAT_IDS) > 1
            else " for one frozen repeat"
        )
    )
    fig.tight_layout(); fig.savefig(figures / "fig03_case_model_delta_heatmap.png", dpi=300); plt.close(fig)

    fig, ax = plt.subplots(figsize=(9.2, 5.0))
    repeat_plot = repeats.groupby(["model", "architecture"]).repeat_sd.mean().unstack().reindex(models)
    repeat_plot[["One-shot", "FlowPilot"]].plot.bar(ax=ax, color=[colors["One-shot"], colors["FlowPilot"]])
    ax.set_ylabel("Mean within-cell score SD"); ax.set_xlabel("")
    ax.set_title("Outcome-score repeatability across fresh calls")
    ax.legend(frameon=False); ax.grid(axis="y", alpha=.2)
    fig.tight_layout(); fig.savefig(figures / "fig04_repeatability.png", dpi=300); plt.close(fig)

    pivot = criteria.pivot(index="criterion_id", columns="model", values="delta").reindex(columns=models)
    fig, ax = plt.subplots(figsize=(9.4, 6.0))
    image = ax.imshow(pivot.values, cmap="RdYlGn", vmin=-.35, vmax=.35, aspect="auto")
    ax.set_xticks(range(len(models)), models); ax.set_yticks(range(len(pivot.index)), pivot.index)
    for y in range(len(pivot.index)):
        for x_index in range(len(models)):
            ax.text(x_index, y, f"{pivot.iloc[y, x_index]:+.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(image, ax=ax, label="FlowPilot minus one-shot normalized criterion score")
    ax.set_title("Universal rubric criterion effects")
    fig.tight_layout(); fig.savefig(figures / "fig05_criterion_delta_heatmap.png", dpi=300); plt.close(fig)

    fig, ax = plt.subplots(figsize=(9.2, 5.0))
    deltas = [pairs[pairs.model == model].paired_delta.values for model in models]
    ax.boxplot(deltas, tick_labels=models, patch_artist=True, boxprops={"facecolor": "#D6E8EA"})
    ax.axhline(0, color="#333333", linewidth=1); ax.set_ylabel("FlowPilot minus one-shot score")
    ax.set_title("Distribution of matched architecture effects")
    ax.grid(axis="y", alpha=.2)
    fig.tight_layout(); fig.savefig(figures / "fig06_delta_distribution.png", dpi=300); plt.close(fig)

    contracts = pd.read_csv(report / "tables/outcome_contract_summary.csv")
    flow_contracts = contracts[contracts.architecture == "FlowPilot"].set_index("model").reindex(models)
    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    executable_rate = flow_contracts.executable / flow_contracts.n_outcomes
    blocked_rate = flow_contracts.blocked / flow_contracts.n_outcomes
    failed_rate = flow_contracts.generation_failed / flow_contracts.n_outcomes
    ax.bar(models, executable_rate, color="#167D8D", label="Executable")
    ax.bar(models, blocked_rate, bottom=executable_rate, color="#C95C54", label="Blocked")
    ax.bar(
        models,
        failed_rate,
        bottom=executable_rate + blocked_rate,
        color="#5C636A",
        label="Generation failed",
    )
    for index, model in enumerate(models):
        ax.text(
            index,
            executable_rate.iloc[index] / 2,
            f"{int(flow_contracts.executable.iloc[index])}/{int(flow_contracts.n_outcomes.iloc[index])}",
            ha="center",
            va="center",
            color="white",
        )
    ax.set_ylim(0, 1); ax.set_ylabel("Fraction of FlowPilot outcomes")
    ax.set_title("Deterministic final-contract closure across repeats")
    ax.legend(frameon=False); ax.grid(axis="y", alpha=.2)
    fig.tight_layout(); fig.savefig(figures / "fig07_flowpilot_contract_closure.png", dpi=300); plt.close(fig)


def create_documentation(output: Path, report: Path, summary: dict[str, Any]) -> None:
    rows = summary["model_architecture_summary"]
    result_lines = [
        f"- {row['model']}: FlowPilot {row['flowpilot_mean_0_1']:.3f} vs one-shot {row['one_shot_mean_0_1']:.3f}; "
        f"matched delta {row['mean_paired_delta']:+.3f} (95% CI {row['paired_delta_95ci_low']:+.3f} to {row['paired_delta_95ci_high']:+.3f}); "
        f"wins/ties/losses {row['wins']}/{row['ties']}/{row['losses']}."
        for row in rows
    ]
    agreement = summary["judge_agreement"]
    closure_by_model = {
        row["model"]: row
        for row in summary["outcome_contract_summary"]
        if row["architecture"] == "FlowPilot"
    }
    closure_lines = [
        f"- {model}: {row['executable']}/{row['n_outcomes']} executable, "
        f"{row['blocked']}/{row['n_outcomes']} blocked, "
        f"{row['generation_failed']}/{row['n_outcomes']} generation failed."
        for model, row in sorted(closure_by_model.items())
    ]
    model_names = ", ".join(row["model"] for row in rows)
    candidate_count = int(summary["candidate_count"])
    judgment_count = int(summary["judgment_count"])
    repeat_word = "call" if len(REPEAT_IDS) == 1 else "calls"
    text = f"""# Repeated Architecture Benchmark

## Frozen Design

- Three held-out chemistries selected before this campaign: {', '.join(SELECTED_CASES)}.
- Generator families: {model_names}.
- Two matched architectures: direct one-shot and the full FlowPilot pipeline.
- {len(REPEAT_IDS)} fresh generation {repeat_word} per case/model/architecture: {candidate_count} outcomes.
- Blinded judges evaluate every outcome: {judgment_count} judgments.
- The unchanged NewGen 2.0 rubric contains 14 equally weighted, architecture-neutral criteria.
- Temperature is {TEMPERATURE}; each call records a distinct seed. Anthropic does not guarantee seed control.

## Results

{chr(10).join(result_lines)}

Overall matched delta: **{summary['overall_mean_paired_delta']:+.3f}**. Generator-family-excluded sensitivity delta: **{summary['overall_generator_family_excluded_delta']:+.3f}**.

FlowPilot final-contract closure in this campaign:

{chr(10).join(closure_lines)}

Blocked and generation-failed outcomes remain in the primary score; the report does not substitute results from earlier campaigns.

UO-09 (solids/slurry handling) is absent from the criterion-effect heatmap because it was not applicable to any of the three frozen cases; no applicable criterion observations were removed.

Judge agreement was {agreement['exact_agreement_rate']:.1%} exact and {agreement['within_one_point_rate']:.1%} within one rubric point; mean pairwise absolute difference was {agreement['mean_pairwise_absolute_difference_0_4']:.3f}/4.

## Interpretation

The paired case-repeat is the unit of comparison. Confidence intervals describe variation across the {len(SELECTED_CASES) * len(REPEAT_IDS)} case-repeat pairs per model, not uncertainty over all flow chemistry. LLM judging remains secondary evidence and does not replace deterministic checks or wet-lab validation. All failures, retries, raw responses, prompts, seeds, checksums, and model metadata are retained in `{output}`.
"""
    (report / "REPEATED_BENCHMARK_REPORT.md").write_text(text, encoding="utf-8")
    (report / "SCORING_METHOD.md").write_text(
        "# Scoring Method\n\nEach applicable universal criterion receives an anchored integer score from 0 to 4 from each of three blinded judges. "
        "Candidate score is the unweighted mean across applicable criteria and judges divided by four. The primary architecture effect is the paired FlowPilot-minus-one-shot candidate score for the same chemistry, model family, and repeat. "
        "No architecture bonuses, post-hoc criterion weights, or score-triggered extra repeats are used. "
        "A criterion marked not applicable is excluded symmetrically from both architectures; UO-09 was not applicable to any frozen case in this campaign.\n",
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
    parser.add_argument("--families", nargs="+", choices=tuple(MODELS), default=list(MODELS))
    parser.add_argument("--judges", nargs="+", choices=JUDGES, default=list(JUDGES))
    parser.add_argument("--pause-between-calls", type=float, default=1.0)
    parser.add_argument("--skip-health", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve(); report = args.report.resolve()
    output.mkdir(parents=True, exist_ok=True)
    cases = selected_cases()
    initialize(output, cases)
    if args.phase == "init":
        print(output); return
    if args.phase in {"health", "generate", "judge", "all"} and not args.skip_health:
        health_check(output)
    if args.phase in {"generate", "all"}:
        run_generation(output, cases, tuple(args.families), args.pause_between_calls)
    if args.phase in {"packets", "all"}:
        build_packets(output, cases)
    if args.phase in {"judge", "all"}:
        run_judges(output, tuple(args.judges), args.pause_between_calls)
    if args.phase in {"report", "all"}:
        build_repeated_report(output, report)
    print(output); print(report)


if __name__ == "__main__":
    main()
