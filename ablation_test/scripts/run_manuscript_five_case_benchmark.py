#!/usr/bin/env python3
"""Run the resumable five-case Qwen/OpenAI manuscript pilot.

Every generation and judging call is checkpointed independently. Re-running
the same command skips valid completed artifacts and continues from the first
missing cell.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import sys
import time
from copy import deepcopy
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ablation_test.scripts.build_newgen_2_0_report import aggregate, documentation, figures
from ablation_test.scripts.run_newgen_2_0_benchmark import call_judge
from ablation_test.src.cases import AblationCase, load_cases_from_path
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
from flora_translate.schemas import LabInventory


RUBRIC = ROOT / "ablation_test/benchmarks/newgen_2_0_outcome_rubric.json"
DEFAULT_OUTPUT = ROOT / "ablation_results/manuscript_benchmark/manuscript_five_case_v1_20260819"
DEFAULT_REPORT = ROOT / "deliverables/manuscript_five_case_v1_20260819"
SOURCE_CASES = {
    "CuAAC": ROOT / "ablation_test/benchmarks/newgen_benchmark_v1_pilot_cuaac/case.json",
    "Hydrogenolysis": ROOT / "ablation_test/benchmarks/newgen_benchmark_v1_pilot_hydrogenolysis/case.json",
    "Two-stage amidation": ROOT / "ablation_test/benchmarks/newgen_benchmark_v1_pilot_multistep/case.json",
    "Exothermic dinitration": ROOT / "ablation_test/benchmarks/newgen_benchmark_v1_pilot_nitration/case.json",
    "Photochemical oxidation": ROOT / "ablation_test/benchmarks/manuscript_five_case_v1/fmoc_case.json",
}
ORACLES = {
    "CuAAC": ROOT / "ablation_test/benchmarks/newgen_benchmark_v1_pilot_cuaac/hidden_oracle.json",
    "Hydrogenolysis": ROOT / "ablation_test/benchmarks/newgen_benchmark_v1_pilot_hydrogenolysis/hidden_oracle.json",
    "Two-stage amidation": ROOT / "ablation_test/benchmarks/newgen_benchmark_v1_pilot_multistep/hidden_oracle.json",
    "Exothermic dinitration": ROOT / "ablation_test/benchmarks/newgen_benchmark_v1_pilot_nitration/hidden_oracle.json",
    "Photochemical oxidation": ROOT / "ablation_test/benchmarks/manuscript_five_case_v1/fmoc_oracle.json",
}
MODELS = {
    "qwen": {
        "provider": "ollama",
        "model": "/models/Qwen3.6-27B",
        "base_url": "http://10.13.24.169:8000/v1",
        "upstream_mode": "always",
        "family": "qwen",
        "display": "Qwen3.6-27B",
    },
    "openai": {
        "provider": "openai",
        "model": "gpt-4o",
        "upstream_mode": "never",
        "family": "openai",
        "display": "GPT-4o",
    },
}
ARCHITECTURES = {
    "one_shot": ("general_one_shot", "One-shot"),
    "flowpilot": ("full", "FlowPilot"),
}
IDENTITIES = {
    "CuAAC": {"confirmed": True, "reaction_name": "copper-catalyzed azide-alkyne cycloaddition", "reaction_class": "cuaac", "mechanism_type": "copper-catalyzed cycloaddition"},
    "Hydrogenolysis": {"confirmed": True, "reaction_name": "N-diphenylmethyl hydrogenolysis", "reaction_class": "hydrogenolysis", "mechanism_type": "heterogeneous catalytic hydrogenolysis"},
    "Two-stage amidation": {"confirmed": True, "reaction_name": "two-stage oxidative amidation", "reaction_class": "amidation", "mechanism_type": "multistep peroxide-mediated oxidative amidation"},
    "Exothermic dinitration": {"confirmed": True, "reaction_name": "pendimethalin dinitration", "reaction_class": "nitration", "mechanism_type": "electrophilic aromatic nitration"},
}
SAFETY_FIXTURES = {
    "CuAAC": [{"equipment_id": "safety_cuaac_shield", "name": "Azide-service blast shield and secondary containment", "type": "shielded enclosure", "capabilities": ["shield_or_containment"], "compatible_hazards": ["energetic_azide"]}],
    "Hydrogenolysis": [
        {"equipment_id": "safety_h2_check_valve", "name": "Hydrogen non-return check valve", "type": "check valve", "capabilities": ["backflow_prevention"], "max_pressure_bar": 40.0, "compatible_hazards": ["hydrogen"]},
        {"equipment_id": "safety_h2_vent", "name": "Dedicated hydrogen separator safe vent", "type": "safe vent", "capabilities": ["vented_separator"], "max_pressure_bar": 25.0, "compatible_hazards": ["hydrogen"]},
    ],
    "Two-stage amidation": [{"equipment_id": "safety_peroxide_shield", "name": "Peroxide-compatible shield and secondary containment", "type": "shielded containment", "capabilities": ["shield_or_containment"], "compatible_hazards": ["peroxide_oxidizer"]}],
    "Exothermic dinitration": [{"equipment_id": "safety_nitration_enclosure", "name": "Temperature-monitored nitration enclosure", "type": "shielded ventilated enclosure", "capabilities": ["shield_or_containment"], "compatible_hazards": ["nitric_acid", "nitro_compound_decomposition", "NOx"]}],
}


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def append_jsonl(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False) + "\n")


def load_frozen_cases() -> list[tuple[str, AblationCase]]:
    output: list[tuple[str, AblationCase]] = []
    for label, path in SOURCE_CASES.items():
        case = load_cases_from_path(path)[0]
        inventory = deepcopy(case.inventory)
        existing = {item.get("equipment_id") for item in inventory.get("safety_accessories", [])}
        for item in SAFETY_FIXTURES.get(label, []):
            if item["equipment_id"] not in existing:
                inventory.setdefault("safety_accessories", []).append(item)
        identity = case.chemistry_identity_confirmation or IDENTITIES.get(label, {})
        case = replace(case, suite_id="manuscript_five_case_v1", inventory=inventory, chemistry_identity_confirmation=identity)
        LabInventory(**inventory)
        output.append((label, case))
    return output


def campaign_cells(cases: list[tuple[str, AblationCase]]) -> list[dict[str, Any]]:
    rows = []
    for label, case in cases:
        for family in ("qwen", "openai"):
            for architecture in ("one_shot", "flowpilot"):
                variant, architecture_label = ARCHITECTURES[architecture]
                rows.append({
                    "case_label": label,
                    "case_id": case.case_id,
                    "generator_family": family,
                    "condition_id": f"{family}_{architecture}",
                    "architecture_key": architecture,
                    "architecture": architecture_label,
                    "variant": variant,
                    "model": MODELS[family]["display"],
                    "model_id": MODELS[family]["model"],
                    "design_input_sha256": case.design_input_sha256,
                })
    return rows


def initialize(output: Path, cases: list[tuple[str, AblationCase]]) -> None:
    frozen = output / "frozen"
    frozen.mkdir(parents=True, exist_ok=True)
    target_rubric = frozen / "outcome_rubric.json"
    if not target_rubric.exists():
        shutil.copy2(RUBRIC, target_rubric)
    manifest = {
        "schema_version": "flowpilot_manuscript_five_case_campaign_v1.0",
        "benchmark_name": "Manuscript five-case architecture pilot",
        "created_at": datetime.now().astimezone().isoformat(),
        "case_count": 5,
        "generator_families": ["qwen", "openai"],
        "architectures": ["One-shot", "FlowPilot"],
        "repeat_count": 1,
        "candidate_count": 20,
        "judge_panel": ["qwen", "openai"],
        "planned_judgments": 40,
        "candidate_budget_flowpilot": 2,
        "temperature": 0.0,
        "rubric_frozen_before_generation": True,
        "rubric_sha256": sha256_file(target_rubric),
        "held_out_source_excluded_from_retrieval": True,
        "selection_policy": "Retain all five pilot cases in the dataset. Rank the three main-text candidates by cross-model FlowPilot advantage, cross-family-judge delta, completeness, and judge agreement; report the other two in ESI.",
    }
    existing = frozen / "campaign_manifest.json"
    if existing.exists():
        prior = read_json(existing)
        for key in ("case_count", "candidate_count", "judge_panel", "rubric_sha256"):
            if prior.get(key) != manifest.get(key):
                raise RuntimeError(f"Frozen campaign mismatch for {key}; use a new output directory")
    else:
        write_json(existing, manifest)
    write_json(frozen / "case_manifest.json", {"suite_id": "manuscript_five_case_v1", "cases": [{"case_label": label, **case.manifest_payload()} for label, case in cases]})
    write_json(frozen / "execution_plan.json", {"cells": campaign_cells(cases)})
    source_dir = frozen / "source_code"
    source_dir.mkdir(exist_ok=True)
    source_files = (
        ROOT / "ablation_test/scripts/run_manuscript_five_case_benchmark.py",
        ROOT / "ablation_test/scripts/build_newgen_2_0_report.py",
        ROOT / "ablation_test/src/runner.py",
        ROOT / "ablation_test/src/newgen_2_outcome.py",
        ROOT / "ablation_test/src/cases.py",
    )
    source_hashes = {}
    for source in source_files:
        target = source_dir / source.name
        shutil.copy2(source, target)
        source_hashes[source.name] = sha256_file(target)
    write_json(frozen / "source_code_checksums.json", source_hashes)
    oracle_dir = frozen / "held_out_oracles"
    oracle_dir.mkdir(exist_ok=True)
    for label, source in ORACLES.items():
        target = oracle_dir / f"{safe_name(label)}.json"
        if not target.exists():
            shutil.copy2(source, target)
    (frozen / "SELECTION_POLICY.md").write_text(
        "# Predeclared Case Selection Policy\n\nAll five cases remain in the released pilot dataset. "
        "Three may be promoted to the manuscript main text using, in order: (1) complete generation and judging, "
        "(2) FlowPilot improvement for both generator families, (3) positive cross-family-judge paired delta, "
        "(4) lower judge disagreement, and (5) complementary chemistry coverage. The remaining cases are reported in ESI.\n",
        encoding="utf-8",
    )


def safe_name(value: str) -> str:
    return "_".join("".join(ch.lower() if ch.isalnum() else " " for ch in value).split())


def completed_run(run_dir: Path) -> bool:
    if not (run_dir / "result.json").is_file() or not (run_dir / "run_summary.json").is_file():
        return False
    return read_json(run_dir / "run_summary.json").get("status") == "completed"


def update_progress(output: Path, phase: str, completed: int, total: int, current: str = "") -> None:
    write_json(output / "progress.json", {
        "phase": phase,
        "completed": completed,
        "total": total,
        "remaining": total - completed,
        "current": current,
        "updated_at": datetime.now().astimezone().isoformat(),
    })


def run_generation(output: Path, cases: list[tuple[str, AblationCase]], pause_s: float) -> None:
    by_id = {case.case_id: case for _, case in cases}
    cells = campaign_cells(cases)
    complete = sum(completed_run(run_directory(output, row)) for row in cells)
    update_progress(output, "generation", complete, len(cells))
    for index, row in enumerate(cells, start=1):
        run_dir = run_directory(output, row)
        if completed_run(run_dir):
            print(f"GEN {index}/{len(cells)} skip {row['case_label']} {row['condition_id']}", flush=True)
            continue
        started = datetime.now().astimezone().isoformat()
        append_jsonl(output / "campaign_events.jsonl", {"event": "generation_started", "cell": row, "at": started})
        seed = stable_seed("manuscript-five-v1", row["case_id"], row["condition_id"], "repeat-01")
        summary = execute_cell(
            case=by_id[row["case_id"]],
            variant=row["variant"],
            bundle_name=row["condition_id"],
            bundle=MODELS[row["generator_family"]],
            run_dir=run_dir,
            candidate_budget=2,
            temperature=0.0,
            seed=seed,
        )
        complete = sum(completed_run(run_directory(output, item)) for item in cells)
        update_progress(output, "generation", complete, len(cells), f"{row['case_label']} / {row['condition_id']}")
        append_jsonl(output / "campaign_events.jsonl", {"event": "generation_finished", "cell": row, "status": summary.get("status"), "at": datetime.now().astimezone().isoformat()})
        print(f"GEN {index}/{len(cells)} {row['case_label']} {row['condition_id']}: {summary.get('status')}", flush=True)
        if pause_s:
            time.sleep(pause_s)
    failed = [row for row in cells if not completed_run(run_directory(output, row))]
    if failed:
        raise RuntimeError(f"Generation incomplete for {len(failed)} cells; rerun to resume")


def run_directory(output: Path, row: dict[str, Any]) -> Path:
    return output / "generation" / safe_name(row["case_label"]) / row["condition_id"] / "repeat_01"


def numeric(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def deterministic_sheet(result: dict[str, Any], metrics: dict[str, Any], case: AblationCase) -> dict[str, Any]:
    final = result.get("final_design") if isinstance(result.get("final_design"), dict) else {}
    parameters = final.get("parameters") if isinstance(final.get("parameters"), dict) else (result.get("proposal") or result.get("raw_proposal") or {})
    q = numeric(parameters.get("flow_rate_mL_min"))
    tau = numeric(parameters.get("residence_time_min"))
    volume = numeric(parameters.get("reactor_volume_mL"))
    closure: dict[str, Any] = {"assessable": all(item is not None for item in (q, tau, volume))}
    if closure["assessable"]:
        expected = q * tau
        closure.update({
            "reported_volume_mL": volume,
            "calculated_Q_tau_mL": round(expected, 8),
            "relative_error": round(abs(volume - expected) / max(abs(volume), 1e-12), 8),
            "passes_10_percent": abs(volume - expected) / max(abs(volume), 1e-12) <= 0.10,
        })
    declared_volumes = sorted(float(item["volume_mL"]) for item in case.inventory.get("reactors", []) if isinstance(item.get("volume_mL"), (int, float)))
    return {
        "schema_version": "flowpilot_architecture_neutral_verification_v1.0",
        "scope": "Mechanical checks only; chemistry quality and operational completeness remain judge-scored.",
        "schema_valid_reported": bool(result.get("schema_valid")),
        "delivered_contract_status": final.get("status") if final else None,
        "reported_disposition": result.get("reported_disposition") or result.get("recommended_disposition"),
        "top_level_volume_flow_time_closure": closure,
        "declared_inventory_reactor_volumes_mL": declared_volumes,
        "reported_reactor_volume_matches_declared": volume in declared_volumes if volume is not None else None,
        "source_exclusion": {
            "expected_record_ids": sorted(case.excluded_record_ids),
            "prompt_leaks": metrics.get("prompt_source_id_leaks", []),
            "retrieval_leaks": metrics.get("retrieval_source_leaks", []),
            "leave_one_source_out_pass": metrics.get("leave_one_source_out_pass"),
        },
    }


def build_packets(output: Path, cases: list[tuple[str, AblationCase]]) -> None:
    case_by_id = {case.case_id: (label, case) for label, case in cases}
    contexts = {}
    for label, case in cases:
        contexts[case.case_id] = build_case_context(
            case_label=label,
            public_input=case.public_payload(),
            inventory=case.inventory,
            oracle=read_json(ORACLES[label]),
        )
    packets = output / "packets"
    packets.mkdir(parents=True, exist_ok=True)
    key_rows = []
    packet_metrics = []
    for row in campaign_cells(cases):
        run_dir = run_directory(output, row)
        if not completed_run(run_dir):
            raise RuntimeError(f"Cannot packet incomplete generation: {run_dir}")
        label, case = case_by_id[row["case_id"]]
        cid = candidate_id(str(run_dir.relative_to(output)))
        packet = build_outcome_packet(
            candidate=cid,
            case_context=contexts[case.case_id],
            result=read_json(run_dir / "result.json"),
            deterministic_verification=deterministic_sheet(read_json(run_dir / "result.json"), read_json(run_dir / "metrics.json"), case),
            has_gas=bool(case.expected_features.get("gas_required")),
            is_multistage="multistep" in case.category or "multistage" in case.category,
        )
        errors = packet_gate(packet)
        if errors:
            raise RuntimeError(f"Packet gate failed for {cid}: {errors}")
        write_json(packets / f"{cid}.json", packet)
        key_rows.append({
            "candidate_id": cid,
            "case": label,
            "case_id": case.case_id,
            "generator_model": row["model"],
            "generator_model_id": row["model_id"],
            "generator_family": row["generator_family"],
            "architecture": row["architecture"],
            "condition_id": row["condition_id"],
            "run_directory": str(run_dir),
            "result_sha256": sha256_file(run_dir / "result.json"),
            "design_input_sha256": case.design_input_sha256,
        })
        packet_metrics.append({"candidate_id": cid, "bytes": (packets / f"{cid}.json").stat().st_size, "gate_errors": []})
    write_json(output / "frozen/candidate_key_confidential.json", {"candidates": key_rows})
    write_json(output / "frozen/packet_metrics.json", packet_metrics)
    manifest = read_json(output / "frozen/campaign_manifest.json")
    manifest["selected_judges"] = ["qwen", "openai"]
    manifest["judge_count"] = 2
    manifest["criteria_per_candidate"] = 14
    manifest["equal_criterion_weights"] = True
    manifest["architecture_label_withheld"] = True
    manifest["architecture_blinding_claimed"] = False
    manifest["generator_identity_withheld"] = True
    write_json(output / "frozen/campaign_manifest.json", manifest)
    update_progress(output, "packets", len(key_rows), len(key_rows))


def valid_judgment(output: Path, judge: str, cid: str) -> bool:
    base = output / "judgments" / judge / cid
    statuses = [base / "status.json", *base.glob("attempts/attempt_*/status.json")]
    return any(path.is_file() and read_json(path).get("status") == "valid" for path in statuses)


def run_judges(output: Path, pause_s: float) -> None:
    candidates = read_json(output / "frozen/candidate_key_confidential.json")["candidates"]
    rubric = read_json(output / "frozen/outcome_rubric.json")
    tasks = [(judge, candidate) for judge in ("qwen", "openai") for candidate in candidates]
    complete = sum(valid_judgment(output, judge, candidate["candidate_id"]) for judge, candidate in tasks)
    update_progress(output, "judging", complete, len(tasks))
    for index, (judge, candidate) in enumerate(tasks, start=1):
        cid = candidate["candidate_id"]
        if valid_judgment(output, judge, cid):
            print(f"JUDGE {index}/{len(tasks)} skip {judge}/{cid}", flush=True)
            continue
        status = call_judge(output, judge, candidate, rubric)
        for _ in range(2):
            if status.get("status") == "valid":
                break
            status = call_judge(output, judge, candidate, rubric)
        complete = sum(valid_judgment(output, name, item["candidate_id"]) for name, item in tasks)
        update_progress(output, "judging", complete, len(tasks), f"{judge} / {cid}")
        append_jsonl(output / "campaign_events.jsonl", {"event": "judgment_finished", "judge": judge, "candidate_id": cid, "status": status.get("status"), "at": datetime.now().astimezone().isoformat()})
        print(f"JUDGE {index}/{len(tasks)} {judge}/{cid}: {status.get('status')}", flush=True)
        if pause_s:
            time.sleep(pause_s)
    incomplete = [(judge, candidate["candidate_id"]) for judge, candidate in tasks if not valid_judgment(output, judge, candidate["candidate_id"])]
    if incomplete:
        raise RuntimeError(f"Judging incomplete for {len(incomplete)} calls; rerun to resume")


def write_case_ranking(report: Path) -> None:
    paired_path = report / "tables/paired_comparisons.csv"
    candidate_path = report / "tables/candidate_consensus_scores.csv"
    paired = list(csv.DictReader(paired_path.open(encoding="utf-8")))
    candidates = list(csv.DictReader(candidate_path.open(encoding="utf-8")))
    rows = []
    for case in sorted({row["case"] for row in paired}):
        case_pairs = [row for row in paired if row["case"] == case]
        deltas = [float(row["paired_delta"]) for row in case_pairs]
        excluded = [float(row["excluded_paired_delta"]) for row in case_pairs]
        sds = [float(row["judge_sd"]) for row in candidates if row["case"] == case]
        rows.append({
            "case": case,
            "models_with_positive_flowpilot_delta": sum(value > 0 for value in deltas),
            "mean_cross_family_delta": round(sum(excluded) / len(excluded), 6),
            "mean_all_judge_delta": round(sum(deltas) / len(deltas), 6),
            "mean_candidate_judge_sd": round(sum(sds) / len(sds), 6),
            "complete_model_pairs": len(case_pairs),
        })
    rows.sort(key=lambda row: (-row["models_with_positive_flowpilot_delta"], -row["mean_cross_family_delta"], -row["mean_all_judge_delta"], row["mean_candidate_judge_sd"], row["case"]))
    for rank, row in enumerate(rows, start=1):
        row["inspection_rank"] = rank
    fields = ["inspection_rank", "case", "models_with_positive_flowpilot_delta", "mean_cross_family_delta", "mean_all_judge_delta", "mean_candidate_judge_sd", "complete_model_pairs"]
    path = report / "tables/case_selection_summary.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows([{key: row[key] for key in fields} for row in rows])
    write_json(report / "case_selection_summary.json", {"policy": "Predeclared inspection ranking; all five cases remain reported.", "cases": rows})


def build_manuscript_diagnostics(output: Path, report: Path) -> None:
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from matplotlib.colors import ListedColormap

    key = read_json(output / "frozen/candidate_key_confidential.json")["candidates"]
    score_rows = list(
        csv.DictReader(
            (report / "tables/candidate_consensus_scores.csv").open(encoding="utf-8")
        )
    )
    scores = {row["candidate_id"]: row for row in score_rows}
    quality_rows = []
    for row in key:
        result = read_json(Path(row["run_directory"]) / "result.json")
        final = result.get("final_design") or {}
        proposal = result.get("proposal") or result.get("raw_proposal") or {}
        score = scores[row["candidate_id"]]
        quality_rows.append(
            {
                "case": row["case"],
                "model": row["generator_model"],
                "architecture": row["architecture"],
                "candidate_id": row["candidate_id"],
                "output_contract": result.get("output_contract") or "production_final_contract",
                "schema_valid": bool(result.get("schema_valid")),
                "assessable_outcome": bool(proposal),
                "final_contract_status": final.get("status") or "not_applicable_one_shot",
                "reported_disposition": result.get("recommended_disposition") or result.get("reported_disposition"),
                "consensus_score_0_1": float(score["mean_score_0_1"]),
                "judge_sd": float(score["judge_sd"]),
                "critical_flags": int(score["total_judge_critical_flags"]),
                "run_directory": row["run_directory"],
            }
        )
    fields = list(quality_rows[0])
    with (report / "tables/run_quality_matrix.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(quality_rows)

    paired = pd.read_csv(report / "tables/paired_comparisons.csv")
    case_order = [
        "CuAAC", "Hydrogenolysis", "Exothermic dinitration",
        "Photochemical oxidation", "Two-stage amidation",
    ]
    model_order = ["Qwen3.6-27B", "GPT-4o"]
    heat = paired.pivot(index="case", columns="model", values="paired_delta").reindex(
        index=case_order, columns=model_order
    )
    fig, ax = plt.subplots(figsize=(8.6, 5.2))
    image = ax.imshow(heat.values, vmin=-0.4, vmax=0.4, cmap="RdYlGn", aspect="auto")
    ax.set_xticks(range(len(model_order)), model_order)
    ax.set_yticks(range(len(case_order)), case_order)
    for y in range(len(case_order)):
        for x in range(len(model_order)):
            ax.text(x, y, f"{heat.iloc[y, x]:+.3f}", ha="center", va="center", fontsize=10)
    fig.colorbar(image, ax=ax, label="FlowPilot minus one-shot score")
    ax.set_title("Matched architecture effect by case and generator")
    fig.tight_layout()
    fig.savefig(report / "figures/fig06_case_model_delta.png", dpi=300)
    plt.close(fig)

    quality = pd.DataFrame(quality_rows)
    columns = [
        (model, architecture)
        for model in model_order
        for architecture in ("One-shot", "FlowPilot")
    ]
    status_values = np.zeros((len(case_order), len(columns)))
    labels: list[list[str]] = []
    for y, case in enumerate(case_order):
        label_row = []
        for x, (model, architecture) in enumerate(columns):
            item = quality[
                (quality.case == case)
                & (quality.model == model)
                & (quality.architecture == architecture)
            ].iloc[0]
            if architecture == "One-shot":
                status = "Assessable" if item.assessable_outcome else "Malformed"
                value = 1 if item.assessable_outcome else 0
            else:
                status = str(item.final_contract_status).title()
                value = 2 if item.final_contract_status == "executable" else 0
            status_values[y, x] = value
            label_row.append(status)
        labels.append(label_row)
    fig, ax = plt.subplots(figsize=(10.5, 5.2))
    ax.imshow(
        status_values,
        vmin=0,
        vmax=2,
        cmap=ListedColormap(["#D95F5F", "#E7B85C", "#3A8D78"]),
        aspect="auto",
    )
    ax.set_xticks(range(len(columns)), [f"{model}\n{architecture}" for model, architecture in columns])
    ax.set_yticks(range(len(case_order)), case_order)
    for y in range(len(case_order)):
        for x in range(len(columns)):
            ax.text(
                x, y, labels[y][x], ha="center", va="center", fontsize=9,
                color="white" if status_values[y, x] in (0, 2) else "black",
            )
    ax.set_title("Delivered-outcome status (not an LLM score)")
    fig.tight_layout()
    fig.savefig(report / "figures/fig07_outcome_status_matrix.png", dpi=300)
    plt.close(fig)

    judgments = pd.read_csv(report / "tables/judge_candidate_scores.csv")
    lookup = judgments.set_index(["judge", "model", "case", "architecture"])["score_0_1"]
    delta_rows = []
    for judge in ("qwen", "openai"):
        for model in model_order:
            values = [
                lookup[judge, model, case, "FlowPilot"]
                - lookup[judge, model, case, "One-shot"]
                for case in case_order
            ]
            delta_rows.append({"judge": judge, "model": model, "mean_delta": float(np.mean(values))})
    delta_frame = pd.DataFrame(delta_rows)
    fig, ax = plt.subplots(figsize=(8.6, 5.0))
    x_values = np.arange(len(model_order))
    width = 0.34
    for index, judge in enumerate(("qwen", "openai")):
        values = [
            delta_frame[(delta_frame.judge == judge) & (delta_frame.model == model)].mean_delta.iloc[0]
            for model in model_order
        ]
        ax.bar(
            x_values + (index - .5) * width,
            values,
            width,
            label=f"{judge.title()} judge",
            color=("#4C78A8", "#E39C37")[index],
        )
    ax.axhline(0, color="#333333", linewidth=1)
    ax.set_xticks(x_values, model_order)
    ax.set_ylabel("Mean matched FlowPilot minus one-shot score")
    ax.set_title("Judge-specific architecture effect")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=.2)
    fig.tight_layout()
    fig.savefig(report / "figures/fig08_judge_specific_delta.png", dpi=300)
    plt.close(fig)

    qwen = paired[paired.model == "Qwen3.6-27B"]
    gpt = paired[paired.model == "GPT-4o"]
    attempt1 = report.with_name(report.name + "_attempt1_nested_string_contract")
    readme = f"""# Manuscript Five-Case Pilot Dataset

## Dataset

- Five held-out chemistry cases, two generator families, and two architectures: 20 matched outcomes.
- One generation repeat per cell; this is a case-selection pilot, not a confirmatory benchmark.
- Two blinded LLM judges scored 14 fixed equal-weight criteria: 40 valid judgments and 560 criterion rows.
- Every case used one identical frozen input hash across all four conditions.
- Held-out source IDs were excluded from retrieval and shown only to judges for fact checking.

## Fairness Repair

The first pass used a JSON string nested inside a JSON envelope for one-shot outputs. This caused serialization failures unrelated to chemistry. That pass is preserved at `{attempt1}` and under `archived_attempts/nested_string_contract_v1`. The reported dataset uses the direct JSON-object v2 baseline. FlowPilot generations were not rerun.

## Primary Findings

- Qwen3.6-27B: FlowPilot {qwen.flowpilot_score_0_1.mean():.3f} vs one-shot {qwen.one_shot_score_0_1.mean():.3f}; delta {qwen.paired_delta.mean():+.3f}.
- GPT-4o: FlowPilot {gpt.flowpilot_score_0_1.mean():.3f} vs one-shot {gpt.one_shot_score_0_1.mean():.3f}; delta {gpt.paired_delta.mean():+.3f}.
- Across all ten pairs, the mean delta is {paired.paired_delta.mean():+.3f}.
- Judge agreement is low. LLM scores are secondary evidence, not the sole endpoint.

## Recommended Cases For Confirmatory Repeats

1. CuAAC: cleanest comparison; both model families show a small non-negative FlowPilot effect and both final contracts are executable.
2. Hydrogenolysis: technically discriminating gas-liquid-solid case.
3. Exothermic dinitration: safety and heat-transfer case.

Photochemical oxidation and two-stage amidation should remain in ESI/failure analysis because both FlowPilot outputs were blocked. Run the selected three with at least three independent repeats per condition. Use deterministic error counts and contract closure as primary endpoints and blinded LLM judging as a secondary sensitivity analysis. Do not claim universal superiority from this single-repeat pilot.
"""
    (report / "MANUSCRIPT_DATASET_README.md").write_text(readme, encoding="utf-8")


def build_report(output: Path, report: Path) -> None:
    report.mkdir(parents=True, exist_ok=True)
    summary = aggregate(output, report)
    figures(report)
    documentation(output, report, summary)
    write_case_ranking(report)
    build_manuscript_diagnostics(output, report)
    frozen = report / "frozen"
    frozen.mkdir(exist_ok=True)
    for name in ("campaign_manifest.json", "case_manifest.json", "execution_plan.json", "outcome_rubric.json", "packet_metrics.json", "SELECTION_POLICY.md"):
        source = output / "frozen" / name
        if source.is_file():
            shutil.copy2(source, frozen / name)
    source_code = output / "frozen/source_code"
    if source_code.is_dir():
        shutil.copytree(source_code, frozen / "source_code", dirs_exist_ok=True)
    checksums = output / "frozen/source_code_checksums.json"
    if checksums.is_file():
        shutil.copy2(checksums, frozen / checksums.name)
    update_progress(output, "complete", 60, 60)


def repair_one_shot(
    output: Path,
    report: Path,
    cases: list[tuple[str, AblationCase]],
    pause_s: float,
) -> None:
    """Replace the nested-string baseline and retain every attempt-1 artifact."""
    archive = output / "archived_attempts/nested_string_contract_v1"
    archive.mkdir(parents=True, exist_ok=True)
    key_path = output / "frozen/candidate_key_confidential.json"
    old_key = read_json(key_path)["candidates"] if key_path.is_file() else []
    one_shot_ids = {
        row["candidate_id"] for row in old_key if row["architecture"] == "One-shot"
    }
    for row in campaign_cells(cases):
        if row["architecture"] != "One-shot":
            continue
        current = run_directory(output, row)
        saved = (
            archive
            / "generation"
            / safe_name(row["case_label"])
            / row["condition_id"]
            / "repeat_01"
        )
        if current.exists() and not saved.exists():
            saved.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(current), str(saved))
    for judge in ("qwen", "openai"):
        for cid in one_shot_ids:
            current = output / "judgments" / judge / cid
            saved = archive / "judgments" / judge / cid
            if current.exists() and not saved.exists():
                saved.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(current), str(saved))
    attempt1_report = report.with_name(
        report.name + "_attempt1_nested_string_contract"
    )
    if report.exists() and not attempt1_report.exists():
        shutil.move(str(report), str(attempt1_report))
    manifest = read_json(output / "frozen/campaign_manifest.json")
    manifest.update(
        {
            "one_shot_output_contract": "direct_json_object_v2",
            "superseded_one_shot_contract": "nested_json_string_v1",
            "repair_reason": (
                "The nested JSON string added serialization failures unrelated "
                "to chemistry or engineering quality."
            ),
            "repair_applied_at": datetime.now().astimezone().isoformat(),
            "superseded_artifacts": str(archive),
        }
    )
    write_json(output / "frozen/campaign_manifest.json", manifest)
    run_generation(output, cases, pause_s)
    build_packets(output, cases)
    run_judges(output, pause_s)
    build_report(output, report)


def health_check(output: Path) -> None:
    status = {name: endpoint_health(bundle) for name, bundle in MODELS.items()}
    write_json(output / "endpoint_health.json", status)
    failed = [name for name, row in status.items() if not row.get("reachable") or not row.get("model_advertised")]
    if failed:
        raise RuntimeError(f"Unavailable benchmark endpoints: {failed}; see {output / 'endpoint_health.json'}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument(
        "--phase",
        choices=("init", "generate", "packets", "judge", "report", "repair-one-shot", "all"),
        default="all",
    )
    parser.add_argument("--pause-between-calls", type=float, default=1.0)
    parser.add_argument("--skip-health", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    report = args.report.resolve()
    output.mkdir(parents=True, exist_ok=True)
    cases = load_frozen_cases()
    initialize(output, cases)
    if args.phase == "init":
        print(output)
        return
    if args.phase == "repair-one-shot":
        if not args.skip_health:
            health_check(output)
        repair_one_shot(output, report, cases, args.pause_between_calls)
        print(output)
        print(report)
        return
    if args.phase in {"generate", "all"}:
        if not args.skip_health:
            health_check(output)
        run_generation(output, cases, args.pause_between_calls)
    if args.phase in {"packets", "all"}:
        build_packets(output, cases)
    if args.phase in {"judge", "all"}:
        if not args.skip_health:
            health_check(output)
        run_judges(output, args.pause_between_calls)
    if args.phase in {"report", "all"}:
        build_report(output, report)
    print(output)
    print(report)


if __name__ == "__main__":
    main()
