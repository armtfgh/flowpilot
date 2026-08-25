#!/usr/bin/env python3
"""Audit that each saved module-ablation run executed its frozen contract."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


SCORING_AGENTS = ("chemistry", "kinetics", "fluidics", "safety")


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def module_disabled(path: Path) -> bool:
    return path.is_file() and bool((read_json(path) or {}).get("module_disabled"))


def audit(root: Path, output: Path) -> dict:
    rows = []
    generation = root / "generation"
    for summary_path in sorted(generation.glob("*/*/*/run_summary.json")):
        run = summary_path.parent
        summary = read_json(summary_path)
        metadata = summary.get("metadata") or {}
        config = metadata.get("council_execution")
        condition = metadata.get("bundle_name", run.parent.name)
        row = {
            "case_id": metadata.get("case_id", run.parents[2].name),
            "condition": condition,
            "repeat_id": run.name,
            "completed": summary.get("status") == "completed",
            "is_control": config is None,
            "council_path": "control" if config is None else "full",
            "config_snapshot_match": True,
            "scoring_contract_match": True,
            "skeptic_contract_match": True,
            "preselection_contract_match": True,
            "chief_contract_match": True,
            "winner_revision_contract_match": True,
            "dfmea_contract_match": True,
            "final_design_status": "",
            "deterministic_score": "",
        }
        result_path = run / "result.json"
        metrics_path = run / "metrics.json"
        if result_path.is_file():
            row["final_design_status"] = (read_json(result_path).get("final_design") or {}).get("status", "")
        if metrics_path.is_file():
            row["deterministic_score"] = read_json(metrics_path).get("deterministic_composite_score", "")
        if config is not None:
            snapshots = run / "snapshots"
            saved_config = read_json(snapshots / "council_execution_config.json")
            row["config_snapshot_match"] = saved_config == config
            scoring_path = snapshots / "stage2_initial_scoring.json"
            early_screen = snapshots / "stage1_screen_required.json"
            if not scoring_path.is_file() and early_screen.is_file():
                row["council_path"] = "early_screen_required"
            else:
                scoring = read_json(scoring_path)
                enabled = set(config["enabled_scoring_agents"])
                row["scoring_contract_match"] = (
                    set(scoring.get("enabled_scoring_agents") or []) == enabled
                    and set(scoring.get("disabled_scoring_agents") or []) == set(SCORING_AGENTS) - enabled
                    and all(not scoring.get(f"{agent}_scores") for agent in set(SCORING_AGENTS) - enabled)
                )
                checks = (
                    ("skeptic_contract_match", "enable_skeptic", "stage3_initial_audit.json"),
                    ("preselection_contract_match", "enable_preselection_refinement", "stage3_5_refinement_summary.json"),
                    ("winner_revision_contract_match", "enable_winner_revision", "stage5_revision_result.json"),
                    ("dfmea_contract_match", "enable_dfmea", "stage6_dfmea.json"),
                )
                for column, flag, filename in checks:
                    if not config[flag]:
                        row[column] = module_disabled(snapshots / filename)
                if not config["enable_chief_llm"]:
                    chief = read_json(snapshots / "stage4_chief_selection.json")
                    row["chief_contract_match"] = bool((chief.get("chief_data") or {}).get("module_disabled"))
        contract_columns = [key for key in row if key.endswith("_match")]
        row["audit_pass"] = bool(row["completed"] and all(row[key] for key in contract_columns))
        rows.append(row)

    output.mkdir(parents=True, exist_ok=True)
    columns = list(rows[0]) if rows else []
    with (output / "module_execution_audit.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    result = {
        "schema_version": "flowpilot_module_execution_audit_v1.0",
        "run_count": len(rows),
        "passed": sum(bool(row["audit_pass"]) for row in rows),
        "failed": sum(not bool(row["audit_pass"]) for row in rows),
        "all_passed": bool(rows) and all(row["audit_pass"] for row in rows),
        "executable_count": sum(row["final_design_status"] == "executable" for row in rows),
        "rows": rows,
    }
    (output / "module_execution_audit.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (output / "MODULE_EXECUTION_AUDIT.md").write_text(
        "# Module Execution Audit\n\n"
        f"- Saved runs: {result['run_count']}\n"
        f"- Contract audits passed: {result['passed']}/{result['run_count']}\n"
        f"- Executable final designs: {result['executable_count']}/{result['run_count']}\n\n"
        "A pass requires the saved execution configuration, active/disabled scoring-agent "
        "sets, empty outputs for disabled specialists, and explicit bypass snapshots for "
        "disabled council stages to agree. One-shot and no-council controls are checked for "
        "completion but do not have a council execution contract.\n",
        encoding="utf-8",
    )
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    outcome = audit(args.root.resolve(), args.output.resolve())
    print(json.dumps({key: outcome[key] for key in ("run_count", "passed", "failed", "all_passed", "executable_count")}, indent=2))
