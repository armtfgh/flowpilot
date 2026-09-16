"""Build one canonical package from the accepted KHU three-protocol runs."""

from __future__ import annotations

import argparse
import hashlib
import csv
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from flora_design.visualizer.flowsheet_builder import FlowsheetBuilder
from flora_translate.diagram_artifacts import render_topology_artifacts
from flora_translate.final_design_contract import build_final_design_contract
from flora_translate.gui_autosave import autosave_gui_result
from flora_translate.inventory_profiles import inventory_profile_from_payload
from flora_translate.main import (
    _build_translate_topology,
    _sync_stage_hardware_from_compiled_topology,
    _topology_matches_serialized_proposal,
)
from flora_translate.multistage_inventory import reconcile_multistage_inventory
from flora_translate.schemas import BatchRecord, ChemistryPlan, FlowProposal
from flora_translate.topology_compiler import compile_inventory_topology


ACCEPTED_RUNS = {
    "case_01_photoredox_giese_aerobic_oxidation": Path(
        "outputs/benchmarks/khu_three_protocols_20260810_122940/"
        "case_01_photoredox_giese_aerobic_oxidation"
    ),
    "case_02_dpdtc_compound_1_procedure_a": Path(
        "outputs/benchmarks/khu_three_protocols_20260810_125530/"
        "case_02_dpdtc_compound_1_procedure_a"
    ),
    "case_03_dpdtc_compound_2_procedure_b": Path(
        "outputs/benchmarks/khu_three_protocols_20260810_125530/"
        "case_03_dpdtc_compound_2_procedure_b"
    ),
}

TITLE_OVERRIDES = {
    "case_01_photoredox_giese_aerobic_oxidation": (
        "Photoredox Giese addition followed by aerobic oxidation"
    ),
    "case_02_dpdtc_compound_1_procedure_a": (
        "DPDTC amide coupling: nitrobenzoic acid + benzylamine"
    ),
    "case_03_dpdtc_compound_2_procedure_b": (
        "DPDTC amide coupling: benzoic acid + morpholine"
    ),
}


def _json_safe(value: Any) -> Any:
    return json.loads(json.dumps(value, default=str))


def _artifact_dir(case_dir: Path) -> Path:
    values = sorted((case_dir / "artifacts").glob("*/result.json"))
    if not values:
        raise FileNotFoundError(f"No result artifact under {case_dir}")
    return values[-1].parent


def _close(left: float, right: float, tolerance: float = 0.01) -> bool:
    scale = max(abs(left), abs(right), 1.0)
    return abs(left - right) / scale <= tolerance


def _audit(result: dict[str, Any], known_ids: set[str]) -> dict[str, Any]:
    final = result["final_design"]
    stages = final.get("stages") or []
    stage_checks = []
    for stage in stages:
        volume = float(stage.get("reactor_volume_mL") or 0.0)
        liquid = float(stage.get("Q_liquid_mL_min") or 0.0)
        gas_stp = float(stage.get("Q_gas_sccm") or 0.0)
        gas_actual = float(stage.get("Q_gas_actual_mL_min") or 0.0)
        expected_inlet = volume / max(liquid + gas_stp, 1e-12)
        expected_channel = volume / max(liquid + gas_actual, 1e-12)
        reported_inlet = float(stage.get("residence_time_inlet_min") or 0.0)
        reported_channel = float(stage.get("residence_time_in_channel_min") or 0.0)
        stage_checks.append(
            {
                "stage_number": stage.get("stage_number"),
                "reactor_equipment_id": stage.get("reactor_equipment_id"),
                "volume_mL": volume,
                "liquid_flow_mL_min": liquid,
                "gas_flow_STP_mL_min": gas_stp,
                "gas_flow_in_channel_mL_min": gas_actual,
                "reported_tau_inlet_min": reported_inlet,
                "computed_tau_inlet_min": round(expected_inlet, 4),
                "reported_tau_in_channel_min": reported_channel,
                "computed_tau_in_channel_min": round(expected_channel, 4),
                "inlet_arithmetic_passed": _close(reported_inlet, expected_inlet),
                "channel_arithmetic_passed": _close(
                    reported_channel, expected_channel
                ),
            }
        )
    unknown_manifest_ids = [
        item.get("equipment_id")
        for item in final.get("instrument_manifest") or []
        if item.get("equipment_id")
        and item.get("equipment_id") not in known_ids
        and not str(item.get("equipment_id")).startswith("ASSUMED-")
    ]
    parameters = final.get("parameters") or {}
    return {
        "final_design_executable": final.get("status") == "executable",
        "consistency_passed": bool((final.get("consistency") or {}).get("passed")),
        "topology_matches_serialized_design": bool(
            (result.get("final_validation") or {}).get("checks", {}).get(
                "topology_matches_serialized_design"
            )
        ),
        "inventory_allocation_status": (result.get("inventory_allocation") or {}).get(
            "status"
        ),
        "unknown_manifest_ids": unknown_manifest_ids,
        "stage_checks": stage_checks,
        "stage_tau_sum_inlet_min": round(
            sum(item["reported_tau_inlet_min"] for item in stage_checks), 4
        ),
        "reported_total_tau_inlet_min": parameters.get("residence_time_inlet_min"),
        "stage_tau_sum_channel_min": round(
            sum(item["reported_tau_in_channel_min"] for item in stage_checks), 4
        ),
        "reported_total_tau_channel_min": parameters.get(
            "residence_time_in_channel_min"
        ),
        "passed": (
            final.get("status") == "executable"
            and bool((final.get("consistency") or {}).get("passed"))
            and not unknown_manifest_ids
            and all(
                item["inlet_arithmetic_passed"]
                and item["channel_arithmetic_passed"]
                for item in stage_checks
            )
        ),
    }


def _canonicalize(
    original: dict[str, Any],
    *,
    profile: Any,
    title: str,
) -> dict[str, Any]:
    result = _json_safe(original)
    proposal = FlowProposal.model_validate(result["proposal"])
    chemistry = ChemistryPlan.model_validate(result["chemistry_plan"])
    proposal, multistage = reconcile_multistage_inventory(
        proposal,
        chemistry,
        profile.lab_inventory,
        operating_limits=profile.design_operating_limits(),
    )
    abstract = _build_translate_topology(proposal, chemistry, BatchRecord())
    compiled, allocation = compile_inventory_topology(
        abstract,
        proposal=proposal,
        inventory=profile.lab_inventory,
    )
    _sync_stage_hardware_from_compiled_topology(proposal, compiled)

    result["proposal"] = proposal.model_dump()
    result["multistage_inventory_plan"] = multistage
    result["multistage_inventory_plan"]["stage_parameters"] = proposal.stage_parameters
    result.setdefault("design_calculations", {})["stage_calculations"] = (
        proposal.stage_parameters
    )
    result["process_requirements_topology"] = abstract.model_dump()
    result["process_topology"] = compiled.model_dump()
    result["inventory_allocation"] = allocation
    result["instrument_manifest"] = allocation.get("instrument_manifest") or []
    checks = result.setdefault("final_validation", {}).setdefault("checks", {})
    checks["topology_matches_serialized_design"] = (
        _topology_matches_serialized_proposal(compiled, proposal)
    )
    checks["inventory_topology_assignment_complete"] = bool(
        allocation.get("checks", {}).get("all_required_operations_assigned")
    )
    unresolved = [name for name, passed in checks.items() if not passed]
    result["final_validation"]["unresolved_reasons"] = unresolved
    result["final_validation"]["status"] = "ready" if not unresolved else "screen_required"

    artifacts = render_topology_artifacts(
        compiled,
        title=title[:72],
        builder=FlowsheetBuilder(),
    )
    result["svg_path"] = artifacts["svg_path"]
    result["png_path"] = artifacts["png_path"]
    result["diagram_artifacts"] = {
        key: value for key, value in artifacts.items() if key != "manifest"
    }
    result["diagram_render_manifest"] = artifacts["manifest"]
    result["final_design"] = build_final_design_contract(result)
    return result


def _checksums(root: Path) -> None:
    rows = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.name != "checksums.sha256":
            rows.append(
                f"{hashlib.sha256(path.read_bytes()).hexdigest()}  "
                f"{path.relative_to(root)}"
            )
    (root / "checksums.sha256").write_text("\n".join(rows) + "\n")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build an audited canonical package for the three KHU cases."
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        help=(
            "Fresh three-case benchmark root. When omitted, use the previously "
            "accepted case runs."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    profile_path = Path("khu_inventory_updated_flowpilot.json")
    profile = inventory_profile_from_payload(json.loads(profile_path.read_text()))
    known_ids = {item.equipment_id for item in profile.lab_inventory.all_equipment()}
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    root = Path("outputs/benchmarks") / f"khu_three_protocols_canonical_{stamp}"
    root.mkdir(parents=True, exist_ok=False)
    shutil.copy2(profile_path, root / "source_inventory_profile.json")
    if Path("2 protocols.pdf").is_file():
        shutil.copy2("2 protocols.pdf", root / "source_2_protocols.pdf")

    accepted_runs = ACCEPTED_RUNS
    if args.source_root is not None:
        source_root = args.source_root.resolve()
        missing = [
            case_id
            for case_id in TITLE_OVERRIDES
            if not (source_root / case_id).is_dir()
        ]
        if missing:
            raise FileNotFoundError(
                f"Source root {source_root} is missing case folders: {missing}"
            )
        accepted_runs = {
            case_id: source_root / case_id for case_id in TITLE_OVERRIDES
        }
    else:
        source_root = None

    summaries = []
    source_paths = {}
    for case_id, source_case in accepted_runs.items():
        source_artifact = _artifact_dir(source_case)
        original = json.loads((source_artifact / "result.json").read_text())
        title = TITLE_OVERRIDES.get(case_id) or str(
            (original.get("chemistry_plan") or {}).get("reaction_name") or case_id
        )
        result = _canonicalize(original, profile=profile, title=title)
        case_root = root / case_id
        case_root.mkdir(parents=True, exist_ok=False)
        shutil.copy2(source_artifact / "result.json", case_root / "raw_run_result.json")
        if (source_case / "summary.json").is_file():
            shutil.copy2(source_case / "summary.json", case_root / "raw_run_summary.json")
        for name in (
            "protocol.txt",
            "intake_package.json",
            "reference_metadata.json",
            "run.log",
            "llm_events.jsonl",
        ):
            source = source_case / name
            if source.is_file():
                shutil.copy2(source, case_root / name)
        canonical_dir = autosave_gui_result(
            result,
            intake_package=json.loads(
                (source_artifact / "intake_package.json").read_text()
            ),
            source="canonical",
            user_input=(source_case / "protocol.txt").read_text(),
            base_dir=case_root / "final",
        )
        for source_name, export_name in (
            ("process.png", "topology.png"),
            ("process.svg", "topology.svg"),
            ("topology.json", "topology.json"),
            ("final_design.json", "final_design.json"),
            ("result.json", "full_result.json"),
            ("inventory_allocation.json", "inventory_allocation.json"),
            ("instrument_manifest.json", "instrument_manifest.json"),
        ):
            shutil.copy2(canonical_dir / source_name, case_root / export_name)
        audit = _audit(result, known_ids)
        (case_root / "audit.json").write_text(json.dumps(audit, indent=2))
        source_paths[case_id] = {
            "source_case_dir": str(source_case.resolve()),
            "source_artifact_dir": str(source_artifact.resolve()),
            "canonical_artifact_dir": str(canonical_dir.resolve()),
        }
        parameters = result["final_design"].get("parameters") or {}
        summaries.append(
            {
                "case_id": case_id,
                "title": title,
                "status": result["final_design"]["status"],
                "recommended_disposition": result.get("recommended_disposition"),
                "confidence": result.get("confidence"),
                "flow_rate_mL_min": parameters.get("flow_rate_mL_min"),
                "flow_rate_basis": parameters.get("flow_rate_basis"),
                "reactor_volume_mL": parameters.get("reactor_volume_mL"),
                "residence_time_inlet_min": parameters.get(
                    "residence_time_inlet_min"
                ),
                "residence_time_in_channel_min": parameters.get(
                    "residence_time_in_channel_min"
                ),
                "temperature_C": parameters.get("temperature_C"),
                "BPR_bar": parameters.get("BPR_bar"),
                "stages": result["final_design"].get("stages") or [],
                "audit_passed": audit["passed"],
                "canonical_artifact_dir": str(canonical_dir.resolve()),
            }
        )

    manifest = {
        "schema_version": "flowpilot_khu_three_protocols_canonical_v1.0",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "all_audits_passed": all(item["audit_passed"] for item in summaries),
        "source_runs": source_paths,
        "source_benchmark_root": str(source_root) if source_root else None,
        "cases": summaries,
        "notes": [
            "Published flow conditions for PDF cases were not passed to the models.",
            "All designs are screening recommendations, not wet-lab validation.",
            "Generic passive T-mixers remain pre-run verification assumptions because the KHU inventory does not catalogue mixers.",
        ],
    }
    (root / "summary.json").write_text(json.dumps(summaries, indent=2))
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2))
    with (root / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "case_id",
                "title",
                "status",
                "recommended_disposition",
                "confidence",
                "flow_rate_mL_min",
                "reactor_volume_mL",
                "residence_time_inlet_min",
                "residence_time_in_channel_min",
                "temperature_C",
                "BPR_bar",
                "audit_passed",
            ],
        )
        writer.writeheader()
        for item in summaries:
            writer.writerow({key: item.get(key) for key in writer.fieldnames})
    (root / "README.md").write_text(
        "# KHU Three-Protocol FlowPilot Package\n\n"
        "This folder contains the canonical outputs for one photoredox case and "
        "two DPDTC amide-coupling cases from `2 protocols.pdf`, all constrained "
        "to `khu_inventory_updated_flowpilot.json`.\n\n"
        "- `manifest.json`: provenance, accepted source runs, and package status.\n"
        "- `summary.json` / `summary.csv`: cross-case final parameters.\n"
        "- `<case>/protocol.txt`: exact protocol passed to FlowPilot.\n"
        "- `<case>/intake_package.json`: frozen standardized intake.\n"
        "- `<case>/run.log` and `llm_events.jsonl`: complete model and pipeline logs.\n"
        "- `<case>/raw_run_result.json`: untouched result from the fresh benchmark.\n"
        "- `<case>/topology.png` / `topology.svg` / `topology.json`: direct topology exports.\n"
        "- `<case>/final_design.json` and `full_result.json`: canonical design contract and complete result.\n"
        "- `<case>/audit.json`: stage arithmetic, topology, and inventory audit.\n"
        "- `<case>/final/<run>/`: canonical result, final design, allocation, "
        "topology, PNG, and SVG.\n\n"
        "The two PDF cases excluded the publication's flow conditions from the "
        "model prompt. All outputs are screening recommendations, not wet-lab "
        "validation. LOW-confidence cases require conservative experimental "
        "screening.\n",
        encoding="utf-8",
    )
    _checksums(root)
    print(json.dumps({"canonical_root": str(root.resolve()), **manifest}, indent=2))
    if not manifest["all_audits_passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
