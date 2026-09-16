#!/usr/bin/env python3
"""Offline Stage 2/3 migration and inventory-allocation benchmark."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from flora_translate.diagram_artifacts import render_topology_artifacts
from flora_translate.inventory_profiles import inventory_profile_from_payload
from flora_translate.schemas import (
    FlowProposal,
    MixerSpec,
    PressureControllerSpec,
    ProcessTopology,
    UnitOperation,
)
from flora_translate.topology_compiler import compile_inventory_topology


def main() -> int:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = ROOT / "outputs/benchmarks/inventory_allocation_v3" / stamp
    output_dir.mkdir(parents=True, exist_ok=False)

    source_profile = json.loads((ROOT / "khu_inventory_flowpilot.json").read_text())
    profile = inventory_profile_from_payload(source_profile)
    inventory = profile.lab_inventory
    topology = _topology()
    proposal = _proposal()

    blocked_topology, blocked_report = compile_inventory_topology(
        topology,
        proposal=proposal,
        inventory=inventory,
    )

    completed_inventory = inventory.model_copy(deep=True)
    completed_inventory.mixers.append(
        MixerSpec(
            equipment_id="khu_t_mixer_1",
            name="KHU PFA T-mixer",
            quantity=1,
            type="T-mixer",
            material="PFA",
            max_inputs=2,
            max_pressure_bar=8,
            compatible_systems=["KHU general flow setup"],
            notes="Benchmark declaration added explicitly for Stage 3 verification.",
        )
    )
    completed_inventory.pressure_controllers.append(
        PressureControllerSpec(
            equipment_id="khu_bpr_6bar_1",
            name="KHU 6 bar BPR",
            quantity=1,
            type="BPR",
            setpoints_bar=[6],
            compatible_systems=["KHU general flow setup"],
            notes="Benchmark declaration added explicitly for Stage 3 verification.",
        )
    )
    compiled_topology, completed_report = compile_inventory_topology(
        topology,
        proposal=proposal,
        inventory=completed_inventory,
    )
    diagram = render_topology_artifacts(
        compiled_topology,
        title="KHU inventory-backed gas-liquid photochemical topology",
        base_dir=output_dir / "diagram",
    )

    _write(output_dir / "migrated_khu_profile_v3.json", profile.model_dump())
    _write(output_dir / "abstract_topology.json", topology.model_dump())
    _write(output_dir / "blocked_compiled_topology.json", blocked_topology.model_dump())
    _write(output_dir / "blocked_allocation_report.json", blocked_report)
    _write(output_dir / "completed_inventory.json", completed_inventory.model_dump())
    _write(output_dir / "compiled_topology.json", compiled_topology.model_dump())
    _write(output_dir / "completed_allocation_report.json", completed_report)

    blocked_categories = {
        item["category"] for item in blocked_report["unresolved_requirements"]
    }
    known_ids = {item.equipment_id for item in completed_inventory.all_equipment()}
    assigned_ids = {
        item_id
        for assignment in completed_report["assignments"]
        for item_id in assignment["equipment_item_ids"]
    }
    checks = {
        "profile_migrated_to_v3": profile.schema_version.endswith("v3.0"),
        "strict_assignment_enabled": inventory.strict_assignment,
        "pump_quantity_migrated": inventory.pumps[0].quantity == 3,
        "missing_mixer_detected": "mixers" in blocked_categories,
        "missing_pressure_controller_detected": "pressure_controllers" in blocked_categories,
        "baseline_design_blocked": blocked_report["status"] == "incomplete",
        "declared_hardware_design_complete": completed_report["status"] == "complete",
        "no_invented_equipment_ids": assigned_ids <= known_ids,
        "no_degasser_in_compiled_topology": not any(
            operation.op_type in {"degas", "degasser", "deoxygenation_unit"}
            for operation in compiled_topology.unit_operations
        ),
        "diagram_complete": diagram["render_status"] == "complete",
    }
    summary = {
        "schema_version": "flowpilot_inventory_allocation_benchmark_v1.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_inventory": "khu_inventory_flowpilot.json",
        "blocked_unresolved_categories": sorted(blocked_categories),
        "completed_instrument_count": len(completed_report["instrument_manifest"]),
        "renderer": diagram["renderer"],
        "checks": checks,
        "passed": all(checks.values()),
    }
    _write(output_dir / "summary.json", summary)
    (output_dir / "README.md").write_text(
        "# FlowPilot Inventory Allocation v3 Benchmark\n\n"
        "The first pass uses the migrated KHU profile unchanged and must remain blocked "
        "because no mixer or pressure controller is declared. The second pass adds "
        "explicit benchmark records for those two items and recompiles the same topology.\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))
    print(f"Output: {output_dir}")
    return 0 if summary["passed"] else 1


def _topology() -> ProcessTopology:
    fixtures = json.loads(
        (ROOT / "flora_translate/tests/fixtures/diagram_stage0_topologies.json").read_text()
    )
    payload = next(
        item["topology"] for item in fixtures
        if item["name"] == "gas_liquid_photochemistry"
    )
    topology = ProcessTopology.model_validate(payload)
    reactor = next(
        item for item in topology.unit_operations if item.op_type == "photoreactor"
    )
    reactor.parameters["ID_mm"] = 1.016
    topology.unit_operations.insert(
        -1,
        UnitOperation(
            op_id="led_1",
            op_type="led_module",
            label="450 nm LED",
            parameters={"wavelength_nm": 450},
        ),
    )
    return topology


def _proposal() -> FlowProposal:
    return FlowProposal(
        flow_rate_mL_min=0.05,
        reactor_volume_mL=10,
        tubing_ID_mm=1.016,
        tubing_material="PFA",
        BPR_bar=6,
        temperature_C=40,
        wavelength_nm=450,
        inventory_selection={
            "system": "KHU general flow setup",
            "equipment_id": "reactor_khu_pfa_coil_10_ml_khu_general_flow_setup_coil_pfa_10_0_1_01c83f6c",
        },
    )


def _write(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
