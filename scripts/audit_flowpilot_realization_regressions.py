#!/usr/bin/env python3
"""Replay frozen FlowPilot candidates through the current final realizer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from flora_translate.design_realizer import realize_executable_design
from flora_translate.schemas import (
    BatchRecord,
    ChemistryPlan,
    FlowProposal,
    LabInventory,
)


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2, default=str) + "\n",
        encoding="utf-8",
    )


def replay(run_dir: Path) -> dict:
    result = _read(run_dir / "result.json")
    public = _read(run_dir / "input_public.json")
    inventory = LabInventory.model_validate(_read(run_dir / "input_inventory.json"))
    candidate = FlowProposal.model_validate(
        result.get("pre_council_proposal") or result.get("proposal") or {}
    )
    plan = ChemistryPlan.model_validate(result.get("chemistry_plan") or {})
    batch = BatchRecord.model_validate(result.get("batch_record") or {})
    realized, report, validation = realize_executable_design(
        candidate,
        batch_record=batch,
        chemistry_plan=plan,
        inventory=inventory,
        hard_constraints=public.get("hard_constraints"),
    )
    gas = next((item for item in realized.streams if item.phase == "gas"), None)
    return {
        "run": str(run_dir),
        "case": public.get("title"),
        "condition": run_dir.parent.parent.name,
        "status": report.get("status"),
        "validation_status": validation.get("status"),
        "unresolved": validation.get("unresolved_reasons") or [],
        "proposal": realized.model_dump(mode="json"),
        "checks": validation.get("checks") or {},
        "gas_summary": (
            {
                "species": gas.contents,
                "equipment_id": gas.pump_equipment_id,
                "inlet_STP_sccm": gas.gas_flow_sccm,
                "in_channel_mL_min": gas.gas_flow_actual_mL_min,
                "equiv": gas.molar_equiv,
            }
            if gas
            else None
        ),
        "liquid_geometry_closure": {
            "volume_mL": realized.reactor_volume_mL,
            "Q_times_tau_mL": round(
                realized.flow_rate_mL_min * realized.residence_time_min, 8
            ),
            "basis": realized.residence_time_basis,
        },
        "decisions": report.get("decisions") or [],
        "issues": report.get("issues") or [],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("campaign", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    runs = sorted(args.campaign.glob("generation/*/*_flowpilot/repeat_01"))
    rows = [replay(path) for path in runs]
    summary = {
        "run_count": len(rows),
        "complete": sum(row["status"] == "complete" for row in rows),
        "ready": sum(row["validation_status"] == "ready" for row in rows),
        "failed": [
            {"case": row["case"], "condition": row["condition"], "issues": row["issues"]}
            for row in rows
            if row["validation_status"] != "ready"
        ],
    }
    _write(args.output / "replay_results.json", rows)
    _write(args.output / "summary.json", summary)
    print(json.dumps(summary, indent=2))
    return 0 if summary["ready"] == len(rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
