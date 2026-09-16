#!/usr/bin/env python3
"""Replay frozen FlowPilot outputs through the canonical v2 release contract."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from copy import deepcopy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from flora_translate.final_design_contract import (
    build_final_design_contract,
    publish_final_design_artifacts,
)
from flora_translate.schemas import ProcessTopology
from flora_translate.topology_semantics import (
    normalize_topology_semantics,
    topology_semantic_issues,
)


DEFAULT_CAMPAIGN = ROOT / "ablation_results/newgen_benchmark/newgen_2_0_qwen_openai_20260818"
DEFAULT_OUTPUT = ROOT / "deliverables/flowpilot_canonical_contract_v2_20260819"
CONFIRMED_FAMILIES = {
    "CuAAC": "cuaac",
    "Hydrogenolysis": "hydrogenolysis/debenzylation",
    "Two-stage amidation": "oxidative amidation",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign", type=Path, default=DEFAULT_CAMPAIGN)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = args.output.resolve()
    contracts_dir = output / "contracts"
    contracts_dir.mkdir(parents=True, exist_ok=True)

    key_path = args.campaign / "frozen/candidate_key_confidential.json"
    candidates = json.loads(key_path.read_text())["candidates"]
    rows = []
    smoke_result_written = False
    for candidate in candidates:
        if candidate["architecture"] != "FlowPilot":
            continue
        run_dir = Path(candidate["run_directory"])
        original = json.loads((run_dir / "result.json").read_text())
        public_input = json.loads((run_dir / "input_public.json").read_text())
        original_topology = ProcessTopology.model_validate(original["process_topology"])
        phase_defects = topology_semantic_issues(original_topology)

        prepared = _prepare(original, public_input)
        original_inventory_contract = build_final_design_contract(prepared)

        confirmed = deepcopy(prepared)
        confirmed["intake_package"] = {
            "chemistry_identity_confirmation": {
                "transformation_family": CONFIRMED_FAMILIES[candidate["case"]],
                "confirmed": True,
                "source": "chemist_confirmed_replay",
            }
        }
        _add_safety_accessories(confirmed["inventory_snapshot"], candidate["case"])
        confirmed_contract = build_final_design_contract(confirmed)
        contract_path = contracts_dir / f"{candidate['candidate_id']}.json"
        contract_path.write_text(
            json.dumps(confirmed_contract, indent=2, ensure_ascii=True),
            encoding="utf-8",
        )
        if confirmed_contract["status"] == "executable" and not smoke_result_written:
            publish_final_design_artifacts(confirmed, confirmed_contract)
            (output / "gui_smoke_result.json").write_text(
                json.dumps(confirmed, indent=2, ensure_ascii=True),
                encoding="utf-8",
            )
            smoke_result_written = True

        rows.append(
            {
                "candidate_id": candidate["candidate_id"],
                "generator_family": candidate["generator_family"],
                "case": candidate["case"],
                "original_topology_semantic_defects": len(phase_defects),
                "old_council_safety_checks": (original.get("safety_report") or {}).get(
                    "total_checks", 0
                ),
                "original_inventory_status_v2": original_inventory_contract["status"],
                "original_inventory_blockers_v2": ";".join(
                    item["code"]
                    for item in original_inventory_contract["consistency"]["issues"]
                ),
                "confirmed_identity_safety_status_v2": confirmed_contract["status"],
                "remaining_blockers_v2": ";".join(
                    item["code"] for item in confirmed_contract["consistency"]["issues"]
                ),
                "canonical_sha256": confirmed_contract.get("canonical_sha256") or "",
            }
        )

    with (output / "replay_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "schema_version": "flowpilot_canonical_contract_replay_v1.0",
        "campaign": str(args.campaign.resolve()),
        "candidate_count": len(rows),
        "original_inventory_executable": sum(
            row["original_inventory_status_v2"] == "executable" for row in rows
        ),
        "confirmed_identity_safety_executable": sum(
            row["confirmed_identity_safety_status_v2"] == "executable" for row in rows
        ),
        "remaining_blocked_candidates": [
            {
                "candidate_id": row["candidate_id"],
                "case": row["case"],
                "generator_family": row["generator_family"],
                "blockers": row["remaining_blockers_v2"].split(";")
                if row["remaining_blockers_v2"]
                else [],
            }
            for row in rows
            if row["confirmed_identity_safety_status_v2"] != "executable"
        ],
        "rows": rows,
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=True), encoding="utf-8"
    )
    (output / "README.md").write_text(_report(summary), encoding="utf-8")
    print(json.dumps(summary, indent=2))


def _prepare(original: dict, public_input: dict) -> dict:
    result = deepcopy(original)
    result["batch_record"] = {
        "raw_text": public_input["protocol"],
        "reaction_description": public_input["protocol"],
    }
    result["inventory_snapshot"] = public_input["inventory"]
    topology = normalize_topology_semantics(
        ProcessTopology.model_validate(result["process_topology"])
    )
    result["process_topology"] = topology.model_dump(mode="json")
    # A current pipeline run renders after phase normalization. The old diagram
    # hash is intentionally omitted rather than compared with a changed graph.
    result["diagram_render_manifest"] = {}
    result.pop("final_design", None)
    return result


def _add_safety_accessories(inventory: dict, case: str) -> None:
    accessories = inventory.setdefault("safety_accessories", [])
    accessories.append(
        {
            "equipment_id": "benchmark_safety_enclosure",
            "name": "Declared blast shield and compatible secondary containment",
            "type": "safety enclosure",
            "capabilities": ["shield_or_containment"],
        }
    )
    if case == "Hydrogenolysis":
        accessories.append(
            {
                "equipment_id": "benchmark_h2_check_valve",
                "name": "Hydrogen-service non-return check valve",
                "type": "check valve",
                "capabilities": ["backflow_prevention"],
            }
        )


def _report(summary: dict) -> str:
    blocked = summary["remaining_blocked_candidates"]
    lines = [
        "# Canonical Contract v2 Frozen Replay",
        "",
        "This replay evaluates the six frozen FlowPilot outputs with the new",
        "post-realization semantic release contract. It does not regenerate model",
        "answers and does not alter the original benchmark files.",
        "",
        "## Result",
        "",
        f"- Original inventory/intake packages executable under v2: "
        f"{summary['original_inventory_executable']}/{summary['candidate_count']}",
        f"- Executable after explicit chemistry confirmation and declared safety "
        f"accessories: {summary['confirmed_identity_safety_executable']}/"
        f"{summary['candidate_count']}",
        f"- Remaining model-output defects: {len(blocked)}",
        "",
        "The original outputs are correctly blocked because chemistry identity was",
        "not an explicit authority field and required safety accessories were absent",
        "from inventory. After supplying those missing authorities, the candidates",
        "listed below remain blocked for genuine output defects:",
        "",
    ]
    for item in blocked:
        lines.append(
            f"- `{item['candidate_id']}` ({item['case']}): "
            + ", ".join(f"`{code}`" for code in item["blockers"])
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "The release contract separates missing user/laboratory authority from",
            "model mistakes. It no longer accepts a model confidence score or a",
            "numerically closed proposal as sufficient evidence of executability.",
            "Topology phase is normalized before rendering, and safety, procedure,",
            "validation experiments, and their canonical SHA-256 are compiled only",
            "after final realization.",
        ]
    )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
