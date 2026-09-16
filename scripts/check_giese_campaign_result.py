"""Audit saved Giese screening output without further model calls."""
import argparse
import csv
import json
import math
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    r = json.loads((args.run / "result.json").read_text())
    request = json.loads((args.run / "request.json").read_text())
    requested_stage = request.get("requested_oxygen_stage", 2)
    p = r["proposal"]
    a = r["scientific_assessment"]
    stages, streams = p["stage_parameters"], p["streams"]
    inventory = r["inventory_snapshot"]
    reactors = {x["equipment_id"]: x for x in inventory["reactors"]}
    lights = {x["equipment_id"]: x for x in inventory["light_sources"]}
    gas = [s for s in streams if s["phase"] == "gas"]
    feed = [s for s in streams if s["phase"] == "liquid" and s["molar_equiv"] == 1]
    close = lambda x, y: math.isclose(x, y, rel_tol=0.0001, abs_tol=0.00001)
    # Independent ideal-gas conversion, not the implementation being tested.
    vm = 0.08314462618 * 273.15 / 1.01325  # mL per mmol at STP
    equiv = sum(s["gas_flow_sccm"] * s["gas_reagent_mole_fraction"] / vm for s in gas) / sum(s["flow_rate_mL_min"] * s["concentration_M"] for s in feed)
    report_stages = r["result_report"]["stages"]
    checks = {
        "source_protocol_unchanged": r["intake_package"]["raw_protocol"] == request["intake_package"]["raw_protocol"],
        "khu_v4": r["intake_package"]["inventory_profile_snapshot"]["version"] == 4,
        "scientific_policy": r["pipeline_runtime"]["design_policy"] == "scientific_v2",
        "12_complete_candidates": a["candidate_count"] == len(a["candidates"]) == 12,
        "48_domain_reviews": len(a["reviews"]) == 4 and all(sorted(v["candidate_id"] for v in rows) == list(range(1, 13)) for rows in a["reviews"].values()),
        "six_council_contributions": len(a["calls"]) == 6,
        "selected_parameters_preserved": a["selected_design_preserved"] is True,
        "numerical_contract_closed": r["final_design"]["status"] == "executable",
        "two_stages": len(stages) == 2,
        "gas_enters_requested_stage": len(gas) == 1 and gas[0]["introduction_stage"] == requested_stage and all(
            (s["Q_gas_sccm"] > 0) == (s["stage_number"] >= requested_stage) for s in stages),
        "gas_equiv_independent_STP_check": len(gas) == 1 and close(equiv, gas[0]["molar_equiv"]),
        "cumulative_liquid_flow": all(close(s["Q_liquid_mL_min"], sum(f["flow_rate_mL_min"] for f in streams if f["phase"] == "liquid" and f["introduction_stage"] <= s["stage_number"])) for s in stages),
        "cumulative_gas_flow_STP": all(close(s["Q_gas_sccm"], sum(f["gas_flow_sccm"] for f in gas if f["introduction_stage"] <= s["stage_number"])) for s in stages),
        "stage_V_over_inlet_flow": all(close(s["reactor_volume_mL"] / (s["Q_liquid_mL_min"] + s["Q_gas_sccm"]), s["residence_time_min"]) for s in stages),
        "inventory_reactor_volume_and_limits": all(s["reactor_equipment_id"] in reactors and close(s["reactor_volume_mL"], reactors[s["reactor_equipment_id"]]["volume_mL"]) and s["temperature_C"] <= reactors[s["reactor_equipment_id"]]["max_temperature_C"] for s in stages),
        "two_separate_inventory_lights": len({s["light_equipment_id"] for s in stages}) == 2 and all(s["light_equipment_id"] in lights and close(s["wavelength_nm"], lights[s["light_equipment_id"]]["wavelength_nm"]) for s in stages),
        "stage_engineering_complete": r["final_stage_engineering"]["complete"],
        "unknown_kinetics_not_fabricated": all(s["calculations"]["rate_constant"] is None and s["calculations"]["intensification_factor"] is None for s in r["final_stage_engineering"]["stages"]),
        "unknown_gas_uptake_not_fabricated": r["final_stage_engineering"]["stages"][1]["calculations"]["o2_transfer_sufficiency"] is None and r["final_stage_engineering"]["stages"][1]["calculations"]["kLa_s"] is None,
        "gas_change_flagged_unapproved": a["source_context"]["gas_delivery"].get("requires_chemist_confirmation") is True and request["gas_change_is_user_confirmed"] is False,
        "no_predicted_yield": a["predicted_yield_pct"] is None,
        "report_matches_final_stages": len(report_stages) == len(stages) and all(t["closure"] is True and close(t["residence_time_min"], s["residence_time_min"]) and close(t["gas_flow_stp_mL_min"], s["Q_gas_sccm"]) for t, s in zip(report_stages, stages)),
        "objective_sent_to_all_reviewers": all(c["request"]["context"]["objective"] == request["intake_package"]["objective"] for c in a["calls"]),
    }
    from flora_translate.engine.council_v4.scientific import pressure_headroom, signature
    from flora_translate.schemas import FlowProposal, LabInventory
    checks["pressure_headroom"] = pressure_headroom(FlowProposal.model_validate(p), r["final_stage_engineering"], LabInventory.model_validate(inventory))["passed"]
    selected = next(c for c in a["candidates"] if c["candidate_id"] == a["selected_candidate_id"])
    checks["selected_signature_matches"] = signature(FlowProposal.model_validate(p)) == signature(FlowProposal.model_validate(selected["proposal"]))
    svg = args.run / "gui_export/process.svg"
    text = svg.read_text() if svg.exists() else ""
    checks["icon_topology_saved"] = "data:image/png;base64," in text
    labels = " ".join("".join(node.itertext()) for node in ET.fromstring(text).iter()
                      if node.tag.rsplit("}", 1)[-1] in {"text", "title", "desc"}).lower() if text else ""
    checks["topology_not_using_sccm_or_channel_labels"] = bool(labels) and "sccm" not in labels and "in-channel" not in labels
    checks["topology_png_saved"] = (args.run / "gui_export/process.png").exists()
    summary = {"checks": checks, "all_passed": all(checks.values()), "independent_O2_equiv": equiv,
               "wet_lab_performance_tested": False, "gas_source_change_approved": False}
    (args.run / "independent_checks.json").write_text(json.dumps(summary, indent=2))
    for name, rows in {"stages": report_stages, "streams": r["result_report"]["streams"]}.items():
        with (args.run / f"{name}.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows({k: json.dumps(v) if isinstance(v, (dict, list)) else v for k, v in row.items()} for row in rows)
    print(json.dumps({"all_passed": summary["all_passed"], "checks": len(checks), "failed": [k for k, v in checks.items() if not v], "independent_O2_equiv": equiv}, indent=2))
    return 0 if summary["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
