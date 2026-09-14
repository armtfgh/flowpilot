"""Independent reporting checks for the supplied DPDTC regression case.

This evaluates representation and arithmetic, not experimental yield or safety.
The manuscript flow time is deliberately absent from every acceptance test.
"""
import argparse
import csv
import json
import math
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("result", type=Path)
    parser.add_argument("--source", type=Path, default=Path("outputs/gui_runs/20260907_145525_webapp/result.json"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    new, old = (json.loads(p.read_text()) for p in (args.result, args.source))
    a = new.get("scientific_assessment") or new.get("proposal", {}).get("scientific_design", {})
    proposal = new.get("proposal", {})
    stages, streams = proposal.get("stage_parameters", []), proposal.get("streams", [])
    checks = {}
    checks["exact_protocol_preserved"] = new.get("intake_package", {}).get("raw_protocol") == old["intake_package"]["raw_protocol"]
    checks["two_source_heating_periods"] = [h["batch_time_min"] for h in a.get("source_context", {}).get("timed_holds", [])] == [30, 30]
    checks["source_stage_temperatures_preserved"] = [h.get("batch_temperature_C") for h in a.get("source_context", {}).get("timed_holds", [])] == [95, 95]
    checks["12_complete_candidates"] = a.get("candidate_count") == len(a.get("candidates", [])) == 12
    checks["48_domain_reviews"] = len(a.get("reviews", {})) == 4 and all(sorted(r["candidate_id"] for r in rows) == list(range(1, 13)) for rows in a.get("reviews", {}).values())
    checks["selected_candidate_not_changed"] = a.get("selected_design_preserved") is True
    checks["benzylamine_only_added_in_stage_2"] = any("benzylamine" in " ".join(s["contents"]).lower() and s["introduction_stage"] == 2 for s in streams) and all("benzylamine" not in " ".join(s["contents"]).lower() for s in streams if s["introduction_stage"] == 1)
    checks["stage_volume_flow_time_closure"] = len(stages) == 2 and all(math.isclose(s["reactor_volume_mL"] / s["Q_liquid_mL_min"], s["residence_time_min"], rel_tol=2e-5) for s in stages)
    checks["cumulative_flow_conserved"] = len(stages) == 2 and all(math.isclose(s["Q_liquid_mL_min"], sum(f["flow_rate_mL_min"] for f in streams if f["introduction_stage"] <= s["stage_number"]), rel_tol=2e-5) for s in stages)
    feeds = {s["introduction_stage"]: s for s in streams}
    ratio = None
    if 1 in feeds and 2 in feeds:
        ratio = feeds[2]["concentration_M"] * feeds[2]["flow_rate_mL_min"] / (feeds[1]["concentration_M"] * feeds[1]["flow_rate_mL_min"])
    checks["benzylamine_1_05_equiv"] = ratio is not None and math.isclose(ratio, 1.05, rel_tol=2e-4)
    inventory = {r["equipment_id"]: r for r in new.get("inventory_snapshot", {}).get("reactors", [])}
    checks["exact_inventory_reactors_and_temperatures"] = bool(stages) and all(s["reactor_equipment_id"] in inventory and
        math.isclose(s["reactor_volume_mL"], inventory[s["reactor_equipment_id"]]["volume_mL"]) and
        s["temperature_C"] <= inventory[s["reactor_equipment_id"]]["max_temperature_C"] for s in stages)
    engineering = new.get("final_stage_engineering", {})
    from flora_translate.engine.council_v4.scientific import pressure_headroom
    from flora_translate.schemas import FlowProposal, LabInventory
    headroom = pressure_headroom(FlowProposal.model_validate(proposal), engineering, LabInventory.model_validate(new["inventory_snapshot"]))
    checks["pump_and_reactor_pressure_headroom"] = headroom["passed"]
    checks["unknown_kinetics_not_fabricated"] = engineering.get("complete") is True and all(s["calculations"].get("rate_constant") is None and s["calculations"].get("intensification_factor") is None for s in engineering.get("stages", []))
    checks["executable_contract_closed"] = new.get("final_design", {}).get("status") == "executable"
    checks["icon_topology_published"] = bool(new.get("svg_path")) and Path(new["svg_path"]).exists() and "data:image/png;base64," in Path(new["svg_path"]).read_text()
    result = {"checks": checks, "all_passed": all(checks.values()), "benzylamine_equiv_recomputed": ratio,
              "old_total_min": old["proposal"]["residence_time_min"], "new_total_min": proposal.get("residence_time_min"),
              "selected_candidate_id": a.get("selected_candidate_id"), "experimental_yield_evaluated": False,
              "caution": "No wet-lab outcome was measured; these are software consistency and provenance checks, not proof of reaction performance."}
    (args.output / "evaluation.json").write_text(json.dumps(result, indent=2))
    with (args.output / "stage_comparison.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["version", "stage", "volume_mL", "liquid_mL_min", "residence_min", "temperature_C", "reactor_id"])
        for label, r in (("frozen_legacy", old), ("scientific_preview", new)):
            for s in r["proposal"]["stage_parameters"]:
                writer.writerow([label, s["stage_number"], s.get("reactor_volume_mL"), s.get("Q_liquid_mL_min"), s.get("residence_time_min"), s.get("temperature_C"), s.get("reactor_equipment_id")])
    lines = ["# DPDTC scientific-policy evaluation", "", result["caution"], "", "| Check | Result |", "| --- | --- |"]
    lines += [f"| {k.replace('_', ' ')} | {'PASS' if v else 'FAIL'} |" for k, v in checks.items()]
    lines += ["", f"Legacy total: {result['old_total_min']} min. New total: {result['new_total_min']} min.",
              "Longer residence time is not itself a success criterion. Preserved evidence, chemistry order, constraints and review coverage are."]
    (args.output / "evaluation.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(result, indent=2))
    return 0 if result["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
