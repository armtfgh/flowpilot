"""Independent numerical/provenance checks and portable campaign exports."""
import argparse
import csv
import json
import math
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    rows = []
    checks_all = []
    for path in sorted(args.root.glob("attempt*/0*/result.json")):
        r = json.loads(path.read_text())
        request = json.loads((path.parent / "request.json").read_text())
        a = r.get("scientific_assessment", {})
        p = r.get("proposal", {})
        stages = p.get("stage_parameters", [])
        streams = p.get("streams", [])
        inv = {x["equipment_id"]: x for x in r.get("inventory_snapshot", {}).get("reactors", [])}
        checks = {
            "original_package_protocol": r.get("intake_package", {}).get("raw_protocol") == request["intake_package"]["raw_protocol"],
            "inventory_v4_bound": r.get("intake_package", {}).get("inventory_profile_snapshot", {}).get("version") == 4,
            "scientific_policy": r.get("pipeline_runtime", {}).get("design_policy") == "scientific_v2",
            "12_candidates": a.get("candidate_count") == len(a.get("candidates", [])) == 12,
            "48_domain_reviews": len(a.get("reviews", {})) == 4 and all(sorted(x["candidate_id"] for x in v) == list(range(1, 13)) for v in a.get("reviews", {}).values()),
            "six_council_calls": len(a.get("calls", [])) == 6,
            "selected_design_preserved": a.get("selected_design_preserved") is True,
            "closed_final_design": r.get("final_design", {}).get("status") == "executable",
            "two_stages": len(stages) == 2,
            "volume_flow_time": bool(stages) and all(math.isclose(s["reactor_volume_mL"] / s["Q_liquid_mL_min"], s["residence_time_min"], rel_tol=2e-5) for s in stages),
            "cumulative_liquid_flow": bool(stages) and all(math.isclose(s["Q_liquid_mL_min"], sum(f["flow_rate_mL_min"] for f in streams if f["phase"] == "liquid" and f["introduction_stage"] <= s["stage_number"]), rel_tol=2e-5) for s in stages),
            "inventory_volume_and_temperature": bool(stages) and all(s["reactor_equipment_id"] in inv and math.isclose(s["reactor_volume_mL"], inv[s["reactor_equipment_id"]]["volume_mL"]) and s["temperature_C"] <= inv[s["reactor_equipment_id"]]["max_temperature_C"] for s in stages),
            "objective_sent_to_all_reviewers": bool(a.get("calls")) and all(c["request"]["context"]["objective"] == request["intake_package"]["objective"] for c in a.get("calls", [])),
            "hypotheses_sent_to_all_reviewers": bool(a.get("calls")) and all(all(h in c["request"]["context"]["authority_labeled_intake"] for h in request["intake_package"]["hypotheses"]) for c in a.get("calls", [])),
        }
        if "dpdtc" in path.parent.name:
            amine = [s for s in streams if any("benzylamine" in x.lower() for x in s["contents"])]
            acid = [s for s in streams if any("nitrobenzoic acid" in x.lower() for x in s["contents"])]
            ratio = sum(s["flow_rate_mL_min"] * s["concentration_M"] for s in amine) / sum(s["flow_rate_mL_min"] * s["concentration_M"] for s in acid) if acid else None
            checks["benzylamine_stage_2_1_05_equiv"] = bool(amine) and all(s["introduction_stage"] == 2 for s in amine) and math.isclose(ratio, 1.05, rel_tol=2e-4)
            checks["source_30_plus_30_at_95_C"] = [(h["batch_time_min"], h.get("batch_temperature_C")) for h in a.get("source_context", {}).get("timed_holds", [])] == [(30, 95), (30, 95)]
        from flora_translate.engine.council_v4.scientific import pressure_headroom
        from flora_translate.schemas import FlowProposal, LabInventory
        checks["pump_and_reactor_headroom"] = pressure_headroom(FlowProposal.model_validate(p), r["final_stage_engineering"], LabInventory.model_validate(r["inventory_snapshot"]))["passed"]
        checks["unknown_kinetics_not_fabricated"] = all(s["calculations"].get("rate_constant") is None and s["calculations"].get("intensification_factor") is None for s in r["final_stage_engineering"]["stages"])
        svg = path.parent / "gui_export" / "process.svg"
        checks["icon_topology"] = svg.exists() and "data:image/png;base64," in svg.read_text()
        report = {"run": str(path.parent), "checks": checks, "all_passed": all(checks.values()), "wet_lab_performance_tested": False}
        (path.parent / "independent_checks.json").write_text(json.dumps(report, indent=2))
        checks_all.append(report)
        for s in stages:
            rows.append({"attempt": path.parent.parent.name, "case": path.parent.name, "stage": s["stage_number"],
                "priority": a.get("objective_policy", {}).get("priority"), "candidate": a.get("selected_candidate_id"),
                "volume_mL": s["reactor_volume_mL"], "liquid_mL_min": s["Q_liquid_mL_min"],
                "residence_min": s["residence_time_min"], "temperature_C": s["temperature_C"],
                "BPR_bar": p["BPR_bar"], "ID_mm": s["d_mm"], "material": s["material"]})
        for field in ("streams", "stage_parameters"):
            values = p[field]
            fields = sorted(set().union(*(x.keys() for x in values)))
            with (path.parent / (field + ".csv")).open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fields)
                writer.writeheader()
                writer.writerows({k: json.dumps(v) if isinstance(v, (list, dict)) else v for k, v in x.items()} for x in values)
    if rows:
        with (args.root / "stage_summary.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    (args.root / "independent_checks.json").write_text(json.dumps(checks_all, indent=2))
    print(json.dumps({"runs_checked": len(checks_all), "all_passed": bool(checks_all) and all(x["all_passed"] for x in checks_all), "failures": [{"run": x["run"], "checks": [k for k, v in x["checks"].items() if not v]} for x in checks_all if not x["all_passed"]]}, indent=2))


if __name__ == "__main__":
    main()
