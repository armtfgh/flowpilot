"""Export immutable GUI verification evidence; no model requests are made."""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from flora_translate.diagram_views import current_diagram
from flora_translate.result_reporting import build_result_report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("sources", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    results = []
    for source in args.sources:
        data = source.read_bytes()
        result = json.loads(data)
        report = build_result_report(result)
        directory = args.output / source.parent.name
        directory.mkdir(exist_ok=False)
        shutil.copy2(source, directory / "result.json")
        (directory / "process_report.json").write_text(json.dumps(report, indent=2))
        for key in ("stages", "streams"):
            with (directory / f"{key}.csv").open("w", newline="") as handle:
                rows = report[key]
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else ["no_published_records"])
                writer.writeheader()
                writer.writerows(rows)
        for kind, filename in [("process-svg", "process.svg"), ("process-png", "process.png")]:
            artifact = current_diagram(result, kind)
            if artifact:
                shutil.copy2(artifact, directory / filename)
        gas_checks = []
        final_streams = result["final_design"].get("streams", [])
        limiting = next((s for s in final_streams if s.get("phase") != "gas" and s.get("molar_equiv") == 1 and s.get("concentration_M")), None)
        for gas in (s for s in report["streams"] if s["phase"] == "gas"):
            if limiting and gas["reagent_fraction"] is not None and gas["flow_mL_min"] is not None:
                molar_feed = limiting["concentration_M"] * limiting["flow_rate_mL_min"]
                calculated = gas["flow_mL_min"] * gas["reagent_fraction"] / (22.414 * molar_feed)
                gas_checks.append({"stream": gas["label"], "independent_equiv": calculated, "reported_equiv": gas["equiv"],
                                   "passed": math.isclose(calculated, gas["equiv"], rel_tol=0.002),
                                   "formula": "Qg_STP * reagent_fraction / (22.414 * C_limiting * Q_limiting)"})
        entry = {"run_id": source.parent.name, "source_sha256": hashlib.sha256(data).hexdigest(),
                 "status": result["final_design"]["status"], "stage_closure_issues": report["issues"],
                 "stage_count": len(report["stages"]), "gas_checks": gas_checks,
                 "final_engineering_complete": result.get("final_stage_engineering", {}).get("complete"),
                 "has_engineering_history": bool(result.get("engineering_history")),
                 "has_new_llm_generation": not bool(result.get("replay_provenance")),
                 "council_record_count": sum(len(r) for r in result.get("deliberation_log", {}).get("rounds", [])),
                 "question_count": len(report["responses"])}
        topology = ((result["final_design"].get("process_graph") or {}).get("topology") or {})
        reactors = [o for o in topology.get("unit_operations", []) if "reactor" in o.get("op_type", "")]
        annotations = {r["stage_number"]: r for r in result.get("final_stage_engineering", {}).get("stages", [])}
        cross_checks = []
        for stage in report["stages"]:
            matches = [o for o in reactors if stage["equipment_id"] and stage["equipment_id"] in
                       [o.get("parameters", {}).get(k) for k in ("inventory_item_id", "inventory_equipment_id")]]
            if len(matches) != 1:
                cross_checks.append({"stage": stage["number"], "passed": False, "reason": "Cannot identify a unique reactor in the final graph"})
                continue
            p = matches[0].get("parameters") or {}
            c = (annotations.get(stage["number"]) or {}).get("calculations") or {}
            for label, reported, diagram, engineering in [
                ("volume_mL", stage["volume_mL"], p.get("volume_mL"), c.get("reactor_volume_mL")),
                ("temperature_C", stage["temperature_C"], p.get("temperature_C"), c.get("temperature_C")),
                ("time_inlet_min", stage["residence_time_min"], p.get("residence_time_inlet_min"), c.get("residence_time_min")),
            ]:
                passed = all(v is not None and math.isclose(reported, v, rel_tol=0.002, abs_tol=0.001) for v in (diagram, engineering))
                cross_checks.append({"stage": stage["number"], "quantity": label, "report": reported,
                                     "diagram": diagram, "final_engineering": engineering, "passed": passed})
        entry["cross_view_checks"] = cross_checks
        results.append(entry)
        assert source.read_bytes() == data
        assert report["status"] == "executable" and not report["issues"]
        assert all(g["passed"] for g in gas_checks)
        assert entry["final_engineering_complete"]
        assert all(check["passed"] for check in cross_checks), cross_checks
    (args.output / "verification.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
