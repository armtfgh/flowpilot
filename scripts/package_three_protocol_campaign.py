"""Collect selected exports without discarding earlier attempts."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    root = args.root
    export = root / "deliverables"
    export.mkdir(exist_ok=False)
    summary = []
    for case in ("01_dpdtc_yield", "02_dpdtc_time"):
        choices = sorted(root.glob(f"attempt*/{case}/result.json"))
        eligible = [p for p in choices if (p.parent / "independent_checks.json").exists()
                    and json.loads((p.parent / "independent_checks.json").read_text())["all_passed"]]
        if not eligible:
            continue
        source = eligible[-1].parent
        destination = export / case
        shutil.copytree(source, destination)
        result = json.loads((source / "result.json").read_text())
        for extension in ("png", "svg"):
            shutil.copy2(source / "gui_export" / f"process.{extension}", export / f"{case}_topology.{extension}")
        assessment = result["scientific_assessment"]
        proposal = result["proposal"]
        stages = proposal["stage_parameters"]
        streams = proposal["streams"]
        summary.append({"case": case, "source_attempt": source.parent.name,
            "selection_rule": "Latest numerically checked attempt, not selection by a favorable yield or residence time.",
            "candidate": assessment["selected_candidate_id"], "priority": assessment["objective_policy"]["priority"],
            "stages": stages, "streams": streams, "BPR_bar": proposal["BPR_bar"]})
        lines = [f"# {case}", "", "Experimental screening proposal, not demonstrated yield or a laboratory-approved SOP.", "",
                 "## Stage conditions", "", "| Stage | Reactor mL | ID mm | Temperature C | Liquid flow mL/min | Residence min |", "|---|---:|---:|---:|---:|---:|"]
        lines += [f"| {s['stage_number']} | {s['reactor_volume_mL']:g} | {s['d_mm']:g} | {s['temperature_C']:g} | {s['Q_liquid_mL_min']:.6f} | {s['residence_time_min']:.2f} |" for s in stages]
        lines += ["", f"BPR: {proposal['BPR_bar']:g} bar. Exact pressure-controller assignment is retained in the JSON.", "",
                  "## Model selection rationale", "", assessment["chief"]["justification"], "", "## Model limitations", ""]
        lines += ["- " + x for x in assessment["chief"].get("limitations", [])]
        lines += ["", "## Generated preparation and procedure", "", "This is preserved model/software output for review, not an independently approved operating procedure.", ""]
        lines += [f"- {x['step_id']}: {x['instruction']}" for x in result.get("operating_procedure", [])]
        (destination / "design_summary.md").write_text("\n".join(lines) + "\n")
        calls = assessment["calls"]
        for i, call in enumerate(calls, 1):
            target = destination / "council_text"
            target.mkdir(exist_ok=True)
            (target / f"{i:02d}_{call['role']}_prompt.txt").write_text(call["system"] + "\n\n" + json.dumps(call["request"], indent=2))
            (target / f"{i:02d}_{call['role']}_response.txt").write_text(call["raw_response"])
    third = root / "attempt_01" / "03_giese_oxidation"
    shutil.copytree(third, export / "03_giese_oxidation_DIAGNOSTIC_ONLY")
    for extension in ("png", "svg"):
        shutil.copy2(third / "diagnostic_export" / f"requirements.{extension}", export / f"03_giese_oxidation_REQUIREMENTS_ONLY.{extension}")
    (export / "selected_designs.json").write_text(json.dumps(summary, indent=2))
    shutil.copy2(root / "attempt_01" / "inventory_profile_v4.json", export / "inventory_profile_v4.json")
    hashes = {str(p.relative_to(export)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(export.rglob("*")) if p.is_file()}
    (export / "sha256_manifest.json").write_text(json.dumps(hashes, indent=2))
    print(json.dumps([{"case": x["case"], "attempt": x["source_attempt"], "candidate": x["candidate"],
                       "stage_times_min": [s["residence_time_min"] for s in x["stages"]]} for x in summary], indent=2))


if __name__ == "__main__":
    main()
