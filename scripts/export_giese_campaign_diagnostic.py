"""Export the failed gas/photo run without manufacturing an executable design."""
import argparse
import json
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("case_folder", type=Path)
    args = parser.parse_args()
    folder = args.case_folder
    out = folder / "diagnostic_export"
    out.mkdir(exist_ok=False)
    from flora_translate.schemas import BatchRecord, ChemistryPlan, DesignInputPackage
    from flora_translate.chemistry_contract import reconcile_chemistry_plan
    from flora_translate.intake_agent import apply_intake_requirements_to_chemistry_plan
    from flora_translate.topology_preflight import build_requirements_topology
    from flora_translate.diagram_artifacts import render_topology_artifacts

    checkpoint = next((folder / "model_artifacts").rglob("upstream.json"))
    original = json.loads(checkpoint.read_text())
    package = DesignInputPackage.model_validate_json((folder / "intake_package.json").read_text())
    plan, reconciliation = reconcile_chemistry_plan(BatchRecord.model_validate(original["batch_record"]),
        ChemistryPlan.model_validate(original["chemistry_plan"]), scientific=True,
        hard_constraints={"inventory_constraints": package.inventory_constraints, "operating_limits": package.operating_limits})
    plan, decisions = apply_intake_requirements_to_chemistry_plan(plan, package)
    topology = build_requirements_topology(plan)
    artifacts = render_topology_artifacts(topology, title="REQUIREMENTS ONLY - scientific_v2 gas/photo design not supported", base_dir=out / "render")
    for key, filename in (("svg_path", "requirements.svg"), ("png_path", "requirements.png")):
        if artifacts.get(key):
            shutil.copy2(artifacts[key], out / filename)
    result = {"artifact_type": "post-run diagnostic replay, not a fresh numerical design",
        "status": "unsupported_policy", "executable": False, "final_design": None,
        "blocking_reason": json.loads((folder / "summary.json").read_text())["error"],
        "policy": "scientific_v2", "legacy_fallback_used": False,
        "intake_package": package.model_dump(), "chemistry_plan": plan.model_dump(),
        "reconciliation": reconciliation, "corrected_intake_binding_decisions": decisions,
        "requirements_topology": topology.model_dump(),
        "limitations": [
            "No operating flow, pressure, residence time, oxygen equivalent or yield has been selected.",
            "The user suggested 'sulfoxide?' tentatively; final oxidation product identity is not verified.",
            "The batch uses ambient air; replacing it with a pure oxygen feed and its purity is a separate unvalidated design choice.",
            "The diagram expresses operations, not assigned inventory equipment or a laboratory procedure.",
            "Offline solvent degassing is allowed by the inventory; no inline degasser is added.",
            "The original interrupted contract incorrectly labeled 1 equiv as chemist-confirmed; this replay removes that authority. Any retained model default is unvalidated and not a selected feed."]}
    (out / "diagnostic_result.json").write_text(json.dumps(result, indent=2))
    (out / "README.md").write_text("# Giese / oxidation diagnostic\n\n" + "\n".join("- " + item for item in result["limitations"]) + "\n")
    print(out)


if __name__ == "__main__":
    main()
