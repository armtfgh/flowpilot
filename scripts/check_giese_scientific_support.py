"""Saved-upstream numerical regression; fixture gas permission is not a user answer."""
import argparse
import json
import logging
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    logging.basicConfig(filename=args.output / "run.log", level=logging.INFO)
    from flora_translate.schemas import BatchRecord, ChemistryPlan, FlowProposal, StreamAssignment, DesignInputPackage, IntakeAnswer
    from flora_translate.chemistry_contract import reconcile_chemistry_plan
    from flora_translate.intake_agent import IntakeAgent, apply_intake_requirements_to_chemistry_plan
    from flora_translate.inventory_profiles import load_inventory_profile
    from flora_translate.engine.council_v4.scientific import build_screen_pool
    source = Path("outputs/flowpilot2/20260907_requested_three_protocols/attempt_01/03_giese_oxidation")
    original = json.loads(next((source / "model_artifacts").rglob("upstream.json")).read_text())
    package = DesignInputPackage.model_validate_json((source / "intake_package.json").read_text())
    package = IntakeAgent().analyze(existing_package=package, use_llm=False, answers=[
        IntakeAnswer(question_id="Q-GAS-001", answer="Pure O2, 100%", source="synthetic_unit_fixture_not_user_confirmation")])
    batch = BatchRecord.model_validate(original["batch_record"])
    plan, audit = reconcile_chemistry_plan(batch, ChemistryPlan.model_validate(original["chemistry_plan"]), scientific=True)
    plan, decisions = apply_intake_requirements_to_chemistry_plan(plan, package)
    plan.scientific_context["operating_limits"] = package.operating_limits
    inv = load_inventory_profile("flora_translate/data/inventory_profiles/khu_laboratory_inventory/v004/inventory_profile.json").lab_inventory
    streams = [StreamAssignment(stream_label=s.stream_label, contents=s.reagents, phase=s.phase, molar_equiv=s.molar_equiv,
                  concentration_M=s.concentration_M, introduction_stage=s.introduction_stage or 1,
                  gas_reagent_mole_fraction=s.gas_reagent_mole_fraction,
                  solvent=batch.solvent if s.phase == "liquid" else None,
                  flow_rate_mL_min=0.02 if s.phase == "liquid" else None) for s in plan.stream_logic if s.accepted_requirement]
    proposal = FlowProposal(streams=streams, flow_rate_mL_min=0.02, concentration_M=0.1, reactor_volume_mL=20,
        residence_time_min=600, BPR_bar=2, temperature_C=40, tubing_ID_mm=1.016, tubing_material="PFA", wavelength_nm=450)
    (args.output / "fixture.json").write_text(json.dumps({"synthetic_gas_permission": True, "batch": batch.model_dump(), "plan": plan.model_dump(), "proposal": proposal.model_dump()}, indent=2))
    try:
        pool, rejected = build_screen_pool(proposal, batch, plan, inv)
    except ValueError as error:
        (args.output / "rejections.json").write_text(json.dumps(getattr(error, "rejections", []), indent=2))
        raise
    (args.output / "pool.json").write_text(json.dumps({"candidates": pool, "rejected": rejected}, indent=2))
    print(json.dumps({"candidates":len(pool), "stages":pool[0]["proposal"]["stage_parameters"]}, indent=2))


if __name__ == "__main__":
    main()
