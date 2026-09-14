"""Fresh requested designs, with immutable inputs and all attempts retained."""
import argparse
import csv
import hashlib
import json
import logging
from pathlib import Path
import shutil
import sys
import time
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

DPDTC = """A 4.0 mL vial equipped with a PTFE stir bar was charged with 3-methyl-4-nitrobenzoic acid (1 equiv, 0.5 mmol), N,N-dimethylpyridin-4-amine (DMAP) (6.1 mg, 0.1 equiv, 0.05 mmol), and DPDTC (130. mg, 1.05 equiv, 0.525 mmol) under ambient atmosphere without argon or nitrogen purging. 2-MeTHF (1.0 mL, 0.5 M) was added, and the vial was capped and sealed with Teflon tape. The reaction vial was placed in a pre-heated oil bath maintained at 95 deg C and stirred at 500-600 rpm for 30 min. The vial was removed from the oil bath and allowed to cool at room temperature for 5-10 min. Once cooled, the cap was removed and benzylamine (1.05 equiv, 0.525 mmol) and 2-MeTHF (0.25 mL, 0.4 M) were quickly added. The vial was capped, sealed with Teflon tape, and returned to the oil bath at 95 deg C, and stirred at 500-600 rpm for another 30 min before workup."""
CHEM = "This is a two-stage DPDTC-mediated amide formation through a thioester intermediate. In Stage 1, 3-methyl-4-nitrobenzoic acid is converted to the corresponding 2-pyridyl thioester using DPDTC and DMAP. The intermediate is not isolated. In Stage 2, benzylamine reacts with the Stage 1 stream to form N-benzyl-3-methyl-4-nitrobenzamide. The amide C(O)-N bond is formed, and the thioester C(O)-S bond is cleaved during Stage 2."
GIESE = """Step 1: Giese reaction
A re-sealable pressure tube (13 x 100 mm) equipped with a magnetic stir bar was charged with (((4-methoxyphenyl)thio)methyl)trimethylsilane (1, 0.2 mmol, 1.0 equiv), acrylonitrile (2, 0.4 mmol, 2.0 equiv), and [Ir(dF(CF3)ppy)2(dtbpy)]PF6 (0.001 mmol, 0.5 mol %) under an argon atmosphere. A degassed solvent mixture (2.0 mL, 0.1 M with respect to 1) of EtOH:pH 9 buffer (5:1, v/v) was added, and the resulting light greenish-yellow mixture, under vigorous magnetic stirring, was positioned 3 cm from a pair of 5 W blue LEDs (lambda max = 452 nm) and irradiated at room temperature (25 deg C, maintained with a cooling fan to counteract heating from the LEDs) for 4 h.

Step 2: Oxidation reaction
After 4 h, the cap of the tube was removed to expose the reaction mixture to ambient air, providing atmospheric oxygen as the terminal oxidant, and stirring was continued under otherwise identical irradiation conditions for an additional 6 h."""

CASES = [
    {"id": "01_dpdtc_yield", "protocol": DPDTC, "answers": {
        "Q-OBJ-001": "Maximize the final amide yield across the complete two-stage sequence. For the initial design, sufficient reaction time should be provided in both stages to achieve high conversion and stable operation.",
        "Q-CHEM-001": CHEM,
        "Q-HYP-001": [
            "High Stage 1 conversion is important because incomplete thioester formation will directly limit the amount of intermediate available for amidation.",
            "Stage 2 should provide sufficient reaction time for conversion of the thioester intermediate to the amide product.",
            "The overall design should be judged by the final amide yield rather than by the performance of either stage alone.",
            "The need for cooling before benzylamine addition should be evaluated rather than assumed."]}},
    {"id": "02_dpdtc_time", "protocol": DPDTC, "answers": {
        "Q-OBJ-001": "Identify a practical two-stage flow process that reduces the overall processing time while maintaining high final amide yield and stable continuous operation. FlowPilot should explore the residence time and interstage configuration without assuming that the batch timing must be retained.",
        "Q-CHEM-001": CHEM,
        "Q-HYP-001": [
            "Continuous heating and mixing in flow may allow the overall processing time to be shortened relative to batch, but the required residence time should be determined from the final process performance.",
            "Any change in Stage 1 conversion will affect the composition of the stream entering Stage 2 and therefore the final amide yield.",
            "The batch cooling step may not be necessary in continuous flow. Direct transfer from Stage 1 to benzylamine addition should be evaluated if chemically and operationally acceptable.",
            "If cooling is required, a short controlled cooling section may be considered.",
            "Temperature and pressure should be selected to maintain a stable liquid-phase process at the reaction temperature."]}},
    {"id": "03_giese_oxidation", "protocol": GIESE, "answers": {
        "Q-OBJ-001": "Identify a practical two-stage flow design that gives the best overall balance among conversion, residence time, oxygen delivery, and process stability. Different operating conditions may be used for the two stages, but the performance of the complete connected process should determine the final design.",
        "Q-HYP-001": [
            "Stage 1 performance may depend on photon delivery and residence time.",
            "Stage 2 performance may depend on oxygen availability and gas-liquid transport.",
            "Excess oxygen may be useful, but its benefit is expected to depend on pressure, gas utilization, and residence time.",
            "Pressure may increase oxygen dissolution while also changing the gas-liquid flow pattern.",
            "FlowPilot should determine the most suitable combination of Stage 1 conversion, oxygen feed, pressure, and Stage 2 residence time for the overall process."],
        "Q-GAS-002": "No fixed value. Please determine and justify an initial oxygen feed based on the reaction requirements and the available operating range.",
        "Q-GAS-003": "Oxygen should first be introduced as the Stage 1 effluent enters the oxidation stage."}},
]


def save(path, value):
    path.write_text(json.dumps(value, indent=2, default=str), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--case", choices=[c["id"] for c in CASES])
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    sources = [Path(__file__), Path("flora_translate/intake_agent.py"),
               Path("flora_translate/scientific_objective.py"), Path("flora_translate/scientific_evidence.py"),
               Path("flora_translate/engine/council_v4/scientific.py"), Path("flora_translate/main.py")]
    source_hashes = {}
    for source in sources:
        destination = args.output / "source_snapshot" / source.name
        destination.parent.mkdir(exist_ok=True)
        shutil.copy2(source, destination)
        source_hashes[str(source)] = hashlib.sha256(source.read_bytes()).hexdigest()
    save(args.output / "source_hashes.json", source_hashes)
    logging.basicConfig(level=logging.INFO, handlers=[logging.FileHandler(args.output / "campaign.log"), logging.StreamHandler()])
    from flora_translate.intake_agent import IntakeAgent, intake_context_block
    from flora_translate.inventory_profiles import load_inventory_profile
    from flora_translate.inventory_resolution import bind_inventory
    from flora_translate.main import translate
    from flora_translate.gui_autosave import autosave_gui_result
    from flora_translate.scientific_objective import resolve_objective
    from flora_translate.engine.llm_agents import set_llm_observer, set_llm_runtime_overrides
    set_llm_runtime_overrides(capture_content=True)

    inventory_path = Path("flora_translate/data/inventory_profiles/khu_laboratory_inventory/v004/inventory_profile.json")
    original_inventory = inventory_path.read_bytes()
    profile = load_inventory_profile(inventory_path)
    assert profile.version == 4
    save(args.output / "inventory_profile_v4.json", profile.model_dump())
    runtime = {"design_policy": "scientific_v2", "candidate_budget": 12,
               "upstream_model": "claude-opus-4-6", "downstream_model": "claude-sonnet-4-6"}
    save(args.output / "manifest.json", {"runtime": runtime, "inventory_source": str(inventory_path),
        "inventory_sha256": hashlib.sha256(original_inventory).hexdigest(),
        "input_transcription": "Whitespace and typography normalized to ASCII; chemical values and supplied answer meanings retained.",
        "history": "No historical measurements supplied for these runs; explicitly unavailable with campaign provenance.",
        "rerun_policy": "Preserve every attempt. Rerun only after diagnosing a technical failure, not to select favorable predictions."})
    summaries = []
    for case in CASES:
        if args.case and case["id"] != args.case:
            continue
        folder = args.output / case["id"]
        folder.mkdir()
        def observe(event):
            with (folder / "llm_calls.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(event, default=str) + "\n")
        set_llm_observer(observe)
        save(folder / "provided_input.json", case)
        (folder / "protocol.txt").write_text(case["protocol"])
        handler = logging.FileHandler(folder / "run.log")
        logging.getLogger().addHandler(handler)
        before = {p for base in ("outputs/scientific_pipeline", "outputs/scientific_council") for p in Path(base).glob("*")}
        started = time.monotonic()
        row = {"case": case["id"], "status": "running"}
        try:
            answers = [{"question_id": k, "answer": v, "source": "user_campaign_input"} for k, v in case["answers"].items()]
            answers.append({"question_id": "Q-HIST-001", "status": "unavailable", "source": "campaign_no_history_supplied"})
            if case["id"].startswith("03"):
                answers.append({"question_id": "Q-CHEM-001", "answer": "Sequential Giese addition then aerobic oxidation. User follow-up: 'sulfoxide?' (tentative, not a confirmed product identity or oxygen stoichiometry).", "source": "provisional_followup_not_verified"})
            package = IntakeAgent().analyze(case["protocol"], answers=answers, use_llm=True)
            package = bind_inventory(package, profile)
            save(folder / "intake_package.json", package.model_dump())
            save(folder / "request.json", {"intake_package": package.model_dump(), "runtime_options": runtime})
            (folder / "prompt.txt").write_text(intake_context_block(package))
            row.update(priority=resolve_objective(package.objective, package)["priority"], ready=package.ready_for_design, missing=package.missing_question_ids)
            save(folder / "summary.json", row)
            if not package.ready_for_design:
                raise ValueError("Intake incomplete: " + str(package.missing_question_ids))
            result = translate(case["protocol"], intake_package=package.model_dump(), runtime_options=runtime)
            save(folder / "result.json", result)
            archive = Path(autosave_gui_result(result, intake_package=package.model_dump(), source="three_protocol_scientific", user_input=case["protocol"]))
            shutil.copytree(archive, folder / "gui_export")
            assessment = result.get("scientific_assessment", {})
            final = result.get("final_design", {})
            row.update(status=final.get("status", "unknown"), gui_archive=str(archive),
                       selected=assessment.get("selected_candidate_id"), stages=final.get("stages"),
                       candidates=assessment.get("candidate_count"), selected_preserved=assessment.get("selected_design_preserved"),
                       council_calls=len(assessment.get("calls", [])),
                       issues=final.get("consistency", {}).get("issues", []))
        except Exception as exc:
            row.update(status="failed", error=str(exc))
            (folder / "failure.txt").write_text(traceback.format_exc())
            logging.exception("Campaign case failed; evidence preserved")
        finally:
            after = {p for base in ("outputs/scientific_pipeline", "outputs/scientific_council") for p in Path(base).glob("*")}
            for path in sorted(after - before):
                shutil.copytree(path, folder / "model_artifacts" / path.parent.name / path.name)
            row["elapsed_seconds"] = time.monotonic() - started
            save(folder / "summary.json", row)
            summaries.append(row)
            save(args.output / "summary.json", summaries)
            logging.getLogger().removeHandler(handler)
            handler.close()
            print("CASE_COMPLETE", case["id"], row["status"], flush=True)
    assert inventory_path.read_bytes() == original_inventory
    return 0 if all(r["status"] == "executable" for r in summaries) else 1


if __name__ == "__main__":
    raise SystemExit(main())
