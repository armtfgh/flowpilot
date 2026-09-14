"""Freeze source-grounded operational examples and selected judge findings."""

import hashlib
import json
import shutil
from pathlib import Path

import pandas as pd
import requests
from playwright.sync_api import sync_playwright

from capture_esi_gui_20260909 import BASE, OUT, ROOT, RUN


def main():
    evidence = OUT / "evidence"
    rpath = ROOT / "outputs/gui_runs" / RUN
    result = json.loads((rpath / "result.json").read_text())
    for name in ["result.json", "intake_package.json", "final_design.json", "summary.json", "topology.json", "instrument_manifest.json", "render_manifest.json", "input.txt"]:
        shutil.copy2(rpath / name, evidence / ("gui_archive_" + name))
    profile = json.loads((evidence / "khu_inventory_v4.json").read_text())
    payload = {"raw_protocol": result["intake_package"]["raw_protocol"], "use_llm": False, "inventory_profile": profile}
    checks = []
    for i in range(3):
        response = requests.post(BASE + "/api/intake/analyze", json=payload, timeout=60)
        response.raise_for_status()
        data = response.json()
        (evidence / f"intake_recheck_{i + 1}.json").write_text(json.dumps(data, indent=2))
        checks.append({"question_ids": [q["question_id"] for q in data["pending_questions"]],
                       "question_set_hash": data["package"]["question_set_hash"]})
    assert checks[0] == checks[1] == checks[2]
    (evidence / "intake_reproducibility_check.json").write_text(json.dumps({"repeats": 3, "identical": True, "mode": "LLM disabled", "observations": checks}, indent=2))

    agents = {}
    for m in result["deliberation_log"]["rounds"][0]:
        if m.get("chain_of_thought"):
            agents[m["agent"]] = json.loads(m["chain_of_thought"])
    assessment = result["scientific_assessment"]
    excerpts = []
    for name in ["DrChemistry", "DrKinetics", "DrSafety"]:
        row = next(v for v in agents[name]["reviews"] if v["candidate_id"] == 6)
        excerpts.append({"speaker": name, "subject": "Candidate 6", "recommendation": row["recommendation"],
                         "exact_text": row["justification"], "source": f"deliberation_log.rounds[0].{name}.reviews[candidate_id=6].justification"})
    excerpts.append({"speaker": "Skeptic", "subject": "Conflicting domain preferences", "recommendation": "No hard veto on this disagreement",
                     "exact_text": assessment["skeptic"]["shared_assessment"]["cross_domain_conflicts"],
                     "source": "scientific_assessment.skeptic.shared_assessment.cross_domain_conflicts"})
    excerpts.append({"speaker": "Chief", "subject": "Final candidate selection", "recommendation": "Candidate " + str(assessment["chief"]["candidate_id"]),
                     "exact_text": assessment["chief"]["justification"], "source": "scientific_assessment.chief.justification"})
    (evidence / "council_exact_excerpts.json").write_text(json.dumps(excerpts, indent=2, ensure_ascii=False))
    (evidence / "council_full_record.json").write_text(json.dumps({"deliberation_log": result["deliberation_log"], "scientific_assessment": assessment}, indent=2, ensure_ascii=False))

    details = pd.read_csv(ROOT / "deliverables/flowpilot_esi_revision_20260902/source_data/S9_error_details.csv")
    ids = ["N2-D1FC65D8", "N2-B6923FF7", "N2-110E4388", "N2-E85D424E"]
    chosen = details[details.candidate_id.isin(ids)]
    chosen.to_csv(evidence / "selected_judge_findings.csv", index=False)
    scores = pd.read_csv(ROOT / "deliverables/manuscript_benchmark_visualizations_20260825/core_tables/model_all_candidate_consensus_scores.csv")
    sources = []
    for row in scores[scores.candidate_id.isin(ids)].itertuples():
        run = Path(row.run_directory)
        dest = evidence / "benchmark_examples" / row.candidate_id
        dest.mkdir(parents=True, exist_ok=True)
        for name in ["result.json", "input_inventory.json", "input_public.json", "raw_response.json"]:
            if (run / name).exists():
                shutil.copy2(run / name, dest / name)
        campaign = run.parents[3]
        for family in ["openai", "qwen", "claude"]:
            judge = campaign / "judgments" / family / row.candidate_id / "parsed_response.json"
            if judge.exists():
                shutil.copy2(judge, dest / (family + "_judge.json"))
        sources.append({"candidate_id": row.candidate_id, "run_directory": str(run.relative_to(ROOT)), "architecture": row.architecture,
                        "case": row.case, "model": row.model, "repeat": row.repeat_id})
    (evidence / "benchmark_example_sources.json").write_text(json.dumps(sources, indent=2))
    q_o2 = 3.3158 * 0.21 / 22.414
    (evidence / "gas_flag_arithmetic_check.json").write_text(json.dumps({
        "candidate_id": "N2-E85D424E", "air_flow_mL_min_STP": 3.3158, "oxygen_fraction": 0.21,
        "molar_volume_mL_per_mmol": 22.414, "oxygen_mmol_min": q_o2,
        "substrate_mmol_min": 0.031068, "oxygen_equiv": q_o2 / 0.031068,
        "interpretation": "The judge's claimed factor-of-1000 error is incorrect: 1 sccm = 1 mL/min at the same standard conditions. No archived score was changed."
    }, indent=2))
    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1000, "height": 1400}, device_scale_factor=2)
        page.goto(BASE + "/?run=" + RUN)
        page.get_by_role("button", name="Council", exact=True).wait_for(timeout=30000)
        page.get_by_role("button", name="Council", exact=True).click()
        page.locator(".resultBody").screenshot(path=str(OUT / "screenshots/12_council_narrow_full.png"))
        body = page.locator(".resultBody")
        body.evaluate("el => window.scrollTo(0, el.getBoundingClientRect().top + window.scrollY - 120)")
        box = body.bounding_box()
        last_round = page.locator(".resultBody .reportStack > details[open]").last.bounding_box()
        box["height"] = last_round["y"] + last_round["height"] + 12 - box["y"]
        page.screenshot(path=str(OUT / "screenshots/12_council_collapsed.png"), clip=box)
        page.set_viewport_size({"width": 1540, "height": 1400})
        page.get_by_role("button", name="Process", exact=True).click()
        page.wait_for_timeout(800)
        page.locator(".resultBanner").screenshot(path=str(OUT / "screenshots/13_disposition_banner.png"))
        page.locator(".warningRow").first.screenshot(path=str(OUT / "screenshots/13_equipment_warning.png"))
        browser.close()
    files = sorted(p for p in evidence.rglob("*") if p.is_file() and p.name != "SHA256SUMS.txt")
    (evidence / "SHA256SUMS.txt").write_text("\n".join(f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.relative_to(evidence)}" for p in files) + "\n")
    print("Saved exact council excerpts, selected judge records, repeatability checks and GUI details.")


if __name__ == "__main__":
    main()
