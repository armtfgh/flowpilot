"""Capture real UI/API interactions; no design/model calls or edits to saved KHU data."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import requests
from playwright.sync_api import sync_playwright, expect

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "manuscript/Submission/inventory_gui_revision_20261001"
BASE = "http://127.0.0.1:8510"
RUN = "20260907_175308_three_protocol_scientific"


def main():
    shots = OUT / "screenshots"
    evidence = OUT / "evidence"
    shots.mkdir(parents=True, exist_ok=True)
    evidence.mkdir(exist_ok=True)
    archive = requests.get(f"{BASE}/api/runs/{RUN}").json()
    intake = archive["intake_package"]
    profile = requests.get(BASE + "/api/inventory/profiles/khu_laboratory_inventory").json()
    (evidence / "source_khu_v4.json").write_text(json.dumps(profile, indent=2))
    manifest = {"captured_utc": datetime.now(timezone.utc).isoformat(), "archive_run": RUN,
                "viewport": {"width": 1440, "height": 1200}, "device_scale_factor": 2,
                "runtime": requests.get(BASE + "/api/runtime").json(),
                "mode": "Live GUI/API, deterministic intake and archived result; no new generation or laboratory experiment",
                "api_calls": [], "console_errors": [], "screenshots": []}
    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True)
        context = browser.new_context(viewport=manifest["viewport"], device_scale_factor=2)
        context.tracing.start(screenshots=True, snapshots=True, sources=True)
        page = context.new_page()
        page.on("dialog", lambda d: d.accept())
        page.on("pageerror", lambda e: manifest["console_errors"].append(str(e)))
        page.on("request", lambda r: manifest["api_calls"].append({"method": r.method, "url": r.url}) if "/api/" in r.url else None)

        def save_manifest():
            (OUT / "capture_manifest.json").write_text(json.dumps(manifest, indent=2))

        def shot(name, selector=None, end=None, height=None):
            target = page.locator(selector) if selector else None
            path = shots / (name + ".png")
            if target:
                target.first.scroll_into_view_if_needed()
                if height or end:
                    target.first.evaluate("e => window.scrollTo(0, e.getBoundingClientRect().top + window.scrollY - 100)")
                page.wait_for_timeout(150)
                if height or end:
                    box = target.first.bounding_box()
                    if height:
                        box["height"] = min(box["height"], height)
                    if end:
                        last = page.locator(end).first.bounding_box()
                        box["height"] = last["y"] + last["height"] - box["y"]
                    page.screenshot(path=str(path), clip=box)
                else:
                    target.first.screenshot(path=str(path))
            else:
                page.screenshot(path=str(path), full_page=True)
            manifest["screenshots"].append({"name": name, "selector": selector, "crop_height_css": height,
                                            "url": page.url, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
            save_manifest()
            print(name, flush=True)

        page.goto(BASE)
        page.get_by_label("Initial batch protocol").fill(intake["raw_protocol"])
        page.locator(".intakePanel .toggle input").uncheck(force=True)
        page.get_by_label("Inventory profile", exact=True).select_option("khu_laboratory_inventory")
        page.locator(".protocolInput").evaluate("e => e.scrollTop = 0")
        shot("01_protocol_full")
        shot("01_protocol", '.intakePanel .field:has(textarea)')
        with page.expect_response(lambda r: "/api/intake/analyze" in r.url) as response:
            page.get_by_role("button", name="Analyze intake", exact=True).click()
        data = response.value.json()
        (evidence / "intake_initial.json").write_text(json.dumps(data, indent=2))
        known = {a["question_id"]: a for a in intake.get("answers", [])}
        for item in page.locator(".question").all():
            qid = item.locator("code").inner_text()
            answer = known.get(qid)
            if answer and answer.get("status") == "answered":
                value = answer.get("answer", answer.get("user_answer", ""))
                item.locator("textarea").fill(value if isinstance(value, str) else json.dumps(value))
            elif item.locator('input[type="checkbox"]').count():
                item.locator('input[type="checkbox"]').check(force=True)
        page.locator(".questionScroll").evaluate("e => e.scrollTop = 0")
        page.locator(".questionsPanel textarea").evaluate_all("es => es.forEach(e => e.scrollTop = 0)")
        shot("02_questions_full")
        shot("02_questions", ".questionsPanel", end=".question:first-child")
        with page.expect_response(lambda r: "/api/intake/analyze" in r.url) as response:
            page.get_by_role("button", name="Save answers", exact=True).click()
        final = response.value.json()
        assert final["package"]["ready_for_design"]
        (evidence / "intake_answered.json").write_text(json.dumps(final, indent=2))
        shot("03_ready_full")
        shot("03_ready", ".questionsPanel .successState")
        repetitions = [requests.post(BASE + "/api/intake/analyze", json={"raw_protocol": intake["raw_protocol"], "inventory_profile": profile, "use_llm": False}).json() for _ in range(3)]
        signatures = [([q["question_id"] for q in x["pending_questions"]], x["package"]["question_set_hash"]) for x in repetitions]
        assert all(x == signatures[0] for x in signatures)
        (evidence / "question_reproducibility.json").write_text(json.dumps(signatures, indent=2))

        page.get_by_role("button", name="Inventory", exact=True).click()
        page.set_viewport_size({"width": 1040, "height": 1200})
        page.get_by_label("Import inventory JSON").set_input_files(str(evidence / "source_khu_v4.json"))
        expect(page.get_by_role("button", name="Export JSON", exact=True)).to_be_enabled()
        shot("04_inventory_full")
        shot("04_inventory_categories", ".equipmentLayout", height=198)
        pump = profile["lab_inventory"]["pumps"][0]
        page.get_by_role("button", name=f"Edit {pump['name']}", exact=True).click()
        shot("05_equipment_dialog_full", ".equipmentDialog")
        shot("05_equipment_fields", ".equipmentFields", height=284)
        page.get_by_role("button", name="Apply equipment", exact=True).click()
        expect(page.get_by_role("button", name="Export JSON", exact=True)).to_be_disabled()
        page.get_by_role("button", name="Constraints", exact=True).click()
        shot("06_constraints_full")
        shot("06_constraints", ".inventoryConstraints .equipmentFields")
        page.get_by_role("button", name="Validate profile", exact=True).click()
        expect(page.get_by_role("button", name="Export JSON", exact=True)).to_be_enabled()
        shot("07_validation", ".inventoryValidation")
        with page.expect_download() as download:
            page.get_by_role("button", name="Export JSON", exact=True).click()
        download.value.save_as(str(evidence / "exported_khu_v4.json"))
        exported = json.loads((evidence / "exported_khu_v4.json").read_text())
        assert exported["lab_inventory"] == profile["lab_inventory"]
        assert exported["equipment_capabilities"] == profile["equipment_capabilities"]
        assert exported["operating_constraints"] == profile["operating_constraints"]
        assert exported["provenance"] == profile["provenance"]
        page.get_by_label("Import inventory JSON").set_input_files(str(evidence / "exported_khu_v4.json"))
        expect(page.get_by_role("button", name="Export JSON", exact=True)).to_be_enabled()
        page.get_by_role("button", name="Use in design", exact=True).click()
        shot("08_bound_inventory_full")
        manifest["inventory_roundtrip_passed"] = True

        page.set_viewport_size({"width": 1440, "height": 1200})
        page.goto(BASE + "/?run=" + RUN)
        page.get_by_role("button", name="Process summary", exact=True).wait_for(timeout=30000)
        shot("09_overview_full")
        page.get_by_role("button", name="Process summary", exact=True).click()
        shot("10_summary_full")
        shot("10_stage_table", ".processTable")
        shot("10_feed_table", ".streamTable")
        page.get_by_role("button", name="Process", exact=True).click()
        image = page.get_by_alt_text("FlowPilot process topology", exact=True)
        expect(image).to_be_visible()
        page.wait_for_function("document.querySelector('img[alt=\"FlowPilot process topology\"]').complete")
        shot("11_topology_full")
        shot("11_topology", ".resultBody")
        page.get_by_role("button", name="Council", exact=True).click()
        page.locator(".agentRecord summary").filter(has_text="DrChemistry").first.click()
        shot("12_council_full")
        shot("12_council", ".agentRecord[open]", height=290)
        # Include the actual source-run identifier without displaying unrelated runs.
        shot("13_result_header", ".resultBanner")
        manifest["intake_ready"] = final["package"]["ready_for_design"]
        assert not manifest["console_errors"]
        assert not any(r["method"] == "POST" and r["url"].endswith("/api/design/jobs") for r in manifest["api_calls"])
        save_manifest()
        context.tracing.stop(path=str(OUT / "browser_trace.zip"))
        browser.close()


if __name__ == "__main__":
    main()
