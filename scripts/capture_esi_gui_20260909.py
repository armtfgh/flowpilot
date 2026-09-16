"""Capture the real web application without mocked responses or new design calls."""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import requests
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "deliverables/esi_gui_additions_20260909"
BASE = "http://127.0.0.1:8513"
RUN = "20260907_175308_three_protocol_scientific"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    screenshots = OUT / "screenshots"
    screenshots.mkdir(exist_ok=True)
    evidence = OUT / "evidence"
    evidence.mkdir(exist_ok=True)
    archive = json.loads((ROOT / "outputs/gui_runs" / RUN / "result.json").read_text())
    intake = archive["intake_package"]
    profile = requests.get(BASE + "/api/inventory/profiles/khu_laboratory_inventory").json()
    (evidence / "khu_inventory_v4.json").write_text(json.dumps(profile, indent=2))
    manifest = {"captured_utc": datetime.now(timezone.utc).isoformat(), "base_url": BASE,
                "runtime": requests.get(BASE + "/api/runtime").json(), "archive_run": RUN,
                "mode": "Real UI and API; deterministic intake; reopened archived design; no new generation",
                "screenshots": [], "api_calls": [], "console_errors": []}
    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True)
        context = browser.new_context(viewport={"width": 1540, "height": 1400}, device_scale_factor=2)
        page = context.new_page()
        page.on("pageerror", lambda error: manifest["console_errors"].append(str(error)))
        page.on("request", lambda req: manifest["api_calls"].append({"method": req.method, "url": req.url}) if "/api/" in req.url else None)

        def shot(name, selector=None, end_selector=None):
            target = page.locator(selector) if selector else page
            if selector:
                target.scroll_into_view_if_needed()
                target.evaluate("el => window.scrollTo(0, el.getBoundingClientRect().top + window.scrollY - 120)")
            page.wait_for_timeout(350)
            path = screenshots / (name + ".png")
            if selector and end_selector:
                box = target.bounding_box()
                end = page.locator(end_selector).bounding_box()
                box["height"] = end["y"] + end["height"] + 14 - box["y"]
                page.screenshot(path=str(path), clip=box)
            elif selector:
                target.screenshot(path=str(path))
            else:
                page.screenshot(path=str(path), full_page=True)
            manifest["screenshots"].append({"name": name, "selector": selector, "url": page.url,
                                            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
            (OUT / "capture_manifest.json").write_text(json.dumps(manifest, indent=2))
            print(name, flush=True)

        page.goto(BASE)
        page.get_by_label("Initial batch protocol").fill(intake["raw_protocol"])
        page.locator(".intakePanel .toggle input").uncheck(force=True)
        page.get_by_label("Inventory profile", exact=True).select_option("khu_laboratory_inventory")
        page.wait_for_timeout(700)
        shot("01_batch_input_full")
        shot("01_batch_input", ".intakePanel", ".intakePanel .inlineControls")
        with page.expect_response(lambda r: "/api/intake/analyze" in r.url) as response:
            page.get_by_role("button", name="Analyze intake", exact=True).click()
        response.value.finished()
        data = response.value.json()
        (evidence / "intake_initial_api.json").write_text(json.dumps(data, indent=2))
        page.locator(".question").first.wait_for()
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
        shot("02_followup_full")
        shot("02_followup_questions", ".questionsPanel", ".questionsPanel > button")
        with page.expect_response(lambda r: "/api/intake/analyze" in r.url) as response:
            page.get_by_role("button", name="Save answers", exact=True).click()
        response.value.finished()
        (evidence / "intake_answered_api.json").write_text(json.dumps(response.value.json(), indent=2))
        shot("03_intake_saved_full")
        shot("03_intake_saved", ".questionsPanel", ".questionsPanel .successState")

        page.get_by_role("button", name="Inventory", exact=True).click()
        page.get_by_label("Profile name", exact=True).fill(profile["name"])
        page.get_by_label("Laboratory", exact=True).fill(profile["laboratory"])
        page.get_by_label("Additional equipment and constraints", exact=True).fill("No inline degasser. Offline pre-degassing and a sealed solvent reservoir are permitted; do not add an inline degasser to the process.")
        page.locator('input[type="file"]').set_input_files(str(evidence / "khu_inventory_v4.json"))
        page.locator(".sourcePanel .toggle input").uncheck(force=True)
        with page.expect_response(lambda r: "/api/inventory/extract" in r.url) as response:
            page.get_by_role("button", name="Extract inventory", exact=True).click()
        response.value.finished()
        (evidence / "inventory_import_api.json").write_text(json.dumps(response.value.json(), indent=2))
        page.get_by_role("button", name="Validate edits", exact=True).wait_for()
        shot("04_inventory_extraction_diagnostic", ".inventoryPreview")
        page.locator(".jsonEditor textarea").fill(json.dumps(profile, indent=2))
        with page.expect_response(lambda r: "/api/inventory/import" in r.url) as response:
            page.get_by_role("button", name="Validate edits", exact=True).click()
        response.value.finished()
        (evidence / "inventory_validated_api.json").write_text(json.dumps(response.value.json(), indent=2))
        shot("04_inventory_full")
        shot("04_inventory_source", ".sourcePanel", ".sourcePanel .inlineControls")
        shot("05_inventory_preview", ".inventoryPreview")
        shot("05_inventory_json", ".jsonEditor")

        page.goto(BASE + "/?run=" + RUN)
        page.get_by_role("button", name="Process summary", exact=True).wait_for(timeout=30000)
        shot("06_result_overview_full")
        page.get_by_role("button", name="Process summary", exact=True).click()
        shot("07_process_summary_full")
        shot("07_process_summary", ".resultBody")
        page.get_by_role("button", name="Process", exact=True).click()
        page.wait_for_timeout(1000)
        shot("08_process_topology_full")
        shot("08_process_topology", ".resultBody")
        page.get_by_role("button", name="Responses", exact=True).click()
        page.locator(".responseRecord summary").filter(has_text="Q-OBJ-001").click()
        shot("09_response_trace_full")
        shot("09_response_trace", ".responseRecord:has-text('Q-OBJ-001')")
        page.get_by_role("button", name="Council", exact=True).click()
        page.locator(".agentRecord summary").filter(has_text="DrChemistry").first.click()
        shot("10_council_full")
        shot("10_council_first", ".agentRecord[open]")
        page.get_by_role("button", name="Saved runs", exact=True).click()
        shot("11_saved_runs_full")
        manifest["intake_final_ready"] = json.loads((evidence / "intake_answered_api.json").read_text()).get("package", {}).get("ready_for_design")
        assert not any(x["method"] == "POST" and x["url"].endswith("/api/design/jobs") for x in manifest["api_calls"])
        browser.close()
    (OUT / "capture_manifest.json").write_text(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
