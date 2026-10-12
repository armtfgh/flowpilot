"""Capture a real intake using an unsaved, neutrally named inventory copy."""
from copy import deepcopy
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import requests
from playwright.sync_api import sync_playwright, expect

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'manuscript/october2_final_review/gui'
BASE='http://127.0.0.1:8513'

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    src=ROOT/'manuscript/Submission/inventory_gui_revision_20261001/evidence/source_khu_v4.json'
    profile=json.loads(src.read_text()); original=deepcopy(profile)
    profile['profile_id']='study_inventory_demonstration'
    profile['name']='Study laboratory inventory'
    profile['laboratory']='Study laboratory'
    for items in profile['lab_inventory'].values():
        if isinstance(items,list):
            for item in items:
                if isinstance(item,dict) and isinstance(item.get('name'),str):
                    item['name']=item['name'].replace('KHU ','')
    (OUT/'demonstration_inventory.json').write_text(json.dumps(profile,indent=2))
    known=requests.get(BASE+'/api/runs/20260907_175308_three_protocol_scientific',timeout=30).json()['intake_package']
    manifest={'captured_utc':datetime.now(timezone.utc).isoformat(), 'modules':[], 'requests':[], 'errors':[],
              'mode':'Live UI; unsaved copy of the archived v4 inventory, display names generalized; equipment IDs and numeric capabilities unchanged; no LLM design call.',
              'source_profile_sha256':hashlib.sha256(src.read_bytes()).hexdigest()}
    with sync_playwright() as pw:
        browser=pw.chromium.launch(headless=True)
        page=browser.new_page(viewport={'width':1360,'height':1040},device_scale_factor=2)
        page.on('dialog',lambda d:d.accept())
        page.on('pageerror',lambda e:manifest['errors'].append(str(e)))
        page.on('request',lambda r:manifest['requests'].append({'method':r.method,'url':r.url}) if '/api/' in r.url else None)
        page.goto(BASE)
        page.get_by_role('button',name='Inventory',exact=True).click()
        page.get_by_label('Import inventory JSON').set_input_files(str(OUT/'demonstration_inventory.json'))
        expect(page.get_by_role('button',name='Use in design',exact=True)).to_be_enabled(timeout=30000)
        page.get_by_role('button',name='Use in design',exact=True).click()
        page.get_by_label('Initial batch protocol').fill(known['raw_protocol'])
        page.locator('.intakePanel .toggle input').uncheck(force=True)
        with page.expect_response(lambda r:'/api/intake/analyze' in r.url) as result:
            page.get_by_role('button',name='Analyze intake',exact=True).click()
        (OUT/'intake_response.json').write_text(json.dumps(result.value.json(),indent=2))
        page.locator('.question').first.locator('textarea').fill('Maximize the final amide yield across the connected activation and amidation stages using the available inventory.')
        page.locator('.protocolInput').evaluate('e => e.scrollTop = 0')
        page.evaluate('window.scrollTo(0, 0)')
        page.wait_for_timeout(250)
        height=page.evaluate('document.documentElement.scrollHeight')
        page.screenshot(path=str(OUT/'front_page.png'),full_page=True)
        for label,selector in [('Navigation','.nav'),('Protocol','.intakePanel'),('Questions','.questionsPanel'),
                               ('Model routing','.modelSelectors'),('Inventory','.launchPanel select[aria-label="Inventory profile"]'),('Run gate','.runButton')]:
            manifest['modules'].append({'label':label,'box':page.locator(selector).bounding_box()})
        manifest['viewport_css']={'width':1360,'height':height}
        assert not manifest['errors'],manifest['errors']
        assert not any(r['method']=='POST' and any(s in r['url'] for s in ['/api/design/jobs','/api/inventory/save']) for r in manifest['requests'])
        assert 'KHU' not in page.locator('.designWorkspace').inner_text() if page.locator('.designWorkspace').count() else True
        browser.close()
    (OUT/'capture_manifest.json').write_text(json.dumps(manifest,indent=2))
    print(json.dumps({'height':height,'errors':manifest['errors'],'calls':len(manifest['requests'])}))

if __name__=='__main__':main()
