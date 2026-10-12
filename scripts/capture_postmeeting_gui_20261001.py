"""Capture an actual intake page and annotate its modules without changing the UI."""
import base64
from hashlib import sha256
import json
from pathlib import Path
from datetime import datetime, timezone
from html import escape

import requests
import cairosvg
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'manuscript/october1_postmeeting_work'
BASE = 'http://127.0.0.1:8512'


def main():
    shots = OUT / 'gui'
    shots.mkdir(parents=True, exist_ok=True)
    intake = requests.get(BASE + '/api/runs/20260907_175308_three_protocol_scientific', timeout=30).json()['intake_package']
    manifest = {'captured_utc': datetime.now(timezone.utc).isoformat(),
                'mode': 'Live application; deterministic intake; no design/model call',
                'requests': [], 'errors': [], 'modules': []}
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(viewport={'width': 1360, 'height': 1040}, device_scale_factor=2)
        page.on('pageerror', lambda e: manifest['errors'].append(str(e)))
        page.on('request', lambda r: manifest['requests'].append({'method': r.method, 'url': r.url}) if '/api/' in r.url else None)
        page.goto(BASE)
        page.get_by_label('Initial batch protocol').fill(intake['raw_protocol'])
        page.locator('.intakePanel .toggle input').uncheck(force=True)
        page.get_by_label('Inventory profile', exact=True).select_option('khu_laboratory_inventory')
        page.get_by_label('Design policy', exact=True).select_option('scientific_v2')
        with page.expect_response(lambda r: '/api/intake/analyze' in r.url) as response:
            page.get_by_role('button', name='Analyze intake', exact=True).click()
        data = response.value.json()
        (shots / 'intake_response.json').write_text(json.dumps(data, indent=2))
        q = page.locator('.question').first
        q.locator('textarea').fill('Maximize the final amide yield across the connected activation and amidation stages using the available inventory.')
        page.locator('.protocolInput').evaluate('e => e.scrollTop = 0')
        page.wait_for_timeout(250)
        full_height = page.evaluate('document.documentElement.scrollHeight')
        page.screenshot(path=str(shots / 'front_page.png'), full_page=True)
        for label, selector in [
            ('Navigation', '.nav'), ('Protocol', '.intakePanel'),
            ('Questions', '.questionsPanel'), ('Model routing', '.modelSelectors'),
            ('Inventory', '.launchPanel select[aria-label="Inventory profile"]'),
            ('Run gate', '.runButton'),
        ]:
            manifest['modules'].append({'label': label, 'box': page.locator(selector).bounding_box()})
        manifest['viewport_css'] = {'width': 1360, 'height': full_height}
        assert not manifest['errors']
        assert not any(r['method']=='POST' and r['url'].endswith('/api/design/jobs') for r in manifest['requests'])
        browser.close()
    raw = (shots / 'front_page.png').read_bytes()
    manifest['screenshot_sha256'] = sha256(raw).hexdigest()
    (shots / 'capture_manifest.json').write_text(json.dumps(manifest, indent=2))
    # Vector annotations are outside the unchanged browser pixels.
    width = 1360
    shot_h = manifest['viewport_css']['height']
    top, bottom = 104, 146
    colors = ['#285d78', '#0e766c', '#976c22', '#285d78', '#0e766c', '#976c22']
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" width="{width}" height="{shot_h+top+bottom}" viewBox="0 0 {width} {shot_h+top+bottom}">',
           '<rect width="100%" height="100%" fill="white"/>',
           '<defs><marker id="arrow" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8" fill="#586573"/></marker></defs>',
           f'<image x="0" y="{top}" width="1360" height="{shot_h}" xlink:href="data:image/png;base64,{base64.b64encode(raw).decode()}"/>']
    specs = [
        ('1', 'Workspace navigation', 'Design, inventory and saved runs', 14, 23, 100, 208),
        ('2', 'Reaction protocol', 'Batch chemistry and stated conditions', 465, 23, 382, 288),
        ('3', 'Standardized questions', 'Fixed IDs; answer or mark unavailable', 917, 23, 826, 280),
        ('4', 'Model routing', 'Upstream and council selections', 14, top+shot_h+28, 1185, 330),
        ('5', 'Laboratory inventory', 'Bind the available-equipment profile', 465, top+shot_h+28, 1210, 550),
        ('6', 'Readiness gate', 'Design runs after required inputs close', 917, top+shot_h+28, 1200, 845),
    ]
    # Numbered markers locate modules; keyed explanations stay out of UI content.
    for i, (number, title, note, x, y, tx, ty) in enumerate(specs):
        color = colors[i]
        svg.append(f'<rect x="{x}" y="{y-10}" width="424" height="70" rx="3" fill="#f6f8f9" stroke="#d4dde1"/>')
        svg.append(f'<circle cx="{x+24}" cy="{y+15}" r="17" fill="{color}"/><text x="{x+24}" y="{y+22}" fill="white" text-anchor="middle" font-family="DejaVu Sans" font-size="23" font-weight="bold">{number}</text>')
        svg.append(f'<text x="{x+50}" y="{y+11}" font-family="DejaVu Sans" font-size="25" font-weight="bold" fill="#17232b">{escape(title)}</text>')
        svg.append(f'<text x="{x+50}" y="{y+42}" font-family="DejaVu Sans" font-size="20" fill="#33434d">{escape(note)}</text>')
        box = manifest['modules'][i]['box']
        px, py = box['x'] + 9, top + box['y'] + 13
        if i in [4, 5]:
            px = box['x'] - 9
        if i in [0, 1, 2]:
            px, py = box['x'] + box['width']/2, top + box['y'] - 7
        svg.append(f'<circle cx="{px}" cy="{py}" r="18" fill="{color}" stroke="white" stroke-width="3"/><text x="{px}" y="{py+8}" text-anchor="middle" font-family="DejaVu Sans" font-size="24" font-weight="bold" fill="white">{number}</text>')
        if i < 3:
            svg.append(f'<path d="M{x+212},{y+60} L{px},{py-21}" stroke="#586573" fill="none" stroke-width="2" marker-end="url(#arrow)"/>')
    svg.append('</svg>')
    path = shots / 'front_page_annotated.svg'
    path.write_text('\n'.join(svg))
    cairosvg.svg2png(url=str(path), write_to=str(shots / 'front_page_annotated.png'), scale=2)
    print(json.dumps({'height': shot_h, 'errors': manifest['errors'], 'path': str(path)}, indent=2))


if __name__ == '__main__':
    main()
