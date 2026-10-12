"""Re-layout existing screenshot annotations; preserve captured UI and data."""
import base64
import hashlib
import json
from html import escape
from pathlib import Path
from zipfile import ZipFile

import cairosvg
from lxml import etree as E
from PIL import Image, ImageFont
from matplotlib.font_manager import FontProperties, findfont

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'manuscript'
OUT = BASE / 'october2_final_review/figures'
FONT = findfont(FontProperties(family='DejaVu Sans'))
BOLD = findfont(FontProperties(family='DejaVu Sans',weight='bold'))

def embed(path, x, y, width, height):
    data = base64.b64encode(path.read_bytes()).decode()
    return f'<image x="{x}" y="{y}" width="{width}" height="{height}" href="data:image/png;base64,{data}"/>'

def svg(width, height):
    return [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
            '<rect width="100%" height="100%" fill="white"/>']

def render(elements, path, scale=2):
    path.with_suffix('.svg').write_text('\n'.join(elements + ['</svg>']))
    cairosvg.svg2png(url=str(path.with_suffix('.svg')),write_to=str(path),scale=scale)

def front_page():
    old = BASE / 'october2_final_review/gui'
    manifest = json.loads((old/'capture_manifest.json').read_text())
    shot_h = manifest['viewport_css']['height']
    top, bottom, width = 152, 152, 1360
    s = svg(width,top+shot_h+bottom)
    s += ['<defs><marker id="arrow" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8" fill="#65717a"/></marker></defs>',
          embed(old/'front_page.png',0,top,width,shot_h)]
    specs = [
        ('Workspace navigation',['Design, inventory and saved runs']),
        ('Reaction protocol',['Batch chemistry and conditions']),
        ('Standardized questions',['Fixed IDs and recorded answers']),
        ('Model routing',['Upstream and council selections']),
        ('Laboratory inventory',['Bind a saved equipment profile']),
        ('Readiness gate',['Required answers resolve','before design generation']),
    ]
    checks = []
    for i, (title, lines) in enumerate(specs):
        x = 20 + (i%3)*448
        y = 20 if i<3 else top+shot_h+20
        color = ['#285d78','#0e766c','#976c22'][i%3]
        s.append(f'<rect x="{x}" y="{y}" width="424" height="106" rx="3" fill="#f6f8f9" stroke="#d4dde1"/>')
        s.append(f'<circle cx="{x+28}" cy="{y+33}" r="16" fill="{color}"/>')
        s.append(f'<text x="{x+28}" y="{y+40}" text-anchor="middle" font-family="DejaVu Sans" font-size="20" fill="white">{i+1}</text>')
        s.append(f'<text x="{x+56}" y="{y+34}" font-family="DejaVu Sans" font-size="20" font-weight="bold" fill="#17232b">{escape(title)}</text>')
        for j,line in enumerate(lines):
            s.append(f'<text x="{x+56}" y="{y+61+j*23}" font-family="DejaVu Sans" font-size="17" fill="#33434d">{escape(line)}</text>')
        widths = [ImageFont.truetype(BOLD,20).getlength(title)]+[ImageFont.truetype(FONT,17).getlength(t) for t in lines]
        assert max(widths)<424-56-20,(title,widths)
        checks.append({'label':title,'right_padding':424-56-max(widths),'top_padding':15,'bottom_padding':20})
        box=manifest['modules'][i]['box']
        px,py=box['x']+9,top+box['y']+13
        if i in (4,5): px=box['x']-9
        if i<3: px,py=box['x']+box['width']/2,top+box['y']-7
        s.append(f'<circle cx="{px}" cy="{py}" r="17" fill="{color}" stroke="white" stroke-width="3"/><text x="{px}" y="{py+7}" text-anchor="middle" font-family="DejaVu Sans" font-size="21" font-weight="bold" fill="white">{i+1}</text>')
        if i<3:
            s.append(f'<path d="M{x+212},{y+106} L{px},{py-20}" stroke="#65717a" fill="none" stroke-width="2" marker-end="url(#arrow)"/>')
    render(s,OUT/'gui_front_page.png')
    return checks

def inventory():
    shots=BASE/'Submission/inventory_gui_revision_20261001/screenshots'
    specs=[('a','Equipment categories','04_inventory_categories.png'),
           ('b','Pump specifications','05_equipment_fields.png'),
           ('c','Availability and constraints','06_constraints.png'),
           ('d','Validation and JSON export','07_validation.png')]
    width,margin,pad,header,gap=1900,48,26,84,46
    panels=[]
    for letter,title,name in specs:
        im=Image.open(shots/name)
        w=width-2*(margin+pad)
        h=round(im.height*w/im.width)
        panels.append((letter,title,shots/name,w,h))
    height=2*margin+sum(header+h+2*pad+gap for _,_,_,w,h in panels)-gap
    s=svg(width,height);y=margin
    for letter,title,path,w,h in panels:
        s.append(f'<text x="{margin}" y="{y+38}" font-family="DejaVu Sans" font-size="34" font-weight="bold" fill="#243934">{letter}  {escape(title)}</text>')
        y+=header
        s.append(f'<rect x="{margin}" y="{y}" width="{width-2*margin}" height="{h+2*pad}" fill="white" stroke="#b9c9c5" stroke-width="2"/>')
        s.append(embed(path,margin+pad,y+pad,w,h))
        y+=h+2*pad+gap
    render(s,OUT/'gui_inventory.png',1.5)

def supplied_figures():
    # White-backed rendering resolves transparent export masks without altering
    # supplied labels, data, or opaque pixels.
    for name in ('esi_fig','esi_fig2'):
        path=BASE/(name+'.png');im=Image.open(path)
        s=svg(im.width,im.height);s.append(embed(path,0,0,im.width,im.height))
        render(s,OUT/(name+'_white.png'),1)

def original_schemes():
    # Keep original vector EMF bytes; no re-drawing of chemical structures.
    source=BASE/'Supplementary Information (KRICT)_final (1).docx'
    with ZipFile(source) as z:
        for old,new in [('image10.emf','experimental_giese.emf'),('image12.emf','experimental_amidation.emf')]:
            (OUT/new).write_bytes(z.read('word/media/'+old))

def neutral_main_labels():
    ns={'s':'http://www.w3.org/2000/svg'}
    for number in (5,6):
        source=BASE/f'Submission/overhaul_20260930/figures/figure{number}_redesigned.svg'
        tree=E.parse(str(source))
        changed=[]
        for node in tree.xpath('//s:text',namespaces=ns):
            if node.text and 'KHU' in node.text:
                old=node.text
                node.text=old.replace('Connected configuration implemented at KHU','Implemented connected configuration').replace('KHU set(s)','Response set(s)')
                changed.append((old,node.text))
        assert len(changed)==2,changed
        target=OUT/f'figure{number}_neutral_labels.svg'
        tree.write(str(target),encoding='UTF-8',xml_declaration=True)
        cairosvg.svg2png(url=str(target),write_to=str(target.with_suffix('.png')),output_width=4320)
        (OUT/f'figure{number}_label_changes.json').write_text(json.dumps(changed,indent=2))

if __name__=='__main__':
    OUT.mkdir(parents=True,exist_ok=True)
    checks=front_page();inventory();supplied_figures();original_schemes();neutral_main_labels()
    (OUT/'asset_checks.json').write_text(json.dumps({'callout_padding':checks,
      'sources_unchanged':True,'gui_capture_date':'2026-10-01',
      'files':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.glob('*') if p.suffix in ('.png','.emf')}},indent=2))
    print(json.dumps({'assets':str(OUT),'callout_checks':checks},indent=2))
