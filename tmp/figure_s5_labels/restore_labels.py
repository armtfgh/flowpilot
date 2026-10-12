from pathlib import Path
import xml.etree.ElementTree as ET
import hashlib
import json
import re
import cairosvg

ROOT=Path('/home/amirreza/Documents/codes/Flow Agent')
source=ROOT/'deliverables/flowpilot_esi_revision_20260902/figures/Figure_S08.svg'
out=ROOT/'output/pdf/Figure_S5_restored_labels'
original=source.read_text()
# The historical Figure S08 is Figure S5 in the current ESI.
# All heatmaps and their text are retained byte-for-byte. Only the shared
# colorbar is translated down, with the canvas extended by the same amount.
shift=14.0
old_h='441.630471'
new_h=f'{float(old_h)+shift:.6f}'
corrected=original.replace(f'height="{old_h}pt"',f'height="{new_h}pt"',1)
corrected=corrected.replace(f'viewBox="0 0 486.96696 {old_h}"',f'viewBox="0 0 486.96696 {new_h}"',1)
corrected=corrected.replace(f'M 0 {old_h} \nL 486.96696 {old_h}',f'M 0 {new_h} \nL 486.96696 {new_h}',1)
assert corrected.count('<g id="axes_4">')==1
corrected=corrected.replace('<g id="axes_4">',f'<g id="axes_4" transform="translate(0 {shift:g})">',1)
ns={'s':'http://www.w3.org/2000/svg'}
a,b=ET.fromstring(original),ET.fromstring(corrected)
checks={}
for axis in ('axes_1','axes_2','axes_3'):
 old=a.find(f'.//s:g[@id="{axis}"]',ns)
 new=b.find(f'.//s:g[@id="{axis}"]',ns)
 assert ET.tostring(old)==ET.tostring(new)
 checks[axis+'_unchanged']=True
labels=[t.text for t in b.findall('.//s:text',ns) if (t.text or '').startswith('UO-')]
expected=[f'UO-{i:02d}' for i in [1,2,3,4,5,6,7,10,11,12,13,14]]
assert labels==expected
out.with_suffix('.svg').write_text(corrected)
cairosvg.svg2pdf(bytestring=corrected.encode(),write_to=str(out.with_suffix('.pdf')))
cairosvg.svg2png(bytestring=corrected.encode(),write_to=str(out.with_suffix('.png')),dpi=450)
checks.update(source=str(source.relative_to(ROOT)),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),displayed_criteria=labels,omitted_criteria=['UO-08','UO-09'],colorbar_shift_pt=shift,labels_baseline_y=406.693812,colorbar_top_y=401.300159+shift,all_data_and_heatmap_styling_unchanged=True)
(ROOT/'tmp/figure_s5_labels/verification.json').write_text(json.dumps(checks,indent=2)+'\n')
print(json.dumps(checks,indent=2))
print('OUTPUTS',*[str(out.with_suffix(e)) for e in ['.png','.svg','.pdf']],sep='\n')
