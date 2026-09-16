"""Rendering-only review of saved KHU diagrams; no LLM calls or numerical changes."""
import hashlib
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from flora_design.visualizer.flowsheet_builder import FlowsheetBuilder
from flora_translate.schemas import ProcessTopology
from flora_translate.diagram_artifacts import _inline_svg_images


def main():
    source=ROOT/'outputs/khu_revised_six_20260914/presentation'
    output=ROOT/'outputs/topology_layout_review_20260914';output.mkdir(exist_ok=True)
    ns={'s':'http://www.w3.org/2000/svg'}
    reviews=[]
    for case in sorted(source.glob('figure*_set*')):
        path=case/'result.json';digest=hashlib.sha256(path.read_bytes()).hexdigest()
        result=json.loads(path.read_text());topology=ProcessTopology.model_validate(result['process_topology'])
        original=topology.model_dump_json()
        builder=FlowsheetBuilder()
        title='Layout only | '+case.name.replace('_',' ')
        if 'figure5' in case.name:title+=' | Reactor/light correction pending - not for execution'
        svg,png=builder.build(topology,title,str(output/f'{case.name}.svg'),str(output/f'{case.name}.png'))
        assert builder.last_render_info['renderer']=='graphviz',builder.last_render_info
        root=ET.parse(svg);positions={}
        for node in root.findall('.//s:g[@class="node"]',ns):
            img=node.find('s:image',ns)
            if img is not None:
                positions[node.find('s:title',ns).text]={'x':float(img.get('x')),
                    'y':float(img.get('y'))+float(img.get('height').removesuffix('px'))/2}
        main=[positions[n.replace('-','_').replace(' ','_')] for n in builder.last_render_info['main_process_path'] if n.replace('-','_').replace(' ','_') in positions]
        span=max(p['y'] for p in main)-min(p['y'] for p in main)
        assert span<=2,(case.name,span)
        assert all(a['x']<b['x'] for a,b in zip(main,main[1:]))
        assert len(root.findall('.//s:g[@class="edge"]',ns))==len(topology.streams)
        assert topology.model_dump_json()==original
        assert hashlib.sha256(path.read_bytes()).hexdigest()==digest
        _inline_svg_images(Path(svg))
        reviews.append({'case':case.name,'source_sha256':digest,'source_unchanged':True,
            'main_path':builder.last_render_info['main_process_path'],'main_icon_center_spread_points':span,
            'all_connections_preserved':True,'png':png,'svg':svg})
    (output/'rendering_review.json').write_text(json.dumps(reviews,indent=2))
    print(json.dumps([{'case':x['case'],'vertical_spread_pt':x['main_icon_center_spread_points']} for x in reviews]))


if __name__=='__main__':main()
