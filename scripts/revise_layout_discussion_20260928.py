"""Unify ESI tables, shorten the Discussion and insert layout-only topology redraws."""
from copy import deepcopy
from hashlib import sha256
import io
import json
from pathlib import Path
import re
from zipfile import ZipFile

import fitz
from lxml import etree as E
from PIL import Image

import revise_prior_work_20260928 as prior

ROOT, BASE = prior.ROOT, prior.BASE
OUT = BASE / "layout_revision_20260928"
W, NS = prior.W, dict(prior.NS)
NS.update(a="http://schemas.openxmlformats.org/drawingml/2006/main", wp="http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing")


def setting(parent, tag, **attrs):
    n = parent.find(W + tag)
    if n is None:
        n = E.SubElement(parent, W + tag)
    for k,v in attrs.items():
        n.set(W+k, str(v))
    return n


def remove(parent, *tags):
    for tag in tags:
        for child in parent.findall(W+tag):
            parent.remove(child)


def order_properties(table):
    sequences = {
        'tblPr': 'tblStyle tblpPr tblOverlap bidiVisual tblStyleRowBandSize tblStyleColBandSize tblW jc tblCellSpacing tblInd tblBorders shd tblLayout tblCellMar tblLook tblCaption tblDescription tblPrChange',
        'tcPr': 'cnfStyle tcW gridSpan hMerge vMerge tcBorders shd noWrap tcMar textDirection tcFitText vAlign hideMark tcPrChange',
        'trPr': 'cnfStyle divId gridBefore gridAfter wBefore wAfter cantSplit trHeight tblHeader tblCellSpacing jc hidden ins del trPrChange',
        'pPr': 'pStyle keepNext keepLines pageBreakBefore framePr widowControl numPr suppressLineNumbers pBdr shd tabs suppressAutoHyphens kinsoku wordWrap overflowPunct topLinePunct autoSpaceDE autoSpaceDN bidi adjustRightInd snapToGrid spacing ind contextualSpacing mirrorIndents suppressOverlap jc textDirection textAlignment textboxTightWrap outlineLvl divId cnfStyle rPr sectPr pPrChange',
        'rPr': 'rStyle rFonts b bCs i iCs caps smallCaps strike dstrike outline shadow emboss imprint noProof snapToGrid vanish webHidden color spacing w kern position sz szCs highlight u effect bdr shd fitText vertAlign rtl cs em lang eastAsianLayout specVanish oMath rPrChange',
    }
    for tag, sequence in sequences.items():
        order = {W+name: i for i,name in enumerate(sequence.split())}
        for props in table.iter(W+tag):
            props[:] = sorted(props,key=lambda n:order.get(n.tag,1000))


def style_tables(doc):
    tables = doc.findall('.//' + W + 'tbl')
    for table in tables:
        before = prior.text(table)
        props = table.find(W+'tblPr')
        if props is None:
            props = E.Element(W+'tblPr')
            table.insert(0, props)
        remove(props, 'tblStyle', 'tblBorders', 'tblCellMar', 'tblCellSpacing', 'shd')
        setting(props, 'tblStyle', val='TableGrid')
        setting(props, 'tblW', w='9360', type='dxa')
        setting(props, 'jc', val='center')
        setting(props, 'tblInd', w='0', type='dxa')
        setting(props, 'tblLayout', type='fixed')
        setting(props, 'tblLook', val='0620', firstRow='1', firstColumn='0', lastRow='0', lastColumn='0', noHBand='1', noVBand='1')
        borders = setting(props, 'tblBorders')
        for edge in ['top', 'left', 'bottom', 'right', 'insideH', 'insideV']:
            setting(borders, edge, val='single', sz='4', color='B8B8B8')
        margins = setting(props, 'tblCellMar')
        for edge in ['top', 'bottom', 'left', 'right']:
            setting(margins, edge, w='60' if edge in ['top','bottom'] else '80', type='dxa')
        grid = table.find(W+'tblGrid')
        original_widths = [int(c.get(W+'w')) for c in grid]
        widths = [round(w*9360/sum(original_widths)) for w in original_widths]
        widths[-1] += 9360-sum(widths)
        # Small width transfers keep header words intact without shrinking text.
        headers = table.find(W+'tr').findall(W+'tc')
        if len(headers) == len(widths):
            font = fitz.Font('tibo')
            minimum = [round((max((font.text_length(word,fontsize=11) for word in re.findall(r'[A-Za-z]+',prior.text(c))),default=0)+10)*20) for c in headers]
            for i in range(len(widths)):
                if widths[i] < minimum[i]:
                    missing = minimum[i]-widths[i]
                    donor = max((j for j in range(len(widths)) if j!=i),key=lambda j:widths[j]-minimum[j])
                    assert widths[donor]-minimum[donor] >= missing
                    widths[donor] -= missing
                    widths[i] += missing
        for child, width in zip(grid, widths):
            child.set(W+'w', str(width))
        for i,row in enumerate(table.findall(W+'tr')):
            rp = row.find(W+'trPr')
            if rp is None:
                rp = E.Element(W+'trPr')
                row.insert(0,rp)
            remove(rp,'trHeight','tblHeader')
            setting(rp,'cantSplit')
            if i == 0:
                setting(rp,'tblHeader',val='true')
            column = 0
            for cell in row.findall(W+'tc'):
                cp = cell.find(W+'tcPr')
                if cp is None:
                    cp = E.Element(W+'tcPr')
                    cell.insert(0,cp)
                remove(cp,'tcBorders','tcMar','shd','noWrap')
                span = int(cp.find(W+'gridSpan').get(W+'val')) if cp.find(W+'gridSpan') is not None else 1
                setting(cp,'tcW',w=str(sum(widths[column:column+span])),type='dxa')
                column += span
                setting(cp,'shd',val='clear',color='auto',fill='E6E6E6' if i == 0 else 'FFFFFF')
                setting(cp,'vAlign',val='top')
                for p in cell.findall(W+'p'):
                    pp = p.find(W+'pPr')
                    if pp is None:
                        pp = E.Element(W+'pPr')
                        p.insert(0,pp)
                    remove(pp,'shd','ind','keepNext')
                    setting(pp,'pStyle',val='Normal')
                    spacing = setting(pp,'spacing',before='0',after='0',line='240',lineRule='auto')
                    remove(pp,'contextualSpacing')
                    setting(pp,'jc',val='left')
                    for r in p.iter(W+'r'):
                        if not prior.text(r):
                            continue
                        properties = r.find(W+'rPr')
                        if properties is None:
                            properties = E.Element(W+'rPr')
                            r.insert(0,properties)
                        prior.previous.Package.font(properties)
                        color = setting(properties,'color',val='000000')
                        for key in list(color.attrib):
                            if key != W+'val':
                                del color.attrib[key]
                        remove(properties,'shd')
                        if i == 0:
                            setting(properties,'b',val='1')
                            setting(properties,'bCs',val='1')
        order_properties(table)
        assert before == prior.text(table)
    return len(tables)


def save_pair(package, stem, changed):
    prior.bold_references(package.doc)
    clean = BASE / f'{stem}_revised_20260928_v2.docx'
    marked = BASE / f'{stem}_revised_20260928_v2_marked.docx'
    prior.intro_tools.write_package(package.parts, package.doc, clean)
    doc = deepcopy(package.doc)
    tree = package.doc.getroottree()
    for node in changed:
        target = doc.xpath(tree.getpath(node),namespaces=NS)
        assert len(target) == 1
        prior.intro_tools.highlight(target[0])
    prior.intro_tools.write_package(package.parts,doc,marked)
    return {'clean': str(clean.relative_to(ROOT)), 'marked': str(marked.relative_to(ROOT))}


def main_manuscript():
    p = prior.previous.Package(BASE/'manuscript_revised_20260928.docx')
    nodes = list(p.body)
    start = next(i for i,n in enumerate(nodes) if prior.text(n)=='DISCUSSION')
    end = next(i for i,n in enumerate(nodes) if prior.text(n)=='CONCLUSION')
    old = nodes[start+1:end]
    template = next(n for n in old if prior.text(n).startswith('The central contribution'))
    new = [prior.clean_paragraph(template,t) for t in (OUT/'discussion_replacement.txt').read_text().strip().split('\n\n')]
    for n in new:
        nodes[end].addprevious(n)
    for n in old:
        p.body.remove(n)
    result = save_pair(p,'manuscript',new)
    result.update(old_discussion_words=sum(len(prior.text(n).split()) for n in old), new_discussion_words=sum(len(prior.text(n).split()) for n in new), new_paragraphs=len(new))
    (OUT/'previous_discussion.txt').write_text('\n\n'.join(prior.text(n) for n in old)+'\n')
    return result


def esi():
    p = prior.previous.Package(BASE/'esi_revised_20260928.docx')
    changed = []
    table_count = style_tables(p.doc)
    for node in p.ps:
        if prior.text(node).startswith('Table S'):
            changed.append(node)
    replacements = {'media/image13.png':'Figure_S13_straight', 'media/image19.png':'Figure_S17a_straight'}
    for i in range(3):
        replacements[f'media/revision_{32+i}_topology.png'] = f'Figure_S20{chr(97+i)}_straight'
        replacements[f'media/revision_{35+i}_topology.png'] = f'Figure_S21{chr(97+i)}_straight'
    rels = {r.get('Id'):r.get('Target') for r in p.rels}
    images = []
    for inline in p.doc.xpath('.//wp:inline',namespaces=NS):
        blips = inline.xpath('.//a:blip/@r:embed',namespaces=NS)
        if not blips or rels[blips[0]] not in replacements:
            continue
        target = rels[blips[0]]
        source = OUT/'figures'/(replacements[target]+'.png')
        data = source.read_bytes()
        p.parts['word/'+target] = data
        with Image.open(io.BytesIO(data)) as im:
            aspect = im.height/im.width
        extent = inline.find('wp:extent',NS)
        cx = int(extent.get('cx'))
        cy = round(cx*aspect)
        extent.set('cy',str(cy))
        for node in inline.xpath('.//a:xfrm/a:ext',namespaces=NS):
            node.set('cy',str(cy))
        images.append(target)
    assert set(images) == set(replacements)
    amendments = [
        ('Figure S13. Archived inventory-assigned topology', None),
        ('Figure S17 reopens the archived DPDTC run', None),
        ('Figure S17. Actual archived-run output:', 'Figure S17. Archived-run output: (a) topology redrawn with a horizontal main process path, right-angle interstage feed and generic pump symbols; equipment names and numerical labels are retained from the archived graph; (b) the original GUI process-summary screenshot, including the interstage feed and cumulative Stage 2 flow. Both panels describe the same archived result. Numerical values are also transcribed in Table S25. The generic T-mixer remains a disclosed verification requirement. This is a software screening output, not a wet-lab validated protocol. Original interface captures and graph files remain in the companion evidence.'),
        ('The companion evidence package separates fresh GUI actions', None)
    ]
    for prefix, replacement in amendments:
        node = next(n for n in p.ps if prior.text(n).startswith(prefix))
        value = prior.previous.prior.marked_text(node)
        if prefix.startswith('Figure S13.'):
            replacement = value.replace('Archived inventory-assigned topology', 'Horizontal redraw of the archived inventory-assigned topology',1) + ' The layout and generic pump symbol have been updated for readability; recorded equipment, connectivity and operating conditions are unchanged.'
        elif prefix.startswith('Figure S17 reopens'):
            replacement = value.replace('Figure S17 reopens the archived DPDTC run and shows its actual topology and process-summary views.', 'Figure S17 pairs a horizontal redraw of the archived DPDTC topology with the original GUI process-summary screenshot.')
        elif prefix.startswith('The companion'):
            replacement = value.replace('Cropping and page arrangement are editorial; no UI text, warnings, numerical values or image contents were altered.', 'The retained GUI screenshots are unaltered apart from cropping and page arrangement.') + ' Figure S17(a) is now an explicitly identified publication redraw; its original screenshot remains in that evidence package.'
        p.revise(node,replacement)
        changed.append(node)
    for node in p.ps:
        value = prior.previous.prior.marked_text(node)
        if 'Actual saved GUI topology for response Set' in value:
            p.revise(node,value.replace('Actual saved GUI topology', 'Horizontal redraw of the archived topology') + ' Only the layout has changed; node labels, equipment icons and process connections are retained.')
            changed.append(node)
    info = save_pair(p,'esi',changed)
    info.update(tables_standardized=table_count,replaced_media=images)
    return info


def main():
    sources = [BASE/f'{stem}_revised_20260928.docx' for stem in ['manuscript','esi']]
    hashes = {str(p.relative_to(ROOT)):sha256(p.read_bytes()).hexdigest() for p in sources}
    audit = {'source_hashes':hashes,'manuscript':main_manuscript(),'esi':esi()}
    assert all(sha256((ROOT/p).read_bytes()).hexdigest()==h for p,h in hashes.items())
    (OUT/'document_revision_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    print(json.dumps(audit,indent=2))


if __name__ == '__main__':
    main()
