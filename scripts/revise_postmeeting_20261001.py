"""October 1 postmeeting revision, preserving the supplied Word packages."""
from copy import deepcopy
from hashlib import sha256
from pathlib import Path
import csv
import json
import re
import shutil

from lxml import etree as E
from PIL import Image
from revise_manuscript_cases_20260922 import Package, W, NS, text
from overhaul_submission_20260930 import with_cites
from revise_manuscript_20260930 import paragraph_like
from revise_layout_discussion_20260928 import order_properties

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'manuscript'
OUT = BASE / 'october1_postmeeting_work'
FIG = OUT / 'figures'
REPORT = []


def para(pkg, template, value):
    return paragraph_like(pkg, template, value)


def revise(pkg, node, value):
    old = text(node)
    pkg.revise(node, value)
    REPORT.append({'document': pkg.source.name, 'before': old, 'after': value})


def remove(pkg, nodes, why):
    for node in nodes:
        if node.getparent() is pkg.body:
            REPORT.append({'document': pkg.source.name, 'removed': text(node), 'reason': why})
            pkg.body.remove(node)


def replace_image(pkg, node, path, max_height=None):
    with Image.open(path) as im:
        width = min(6.5, max_height * im.width / im.height) if max_height else 6.5
    new = pkg.image(path, width)
    pp = new.find(W + 'pPr')
    E.SubElement(pp, W + 'jc', {W + 'val': 'center'})
    if node.xpath('./w:pPr/w:pageBreakBefore[not(@w:val="0")]', namespaces=NS):
        E.SubElement(pp, W + 'pageBreakBefore')
    pkg.body.replace(node, new)


def caption(pkg, template, value, table=False):
    p = para(pkg, template, value)
    pp = p.find(W + 'pPr')
    for child in list(pp):
        if child.tag in {W + 'pageBreakBefore', W + 'keepNext'}:
            pp.remove(child)
    E.SubElement(pp, W + 'keepNext', {W + 'val': '1' if table else '0'})
    return p


def table(pkg, headers, rows, widths):
    node = pkg.table(headers, rows, widths)
    for shd in node.findall('.//' + W + 'shd'):
        shd.set(W + 'fill', 'E6E6E6')
    for border in node.xpath('./w:tblPr/w:tblBorders/*', namespaces=NS):
        border.set(W + 'color', 'B8B8B8')
    return node


def response_rows(nodes, sets, common):
    rows, evidence = [], []
    shared = {}
    for set_id, indices in sets:
        for index in indices:
            value = text(nodes[index])
            qid = re.search(r'Q-[A-Z]+-\d+', value).group()
            answer = re.sub(r'^.*?Please enter here\s*(?:\u2014>)?\s*', '', value).rstrip('.') + '.'
            evidence.append({'set': set_id, 'question_id': qid, 'answer': answer})
            if qid in common:
                if qid in shared:
                    assert shared[qid] == answer
                shared[qid] = answer
            else:
                rows.append([str(set_id), qid, answer])
    rows = [['All three', qid, value] for qid, value in shared.items()] + rows
    return rows, evidence


def topology_pages(pkg, nodes, chemistry):
    figure_nums = [102, 103, 104] if chemistry == 5 else [105, 106, 107]
    folder = (ROOT / 'outputs/figure5_pure_oxygen_physics_20260918/slides' if chemistry == 5 else
              ROOT / 'outputs/khu_revised_six_20260915/collaborator_slides_20260915_163845')
    group = '20' if chemistry == 5 else '21'
    out = []
    for i, number in enumerate(figure_nums, 1):
        source = BASE / f'layout_revision_20260928/figures/Figure_S{group}{chr(96+i)}_straight.png'
        target = FIG / f'case_{chemistry-4}_set_{i}_proposed_topology.png'
        shutil.copy2(source, target)
        rawdir = OUT / 'raw' / f'figure{chemistry}_set{i}'
        rawdir.mkdir(parents=True, exist_ok=True)
        for name in ['stage_parameters.csv', 'feed_parameters.csv', 'process_topology.json']:
            shutil.copy2(folder / f'figure{chemistry}_set{i}' / name, rawdir / name)
        out.append(pkg.image(target, 6.5, page_break=True))
        detail = ('Oxygen is introduced only after R1; the gas-line check valve is not proof of upstream liquid-branch protection. R1 uses liquid flow; R2 uses the nominal inlet/STP time index, not pressure-dependent residence time.'
                  if chemistry == 5 else 'Benzylamine enters between the coils. R1 time uses Feed A alone; R2 time uses the combined liquid flow. Feed concentrations and equipment labels are retained from the proposal.')
        out.append(caption(pkg, nodes[260], f'Figure S{number}. Proposed {"Giese addition/oxidation" if chemistry == 5 else "DPDTC-mediated amidation"} topology for response Set {i} (main-text Figure {chemistry}). Horizontal layout of the archived design; no new model run was used. {detail} This is a proposal, not a diagram of independently verified operating conditions; implementation differences are recorded in Tables S18–S19.'))
        data = list(csv.DictReader((rawdir / 'stage_parameters.csv').open()))
        values = []
        for row in data:
            values.append([str(int(float(row['stage']))), f"{float(row['volume_mL']):g}",
                           f"{float(row['liquid_flow_mL_min']):g}", f"{float(row['gas_inlet_STP_mL_min']):g}",
                           f"{float(row['nominal_inlet_residence_min']):.2f}", f"{float(row['temperature_C']):g}"])
        # A figure key, not an independently numbered/duplicated results table.
        out.append(para(pkg, nodes[267], 'Reading the scheme: ' + '; '.join(
            f'R{v[0]}: {v[1]} mL, liquid {v[2]} mL min\u22121, gas {v[3]} mL min\u22121 at STP, time {v[4]} min, {v[5]} \u00b0C' for v in values) + '.'))
        feeds = list(csv.DictReader((rawdir / 'feed_parameters.csv').open()))
        for stream in dict.fromkeys(r['stream'] for r in feeds):
            entries = [r for r in feeds if r['stream'] == stream]
            first = entries[0]
            details = []
            for row in entries:
                component = row['component']
                if chemistry == 5 and component.startswith('((('):
                    component = 'substrate 1a'
                elif chemistry == 5 and component.startswith('[Ir'):
                    component = 'iridium photocatalyst'
                elif chemistry == 6 and component.startswith('3-methyl'):
                    component = 'acid 1b'
                c = f"{float(row['concentration_M']):g} M; " if row['concentration_M'] else ''
                details.append(f"{component} ({c}{float(row['molar_flow_mmol_min']):.4g} mmol min\u22121)")
            basis = ' at STP' if first['phase'] == 'gas' else ''
            out.append(para(pkg, nodes[267], f"Feed {stream}, {float(first['flow_mL_min']):g} mL min\u22121{basis}: " + '; '.join(details) + '.'))
    return out


def main_text(pkg):
    n = list(pkg.body)
    revise(pkg, n[8], with_cites(n[8]).replace('[[5]]-10', '[[5-10]]'))
    experience = ('Choosing among these coupled options also requires practical flow-chemistry experience: recognizing when a familiar reaction becomes limited by photon penetration, anticipating precipitation or catalyst retention, and deciding which batch-handling steps have a chemical purpose. Such judgments concern which model or operating assumption is appropriate, not only whether an equation has been evaluated correctly. Flow-development guides and reactor-selection toolboxes organize part of this expertise, but applying it still requires attention to the specific chemistry and apparatus.[[3-4,12,21]] A data-driven predictor, a simulator or a fixed calculation can support that decision without, on its own, supplying the complete experimental context. For an agentic assistant, a central challenge is therefore to make documented experience available at the point where a design choice is made, while retaining its source, assumptions and limits of transferability.')
    pkg.after(n[9], [para(pkg, n[9], experience)])
    revise(pkg, n[14], 'Here we introduce FlowPilot as an agentic research assistant for end-to-end flow-process design. The researcher supplies a batch procedure, available instruments and a design objective; standardized follow-up questions establish missing requirements, experimental evidence and hypotheses. FlowPilot combines three complementary knowledge functions: retrieval supplies relevant published examples, a handbook-derived rule base supplies engineering principles and cautions, and LLM reasoning interprets their relevance to the current reaction and inventory. Together, these resources provide an explicit, source-linked representation of documented flow-design experience, rather than a substitute for the researcher\u2019s experimental judgment. A specialist council then develops and reviews alternative connected configurations with quantitative tools, from deciding where reagents and gases enter to selecting stage conditions and assigning equipment. End-to-end denotes this integrated design workflow; experimental implementation and interpretation remain researcher-led.')
    revise(pkg, n[36], 'We next asked whether coordinating these capabilities improves the delivered process design beyond what the same LLM can produce in one response (Figure 4a). The three benchmark chemistries were selected to expose complementary process-design demands while using one unchanged assessment rubric. Cu/C-catalyzed azide\u2013alkyne cycloaddition (CuAAC) tests whether catalyst strategy, liquid-feed preparation and heated, pressurized operation remain consistent when a batch procedure is converted into a proposed apparatus. Oxidation of Fmoc-L-methionine tests the coordination of photocatalyst, irradiation and oxygen delivery, including the distinction between total air flow and reactive oxygen equivalents. Hydrogenolysis of N-diphenylmethylazetidin-3-ol adds a gas\u2013liquid\u2013solid contact problem, for which catalyst-bed placement, pressure control and the definition of contact time must agree. These contrasts challenge different assumptions even when a model correctly recognizes the transformation.')
    pkg.after(n[36], [para(pkg, n[36], 'The cases therefore act as a controlled stress test of connected design, rather than a representative survey of synthetic chemistry or a competition to predict published yields. For each case, one-shot generation and FlowPilot received the same protocol, objective, authoritative inventory and requested final-output structure. The one-shot baseline received the complete design brief, not merely a short chemistry question. Applicable criteria were selected by fixed rules rather than adjusted to favor a chemistry or architecture; source records were excluded from FlowPilot retrieval, although this does not establish their absence from model pretraining. The original prompts and source references in ESI Section 4 make these boundaries explicit.')])
    revise(pkg, n[43], 'The critical-flag analysis asks whether the higher average scores correspond to fewer consequential defects in the proposed process (Figure 4c; Table S12). Judges recorded 98 flags across 45 one-shot outcomes and 4 across 45 FlowPilot outcomes; 18 and 43 outcomes, respectively, had no flag. Panel c groups these findings by generator: each point is the mean total across the three chemistries within a repeat, not a measured failure frequency. The large reductions for the Qwen models and GPT-4o are consistent with the larger score gains in panel b, whereas the Claude one-shot designs already contained fewer flagged defects. This makes the reduction more informative than a score increase alone, but does not convert the assessment into a prediction of laboratory success.')
    pkg.after(n[43], [para(pkg, n[43], 'The underlying examples illustrate why these defects matter chemically and operationally. A one-shot hydrogenolysis procedure placed a vented gas\u2013liquid separator before its only pressure-defining BPR while describing pressurized reaction, leaving the proposed gas contact and pressure boundary incompatible as written. Another mixed an empty-bed time with a different reported residence-time value, obscuring the exposure assigned to the substrate. FlowPilot was not defect-free: one CuAAC output placed Cu/C in the liquid preparation despite specifying a packed catalyst bed, a contradiction that changes how the catalyst is contacted and retained. Moreover, one judge incorrectly treated equal-reference sccm and mL min\u22121 values as differing by a factor of 1,000. The archived counts retain that false positive, and multiple flags can refer to one underlying defect. They are therefore reported as judge findings, not as independently confirmed physical errors or safety probabilities.')])
    revise(pkg, n[45], 'To examine the role of coordinated review, we separately tested 15 Qwen3.8-27B configurations over the same three cases (Figure 4d; Table S10). The selected conditions in panel d compare the full pipeline with one-shot generation, removal of the council and removal of individual domain reviewers. Removing the council reduced the mean score from 0.928 to 0.670, below the 0.743 one-shot result in this screen. Thus splitting the task into an initial sequence of agent calls was not sufficient: the benefit depended on reviewing whether the chemical and engineering choices formed a coherent process. The full condition table is retained because the displayed subset alone does not describe every architecture tested.')
    pkg.after(n[45], [para(pkg, n[45], 'The domain removals help interpret that result. Without the Chemistry reviewer, the mean was 0.902; without Fluidics, Safety or Kinetics it was 0.913, 0.915 or 0.924, respectively. These reviewers examine coupled decisions such as whether a supported catalyst remains in the intended phase, whether gas and liquid feeds support the stated contact time, and whether thermal and pressure assumptions are compatible with the solvent and hardware. Their overlapping responsibilities allow another reviewer or a deterministic check to catch some omissions, so the smaller changes do not establish that a removed specialty is unimportant. Conversely, one-pass review (0.932) and omission of preselection refinement (0.935) slightly exceeded the complete configuration. The result favors coordinated cross-domain checking, not an assumption that more discussion always improves the design. Each condition was generated once per chemistry; the bars show between-case variation, not replicate uncertainty, and the screen cannot resolve a causal ranking of individual agents.')])
    pkg.after(n[46], [para(pkg, n[46], 'Connected design extends beyond reaction conditions to subsequent material handling, as illustrated by continuous isoxazole synthesis coupled to inline solvent separation and recovery.[[56]] An earlier FlowPilot computational dry run examined aldehyde \u03b1-bromination followed by an inline quench, connecting a reaction step to downstream reagent deactivation (ESI Section 6.4; Figure S101). This archived illustration is retained as a pre-experimental design example, including its unresolved mixing warnings, rather than as an additional laboratory validation. The two studies below instead examine actual implementation of connected reaction sequences.')])
    revise(pkg, n[47], with_cites(n[47]).replace('Full objectives, hypotheses and fixed-ID answers are reproduced in ESI Section 7.', 'The fixed-ID responses are compared in Tables S101–S102, and all six archived proposed topologies are reproduced in Figures S102–S107 (ESI Section 7).'))
    revise(pkg, n[80], with_cites(n[80]).replace('Council terminology, configuration studies, and examples of actual deliberation are documented in ESI Section 6 and Section 7.5.', 'The component responsibilities are specified in ESI Section 2.1, the ablation configurations in Table S10, and an archived discussion example in ESI Section 6.4.'))


def esi_text(pkg):
    n = list(pkg.body)
    # Remove the named subsection and the redundant retrieval schematic. Retain
    # a data-only S4 audit and the numeric retrieval settings, not a second workflow.
    remove(pkg, n[66:75], 'Remove old Section 3.2 and the category-count table duplicated by Figure S4.')
    replace_image(pkg, n[75], FIG / 'Figure_S4_rules_three_panels.png', max_height=7.0)
    revise(pkg, n[76], 'Figure S4. Audit of the handbook-derived rule base (2,537 rules; 29 categories). (a) Counts and stored severity labels for the 14 largest categories. Category and severity are distinct: the Safety category is not the same as the safety severity label. (b) Percentage of rules containing a machine-detected quantitative expression, in the same category order as (a). Expression detection does not verify an equation, and a hard_rule label does not by itself implement a deterministic gate. (c) Non-exclusive category associations with chemistry classes, shown as log(1 + association count). The distribution identifies the engineering topics available to prompt construction, including heat transfer, photochemistry and catalyst handling; it is not a performance metric. The source CSVs retain all category counts.')
    remove(pkg, [n[77], n[78], n[81], n[82]], 'Remove the retrieval workflow already shown in main-text Figure 3.')
    # Place settings inside the retained methods/results subsection.
    pkg.body.remove(n[79]); pkg.body.remove(n[80])
    pkg.after(n[83], [para(pkg, n[84], 'Main-text Figure 3 presents the retrieval workflow. Table S6 records the operational numerical settings needed for reproduction; the offline evaluations below used the explicitly stated, reduced scoring functions and should not be confused with a run of every operational tier.'), n[79], n[80]])
    revise(pkg, n[83], 'Retrieval settings, benchmark and source-exclusion controls')
    revise(pkg, n[248], 'Figure S11 shows the complete design-studio front page during a live intake interaction. Numbered callouts identify workspace navigation, protocol entry, standardized questions, model routing, inventory binding and the readiness gate. The screenshot uses a DPDTC protocol and the saved inventory profile; it was captured without a new design-generation call. It shows a question-answering state, not a completed experiment or a fabricated design result.')
    replace_image(pkg, n[250], OUT / 'gui/front_page_annotated.png', max_height=7.25)
    revise(pkg, n[251], 'Figure S11. Annotated front page of the FlowPilot web application. (1) Navigation to design, inventory and saved runs. (2) Batch-protocol entry and the current intake exchange. (3) Fixed-ID questions and answer fields. (4) Upstream and downstream model selections. (5) Bound laboratory inventory. (6) Run control, disabled until required intake items are resolved. The browser image is an actual October 1 capture; only explanatory shapes, arrows and labels were added. The complete-resolution image and capture record are supplied separately. Inventory editing is shown in Figure S12.')
    # Preserve the accepted inventory screenshot (n[259]) byte-for-byte.
    dry = BASE / 'layout_revision_20260928/original_figures/29_revision_38_figureS22_preprint_figure6_exact.png'
    shutil.copy2(dry, FIG / 'archived_preprint_figure6_dry_run.png')
    dry_nodes = [para(pkg, n[261], 'Computational dry-run example: bromination and inline quench'),
        para(pkg, n[262], 'Before the connected reactions in Section 7 were implemented, a computational example examined \u03b1-bromination of 3-phenylpropanal with bromine in acetonitrile followed by aqueous thiosulfate quenching, based on the chemistry reported by Lu et al. [13]. The design question was how to connect reaction and residual-reagent deactivation without reproducing the batch addition and work-up as disconnected instructions. Figure S101 reproduces the former preprint Figure 6 unchanged, retaining the displayed council exchange, process scheme and contextual literature comparison.'),
        para(pkg, n[262], 'The displayed scheme assigns 2.2 min to the reaction segment and 1.0 min to quench contact. At the displayed feed rates, these correspond to approximately 5.30 and 2.51 mL, respectively; the downstream calculation includes the added quench stream. All 12 candidates nevertheless retained a mixing warning in the archived discussion. This was a dry run, not a measured yield, conversion or safety demonstration, and agreement with a literature time range at another scale does not establish successful process transfer.'),
        para(pkg, n[262], 'The unedited historical artwork also exposes limitations of that early output: the selected-candidate dialogue and the final scheme use different flow values, and the safety dialogue refers to a BPR absent from the scheme and comparison table. Its legacy label \u201cValidated Process Flow Diagram\u201d and estimated conversion statements must therefore not be read as experimental validation or a closed executable design. This record illustrates the need for the shared final-design record described in Section 2; it is excluded from the architecture-score aggregates and from the wet-lab evidence.'),
        pkg.image(dry, 6.5, page_break=True),
        caption(pkg, n[260], 'Figure S101. Historical computational demonstration of aldehyde \u03b1-bromination followed by inline quenching, reproduced unchanged from preprint Figure 6. (a) Archived council discussion and unresolved mixing concerns. (b) Proposed equipment arrangement. (c) Contextual comparison with the literature process [13]. The legacy validation wording refers to the historical computational illustration, not laboratory verification. Conflicting dialogue/scheme values are retained and discussed in Section 6.4; the image must not be used as an execution protocol.')]
    pkg.before(n[266], dry_nodes)
    giese, ge = response_rows(n, [(1, range(276,280)), (2, range(281,285)), (3, range(286,290))], {'Q-GAS-002','Q-GAS-003'})
    amide, ae = response_rows(n, [(1, range(300,303)), (2, range(304,307)), (3, range(308,311))], {'Q-CHEM-001'})
    (OUT / 'response_table_source_audit.json').write_text(json.dumps({'case1':ge,'case2':ae}, indent=2, ensure_ascii=False))
    pkg.before(n[274], [caption(pkg,n[257], 'Table S101. Standardized responses for Giese addition/oxidation (case 1; main-text Figure 5). Shared answers are listed once; objectives and hypotheses retain the supplied wording, with form prefixes and duplicate punctuation removed.', table=True), table(pkg,['Set','Question ID','Confirmed response'],giese,[.65,1.0,4.85])])
    remove(pkg, n[274:290], 'Replace repeated response paragraphs with the case-1 response table.')
    revise(pkg, n[290], with_cites(n[290]) + ' The later pure-oxygen requirement was a researcher-specified revision following the gas-flow observation, not a different wording silently substituted into Q-GAS-002. Figures S102–S104 preserve all three proposed topologies; their convergence is not evidence of independent experimental repeats.')
    pkg.before(n[293], topology_pages(pkg,n,5))
    pkg.before(n[298], [caption(pkg,n[257], 'Table S102. Standardized responses for DPDTC-mediated amidation (case 2; main-text Figure 6). The chemistry-identity answer is common to all sets. Supplied wording is retained apart from form prefixes and duplicate punctuation.', table=True), table(pkg,['Set','Question ID','Confirmed response'],amide,[.65,1.0,4.85])])
    remove(pkg, n[298:311], 'Replace repeated response paragraphs with the case-2 response table.')
    revise(pkg,n[311], with_cites(n[311]) + ' Figures S105–S107 reproduce the three proposed topologies. The laboratory grouped Sets 2/3 into one operating-point result; proposal multiplicity must not be treated as additional yield replicates.')
    pkg.before(n[312], topology_pages(pkg,n,6))
    # Introduce the image before it appears, rather than merely restore an orphan.
    revise(pkg,n[267], 'The two experimentally examined chemistries used a common protocol and inventory within each case, with three response sets varying the design priorities. Tables S101–S102 provide the exact answers; Figures S102–S107 retain the corresponding proposed topologies. This section documents the inputs and proposals. Section 8 separately records the actual feed preparation, operating points, yield measurements and implementation changes; the three sets are not three independent experimental replicates.')
    pkg.before(n[383], [para(pkg,n[382], '13. Lu, Y.; Yan, Z.; Jiang, Y.; He, H.; Zhou, J.; Gao, S.; Xing, D. \u03b1-Bromination of Aldehydes by Continuous Flow Chemistry and Its Application to the Flow Synthesis of 2-Aminothiazoles. Organic Process Research & Development 2025, 29, 2047\u20132055. https://doi.org/10.1021/acs.oprd.5c00076.')])


REF = re.compile(r'\b(Figures?|Tables?) S(\d+)([a-e]?)(?:[\u2013\u2014-]S?(\d+)([a-e]?))?')


def renumber(esi, main):
    mappings = {'Figure':{}, 'Table':{}}
    for node in esi.body:
        m = re.match(r'^(Figure|Table) S(\d+)\.', text(node))
        if m:
            kind, old = m.group(1), int(m.group(2))
            assert old not in mappings[kind], (kind, old)
            mappings[kind][old] = len(mappings[kind]) + 1
    def replace(m):
        prefix,start,suffix,end,end_suffix = m.groups()
        mapping = mappings['Figure' if prefix.startswith('Figure') else 'Table']
        numbers = list(range(int(start),int(end)+1)) if end else [int(start)]
        kept = [mapping[i] for i in numbers if i in mapping]
        assert kept, ('Unresolved removed reference',m.group())
        if len(kept)==1:
            return prefix.rstrip('s')+' S'+str(kept[0])+suffix
        if all(b==a+1 for a,b in zip(kept,kept[1:])):
            return prefix.rstrip('s')+'s S'+str(kept[0])+'\u2013S'+str(kept[-1])
        return prefix.rstrip('s')+'s '+', '.join('S'+str(i) for i in kept)
    for pkg in [esi,main]:
        for p in list(pkg.doc.iter(W+'p')):
            # TOC cache is regenerated after rendering, not treated as body text.
            if p.xpath('ancestor::w:sdt',namespaces=NS):
                continue
            value = with_cites(p)
            value = value.replace('now lists all 15 conditions', 'lists all 15 conditions')
            value = value.replace('Figure S5', 'main-text Figure 3' if pkg is esi else 'Figure 3') if re.search(r'Figure S5(?!\d)',value) else value
            value = value.replace('Section 3.4','Section 3.2').replace('section 3.4','section 3.2')
            changed = REF.sub(replace,value)
            if changed != with_cites(p):
                pkg.revise(p,changed)
    (OUT/'numbering_map.json').write_text(json.dumps(mappings,indent=2))
    return mappings


def restrained_bold(pkg):
    removed = []
    for p in pkg.body.findall(W+'p'):
        value = text(p)
        style = p.xpath('./w:pPr/w:pStyle/@w:val',namespaces=NS)
        if not value or any(s.startswith('Heading') or s.startswith('TOC') or s in {'Title','Subtitle'} for s in style):
            continue
        # Preserve short unstyled section labels and the front matter.
        if len(value.split()) < 12 and not re.match(r'^(Figure|Table) ',value):
            continue
        boldtext=[]
        for r in p.findall(W+'r'):
            props=r.find(W+'rPr')
            if props is None: continue
            for b in list(props):
                if b.tag in {W+'b',W+'bCs'}:
                    boldtext.append(text(r)); props.remove(b)
        if boldtext: removed.append({'paragraph':value[:120], 'removed_direct_bold':boldtext})
        # Caption label only; full caption bodies remain regular.
        m=re.match(r'^(Figure \d+\.|(?:Figure|Table) S\d+\.)',value)
        if m:
            label_len=len(m.group())
            offset=0
            for r in list(p.findall(W+'r')):
                rv=text(r)
                if offset>=label_len: break
                if offset+len(rv)>label_len:
                    tail=deepcopy(r)
                    tail.find(W+'t').text=rv[label_len-offset:]
                    r.find(W+'t').text=rv[:label_len-offset]
                    r.addnext(tail)
                rp=r.find(W+'rPr')
                if rp is None: rp=E.Element(W+'rPr');r.insert(0,rp)
                E.SubElement(rp,W+'b')
                offset+=len(rv)
    return removed


def order_esi_references(pkg):
    """The restored dry run is cited before the two experimental sources."""
    mapping={10:11,11:12,12:13,13:10}
    nodes=list(pkg.body)
    start=next(i for i,n in enumerate(nodes) if text(n)=='References')
    refs={int(m.group(1)):n for n in nodes[start+1:]
          if (m:=re.match(r'^(\d+)\. ',text(n)))}
    assert set(refs)==set(range(1,14))
    for node in nodes[:start]:
        for p in ([node] if node.tag==W+'p' else node.findall('.//'+W+'p')):
            value=with_cites(p)
            changed=re.sub(r'\[(10|11|12|13)\]',lambda m:'['+str(mapping[int(m[1])])+']',value)
            if changed!=value:pkg.revise(p,changed)
    anchor=nodes[start]
    for old in list(range(1,10))+[13,10,11,12]:
        p=refs[old]
        if old in mapping:
            pkg.revise(p,re.sub(r'^\d+\.',str(mapping[old])+'.',with_cites(p)))
        pkg.body.remove(p)
        anchor.addnext(p)
        anchor=p
    (OUT/'ESI_literature_numbering_map.json').write_text(json.dumps(mapping,indent=2))


def main():
    source_hashes=json.loads((OUT/'source_manifest.json').read_text())
    manuscript=Package(BASE/'FlowPilot_manuscript__20261001.docx')
    esi=Package(BASE/'FlowPilot_ESI__20261001.docx')
    # Earlier edits left unused image relationships whose IDs can exceed the
    # largest displayed drawing ID. Never reuse those relationship identifiers.
    for pkg in [manuscript, esi]:
        used = [int(m.group(1)) for r in pkg.rels
                if (m := re.fullmatch(r'rIdRevision(\d+)', r.get('Id', '')))]
        pkg.next_id = max([pkg.next_id] + [x + 1 for x in used])
    main_text(manuscript)
    esi_text(esi)
    maps=renumber(esi,manuscript)
    order_esi_references(esi)
    bold={}
    for key,pkg in [('manuscript',manuscript),('ESI',esi)]:
        bold[key]=restrained_bold(pkg)
        for highlight in list(pkg.doc.iter(W+'highlight')):
            highlight.getparent().remove(highlight)
        order_properties(pkg.doc)
        path=BASE/f'FlowPilot_{key}_october1_postmeeting.docx'
        pkg.save(path)
        marked=OUT/'marked'
        marked.mkdir(exist_ok=True)
        pkg.save(marked/f'FlowPilot_{key}_october1_postmeeting_marked.docx',marked=True)
        print(path)
    (OUT/'text_change_log.json').write_text(json.dumps(REPORT,indent=2,ensure_ascii=False))
    (OUT/'bold_cleanup.json').write_text(json.dumps(bold,indent=2,ensure_ascii=False))
    for path,digest in source_hashes.items():
        assert sha256((ROOT/path).read_bytes()).hexdigest()==digest
    print(json.dumps(maps,indent=2))


if __name__=='__main__':
    main()
