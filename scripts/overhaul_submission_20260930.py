"""Streamline the submission without deleting raw evidence or changing scores."""
from copy import deepcopy
from hashlib import sha256
from pathlib import Path
import csv
import json
import re
import shutil

from lxml import etree as E
import professional_review_20260930 as old
from revise_layout_discussion_20260928 import style_tables, order_properties
from revise_submission_20260929 import science_typography
from revise_manuscript_cases_20260922 import compressed_cites, expanded_cites

ROOT, BASE, W, NS = old.ROOT, old.BASE, old.W, old.NS
Package, text = old.Package, old.text
OUT = BASE / 'overhaul_20260930'
MAIN = BASE / 'manuscript_submission_reviewed_20260930.docx'
ESI = BASE / 'esi_submission_reviewed_20260930.docx'
FIG_KEEP = list(range(1, 11)) + [12, 14, 15, 16, 17, 18, 23, 25, 27, 28, 29, 30, 31]
TABLE_KEEP = [36, 1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 18, 19, 20, 21, 24, 25, 26, 27, 31, 34, 38, 39]
MAPS = {'Figure': {n: i+1 for i, n in enumerate(FIG_KEEP)}, 'Table': {n: i+1 for i, n in enumerate(TABLE_KEEP)}}
SECTION_MAP = {1: 2, 2: 3, 3: 4, 4: 5, 5: 5, 6: 6, 7: 7, 8: 8, 9: None, 10: None, 11: 1, 12: 9, 13: 10}
log = []


def with_cites(p):
    chunks = []
    runs = list(p.iter(W + 'r'))
    for i, r in enumerate(runs):
        value = text(r)
        following = text(runs[i+1]) if i+1 < len(runs) else ''
        is_cite = r.xpath('./w:rPr/w:vertAlign[@w:val="superscript"]', namespaces=NS) and re.fullmatch(r'\d+(?:[,–-]\d+)*', value) and not re.match(r'^(?:H|C)\b', following)
        chunks.append('[[' + value + ']]' if is_cite else value)
    return ''.join(chunks)


def edit(pkg, prefix, value, why):
    ps = [p for p in pkg.body if text(p).startswith(prefix)]
    assert len(ps) == 1, (prefix, len(ps))
    p = ps[0]
    log.append({'document': pkg.source.name, 'before': text(p), 'after': value, 'reason': why})
    pkg.revise(p, value)
    return p


def replace_section(pkg, start, end, paragraphs):
    old_nodes = old.base.section(pkg, start, end)
    nodes = [old.base.paragraph_like(pkg, old_nodes[0], v) for v in paragraphs]
    log.append({'document': pkg.source.name, 'before': '\n'.join(text(p) for p in old_nodes), 'after': '\n'.join(paragraphs), 'reason': 'Rewrite section to strengthen synthesis and avoid repetition.'})
    old.base.replace_section(pkg, start, end, nodes)


def caption(pkg, value):
    return pkg.paragraph(value, keep=True)


def compact(nums):
    groups = []
    i = 0
    while i < len(nums):
        j = i
        while j+1 < len(nums) and nums[j+1] == nums[j]+1:
            j += 1
        groups.append(f'S{nums[i]}' if i == j else f'S{nums[i]}–S{nums[j]}')
        i = j+1
    return ', '.join(groups)


LABEL = re.compile(r'\b(Figures?|Tables?)\s+(S\d+[a-z]?(?:\s*(?:[–-]|,\s*(?:and\s+)?|\band\b)\s*S?\d+[a-z]?)*)')


def remap_labels(value):
    def sub(m):
        kind = 'Figure' if m[1].startswith('Figure') else 'Table'
        s = m[2]
        numbers = []
        for bit in re.split(r'\s*(?:,\s*(?:and\s+)?|and)\s*', s):
            found = [int(n) for n in re.findall(r'\d+', bit)]
            numbers.extend(range(found[0], found[1]+1) if len(found) == 2 else found)
        missing = [n for n in numbers if n not in MAPS[kind]]
        if missing:
            raise ValueError(f'Unresolved removed reference: {m[0]} in {value[:200]}')
        mapped = [MAPS[kind][n] for n in numbers]
        label = kind + ('s' if len(mapped)>1 else '')
        suffix = re.search(r'S\d+([a-z])$', s)
        return label + ' ' + compact(mapped) + (suffix[1] if suffix and len(mapped)==1 else '')
    return LABEL.sub(sub, value)


def remap_sections(value):
    def sub(m):
        number, tail = int(m[2]), m[3] or ''
        if number == 6 and tail == '.1':
            return m[1] + ' 6'
        mapped = SECTION_MAP[number]
        assert mapped is not None, ('removed section', m[0], value)
        return m[1] + ' ' + str(mapped) + tail
    return re.sub(r'\b(Sections?)\s+(1[0-3]|[1-9])(\.\d+)?\b', sub, value)


def archive_inputs(pkg, nodes):
    archive = OUT / 'companion_archive'
    archive.mkdir(exist_ok=True)
    shutil.copy2(ESI, archive / 'esi_before_streamlining.docx')
    for name, start, end, index in [('hydrogenolysis', 146, 164, 161), ('photochemical_oxidation', 168, 186, 183), ('cuaac', 190, 205, 201)]:
        (archive / f'{name}_design_input.txt').write_text('\n\n'.join(text(p) for p in nodes[start:end]) + '\n')
        inventory = json.loads(text(nodes[index]))
        (archive / f'{name}_inventory.json').write_text(json.dumps(inventory, indent=2) + '\n')
    (archive / 'oneshot_response_contract.txt').write_text(text(nodes[114]) + '\n')
    (archive / 'legacy_matched_run_full_text.txt').write_text('\n\n'.join(text(p) for p in nodes[283:714]) + '\n')
    (archive / 'pipeline_history.txt').write_text('\n\n'.join(text(p) for p in nodes[864:871]) + '\n')


def build_esi():
    s = Package(ESI)
    n = list(s.body)
    assert text(n[90]) == 'Ablation test and architecture comparison'
    assert text(n[880]) == 'Laboratory implementation, experimental procedures, and analytical data'
    archive_inputs(s, n)
    keep = []
    def take(a, b, exclude=()):
        keep.extend(n[i] for i in range(a,b) if i not in exclude)
    take(0, 20)
    # Direct literature comparison precedes workflow tables, as in the Introduction.
    take(871, 874)
    comparison = [
        ['Chat-microreactor [7]', 'Flow-pattern prediction and microreactor guidance.', 'Local hydrodynamics and a piperacillin design application.'],
        ['LLM-RDF [8]', 'Synthesis development, reactor design and experimental workflows.', 'Experimental execution, kinetic analysis, optimization and purification.'],
        ['SapoMind [9]', 'Condition recommendation for continuous-flow saponification.', 'Experimental optimization of the selected microreactor reaction.'],
        ['Text-to-Simulation [10]', 'Process topology and parameter generation for simulation.', 'Executable flowsheets and simulator feedback.'],
        ['CeProAgents [11]\nPreprint', 'Hierarchical conceptual and parameter design for chemical processes.', 'Industrial-process case studies and simulation.'],
        ['CAAF [12]\nPreprint', 'Deterministic validation and bounded candidate revision.', 'Constructed reactor-design constraints; not a laboratory-inventory trial.'],
        ['FlowPilot\nPresent work', 'Reaction-level input to feed, stage and equipment assignments.', 'Matched architecture comparison; two connected-stage laboratory implementations.'],
    ]
    keep.append(s.table(['Study', 'Design task and output', 'Reported scope and validation'], comparison, [1.30,2.40,2.80]))
    take(878, 879)
    take(20, 46)
    take(49, 90)
    take(90, 227, [160,161,182,183,200,201])
    take(227, 240)
    take(242, 245)
    take(249, 252)
    take(265, 283)
    n[265].find(W+'pPr').find(W+'pStyle').set(W+'val', 'Heading1')
    take(714, 720)
    take(722, 733)
    take(733, 763)
    take(770, 777)
    take(779, 797)
    # Keep the scientific safety qualification, not duplicated numeric exports.
    take(803, 805)
    take(818, 822)
    take(824, 839)
    take(880, 889)
    keep.append(s.image(OUT/'figures/figureS17_irradiation_sources.png', width=6.5, page_break=True))
    keep.append(caption(s, 'Figure S23. Irradiation sources supplied by KHU: (a) two MR16 5 W batch lamps; (b) Vapourtec 450 nm LED for R1; (c) blue strip LED for R2. The original photographs and emission traces are retained. Reported peak wavelengths are 452.4, 450.32 and 447.50 nm, respectively. Electrical inputs are 5 W per MR16 lamp, 60 W for the Vapourtec source, and 10 W for the strip. The Vapourtec radiant-power specification is 24 W; none of these values establishes irradiance at the reacting fluid.'))
    take(897, 904)
    keep.append(caption(s, 'Table S31. Giese addition/oxidation operating points and yield. Yield is determined by 1H NMR. Sets 1/3 share one supplied entry; no independent replicate count is inferred. Stage 2 is a nominal inlet/STP time index.'))
    keep.append(s.table(['Parameter', 'KHU sets 1 / 3', 'KHU set 2'], [
        ['Yield (%)','97','98'], ['R1 / R2 volume (mL)','2 / 20','5 / 20'],
        ['Liquid flow (mL/min)','0.020','0.020'],['O2 controller setting (mL/min)','0.090','0.090'],
        ['R1 time / R2 index (min)','100 / 181.82','250 / 181.82'],
        ['Temperature (both stages)','25 ± 1 °C','25 ± 1 °C'],
        ['Recorded pressure (bar)','5.1–5.4','5.2–5.6'],['BPR cartridge','7 bar','7 bar'],
    ], [3.1,1.7,1.7]))
    keep.append(s.paragraph('The literature flow reference in main-text Figure 5 is 90% NMR yield (Park et al. [4], Table 4 entry 5). Literature and KHU configurations differ in gas delivery, irradiation and other operating conditions; the reference provides context rather than a matched experimental control.'))
    take(906, 915)
    keep.append(caption(s, 'Table S34. DPDTC-mediated amidation operating points and yield. Yield is determined by 1H NMR. Sets 2/3 share one supplied entry. Stage times are nominal liquid-volume/flow calculations.'))
    keep.append(s.table(['Parameter','KHU set 1','KHU sets 2 / 3'], [
        ['Yield (%)','86','68'],['R1 / R2 volume (mL)','5 / 5','10 / 5'],
        ['Feed A / B (mL/min)','0.160 / 0.040','0.280 / 0.070'],
        ['R1 / R2 time (min)','31.25 / 25.00','35.71 / 14.29'],
        ['Temperature (both stages)','95 °C','95 °C'],['Recorded pressure (bar)','5.2–5.3','5.5–5.8'],
        ['BPR cartridge','7 bar','7 bar'],
    ], [3.1,1.7,1.7]))
    keep.append(s.paragraph('The literature reference in main-text Figure 6 is 98% NMR yield in batch (Saunders et al. [5], Table 1 entry 5). This measurement basis matches the KHU yield determinations, but the operating mode and conditions differ; the comparison does not isolate the effect of batch-to-flow translation.'))
    take(917, 925)
    take(926, len(n))
    for node in list(s.body):
        s.body.remove(node)
    for node in keep:
        s.body.append(node)
    # Separate drawings from captions before editing text-only paragraphs.
    for p in list(s.body):
        if not text(p).startswith('Figure S') or not p.xpath('.//w:drawing', namespaces=NS):
            continue
        picture = E.Element(W+'p')
        pp = deepcopy(p.find(W+'pPr'))
        if pp is None:
            pp = E.Element(W+'pPr')
        E.SubElement(pp, W+'keepNext')
        picture.append(pp)
        for run in list(p.findall(W+'r')):
            if run.xpath('.//w:drawing', namespaces=NS):
                assert not text(run)
                p.remove(run)
                picture.append(run)
        for flag in p.xpath('./w:pPr/w:pageBreakBefore', namespaces=NS):
            flag.getparent().remove(flag)
        p.addprevious(picture)
    # Condense repeated administration without removing known failure evidence.
    edit(s, 'Table S36.', 'Table S36. Scope of FlowPilot and related flow-chemistry/process-design agents. This comparison summarizes reported tasks and validation; it is not a performance ranking. Source numbers refer to the ESI bibliography.', 'Merge two overlapping literature comparison panels.')
    edit(s, 'This section compares FlowPilot with six studies', 'Table S36 summarizes the tasks, outputs and reported validation of six related flow-chemistry or process-design agents [7-12]. Journal articles, a conference paper and preprints are distinguished. Different published evaluations are not treated as a common performance benchmark.', 'Condense the related-work preamble to match the retained table.')
    edit(s, 'A common input-output description', 'The common scope description does not imply equal capabilities or benchmarks. FlowPilot focuses on reconciling reaction order, feed composition, stage calculations and available equipment in one design record. The matched architecture test compares the recorded one-shot and FlowPilot configurations, not the external systems in Table S36.', 'Remove repetitive qualifications and forward references.')
    edit(s, 'Table S8.', 'Table S8. Fixed 0–4 rating anchors applied by each LLM judge.', 'Use accurate judge terminology.')
    edit(s, 'Figure S4.', 'Figure S4. Engineering rule-base structure. (a) Counts by category and severity for the 14 largest categories. (b) Fraction of rules containing a machine-detected quantitative expression; this is expression coverage, not independent equation verification. (c) Non-exclusive rule-category associations across chemistry classes. (d) Computed concept co-occurrence network using the 12 most frequent concepts and 22 highest-weight links. Node area is 35 + 420 sqrt(f_i/f_max) points squared, where f_i is concept frequency; node color is the dominant rule category. Edge width is 0.25 + 2.4(w_ij/w_max) points, where w_ij is the rule-level co-occurrence count. The circular layout is for legibility.', 'Retain quantitative network definitions without a malformed inline equation.')
    for table in s.body.findall(W+'tbl'):
        for row in table.findall(W+'tr'):
            cells = row.findall(W+'tc')
            if cells and text(cells[0]) == 'Limiting molar flow':
                s.revise(cells[1].find(W+'p'), 'Molar feed = C × Q; M × mL min−1 = mmol min−1')
    edit(s, 'Source of the photochemical benchmark case:', 'Source of the photochemical benchmark case: Thomson et al., reference 2. Its supporting information includes Fmoc-L-methionine oxidation. The benchmark uses the frozen extracted input reproduced below; internally inconsistent reference-flow fields were excluded from numerical scoring.', 'Retain the source qualification without document-editing narration.')
    edit(s, 'This ESI distinguishes the historical benchmark configuration', 'Computational reconstruction requires the frozen inputs and inventory, exact model identifiers, generation parameters, software revision, raw model events, deterministic audit and final record. Historical liquid-volume, inlet-reference and pressure-corrected times retain their recorded definitions; later GUI defaults do not alter archived scores. Complete case inventories and the long worked-run transcripts are supplied in the companion archive.', 'Replace administrative replay table with concise reproducibility text.')
    edit(s, 'Council model matrix and candidate-budget sensitivity', 'Council configuration and candidate-budget sensitivity', 'Make the retained council analysis a stand-alone section.')
    edit(s, 'Candidate budgets B = 1, 6, 12, and 24', 'Candidate budgets B = 1, 6, 12 and 24 were evaluated with five repeats. Larger budgets increased design-family diversity and revision activity, but also disqualifications (Table S19). The normalized radar-area score was 0.31 ± 0.03 before and 0.48 ± 0.23 after council review (Table S20). Metric transformations are given in Table S21.', 'Remove superseded single-run rhetoric.')
    edit(s, 'This section complements the architecture schematics', 'Actual browser captures document protocol entry, answer saving and inventory validation on 9 September 2026. Result and council views reopen the archived DPDTC run. These are software-interaction examples, not additional benchmark outcomes or chemical experiments.', 'Remove unnecessary capture-process narration.')
    inv_caption = next(p for p in s.body if text(p).startswith('Table S24.'))
    inv_table = inv_caption.getnext()
    assert inv_table.tag == W+'tbl'
    s.body.replace(inv_table,s.table(['Inventory item','Stored information','Design implication'],[
        ['Degassing','No inline degasser; offline pre-degassing is allowed.','Omit the device, not the oxygen-exclusion requirement.'],
        ['Reactors','PFA: 2, 5 and 10 mL; two 5 mL/1.016 mm units and one 5 mL/0.762 mm unit. FEP: 10 and 20 mL.','Use declared volume, material and ID combinations.'],
        ['Temperature','Profile limits: PFA 80 °C; FEP 50 °C.','Equipment-specific limits, not universal material limits.'],
        ['Pumps','Three Chemyx Fusion 100 units; Vapourtec SF-10 and R2C+.','Check the selected pump and syringe range.'],
        ['Pressure control','Confirmed BPR settings: 2, 8 and 9 bar; maximum 10 bar.','Check the full flow path and pressure basis.'],
        ['Mixing','One connector; no dedicated mixer.','Generic T-mixer requires verification; a two-port union cannot merge feeds.'],
        ['Source caveats','Conflicting tubing units, missing radiant power and original BPR omission.','Retain uncertainty; a missing-value sentinel is not measured zero.'],
    ],[1.20,2.85,2.45]))
    edit(s, 'This section preserves the supplied batch protocols', 'The following batch protocols and fixed-ID response sets define the two laboratory-design cases. They are separate from the three architecture-benchmark cases in Section 3. Full run packages and all six archived topologies remain in the companion records; the implemented configurations, experimental procedures and yields are reported in Section 12.', 'Separate inputs from experimental evidence and avoid duplicate condition tables.')
    edit(s, 'Archived execution: outputs/figure5', 'Design records: outputs/figure5_pure_oxygen_physics_20260918, attempt_01 for each set. The routing was Claude Opus 4.6 upstream and Claude Sonnet 4.6 downstream, scientific_v2 policy, with 12 candidates. The source response sets specify integrated process performance, conversion and throughput, respectively; all require oxygen introduction only at Stage 2.', 'Retain reproducibility and response-set distinctions in one paragraph.')
    edit(s, 'Archived numerical source:', 'Design records: outputs/khu_revised_six_20260915/presentation/figure6_set1 through figure6_set3; collaborator export 20260915_163845. The routing was Claude Opus 4.6 upstream and Claude Sonnet 4.6 downstream, scientific_v2 policy, with 12 candidates. The response sets prioritize integrated performance, conversion and throughput, respectively.', 'Remove repeated source-package administration.')
    edit(s, 'The batch irradiation used two MR16', 'The batch source comprised two MR16 blue LEDs; flow R1 used the UV-150 450 nm LED and R2 a separate blue strip. Figure S23 combines the supplied photographs and emission spectra. The caption records electrical-input and radiant-power specifications separately; neither is a measured photon flux at the fluid.', 'Replace redundant lamp table and three separate pages with a single composite.')
    edit(s, 'The transformation converts (((4-methoxyphenyl)', 'The sequence converts (((4-methoxyphenyl)thio)methyl)trimethylsilane (1a) and acrylonitrile (2a) through sulfide 3a to sulfoxide 4a, following Park et al. [4]. Main-text Figure 5 shows the connected configuration; Figure S25 shows KHU’s apparatus.', 'Remove duplicate source-schematic reference.')
    edit(s, 'The sequence converts 3-methyl-4-nitrobenzoic acid', 'The sequence converts 3-methyl-4-nitrobenzoic acid (1b) through a 2-pyridyl thioester to N-benzyl-3-methyl-4-nitrobenzamide (3b), following Saunders et al. [5]. The intermediate is not isolated. Main-text Figure 6 shows the connected configuration; Figure S27 shows KHU’s apparatus.', 'Remove duplicate source-schematic reference.')
    p = next(p for p in s.body if text(p).startswith('An aliquot of the collected reactor effluent'))
    edit(s, 'An aliquot of the collected reactor effluent', text(p).replace('97% NMR yield and 94% isolated yield', '97% NMR yield'), 'Use NMR yields consistently in outcome summaries.')
    p = next(p for p in s.body if text(p).startswith('An aliquot of the collected effluent'))
    edit(s, 'An aliquot of the collected effluent', text(p).replace('86% NMR yield and 84% isolated yield (227.0 mg)', '86% NMR yield'), 'Use NMR yields consistently; retain isolation records only in characterization.')
    edit(s, 'The supplied reference tables also compare', 'The literature benchmarks in main-text Figures 5 and 6 use verified NMR yields. They provide context rather than controlled comparisons: feed concentrations, irradiation, reactor size, reagent addition and gas delivery differ. No causal yield advantage or intensification factor is inferred. For the gas–liquid stage, the inlet-reference index is not a measurement of operating-pressure residence time or phase holdup.', 'State a single measurement basis and appropriate comparison scope.')
    # Fix references to archived rather than retained legacy material in a failure example.
    for p in s.doc.iter(W+'p'):
        value = text(p)
        if 'worked example in Section 6' in value or '(Table S22)' in value:
            s.revise(p, value.replace('worked example in Section 6', 'archived worked-run transcript').replace('(Table S22)', '(companion archive)'))
        elif '(Table S3)' in value:
            s.revise(p, value.replace(' (Table S3)', ''))
    # Caption numbering is applied only after all old-label references are reconciled.
    return s


def build_main():
    m = Package(MAIN)
    p = next(p for p in m.body if text(p).startswith('Translating a batch protocol'))
    value = with_cites(p).replace('with isolated yields of 94% and 84% at selected operating points.', 'under the reported laboratory settings.')
    edit(m, 'Translating a batch protocol', value, 'Remove isolated-yield endpoints from the main outcome narrative.')
    p = next(p for p in m.body if text(p).startswith('Here we present FlowPilot,'))
    edit(m, 'Here we present FlowPilot,', with_cites(p).replace(' (Figure 5)', '').replace(' (Figure 6)', ''), 'Reserve first main-figure citations for sequential presentation in Results.')
    edit(m, 'The aggregate comparison is complemented',
         'Chemistry-specific scores are shown in Figure S7, criterion-level differences in Figure S8, and run-level critical flags in Figure S9. Figure S10 reports generation tokens, estimated costs and runtime. The score-per-cost values use the archived token-price schedule, not total deployment costs. Table S13 defines the numerical closure checks applied to stage and feed calculations.', 'Remove legacy heuristic comparisons and duplicate administrative references.')
    p = next(p for p in m.body if text(p).startswith('A separate council model-matrix'))
    value = with_cites(p)
    value = value[:value.index('Figure S13 and Table S22')].rstrip()
    edit(m, 'A separate council model-matrix', value, 'Keep the quantitative council study; move the legacy worked-output dump to the archive.')
    edit(m, 'The GUI examples connect these assessments',
         'The interface links these assessments to the chemist’s workflow. Figures S14–S15 show protocol entry and fixed-ID follow-up questions. Figure S16 and Table S24 document inventory review, while Figure S17 and Table S25 show stage-resolved output. Table S26 retains concrete one-shot and FlowPilot problems, including an evaluator false positive; Figure S18 and Table S27 show council disagreement and selection. These records make assumptions and unresolved requirements inspectable before laboratory implementation.', 'Retain meaningful GUI and failure evidence without exhaustive administrative tables.')
    p = next(p for p in m.body if text(p).startswith('The first laboratory case couples'))
    value = with_cites(p).replace('The supplied batch input specified 4 h under argon followed by 6 h in air; the distinct literature-comparison timings are retained in ESI Section 12.3.', 'The batch input specified 4 h under argon followed by 6 h in air. The NMR-yield comparison uses the published flow result, providing context for implementation rather than a controlled test of identical hardware.')
    edit(m, 'The first laboratory case couples', value, 'Connect the chemical challenge to the verified reference endpoint.')
    edit(m, 'KHU had previously reported backflow',
         'KHU initially observed backflow toward the first reactor with pure oxygen at 0.43 mL min−1 and liquid at 0.020 mL min−1. The revised point retained pure oxygen at 0.090 mL min−1. A gas-line check valve was included in the proposal, but its presence alone does not establish protection of the upstream liquid branch. The installed BPR was a 7 bar cartridge; recorded system pressures were 5.1–5.6 bar. Sets 1/3, reported together for the 2 mL first reactor, gave 97% yield; Set 2, with a 5 mL first reactor, gave 98%. These and the 90% literature flow reference are NMR yields.[[57]] The difference between KHU configurations is too small to attribute to residence time without replicates, and the comparison with literature is not matched for operating conditions. Table S31 records the KHU settings; Figures S23 and S25 show irradiation sources and the assembled apparatus.', 'Use one yield basis, retain the actual backflow history, and introduce ESI material in order.')
    edit(m, 'Set 1 afforded amide 3b',
         'Set 1 gave 86% yield. Sets 2/3 were reported together for a 10 mL first reactor and 5 mL second reactor at feed rates of 0.280 and 0.0700 mL min−1, giving stage times of 35.71 and 14.29 min and a yield of 68%. Both outcomes are NMR yields, as is the 98% literature batch reference plotted in Figure 6.[[58]] Relative to Set 1, the total nominal stage time decreased from 56.25 to 50.00 min while the nominal substrate feed increased from 0.0800 to 0.140 mmol min−1. The lower yield at higher flow reinforces the need to evaluate the connected sequence, but does not isolate a kinetic cause because several operating variables changed together. Table S34 and Figure S27 report the operating points and apparatus; Tables S38–S39 document implementation changes and component molar feeds.', 'Report comparable NMR endpoints and retain the less successful operating point.')
    p = next(p for p in m.body if text(p).startswith('An additional alpha-bromination design'))
    m.body.remove(p)
    for num in (5, 6):
        picture, cap = old.base.image_before_caption(m, num)
        m.replace_image(picture, OUT/f'figures/figure{num}_redesigned.png')
    edit(m, 'Figure 5.',
         'Figure 5. Photoredox Giese addition/oxidation from chemistry to laboratory implementation. (a) Precursor 1a is converted through sulfide 3a to sulfoxide 4a. (b) Connected equipment arrangement; oxygen is introduced after R1. Feed A contains 0.100 M 1a, 0.200 M acrylonitrile and 0.500 mM iridium photocatalyst. The process schematic is not a piping or safety-certification diagram. (c) KHU operating points and yield, with the literature flow reference from Park et al.[[57]] All plotted yields are determined by 1H NMR; grouped Sets 1/3 are one supplied result entry, not independent repeats. R1 time is V/Qliquid. The R2 value is the nominal inlet/STP index V/(Qliquid + Qgas,STP), not residence time under operating pressure. The MFC reference state requires confirmation. Literature and KHU experiments used different conditions.', 'Replace the cluttered four-panel table layout with chemistry, connected hardware and one yield comparison.')
    edit(m, 'Figure 6.',
         'Figure 6. DPDTC-mediated amidation from reaction sequence to connected operation. (a) Acid 1b is activated to a 2-pyridyl thioester and undergoes aminolysis to amide 3b. (b) Two heated ETFE coils receive acid/DPDTC/DMAP in Feed A (0.500/0.525/0.0500 M) and benzylamine in Feed B (2.10 M), added between the stages. (c) KHU operating points and yield, with the literature batch reference from Saunders et al.[[58]] All plotted yields are determined by 1H NMR. Sets 2/3 share one supplied result entry. Stage times use V1/QA and V2/(QA + QB); recorded pressures are distinct from the 7 bar BPR cartridge rating. The literature reference is contextual rather than a matched flow-control experiment.', 'Use a readable matched style and an analytically comparable reference endpoint.')
    replace_section(m, 'CONCLUSION', 'METHODS', [
        'FlowPilot addresses batch-to-flow translation by treating chemistry, stream composition, reactor operation and available equipment as parts of one connected design. Standardized intake makes the chemist’s objectives and constraints explicit; retrieval and specialist review supply chemical context; deterministic calculations reconcile the quantities that must agree. A shared final record links the resulting stream and stage conditions to equipment assignments and the process diagram, allowing the proposal to be inspected and revised rather than accepted solely as a fluent explanation.',
        'The computational and laboratory evaluations test complementary aspects of this approach. Across the retained five-model, three-chemistry comparison, FlowPilot achieved a mean fixed-criteria score of 0.917, compared with 0.794 for matched one-shot generation, and received fewer judge-reported critical flags. The module screen identifies the complete council as an important contributor within those cases, without establishing that every specialist or larger candidate budget always improves the result. The two laboratory implementations extend the assessment to connected chemistry: staged oxygen delivery supported Giese addition/oxidation at 97–98% yield, while activation followed by amidation gave 68–86% yield. These values were determined by NMR and represent the reported operating points, not optimized or replicated performance across a broad reaction space.',
        'The practical contribution is therefore a traceable starting point for experimental process development, not a replacement for it. The observed backflow and the yield change between amidation conditions show why apparatus behavior and complete-sequence performance must remain part of evaluation. Returning actual operating conditions and measured outcomes to the same design record provides a basis for subsequent refinement. Further prospective studies can test whether this design-and-feedback workflow reduces experimental effort across a broader range of chemistries and laboratory inventories.',
    ])
    # Remove references to retired administrative tables and map Methods to the compact ESI.
    for p in list(m.body):
        value = with_cites(p)
        updated = value.replace('Table S35', 'the companion version-history record')
        updated = updated.replace('The intake contract and question rules are given in ESI Section 1 and Tables S1–S3.', 'The intake contract and question rules are given in ESI Section 1 and Tables S1–S2.')
        updated = updated.replace('(Table S14)', '(Table S24)')
        updated = updated.replace('and Tables S15–S16 specify the feedback and provenance records.', 'with the machine-readable feedback and provenance records retained in the companion archive.')
        updated = updated.replace('ESI Sections 6.1 and 7.5', 'ESI Section 6.1 and Section 7.5')
        updated = updated.replace('Tables S1–S3', 'Tables S1–S2')
        updated = updated.replace('Tables S22 and S26', 'Table S26')
        if updated != value:
            m.revise(p, updated)
    return m


def renumber(pkg):
    for p in list(pkg.doc.iter(W+'p')):
        value = with_cites(p)
        updated = remap_sections(remap_labels(value))
        # A range containing the two same-parent subsections needs both prefixes changed.
        updated = updated.replace('Sections 2.4–1.5', 'Sections 2.4–2.5')
        if updated != value:
            pkg.revise(p, updated)


def normalize_captions(pkg):
    for p in pkg.body:
        value = text(p)
        match = re.match(r'^(Figure|Table) S\d+[a-z]?(?: \(continued\))?\.', value)
        if not match:
            continue
        # Rebuild only caption text; references in captions remain normal weight.
        pkg.revise(p, with_cites(p))
        if match[1] == 'Figure':
            pp = p.find(W+'pPr')
            if pp is None:
                pp = E.Element(W+'pPr')
                p.insert(0,pp)
            for flag in pp.findall(W+'keepNext'):
                pp.remove(flag)
            E.SubElement(pp,W+'keepNext',{W+'val':'0'})
            if pp.find(W+'keepLines') is None:
                E.SubElement(pp,W+'keepLines')
        consumed = 0
        for r in list(p.findall(W+'r')):
            rp = r.find(W+'rPr')
            for name in ('b', 'bCs', 'i', 'iCs'):
                element = rp.find(W+name)
                if element is None:
                    element = E.SubElement(rp,W+name)
                element.set(W+'val', '0')
            length = len(text(r))
            if consumed < match.end() and consumed + length <= match.end():
                rp.find(W+'b').set(W+'val','1')
            elif consumed == 0 and length > match.end():
                t = r.find(W+'t')
                if t is not None:
                    following = deepcopy(r)
                    following.find(W+'t').text = t.text[match.end():]
                    t.text = t.text[:match.end()]
                    rp.find(W+'b').set(W+'val','1')
                    r.addnext(following)
            consumed += length


def bibliography_main(m):
    reference_heading = old.base.find(m, 'REFERENCES')
    refs = list(m.body)[list(m.body).index(reference_heading)+1:]
    refs = [p for p in refs if text(p).strip()]
    assert len(refs) == 62
    assert text(refs[58]).startswith('Lu, Y.;')
    m.body.remove(refs[58])
    for p in list(m.body)[:list(m.body).index(reference_heading)]:
        for r in p.iter(W+'r'):
            if not r.xpath('./w:rPr/w:vertAlign[@w:val="superscript"]', namespaces=NS):
                continue
            v = text(r)
            if not re.fullmatch(r'\d+(?:[,–-]\d+)*',v):
                continue
            numbers = []
            for part in v.split(','):
                edges = re.split('[–-]', part)
                numbers.extend(range(int(edges[0]),int(edges[-1])+1))
            assert 59 not in numbers, ('Removed source still cited', text(p))
            numbers = [x-1 if x>59 else x for x in numbers]
            new = compressed_cites(numbers)
            r.find(W+'t').text = new


def bibliography_esi(s):
    heading = next(p for p in s.body if text(p) == 'References')
    index = list(s.body).index(heading)
    paragraphs = list(s.body)[:index]
    references = {int(re.match(r'^(\d+)\.', text(p))[1]): p
                  for p in list(s.body)[index+1:] if re.match(r'^\d+\.', text(p))}
    pattern = re.compile(r'\[([0-9]+(?:[,–-][0-9]+)*)\]|\breference ([0-9]+)\b')
    ordered = []
    for node in paragraphs:
        for match in pattern.finditer(text(node)):
            for number in expanded_cites(match[1] or match[2]):
                assert number in references
                if number not in ordered:
                    ordered.append(number)
    mapping = {number: i+1 for i, number in enumerate(ordered)}
    for node in paragraphs:
        for p in ([node] if node.tag == W+'p' else node.iter(W+'p')):
            value = with_cites(p)
            def replacement(match):
                mapped = compressed_cites([mapping[n] for n in expanded_cites(match[1] or match[2])])
                return '[' + mapped + ']' if match[1] else 'reference ' + mapped
            updated = pattern.sub(replacement, value)
            if value != updated:
                s.revise(p, updated)
    for p in references.values():
        s.body.remove(p)
    previous = heading
    for number in ordered:
        p = references[number]
        s.revise(p, re.sub(r'^\d+\.', str(mapping[number])+'.', text(p)))
        previous.addnext(p)
        previous = p
    (OUT/'esi_bibliography_map.json').write_text(json.dumps(mapping, indent=2)+'\n')


def pagination(pkg, stem):
    def props(p):
        pp=p.find(W+'pPr')
        if pp is None:
            pp=E.Element(W+'pPr');p.insert(0,pp)
        return pp
    if stem=='manuscript':
        # Keep each explanation together before its full-page figure, without
        # changing artwork, picture dimensions, fonts or page geometry.
        groups={
            1:['The downstream council reviews alternatives','The selected proposal is reconciled'],
            2:['The engineering knowledge base contained'],
            3:['In the two query-enrichment examples','In the family-alignment analysis'],
        }
        for number,prefixes in groups.items():
            picture,_=old.base.image_before_caption(pkg,number)
            for prefix in prefixes:
                p=next(p for p in pkg.body if text(p).startswith(prefix))
                pkg.body.remove(p);picture.addprevious(p)
    else:
        for number in [17,18,22,23,25]:
            cap=next(p for p in pkg.body if text(p).startswith(f'Table S{number}.'))
            table=cap.getnext()
            assert table.tag==W+'tbl'
            paragraphs=list(table.iter(W+'p'))
            for i,p in enumerate(paragraphs):
                pp=props(p)
                for flag in pp.findall(W+'keepNext'):pp.remove(flag)
                E.SubElement(pp,W+'keepNext',{W+'val':'1' if i<len(paragraphs)-1 else '0'})
    title='REFERENCES' if stem=='manuscript' else 'References'
    ref=next(p for p in pkg.body if text(p)==title)
    for p in list(pkg.body)[list(pkg.body).index(ref)+1:]:
        if p.tag!=W+'p' or not text(p).strip():continue
        pp=props(p)
        if pp.find(W+'keepLines') is None:E.SubElement(pp,W+'keepLines')
        if stem=='esi':
            spacing=pp.find(W+'spacing')
            if spacing is None:spacing=E.SubElement(pp,W+'spacing')
            spacing.set(W+'after','40')


def save(pkg, stem):
    science_typography(pkg)
    old.base.prior.bold_references(pkg.doc)
    if stem == 'esi':
        style_tables(pkg.doc)
        normalize_captions(pkg)
    pagination(pkg,stem)
    order_properties(pkg.doc)
    for suffix, marked in [('',False),('_marked',True)]:
        pkg.save(BASE / f'{stem}_submission_overhauled_20260930{suffix}.docx',marked=marked)


def main():
    OUT.mkdir(exist_ok=True)
    inputs = {str(p.relative_to(ROOT)): sha256(p.read_bytes()).hexdigest() for p in (MAIN,ESI)}
    s,m = build_esi(),build_main()
    renumber(s);renumber(m)
    bibliography_main(m);bibliography_esi(s)
    save(s,'esi');save(m,'manuscript')
    mapping = [{'kind':k,'old':f'S{oldn}','new':f'S{new}'} for k,mp in MAPS.items() for oldn,new in mp.items()]
    with (OUT/'numbering_map.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['kind','old','new']);writer.writeheader();writer.writerows(mapping)
    removed = {'figures':[f'S{i}' for i in range(1,32) if i not in FIG_KEEP], 'tables':[f'S{i}' for i in range(1,40) if i not in TABLE_KEEP], 'archive':'companion_archive/esi_before_streamlining.docx'}
    (OUT/'removed_content_manifest.json').write_text(json.dumps(removed,indent=2)+'\n')
    (OUT/'source_manifest.json').write_text(json.dumps(inputs,indent=2)+'\n')
    (OUT/'editorial_changes.json').write_text(json.dumps(log,indent=2,ensure_ascii=False)+'\n')
    assert all(sha256((ROOT/k).read_bytes()).hexdigest()==v for k,v in inputs.items())
    print('Saved overhauled manuscript and ESI with',len(FIG_KEEP),'supplementary figures and',len(TABLE_KEEP),'tables.')


if __name__ == '__main__':
    main()
