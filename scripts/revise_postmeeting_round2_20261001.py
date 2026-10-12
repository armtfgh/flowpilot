"""Second evidence-led review; preserve original figures and archived observations."""
from copy import deepcopy
from hashlib import sha256
from pathlib import Path
import csv
import json
import re

from lxml import etree as E
from revise_manuscript_cases_20260922 import Package, W, NS, text
from overhaul_submission_20260930 import with_cites
from revise_manuscript_20260930 import paragraph_like
from revise_postmeeting_20261001 import restrained_bold, table
from revise_layout_discussion_20260928 import order_properties

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'manuscript'
OUT = BASE / 'october1_postmeeting_round2_work'
CHANGES = []


def change(pkg, node, value, reason):
    CHANGES.append({'document': pkg.source.name, 'reason': reason,
                    'before': with_cites(node), 'after': value})
    pkg.revise(node, value)


def add(pkg, anchor, value, reason):
    node = paragraph_like(pkg, anchor, value)
    pkg.after(anchor, [node])
    CHANGES.append({'document': pkg.source.name, 'reason': reason, 'added': value})
    return node


def replace_table(pkg, old, headers, rows, widths, reason):
    new = table(pkg, headers, rows, widths)
    pkg.body.replace(old, new)
    pkg.changed.append(new)
    CHANGES.append({'document': pkg.source.name, 'reason': reason,
                    'table_before': text(old), 'headers': headers, 'rows': rows})
    return new


def scientific_subscripts(node):
    tokens = ['Qg,STP', 'Qg,ch', 'Vm,STP', 'TSTP', 'PSTP', 'Pabs', 'Tch',
              'yactive', 'Clim', 'Qlim', 'Awall', 'QL', 'VL']
    pattern = re.compile(r'(?<![A-Za-z])(' + '|'.join(tokens) + r')(?![A-Za-z])')
    for p in ([node] if node.tag == W+'p' else node.iter(W+'p')):
        for run in list(p.findall(W+'r')):
            value = text(run)
            if not pattern.search(value):continue
            if len(run.findall(W+'t')) != 1:continue
            anchor = run
            for piece in pattern.split(value):
                if not piece:continue
                pieces = [(piece[0], False), (piece[1:], True)] if piece in tokens else [(piece, False)]
                for content, sub in pieces:
                    r = deepcopy(run)
                    r.find(W+'t').text = content
                    r.find(W+'t').set('{http://www.w3.org/XML/1998/namespace}space','preserve')
                    if sub:
                        rp = r.find(W+'rPr')
                        if rp is None:rp=E.SubElement(r,W+'rPr')
                        E.SubElement(rp,W+'vertAlign',{W+'val':'subscript'})
                    anchor.addnext(r);anchor=r
            p.remove(run)


def frozen_inputs():
    cohorts = ['manuscript_three_model_three_repeat_postfix_20260820',
               'qwen38_three_repeat_20260824', 'alternative_frontier_three_repeat_20260824']
    raw = OUT / 'frozen_benchmark_inputs'
    raw.mkdir(exist_ok=True)
    hashes = {}
    sources = []
    cases = None
    for cohort in cohorts:
        path = ROOT / 'ablation_results/manuscript_benchmark' / cohort / 'frozen/case_manifest.json'
        source = json.loads(path.read_text())
        sources.append({'path': str(path.relative_to(ROOT)), 'sha256': sha256(path.read_bytes()).hexdigest()})
        for case in source['cases']:
            task = {k: case[k] for k in ['protocol', 'objective', 'hard_constraints', 'inventory']}
            digest = sha256(json.dumps(task, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode()).hexdigest()
            if case['case_id'] in hashes:
                assert hashes[case['case_id']] == digest, 'Different task inputs across retained cohorts'
            hashes[case['case_id']] = digest
        if cases is None:
            cases = source['cases']
    files = []
    for case in cases:
        for suffix, value in [('inventory.json', case['inventory']), ('case.json', case)]:
            p = raw / f"{case['case_id']}_{suffix}"
            p.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
            files.append({'file': p.name, 'sha256': sha256(p.read_bytes()).hexdigest()})
        p = raw / f"{case['case_id']}_design_input.txt"
        p.write_text(case['design_input_text'])
        files.append({'file': p.name, 'sha256': sha256(p.read_bytes()).hexdigest()})
    manifest = {'sources': sources, 'canonical_task_sha256': hashes, 'exports': files,
                'note': 'Unmodified archived values; no new generation or re-scoring. These are benchmark inventories, not the wet-lab KHU profile.'}
    (raw / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    return cases


def main_text(pkg):
    n = list(pkg.body)
    change(pkg, n[24], with_cites(n[24]).replace(
        'The illustrated recipe card is schematic; computational validation does not establish experimental performance.',
        'The illustrated recipe card and candidate/revision budgets are schematic, not common settings for every campaign. The audited rule-base export derives from six source texts (ESI Section 3); the approximate handbook count in the overview is illustrative. Computational validation denotes consistency checking, not experimental performance or authorization to operate.'),
        'Separate illustrative architecture labels from audited counts and operating authorization.')
    change(pkg, n[38], with_cites(n[38]).replace(
        'The cases therefore act as a controlled stress test of connected design,',
        'Selected after pilot development, the cases act as a retrospective stress test of connected design,').replace(
        'The original prompts and source references in ESI Section 4 make these boundaries explicit.',
        'ESI Section 4 retains the original prompts, source references and case-specific inventory limits, including ambiguities in the frozen inputs.'),
        'Disclose case selection and avoid implying a prospective unseen test.')
    change(pkg, n[45], with_cites(n[45]).replace(
        '98 flags across 45 one-shot outcomes and 4 across 45 FlowPilot outcomes',
        '98 judge-level critical flags across 45 one-shot outcomes and 4 across 45 FlowPilot outcomes'),
        'Define counted unit consistently with the evidence register.')
    change(pkg, n[48], with_cites(n[48]).replace(
        'Thus splitting the task into an initial sequence of agent calls was not sufficient: the benefit depended on reviewing whether the chemical and engineering choices formed a coherent process.',
        'In this screen, the initial sequence of agent calls without council review did not recover the quality of the complete configuration. This is consistent with a benefit from reviewing whether chemical and engineering choices form a coherent process, although one generation per case cannot isolate that mechanism from generation variability.'),
        'Separate descriptive ablation result from a causal mechanism claim.')
    change(pkg, n[50], with_cites(n[50]).replace(
        'the categorized inventory and constraints (Figure S11; Table S13).',
        'and the categorized inventory and constraints (Figure S11; Table S13).'),
        'Repair sentence coordination.')
    change(pkg, n[62], with_cites(n[62]).replace(
        'Its lower final yield despite the longer first-stage time illustrates why upstream residence time or total process time alone is an inadequate design objective.',
        'The lower yield at this operating point, despite its longer nominal first-stage time, shows that neither upstream exposure nor total nominal time alone explains the outcome.'),
        'Keep interpretation tied to the two measured, confounded operating points.')
    change(pkg, n[76], with_cites(n[76]).replace(
        'Follow-up questions use stable identifiers and predefined readiness rules; optional evidence can be explicitly marked unavailable.',
        'Deterministic protocol triggers activate questions from a versioned bank of stable identifiers, and confirmed package state determines which remain pending. The LLM extracts and interprets answers but cannot introduce new identifiers or override readiness. Information such as historical evidence can be explicitly marked unavailable; conditional gas, irradiation and stage questions follow the resolution rules in Table S3.'),
        'Align intake methods with implemented deterministic activation and conditional readiness.')
    change(pkg, n[83], with_cites(n[83]).replace(
        'The inlet-reference index V/(Qliquid + Qgas,STP) was kept separate from pressure-corrected apparent time and from an experimentally measured residence time.',
        'The inlet-reference index V/(Qliquid + Qgas,STP) was kept separate from pressure-corrected nominal time and from measured residence time. Ideal-gas correction alone does not determine liquid holdup or residence-time distribution; a no-slip estimate additionally neglects phase slip, gas dissolution or consumption, and axial pressure variation.'),
        'State physical assumptions rather than equating pressure correction with measured contact time.')
    change(pkg, n[90], with_cites(n[90]).replace(
        'Exact case prompts, inventories, and model records are retained in ESI Section 4.',
        'ESI Section 4 describes the case prompts, inventory limits and model records; the accompanying frozen-input files retain the exact JSON and complete design-input strings.'),
        'Distinguish readable ESI summary from exact machine-readable inputs.')
    change(pkg, n[96], with_cites(n[96]).replace(
        'Both setups used a 7 bar cartridge BPR;',
        'Reported concentrations and molar feeds describe the nominal sample solution, not a measured concentration profile throughout the dispersed injection. Both setups used a 7 bar cartridge BPR;'),
        'Make finite-loop implementation explicit in experimental methods.')


def esi_text(pkg):
    n = list(pkg.body)
    change(pkg, n[39],
        'The intake layer converts the raw protocol and confirmed answers into a frozen DesignInputPackage. The question bank (flowpilot_questions_v1.1) contains eight core identifiers and five conditional identifiers (Table S3). The protocol activates conditional questions through deterministic gas, photochemistry and multistep triggers; the confirmed package state determines which answers are still missing. The LLM extracts information and interprets responses but cannot invent an identifier or override readiness. Protocol, objective and chemistry identity must be established; history, inventory, operating limits and hypotheses require either supplied information or an explicit unavailable status. Output preference is optional. Active conditional questions have their own resolution rules: unavailable is allowed for a gas-equivalent target or wavelength, but not for unresolved gas identity, gas-introduction stage or stage definition. Allowing an unavailable answer records uncertainty; it does not authorize unsupported hardware or safe operation. The saved package retains effective answers and superseded history, question-bank version, active IDs, question-set hash, engineering requirements and provenance. For unchanged protocol, confirmed state and question-bank version, the ordered pending IDs are reproducible; extracted values, generated designs and conversational phrasing are not thereby guaranteed to be identical.',
        'Replace outdated LLM-selected question description and define conditional readiness.')
    questions = [
        ['Q-BATCH-001\nBatch protocol', 'Core; required', 'Confirmed protocol', 'Chemical facts and batch conditions'],
        ['Q-OBJ-001\nDesign objective', 'Core; required', 'Answered', 'Design priorities and candidate ranking'],
        ['Q-CHEM-001\nChemistry identity', 'Core; required', 'Confirmed transformation', 'Substrate/product identity and bond changes'],
        ['Q-HIST-001\nHistory', 'Core; required resolution', 'Supplied or unavailable', 'Measured evidence for refinement'],
        ['Q-INV-001\nInventory', 'Core; required resolution', 'Supplied or unavailable', 'Equipment and capability constraints'],
        ['Q-CONSTR-001\nOperating limits', 'Core; required resolution', 'Supplied or unavailable', 'Operating and safety boundaries'],
        ['Q-HYP-001\nHypotheses', 'Core; required resolution', 'Supplied or unavailable', 'Ideas to test, not measured facts'],
        ['Q-PREF-001\nOutput preference', 'Core; optional', 'Answered or omitted', 'Requested presentation and screening format'],
        ['Q-GAS-001\nGas identity', 'Gas domain; identity missing', 'Gas and active fraction established', 'Reactive-gas molar delivery'],
        ['Q-GAS-002\nGas equivalents', 'Gas domain; positive target missing', 'Positive target or unavailable', 'Inlet/STP feed target; unknown remains provisional'],
        ['Q-GAS-003\nGas-introduction stage', 'Multistep gas domain; stage missing', 'Introduction stage established', 'Gas mixer placement and downstream accounting'],
        ['Q-PHOTO-001\nWavelength', 'Photochemistry; wavelength missing', 'Wavelength or unavailable', 'Light matching; unknown remains provisional'],
        ['Q-MULTI-001\nOrdered stages', 'Multistep domain; definition missing', 'Ordered stages and interstage operations', 'Stage boundaries and cumulative flows'],
    ]
    question_table = replace_table(pkg, n[43], ['Question ID / information', 'Activation / readiness', 'Resolution', 'Design use'], questions,
                                   [1.35, 1.65, 1.65, 1.85], 'Restore all fixed IDs and conditional question rules.')
    for spacing in question_table.findall('.//'+W+'spacing'):
        spacing.set(W+'before','40')
        spacing.set(W+'after','40')
    cell = n[64].findall(W+'tr')[5].findall(W+'tc')[3].find(W+'p')
    change(pkg, cell, '1 stream: 22 (14.3%); 2: 91 (59.1%); 3: 30 (19.5%); 4: 5 (3.2%); 5: 3 (1.9%); 6, 10 and 12: 1 each (0.6% each)',
           'Complete the inlet-stream distribution and fix singular grammar.')
    change(pkg, n[84],
        'This is a retrospective, development-stage comparison on three selected chemistry cases, not a prospective test on unseen reactions. The frozen campaign record states that cases were selected after a five-case pilot. The same case inputs were subsequently used for both architectures; that pairing controls supplied task information, not case-selection bias or development exposure. Scores describe assessed design quality for these tasks, not experimentally measured robustness across flow chemistry.',
        'Disclose pilot-based selection and delimit generalization.')
    change(pkg, n[95],
        'The paired unit was the same generator model, chemistry case and repeat index under the two architectures. Three generation repeats per case were retained, with a recorded temperature of 0.2. The campaign manifests and later repeat-extension records identify their provenance; repeat indexing does not imply shared random draws or statistically independent evaluator judgments. The designated source record was excluded from final retrieval selection. This controls direct use of that record by RAG but does not establish absence of related information from the corpus or model pretraining.',
        'Remove unsupported prospective stopping-rule language and clarify pairing.')
    change(pkg, n[106], with_cites(n[106]).replace(
        'All internal prompts and stage events were retained in the archived run directories.',
        'The archived run directories retain generation outputs, available prompt/stage records, evaluator packets and judgments; the common input is preserved separately from architecture-specific traces.'),
        'Avoid an unaudited claim of complete per-stage prompt retention.')
    change(pkg, n[123], with_cites(n[123]) +
        ' In these three cases, UO-09 is inapplicable because there is only one reactive stage; gas mixing or product separation alone does not create a second reactive stage. UO-08 is inapplicable to CuAAC but applies to the two gas-using cases. Their omission from the cross-chemistry gain display does not remove applicable UO-08 ratings from the overall score.',
        'Explain precisely why UO-08 and UO-09 differ in applicability.')
    # Readable limits accompany the unmodified JSON rather than replacing it.
    add(pkg, n[152],
        'Frozen inventory summary. The hydrogenolysis task supplies one plunger pump (0.1-2.0 mL min\u22121; 40 bar limit), a stainless-steel packed bed (3.0 mL; 4.35 mm ID; 200 mm declared tube length), a 60 \u00b0C operating temperature and 21 bar BPR setpoint. H2 and N2 supplies, an H2 MFC (1-100 mL min\u22121 at its standard reference), a T-mixer, heater, packed-bed connectors, gas\u2013liquid separator, vented collector, H2 check valve and safe vent are declared. No light source or degasser is supplied. The volume is nominal bed volume, not measured liquid holdup.',
        'Supply the previously missing benchmark inventory next to the hydrogenolysis prompt.')
    add(pkg, n[172],
        'Frozen inventory summary. The oxidation task supplies one HPLC pump (0.01-1.0 mL min\u22121), a 1.0 mL PFA coil (1.0 mm ID; 1.273 m tubing), a 420 nm, 1.5 W light source, temperatures of 21 or 25 \u00b0C and a 3 bar BPR setpoint. A dry-air supply and MFC (0.1-10 mL min\u22121 at its standard reference), T-mixer, thermostat, connectors, gas\u2013liquid separator, vented collector and interlocked enclosure/vent are supplied; no degasser is available. The coil and pump are rated to 10 bar. The frozen instruction calls V/QL a liquid contact time; physically this is a nominal liquid-feed space time, not a measured phase-contact time. Its original wording is retained for reproducibility, without implying known liquid holdup.',
        'Supply oxidation inventory and expose the time-basis ambiguity without silently changing the test.')
    anchor = add(pkg, n[189],
        'Frozen inventory summary. The CuAAC task supplies an X-Cube HPLC pump (0.1-3.0 mL min\u22121; 40 bar limit), a stainless-steel Cu/C cartridge with declared volume 0.30 mL and ID 4.0 mm, a heater with a 200 \u00b0C upper limit, a 20 bar BPR, cartridge connector, vented collector and azide-service shield. Gas, light, mixer and degasser lists are empty. The separate cartridge-body record gives a length of 60 mm: its cylindrical volume is approximately 0.754 mL, not 0.30 mL. The inventory does not identify whether 0.30 mL denotes accessible liquid volume, packed volume or a different active section. These two values are retained as an unresolved input-basis ambiguity, not reconciled by inventing a porosity. The protocol\u2019s phrase \u201chomogeneous liquid reaction mixture\u201d also does not mean that the Cu/C catalyst is dissolved; the hard constraint requires catalyst retention in the packed bed.',
        'Supply CuAAC inventory and disclose contradictory nominal geometry.')
    add(pkg, anchor,
        'Machine-readable benchmark inputs. The companion frozen_benchmark_inputs folder contains each complete case, authoritative inventory JSON and exact design-input string, named with hydrogenolysis_oprd_9b00416, aerobic_oxidation_fmoc_methionine or cuaac_adsc_200900726, together with SHA-256 hashes and source paths in manifest.json. Canonical hashes of protocol, objective, hard constraints and inventory agree across the three retained campaign sources supplying the five generators. These inventories define the computational tasks and are not the laboratory inventory used for Figures 5 and 6. The source manifest marks hydrogenolysis and CuAAC as page-verified and the photochemical case as machine-extracted/unverified; the latter designation is retained rather than promoted to verified ground truth. Original inputs and scores have not been changed during manuscript review.',
        'Make exact archive provenance available and preserve the source-verification boundary.')
    # Remove an architecture-specific flag, not any benchmark score or outcome.
    rows = [[text(c) for c in tr.findall(W+'tc')] for tr in n[193].findall(W+'tr')]
    column = next(i for i, v in enumerate(rows[0]) if v.startswith('Exec'))
    rows = [[v for i,v in enumerate(r) if i != column] for r in rows]
    widths = [2.15] + [4.35/(len(rows[0])-1)] * (len(rows[0])-1)
    replace_table(pkg, n[193], rows[0], rows[1:], widths,
                  'Remove the architecture-specific executable-contract flag from the comparative table; retain every condition and score.')
    change(pkg, n[191], with_cites(n[191]) + ' DFMEA denotes design failure mode and effects analysis.',
        'Define the abbreviated ablation condition.')
    change(pkg, n[192],
        'Table S9. All 15 Qwen3.8-27B module-screen conditions. Score means and sample SD summarize three different cases, with one generation per condition-case pair. Calls are mean generation LLM calls per case; flags are total judge reports across the three cases. The architecture-specific final-contract execution flag is not used as a comparative outcome; it remains in the raw archive.',
        'Match caption to the neutral comparison table.')
    change(pkg, n[194],
        'The full configuration scored 0.928, compared with 0.670 without the council and 0.743 for one-shot generation in this screen. Each individual domain-reviewer removal had a lower mean than the full configuration (Chemistry 0.902; Fluidics 0.913; Safety 0.915; Kinetics 0.924). Conversely, one-pass review (0.932) and omission of preselection refinement (0.935) slightly exceeded it. The screen therefore supports examining coordinated review rather than assuming that every added step improves quality. Between-case SD and one generation per condition-case pair cannot establish a causal ranking of reviewers or distinguish small effects from generation variability.',
        'Correct the interpretation against all actual ablation means.')
    change(pkg, n[207],
        'Figure S7. Mean paired FlowPilot-minus-one-shot score difference by criterion and chemistry; positive values favor FlowPilot. UO-08 is omitted from this cross-chemistry display because gas bookkeeping applies to oxidation and hydrogenolysis but not CuAAC. UO-09 is omitted because none of these cases has more than one reactive stage. Applicable gas-bookkeeping scores remain included in the overall benchmark score.',
        'Replace inaccurate general explanation of the omitted criteria.')
    change(pkg, n[209],
        'Figure S8. Critical flags for (a) CuAAC, (b) photochemical oxidation and (c) hydrogenolysis. Each model has adjacent one-shot (OS) and FlowPilot (FP) blocks with repeats 1-3. Rows identify criteria; cell values count judges flagging an issue, not unique physical errors. Pale cells have no flag; gray cells with a dash are inapplicable. The display retains 90 outcomes and 990 cells across 11 criteria. UO-01 and UO-14 had no critical flags and are not plotted; UO-09 is inapplicable throughout. All 98 one-shot and 4 FlowPilot judge flags are retained. No flag is not evidence of experimental success.',
        'Explain the 11-row display and reconcile it with the 14-criterion rubric and total flags.')
    change(pkg, n[216],
        'Liquid component molar delivery is C \u00d7 Q. In a fully liquid-filled, unpacked reactor, V/QL is nominal liquid space time. Gas molar delivery uses the stated reference temperature and pressure; reactive-gas equivalents additionally include the active mole fraction. The inlet-reference index V/(QL + Qg,STP) is a bookkeeping index, not a physical residence time under operating pressure. Ideal-gas scaling gives Qg,ch = Qg,STP(Tch/TSTP)(PSTP/Pabs), with absolute temperature and pressure. A time formed using this local gas volume remains a nominal homogeneous-flow estimate: it assumes negligible phase slip and neglects dissolution, gas consumption and axial variations unless separately modeled. The ratio Qg,ch/(QL + Qg,ch) is a no-slip volumetric gas fraction, not measured holdup. For a packed bed or gas\u2013liquid reactor, liquid contact time instead requires accessible liquid volume or holdup, with residence-time-distribution measurements needed to characterize dispersion. A standard-volume gas feed and a compressed volume must never be substituted into the same equivalent calculation without converting their reference states.',
        'Separate nominal indices, no-slip approximations and physical phase-contact measurements.')
    closure_rows = [
        ['Component molar feed', '\u1e45 = C Q; M \u00d7 mL min\u22121 = mmol min\u22121', 'Component concentration, stream flow and units agree'],
        ['Liquid space time / contact time', 'V/QL for a filled liquid coil; VL/QL for known liquid holdup', 'State whether V is total, empty-bed, accessible or liquid volume'],
        ['Inlet/STP time index', 'V/(QL + Qg,STP)', 'Label as a reference-volume index, not operating residence time'],
        ['Local gas volume', 'Qg,ch = Qg,STP(Tch/TSTP)(PSTP/Pabs)', 'Absolute temperature and pressure; stated ideal-gas assumption'],
        ['Reactive-gas equivalents', 'eq = yactive (Qg,STP/Vm,STP)/(Clim Qlim)', 'Use consistent molar units; Vm,STP = 22.414 mL mmol\u22121 at 273.15 K, 1 atm'],
        ['No-slip gas-fraction estimate', 'Qg,ch/(QL + Qg,ch)', 'Homogeneous-flow approximation only; not measured holdup'],
        ['Multistage time total', 'Sum of stage times on a common basis', 'Use cumulative local flow; do not call mixed-basis sums physical residence time'],
        ['Pressure drop', 'Single- or two-phase relation for the declared regime', 'Check total operating pressure against each component rating; a BPR is not pressure relief'],
        ['Thermal screening', 'UA = U Awall; heat-generation/removal comparison', 'Report assumed heat-transfer and enthalpy inputs; no calorimetric validation implied'],
    ]
    closure_table = replace_table(pkg, n[218], ['Quantity', 'Definition / basis', 'Closure or qualification'], closure_rows,
                                  [1.45,2.40,2.65], 'Correct molar-flow notation and the physical meaning of holdup/time checks.')
    scientific_subscripts(closure_table)
    scientific_subscripts(n[216])
    for idx in [222,224,227,229]:
        pp = n[idx].find(W+'pPr')
        st = pp.find(W+'pStyle')
        if st is None: st = E.SubElement(pp,W+'pStyle')
        st.set(W+'val','Heading2')
        for run in n[idx].findall(W+'r'):
            rp = run.find(W+'rPr')
            if rp is None:rp=E.SubElement(run,W+'rPr')
            pkg.font(rp)
        pkg.changed.append(n[idx])
    CHANGES.append({'document':pkg.source.name,'reason':'Restore engineering subsection hierarchy and include 5.3-5.6 in contents.'})
    change(pkg, n[351], with_cites(n[351]).replace('[11-12]', '[12-13]'),
        'Repair the stale characterization reference range after the bromination citation was inserted.')
    for idx in [40,41,202,203]:
        assert not text(n[idx]).strip() and not n[idx].xpath('.//w:drawing',namespaces=NS)
        pkg.body.remove(n[idx])
    for idx in [191,194,198]:
        pp = n[idx].find(W+'pPr')
        spacing = pp.find(W+'spacing')
        if spacing is None:spacing=E.SubElement(pp,W+'spacing')
        spacing.set(W+'after','120')
    CHANGES.append({'document':pkg.source.name,'reason':'Remove empty intake/benchmark spacers and excessive local paragraph gaps so the question table and small aggregate table do not leave orphan rows/pages.'})
    cap = next(p for p in pkg.body if text(p).startswith('Figure S19.'))
    image = cap.getprevious()
    for br in image.xpath('./w:pPr/w:pageBreakBefore',namespaces=NS):br.getparent().remove(br)
    CHANGES.append({'document':pkg.source.name,'reason':'Allow the irradiation figure to follow its short introduction instead of forcing a nearly empty introductory page.'})


def main():
    OUT.mkdir(exist_ok=True)
    cases = frozen_inputs()
    # Keep exported filenames honest if an archived identifier differs from a shorthand.
    assert cases[1]['case_id'] == 'aerobic_oxidation_fmoc_methionine'
    for name, method in [('manuscript',main_text),('ESI',esi_text)]:
        pkg = Package(BASE / f'FlowPilot_{name}_october1_postmeeting.docx')
        method(pkg)
        restrained_bold(pkg)
        for h in list(pkg.doc.iter(W+'highlight')): h.getparent().remove(h)
        order_properties(pkg.doc)
        target = BASE / f'FlowPilot_{name}_october1_postmeeting_round2.docx'
        pkg.save(target)
        (OUT/'marked').mkdir(exist_ok=True)
        pkg.save(OUT/'marked'/f'FlowPilot_{name}_october1_postmeeting_round2_marked.docx',marked=True)
        print(target)
    (OUT/'text_change_log.json').write_text(json.dumps(CHANGES,indent=2,ensure_ascii=False))


if __name__ == '__main__':main()
