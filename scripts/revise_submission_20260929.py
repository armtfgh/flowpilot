"""Apply the collaborator's submission revisions to the user's authoritative files.

Original packages are immutable inputs. Unchanged XML/media are preserved; numerical
results come from archived CSVs or KHU's supplied experimental document.
"""
from pathlib import Path
from copy import deepcopy
from zipfile import ZipFile
from hashlib import sha256
import csv
import json
import re
from lxml import etree as E

import revise_prior_work_20260928 as prior
from revise_layout_discussion_20260928 import style_tables, order_properties, setting

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'manuscript/Submission'
OUT=BASE/'revision_20260929'
FIG=OUT/'figures'
RAW=OUT/'source_data'
W,NS=prior.W,dict(prior.NS)
NS.update(a='http://schemas.openxmlformats.org/drawingml/2006/main',wp='http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing')
Package=prior.previous.Package
text=prior.text

AUTHORS='Amirreza Mottafegh[[a,†]], Mincheol Park[[b,†]], Jun-young Jo[[b]], Jin Suk Myung[[a]], Boyoung Y. Park[[b,*]], and Gwang-Noh Ahn[[a,*]]'
AFFIL_A='a Digital Chemical Research Center, Korea Research Institute of Chemical Technology, 141 Gajeong-ro, Yuseong-gu, Daejeon 34114, Republic of Korea'
AFFIL_B='b Department of Biomedical and Pharmaceutical Sciences, Kyung Hee University, Seoul 02447, South Korea'
EQUAL='†Amirreza Mottafegh and Mincheol Park contributed equally to this work.'
CORR_PARK='*Boyoung Y. Park, E-mail: boyoungy.park@khu.ac.kr'
CORR_AHN='*Gwang-Noh Ahn, E-mail: gnahn@krict.re.kr'
AVAIL=('Source code is publicly accessible at https://github.com/armtfgh/flowpilot. '
       'Benchmark scripts and the data-access index are available at '
       'https://github.com/armtfgh/flowpilot/tree/master/ablation_test and '
       'https://github.com/armtfgh/flowpilot/tree/master/ablation_results, respectively. '
       'The latter is currently an index, not the complete outcome archive. The ESI contains '
       'the reported numerical summaries, experimental procedures, and characterization spectra. '
       'As of 29 September 2026, the repository is public but has no tagged manuscript release; '
       'the complete benchmark archive, raw experimental analytical files, and a persistent data '
       'DOI have not yet been deposited. These release items remain to be completed before submission.')


def replace_table(pkg,old,headers,rows,widths=None):
    new=pkg.table(headers,rows,widths=widths)
    pkg.body.replace(old,new)
    return new


def replace_art(pkg,p,path):
    fresh=pkg.image(path,width=6.5)
    oldprops=p.find(W+'pPr')
    if oldprops is not None:
        fresh.remove(fresh.find(W+'pPr'));fresh.insert(0,deepcopy(oldprops))
    setting(fresh.find(W+'pPr'),'keepNext')
    pkg.body.replace(p,fresh)
    pkg.changed.append(fresh)


def science_typography(pkg):
    """Apply isotope/formula typography only to newly revised text, not raw logs."""
    pattern=re.compile(r'\b(?:C11H14NO2S|C11H13NO2SK|C15H15N2O3|C15H14N2O3Na|CDCl3|DMSO-d6|KMnO4|CF3|PF6|O2|1H|13C)\b')
    for node in pkg.changed:
        if node.getroottree().getroot() is not pkg.doc:continue
        paragraphs=[node] if node.tag==W+'p' else list(node.iter(W+'p'))
        for p in paragraphs:
            ts=prior.paragraph_text_nodes(p);value=''.join(t.text or '' for t in ts)
            mask=[None]*len(value)
            for m in pattern.finditer(value):
                for i in range(m.start(),m.end()):
                    if value[i].isdigit():mask[i]='superscript' if m[0] in ('1H','13C') else 'subscript'
            if not any(mask):continue
            offset=0
            for t in ts:
                v=t.text or '';flags=mask[offset:offset+len(v)];offset+=len(v)
                if not any(flags):continue
                r=t.getparent()
                if len(r.findall(W+'t'))!=1:continue
                pos=r.getparent().index(r);parent=r.getparent();start=0
                for stop in range(1,len(v)+1):
                    if stop==len(v) or flags[stop]!=flags[start]:
                        nr=deepcopy(r);nt=nr.find(W+'t');nt.text=v[start:stop];nt.set('{http://www.w3.org/XML/1998/namespace}space','preserve')
                        if flags[start]:
                            rp=nr.find(W+'rPr')
                            if rp is None:rp=E.Element(W+'rPr');nr.insert(0,rp)
                            setting(rp,'vertAlign',val=flags[start])
                        parent.insert(pos,nr);pos+=1;start=stop
                parent.remove(r)


def finish(pkg,stem):
    science_typography(pkg)
    style_tables(pkg.doc)
    prior.bold_references(pkg.doc)
    order_properties(pkg.doc)
    # Imported scientific PNGs use the existing content type. Keep each package's
    # section geometry, headers, footers, numbering and other styles unchanged.
    pkg.save(BASE/f'{stem}_submission_revised_20260929.docx')
    pkg.save(BASE/f'{stem}_submission_revised_20260929_marked.docx',marked=True)


def main_document():
    m=Package(BASE/'manuscript_revised_20260928.docx');p=list(m.body)
    revisions={
      1:AUTHORS,2:AFFIL_A,3:EQUAL,
      5:('Translating a batch protocol into continuous flow requires chemical reasoning, engineering calculations, and equipment choices to remain consistent. We present FlowPilot, a multi-agent AI system connecting these decisions through standardized chemist intake, literature retrieval, deterministic calculation, specialist review, and inventory reconciliation. A corpus of 464 classified literature records and 2,537 handbook-derived rules supports the workflow. A common design record links stage conditions, stream assignments, equipment, and the process diagram. Matched one-shot and full-pipeline generation across three chemistry cases, five retained generator models, and three repeats yields 90 outcomes. Separate LLM judges apply 14 fixed criteria informed by deterministic checks. The mean benchmark score increases from 0.794 to 0.917, while raw judge critical flags decrease from 98 to 4. Laboratory implementation examines two connected-stage reactions: photoredox Giese addition followed by oxidation affords a sulfoxide in 97–98% NMR yield, and DPDTC-mediated amidation affords an amide in 68–86% NMR yield. Selected conditions give isolated yields of 94% and 84%, respectively. These results connect computational design consistency with experimentally demonstrated synthesis, while distinguishing proposed settings, implemented operation, and measured chemical outcomes.'),
      13:('FlowPilot focuses on translating reaction-level instructions into a connected, laboratory-specific flow configuration. This requires relating the order of chemical transformations to stream introduction, reagent concentrations and molar feeds, reactor volumes, stage-specific residence-time conventions, and compatible equipment assignments. FlowPilot addresses this integration task by constructing and reconciling these quantities in one inspectable process record. This differs in emphasis from predicting a local flow pattern, optimizing an established platform, or configuring a general process simulator, while overlapping with parts of each. Table 1 summarizes the task-level comparison; Table S36 provides the fuller source-based analysis. Neither comparison ranks the performance of the published systems.'),
      16:('The study separates information support, computational design quality, and experimental chemical performance. Corpus and retrieval analyses evaluate the context supplied to the generator. Matched architecture comparisons use a fixed rubric, deterministic verification evidence, and separate LLM judges; module ablations and a historical isoxazole study examine configuration sensitivity.[[56]] Laboratory implementation then examines two connected-stage cases. Photoredox Giese addition followed by aerobic oxidation tests the transition from an oxygen-free stage to an oxygen-fed stage within one equipment configuration (Figure 5), while DPDTC-mediated amidation tests staged reagent addition, coupled flow calculations, and intermediate transfer without isolation (Figure 6).[[57-58]] The experiments assess implementability and product formation separately from the computational benchmark.'),
      37:('Figure 4. Ablation benchmarking and cost efficiency. (a) Architecture comparison under shared chemistry, inventory, and evaluation criteria. (b) Benchmark scores for five generator models: mean and sample SD of three repeat-level means, each averaging three chemistry cases. (c) Raw judge critical flags: mean and sample SD of the total flags per three-case repeat; flags are not independently confirmed physical errors. The same five-model cohort is used in both panels and Tables S11–S12. (d) Selected Qwen3.8-27B module ablations: mean and between-case SD, with one generation per case-condition. Table S10 reports all 15 conditions. (e) Ratio of mean FlowPilot score to mean generation cost across nine outcomes per model, under the recorded price schedule (log scale), excluding judge and deployment costs. Full definitions and case-level results are in ESI Section 3.'),
      41:('The ESI also shows how the chemist supplies and inspects the information behind a design. Figures S14–S15 and Table S23 document protocol entry and fixed-ID follow-up questions. Figure S16 and Table S24 show inventory import, normalization, and equipment warnings; Figure S17 and Table S25 show stage-resolved topology and stream parameters from a reopened run. Concrete failures are retained in Table S26, and Figure S18 and Table S27 expose actual council disagreement and selection records. Figure S19 and Table S28 place these views in the complete GUI and saved-run context. These operational records connect the computational benchmark to an inspectable workflow. The laboratory cases below examine how selected proposals translate into connected reactions and measured product yields.'),
      43:('The first connected-process case couples photoredox Giese addition with oxidation of the resulting sulfide to a sulfoxide (Figure 5). Photoinduced single-electron transfer generates an alpha-thiomethyl radical from an alpha-silyl sulfide; addition to acrylonitrile constructs the C–C bond, and subsequent oxygen-dependent photochemistry changes the sulfur oxidation state.[[57]] This sequence requires the oxygen-free first stage and oxidizing second stage to remain distinct within a connected process. The supplied design input specifies 4 h under argon followed by 6 h in air. These input durations are retained separately from the published batch comparison reproduced by KHU, which lists 180 and 360 min. Neither is treated as a kinetic law or converted using a universal intensification factor.'),
      44:('The revised proposal introduces pure oxygen only between the stages and binds each coil to a compatible irradiation module. KHU implemented a 2 or 5 mL PFA coil in the Vapourtec UV-150 module followed by a separately illuminated 20 mL FEP coil, both at 25 ± 1 °C. A 2.0 mL sample loop introduced a solution containing substrate 1a (0.100 M), acrylonitrile (0.200 M), and iridium photocatalyst (0.500 mM) into a carrier stream at 0.020 mL min−1. Stage 1 therefore has a nominal liquid residence time of 100 or 250 min. The oxygen-controller setting was 0.090 mL min−1. On the proposal\'s explicit STP basis (273.15 K, 1 atm), this corresponds to 0.00402 mmol min−1 of oxygen and 2.01 equivalents relative to the nominal substrate molar flow. The laboratory MFC reference conditions require confirmation before this conversion is taken as a calibrated experimental value. The Stage 2 index, V/(Qliquid + Qgas,STP), is 181.82 min. It combines reference-state gas volume with liquid flow and is not the residence time of either phase at operating pressure.'),
      45:('An earlier high-gas-flow trial showed upstream gas entry, prompting the move from air-based delivery to pure oxygen and a lower total gas flow. This change reduces the carrier-gas burden at a specified oxygen molar feed; it does not by itself establish hydraulic protection. The proposal includes a gas-line check valve, and the KHU photograph identifies a check valve in the assembled apparatus. Exact placement and startup-pressure behavior remain to be documented. KHU used a 7 bar cartridge BPR but recorded system pressures of 5.1–5.6 bar; the hardware rating and measured pressure are reported separately. The 2 mL first-stage configuration, grouped as Sets 1/3 in the supplied record, afforded sulfoxide 4a in 97% NMR yield and 94% isolated yield. Increasing the first-stage volume to 5 mL (Set 2) gave 98% NMR yield. Without replicate statistics, the one-percentage-point difference does not establish a benefit from the longer first stage. Figure S20 and Tables S29–S31 retain the design inputs and proposed settings; ESI Section 12, Figures S23–S25, Figures S28–S29, and Tables S37–S39 document irradiation, implementation, and characterization. These measurements demonstrate productive telescoping, not a measured residence-time distribution or an independently optimized operating point.'),
      47:('Figure 5. Photoredox Giese addition followed by sulfoxidation: proposal and laboratory implementation. (a) Chemical sequence from KHU\'s supplied experimental artwork. (b) Simplified FlowPilot proposal; full archived equipment topologies are in Figure S20. (c) KHU-reported reactor volumes and operating pressures, with stage times calculated from the stated settings. Stage 1 uses V/Qliquid. The Stage 2 value is a nominal inlet/STP index, V/(Qliquid + Qgas,STP), not an operating-pressure or measured residence time; MFC reference conditions remain to be confirmed. The 7 bar cartridge BPR is not equated with the recorded pressure. (d) Reported NMR and isolated yields. Sets 1/3 share one reported entry, not independently documented replicates; no error bars are inferred. Actual sample-loop injection, photographs, full procedures, and analytical data are given in ESI Section 12.'),
      49:('The second case is a thermal two-stage amidation of 3-methyl-4-nitrobenzoic acid with benzylamine (Figure 6). DPDTC and DMAP first generate a 2-pyridyl thioester; downstream aminolysis forms the amide C(O)–N bond while displacing the sulfur-containing leaving group.[[58]] The input specifies 30 min at 95 °C for each batch stage, separated by cooling and amine addition. In flow, the thioester is neither isolated nor assumed to form quantitatively. Stage 1 conversion limits the intermediate entering Stage 2, making final amide yield a response of the complete sequence rather than of either reactor alone.'),
      50:('KHU implemented the proposed separate-feed arrangement using two channels of the Vapourtec E-series system. Feed A contained acid (0.500 M), DPDTC (0.525 M), and DMAP (0.0500 M) in 2-MeTHF; Feed B contained benzylamine (2.10 M). In Set 1, QA = 0.160 and QB = 0.0400 mL min−1 correspond to nominal acid and amine molar flows of 0.0800 and 0.0840 mmol min−1, preserving 1.05 equivalents of amine. Two 5 mL ETFE coils give nominal stage times of 31.25 and 25.00 min because the second reactor receives both feeds. The reported implementation used a 2.0 mL sample loop for Feed A, direct interstage addition, and a 95 °C water bath for both coils. A 7 bar cartridge BPR was fitted; the recorded pressure for Set 1 was 5.2–5.3 bar. Thus hardware selection, nominal stage times, and measured pressure remain distinct parts of the process record.'),
      51:('Set 1 afforded amide 3b in 86% NMR yield and 84% isolated yield. Sets 2/3 share a second reported operating point: a 10 mL first reactor and 5 mL second reactor with feeds of 0.280 and 0.0700 mL min−1, giving nominal stage times of 35.71 and 14.29 min and 68% NMR yield. The nominal combined stage time decreases from 56.25 to 50.00 min while the substrate molar feed rises from 0.0800 to 0.140 mmol min−1. The lower final yield despite a longer first stage is consistent with a coupled yield–throughput trade-off, but cannot identify the limiting step because both flow and reactor configuration changed and intermediate conversion was not measured. Figure S21 and Tables S32–S34 retain the response sets and design calculations. Figures S26–S27 and Figures S30–S31 provide the supplied experimental scheme, apparatus photograph, and product spectra; Tables S38–S39 reconcile implementation and molar feeds. The measurements support intermediate transfer without isolation while showing why numerical closure alone does not determine the best chemical operating point.'),
      54:('Figure 6. DPDTC-mediated telescoped amidation: proposal and laboratory implementation. (a) Acid activation to a 2-pyridyl thioester followed by benzylamine addition and amide formation. (b) Simplified proposed connected train; complete archived topologies are in Figure S21. (c) Implemented flow rates, reactor volumes, nominal liquid residence times, and KHU-reported system pressures. R1 uses V1/QA, whereas R2 uses V2/(QA + QB). Both stages were held at 95 °C; the 7 bar cartridge rating is distinct from recorded pressure. (d) Reported NMR and isolated yields. Sets 2/3 share one experimental entry; independent replicate statistics were not supplied. Sample-loop operation, detailed preparation, photographs, and analytical records are in ESI Section 12.'),
      58:('This emphasis complements related advances in microreactor design, experimental synthesis development, process simulation, and deterministic constraint validation (Table 1).[[46-51]] FlowPilot links the reaction sequence to explicit feed compositions, molar flows, stage calculations, and available equipment in one record. Its focus on coupled stream, stage, and equipment consistency also complements process-optimization and materials-development agents.[[60,61]] Table S36 describes these differences in task and output; the one-shot benchmark is an architecture comparison within this study, not a head-to-head evaluation of the published systems.'),
      60:('The laboratory cases connect this computational assessment to chemical outcomes. Giese addition/oxidation demonstrates a productive sequence with oxygen introduced downstream, while the closely similar yields at two first-stage volumes do not establish an optimum. DPDTC amidation demonstrates intermediate transfer without isolation, but its lower yield at higher feed rate shows that a numerically closed proposal need not be the chemically preferred operating point. Both were implemented using sample-loop injections; extended steady-state operation and repeatability are separate questions. The feedback interface can incorporate measured yields together with the actual settings and deviations, rather than treating a proposed design as experimental evidence. Integration with Bayesian optimization[[30-34]] and language-guided priors[[62]] offers a route to test chemist hypotheses against such accumulated records.'),
      63:('For the five-model, three-chemistry cohort, the mean fixed-criteria score increased from 0.794 for one-shot generation to 0.917 for FlowPilot, with fewer raw judge critical flags. Module ablations support a contribution from council review, while historical model-matrix and budget studies show configuration sensitivity. Laboratory implementation produced a sulfoxide in 94% isolated yield and an amide in 84% isolated yield under selected connected-stage conditions. The different amidation outcomes at two operating points reinforce the need to evaluate chemical performance separately from computational consistency. Together, these findings support a traceable workflow from reaction-level input to an experimentally reviewable flow configuration within the scope of the cases examined.'),
      69:('The translation workflow comprises standardized intake, chemistry interpretation, retrieval, engineering calculation, candidate generation, council review, inventory realization, and topology rendering. LLMs propose chemical interpretations and independent design choices; deterministic routines recalculate dependent quantities and check their consistency. Chemist objectives and available evidence guide residence-time and throughput choices. Stage parameters, stream assignments, and topology are reconciled before final publication. Major development milestones and dated evidence are summarized in Table S35 (ESI Section 10). Later scientific-policy and backflow extensions are separated from the frozen software used for the earlier architecture benchmarks. Table 1 and Table S36 compare six directly related systems with FlowPilot by task, output, equipment context, and reported validation; this literature-based comparison is not an additional performance benchmark.'),
      74:'Corresponding Authors',75:CORR_PARK+'; '+CORR_AHN,
      80:AVAIL,
      82:text(p[82])+' This work was also supported by the National Research Foundation of Korea (NRF) grant funded by the Korea government (MSIT) (RS-2026-25594673).',
      84:('BPR, back-pressure regulator; DPDTC, dipyridyldithiocarbonate; DMAP, 4-(dimethylamino)pyridine; Da, Damköhler number; ETFE, ethylene tetrafluoroethylene; FEP, fluorinated ethylene propylene; IF, intensification factor; LLM, large language model; LRMS, low-resolution mass spectrometry; MFC, mass flow controller; 2-MeTHF, 2-methyltetrahydrofuran; NMR, nuclear magnetic resonance; Pe, Péclet number; PFA, perfluoroalkoxy; PFD, process flow diagram; PTFE, polytetrafluoroethylene; RAG, retrieval-augmented generation; Re, Reynolds number; STP, standard temperature and pressure (273.15 K and 1 atm for the proposal calculations); STY, space-time yield.'),
    }
    revisions[37]=('Figure 4. Architecture comparison and cost efficiency. (a) Shared-input generation and fixed evaluation. (b) Benchmark score and (c) raw judge flags for five models (Tables S11–S12): mean ± sample SD over three repeats, each containing three chemistries; flags are summed within each repeat. (d) Selected Qwen3.8-27B ablations: mean and between-chemistry SD for one generation per cell (all 15 conditions: Table S10). (e) Mean FlowPilot score divided by mean generation cost over nine outcomes per model, at recorded prices. Flags are judge findings, not independently confirmed physical errors; costs exclude judging and deployment.')
    revisions[47]=('Figure 5. Giese addition/oxidation: proposed configuration and measured outcomes. (a) Chemical sequence. (b) Simplified FlowPilot proposal (full topologies: Figure S20). (c) Implemented volumes and recorded pressures. R1 time is V/Qliquid; R2 uses the nominal inlet/STP index V/(Qliquid + Qgas,STP), not operating-pressure residence time. MFC reference conditions require confirmation. The 7 bar cartridge rating is distinct from recorded pressure. (d) NMR and isolated yields supplied by KHU. Sets 1/3 share one reported entry, not documented independent repeats. Actual sample-loop operation, photographs, procedures, and analytical data are in ESI Section 12.')
    revisions[45]=revisions[45].replace('Figures S23–S25, Figures S28–S29, and Tables S37–S39','Figures S23–S25 and Tables S37–S39')
    revisions[51]=revisions[51].replace('Figures S26–S27 and Figures S30–S31 provide the supplied experimental scheme, apparatus photograph, and product spectra','Figures S26–S27 provide the supplied experimental scheme and apparatus photograph')
    for i,value in revisions.items():m.revise(p[i],value)
    # Compact only the three expanded captions so figure and caption stay together;
    # retain Times New Roman 11 pt and leave other paragraph styles untouched.
    for i in (37,47,54):
        props=p[i].find(W+'pPr')
        setting(props,'keepLines');setting(props,'spacing',before='0',after='120',line='240',lineRule='auto')
    for i in (37,47,54):
        for n in p[i].xpath('./w:pPr/w:keepNext',namespaces=NS):n.getparent().remove(n)
    m.before(p[3],[m.paragraph(AFFIL_B)])
    for i,name in [(19,'figure1_unit_corrected'),(36,'figure4_five_model_cohort'),(46,'figure5_experimental_revision'),(53,'figure6_experimental_revision')]:replace_art(m,p[i],FIG/f'{name}.png')
    # The table is inserted after the six systems have been cited in the prose,
    # preserving the source manuscript's reference numbering.
    literature=[
      ['Chat-microreactor\nPan et al., 2025','Flow information, flow-pattern prediction, and microreactor guidance.','Microreactor and multiphase-flow context; not presented as an inventory-bound multistage translator.','Flow-pattern tests and a piperacillin design example.'],
      ['LLM-RDF\nRuan et al., 2024','Synthesis development, kinetic analysis, optimization, and reactor design.','Automated synthesis/analysis platform; experimental operations and scale-up.','Wet-lab synthesis, kinetics, photochemistry, and reactor-design studies.'],
      ['SapoMind\nWang et al., 2026','Condition recommendation and process optimization for lanolin saponification.','An implemented continuous-flow microreactor; reaction-specific development.','Wet-lab product quality and process-greenness assessment.'],
      ['Text-to-Simulation\nTian et al., 2026','Process topology and parameter generation for executable flowsheets.','Simulator unit operations and connected flowsheets, rather than laboratory inventory allocation.','Computational simulation convergence and design-time evaluation.'],
      ['CeProAgents\nYang et al., 2026*','Knowledge, conceptual design, and parameter-optimization tasks.','Hierarchical process-development agents and simulator tools.','CeProBench tasks and Aspen Plus optimization.'],
      ['CAAF\nZhang, 2026*','Constraint checking and controlled candidate revision.','Formal engineering invariants; constructed flow-reactor constraint problem.','Computational constraint-validation benchmarks.'],
      ['FlowPilot\nThis study','Batch-to-flow proposal with topology, feeds, stage conditions, and provenance.','Supplied laboratory inventory; coupled multistage feeds, volumes, and equipment.','Matched architecture benchmark and two wet-lab connected-stage cases.'],
    ]
    nodes=[m.paragraph('Table 1. Scope of related LLM/agent systems and FlowPilot. Rows summarize the checked sources, not a performance ranking. The first six rows correspond to references 46, 48, 47, 49, 50, and 51, respectively; full comparison and evidence boundaries are in Table S36. *Preprint.',bold=False,keep=True),m.table(['System','Design task and output','Equipment and stage scope','Reported validation'],literature,[1.04,1.94,1.88,1.64])]
    m.after(p[13],nodes)
    m.before(p[73],[m.paragraph('Laboratory experiments were performed by KHU using a Vapourtec E-series system. The photochemical sequence used PFA and FEP coils, an interstage oxygen feed, and separate irradiation modules; the amidation sequence used two heated ETFE coils and interstage benzylamine addition. Both used sample-loop injection and a 7 bar cartridge BPR. NMR yields were determined using 1,3-benzodioxole as the internal standard; isolated products were characterized by 1H/13C NMR and LRMS. ESI Section 12 contains the supplied apparatus photographs, procedures, numerical reconciliation, and outstanding metrology details. Figures S28–S31 contain the product-characterization spectra. No replicate SD or measured two-phase residence time is inferred where these records were not supplied.')])
    finish(m,'manuscript')
    return m


def esi_document():
    s=Package(BASE/'esi_revised_20260928.docx');p=list(s.body)
    # The original spacer paragraphs after the TOC become a blank page when
    # its expanded cache gains lines. Preserve the cover, remove only this gap.
    for node in p[20:35]:
        if node.tag==W+'p' and not text(node).strip() and not node.xpath('.//w:drawing | .//w:sectPr',namespaces=NS):
            s.body.remove(node)
    for i,v in {1:AUTHORS,2:AFFIL_A,3:AFFIL_B,4:CORR_PARK,5:CORR_AHN,6:EQUAL,
      219:('The internal screen used Qwen3.8-27B as the fixed generator and the same three chemistry cases. Fifteen architecture conditions were tested, giving 45 condition-case outcomes. Each cell was generated once; SD therefore describes between-case variation, not generation-repeat uncertainty. Disabled specialists received no LLM call or default-score penalty, and active council weights were renormalized. Table S10 now lists all 15 conditions from the archived module summary. Main-text Figure 4d displays a selected subset. The archive\'s executable-outcome fraction is retained as metadata and should not be interpreted as independent experimental implementability or as a fair cross-architecture safety rating.'),
      220:('Table S10. All 15 internal module-ablation conditions, restored from the archived 45-outcome summary. Mean score and sample SD summarize three chemistries with one generation per cell. Exec. is the archive\'s executable-status fraction, not experimental success or a safety verdict. LLM calls are means per outcome; flags are totals over the three cases.'),
      784:('This section preserves the supplied batch protocols, frozen chemist responses, and archived September design proposals separately from the August architecture-comparison campaigns in Section 3. Tables S31 and S34 summarize the new KHU experimental outcomes without treating shared response-set entries as replicates. Section 12 provides the full procedures, photographs, characterization, and implementation reconciliation. Proposed conditions remain proposals even where the laboratory implemented matching settings; no measured residence-time distribution or optimized operating point is inferred.'),
      818:('Table S31. KHU-reported Figure 5 outcomes and actual settings. Sets 1/3 share a single entry in the supplied result table; this is not evidence of independent repeats. NR means not reported. Calculated times are identified separately from measurements.'),
      860:('Table S34. KHU-reported Figure 6 outcomes and actual settings. Sets 2/3 share one entry in the supplied result table. Times are nominal V/Q calculations. NR means not reported; the table does not imply independent experimental replicates.'),
      863:('The laboratory record groups Sets 2 and 3 at the same operating point and gives 68% NMR yield for that grouped entry. It does not supply replicate identifiers or a distribution of yields. Set 1 gives 86% NMR and 84% isolated yield. Section 12 reports the supplied procedures and characterization; quantitative NMR calculation records, sample collection windows, and intermediate composition remain to be supplied.'),
      892:('Laboratory implementation and chemical response are reported separately in Section 12. Figure 5 concerns an oxygen-free/oxygen-fed photochemical sequence; Figure 6 concerns telescoped activation and amidation. Tables S31 and S34 summarize the measured outcomes, while Tables S38–S39 reconcile proposals with the supplied operating records and feed calculations. These laboratory observations do not convert the literature comparison into a controlled head-to-head test of the six external systems.'),
    }.items():s.revise(p[i],v)
    module=ROOT/'deliverables/newgen_2_0_module_attribution_qwen38_screen3_matched_v1_20260821/tables/module_summary.csv'
    with module.open() as f:mr=list(csv.DictReader(f))
    assert len(mr)==15 and sum(int(r['n_outcomes']) for r in mr)==45
    RAW.mkdir(exist_ok=True);(RAW/'tableS10_all15_module_conditions.csv').write_bytes(module.read_bytes())
    replace_table(s,p[221],['Condition','Mean','SD','Exec.','Calls','Flags'],[[r['condition'],f"{float(r['mean_score_0_1']):.3f}",f"{float(r['score_sd']):.3f}",f"{100*float(r['executable_rate']):.0f}%",f"{float(r['mean_llm_calls']):.1f}",r['critical_flags']] for r in mr],[2.65,.79,.76,.78,.76,.76])
    f5rows=[
      ['NMR yield of 4a (%)','97','98'],['Isolated yield / mass','94% / 42.0 mg','NR'],
      ['R1 / R2 volume (mL)','2 / 20','5 / 20'],['Liquid feed (mL/min)','0.020','0.020'],
      ['Oxygen controller setting (mL/min)','0.090','0.090'],['R1 nominal liquid time (min)','100','250'],
      ['R2 inlet/STP index (min), calculated','181.82','181.82'],['Reported temperature','25 ± 1 °C, both stages','25 ± 1 °C, both stages'],
      ['Recorded system pressure (bar)','5.1–5.4','5.2–5.6'],['BPR cartridge rating','7 bar','7 bar'],
      ['NMR internal standard','1,3-Benzodioxole','1,3-Benzodioxole'],['Replicate count / SD / sample window','NR','NR'],
      ['Substrate conversion / selectivity / intermediates','NR','NR']]
    replace_table(s,p[819],['Reported item','Sets 1 and 3 (grouped)','Set 2'],f5rows,[2.68,1.91,1.91])
    f6rows=[['NMR yield of 3b (%)','86','68'],['Isolated yield / mass','84% / 227.0 mg','NR'],
      ['R1 / R2 volume (mL)','5 / 5','10 / 5'],['Feed A / B flow (mL/min)','0.160 / 0.040','0.280 / 0.070'],
      ['R1 / R2 nominal time (min)','31.25 / 25.00','35.7143 / 14.2857'],['Temperature of both reactors','95 °C','95 °C'],
      ['Recorded system pressure (bar)','5.2–5.3','5.5–5.8'],['BPR cartridge rating','7 bar','7 bar'],
      ['Acid / DPDTC / DMAP in Feed A (M)','0.500 / 0.525 / 0.0500','0.500 / 0.525 / 0.0500'],
      ['Benzylamine in Feed B (M)','2.10','2.10'],['NMR internal standard','1,3-Benzodioxole','1,3-Benzodioxole'],
      ['Replicate count / SD / sample window','NR','NR'],['Stage 1 conversion / residual intermediate','NR','NR']]
    replace_table(s,p[861],['Reported item','Set 1','Sets 2 and 3 (grouped)'],f6rows,[2.68,1.91,1.91])
    # Update only the present-study validation cell in the literature table.
    last=p[887].findall(W+'tr')[-1].findall(W+'tc')[-1]
    cp=last.find(W+'p');s.revise(cp,'Matched architecture benchmarks, archived run audits, and two connected-stage laboratory cases. Reported isolated yields: sulfoxide 4a, 94%; amide 3b, 84% (Section 12).')
    add_experiments(s,p[893])
    newref=s.paragraph('13. Wang, A.; Xie, Y.; Wang, J.; Shi, D.; Yu, H. Atom-economic amide synthesis by using an iron-substituted polyoxometalate catalyst. Chemical Communications 2022, 58, 1127–1130. https://doi.org/10.1039/D1CC05417A.')
    s.after(p[905],[newref])
    finish(s,'esi')
    return s


def add_experiments(s,anchor):
    nodes=[]
    def para(v,**kwargs):nodes.append(s.paragraph(v,**kwargs))
    def heading(v,level=2):para(v,style=f'Heading{level}',page_break=v!='General materials, instrumentation, and analytical methods',keep=True)
    def figure(path,caption,width=6.5):
        nodes.append(s.image(path,width=width,page_break=True));para(caption)
    def table(caption,heads,values,widths):para(caption,keep=True);nodes.append(s.table(heads,values,widths))
    heading('Laboratory implementation, experimental procedures, and analytical data',1)
    para('The KHU experimental supplement supplied on 29 September 2026 is the primary source for this section. The supplied setup photographs, emission spectra, reaction schemes, and product NMR spectra are retained; typographical corrections and the reconciliation of nominal time/pressure conventions are identified in the text. No new experiment or computational design campaign was performed for this document revision. Reported grouped response sets are not relabeled as independent measurements. Tables S31 and S34 provide the outcome summaries; the following methods and images establish their experimental context.')
    heading('General materials, instrumentation, and analytical methods')
    khu=json.loads((OUT/'sources/khu/document.json').read_text())
    for i in (6,7,8):para(khu[i]['text'].strip())
    para('The 7 bar BPR values above identify the installed cartridge, whereas the system-pressure ranges in Tables S31 and S34 are the values supplied by KHU. The pressure-reference convention (gauge or absolute), sensor location, and calibration record were not specified. The bath is described as a water bath in the supplied methods and has not been silently relabeled as an oil bath. Laboratory confirmation of this detail, and of the pressure/MFC metrology, remains pending.')
    heading('Irradiation sources and recorded specifications')
    para('The batch irradiation used two MR16 5 W blue LED spotlights. Flow Stage 1 used the Vapourtec 450 nm LED in the UV-150 module, and Stage 2 used a separate blue LED strip. Figure S23 reproduces the supplied photographs and emission spectra. Table S37 distinguishes electrical input from the radiant-power specification; these values are not photon fluxes at the reacting solution. No actinometric calibration or position-dependent irradiance measurement was supplied.')
    table('Table S37. Irradiation-source specifications transcribed from the KHU supplement. Peak wavelengths are those reported in the supplied specification/spectrum images, not an independently repeated calibration.',
      ['Light source / use','Electrical specification','Reported radiant power','Peak wavelength'],[
      ['MR16 spotlight / batch, two lamps','5 W per lamp; 12 V','Not reported','452.4 nm'],
      ['Vapourtec LED / flow R1','60 W input','24 W','450.32 nm'],
      ['Blue strip / flow R2','10 W; 24 V','Not reported','447.50 nm']],[1.9,1.7,1.5,1.4])
    for letter,label in [('a','MR16 5 W batch lamp'),('b','Vapourtec flow-stage-1 LED'),('c','blue strip LED for flow-stage-2 irradiation')]:
        figure(FIG/f'figureS23{letter}_irradiation.png',f'Figure S23{letter}. KHU photograph and emission spectrum of the {label}. Artwork and spectral traces are reproduced from the supplied experimental supplement. Table S37 gives the corresponding specifications; lamp wattage is not assumed to equal incident optical power at the reaction mixture.')
    heading('Photoredox Giese addition and oxidation: preparation and operation')
    para('The transformation converts (((4-methoxyphenyl)thio)methyl)trimethylsilane (1a) and acrylonitrile (2a) to sulfide 3a and then sulfoxide 4a, following the chemistry of Park et al. [4]. Figure S24 reproduces KHU\'s supplied reaction/setup comparison and Figure S25 shows the actual apparatus. The compared literature values belong to the cited publication, not to new FlowPilot experiments.')
    for i in (44,45):para(khu[i]['text'].strip())
    para('The Stage 1 liquid feed therefore contains nominal concentrations of 0.100 M 1a, 0.200 M acrylonitrile, and 0.000500 M iridium photocatalyst. The 2.0 mL sample loop contains 0.200 mmol 1a and corresponds to a nominal injection duration of 100 min at 0.020 mL min−1. This is a finite sample-loop experiment in a continuously flowing carrier, not evidence of indefinitely sustained reagent-fed operation. Exact leading/trailing collection windows and the degassing/argon-handling procedure used during the implemented runs were not specified in the new methods. The frozen design input retains an oxygen-free, argon-prepared first stage; oxygen is introduced only at the interstage T-mixer.')
    para('At the stated settings, nominal Stage 1 times are 2/0.020 = 100 min and 5/0.020 = 250 min. The Stage 2 value labeled “181 min” in the supplied artwork is interpreted here as the rounded inlet-reference index 20/(0.020 + 0.090) = 181.82 min. On the proposal\'s STP convention (273.15 K, 1 atm), 0.090 mL min−1 of pure O2 corresponds to 0.004015 mmol min−1, or 2.008 equivalents relative to 0.00200 mmol min−1 of 1a. These are calculated proposal-basis values; the actual MFC reference temperature/pressure must be confirmed. Gas compression, dissolution, consumption, and phase holdup prevent this index from being identified with physical gas or liquid residence time. No tracer or residence-time-distribution measurement was supplied.')
    para('An aliquot of the collected reactor effluent was analyzed by 1H NMR spectroscopy using 1,3-benzodioxole as an internal standard to determine the NMR yield of 4a. The remaining effluent was concentrated under reduced pressure and purified by flash column chromatography on silica gel. KHU reports 97% NMR yield and 94% isolated yield for the grouped Sets 1/3 entry (2 mL R1), and 98% NMR yield for Set 2 (5 mL R1). The recorded system pressures are 5.1–5.4 and 5.2–5.6 bar, respectively. Replicate counts and quantitative-NMR integration/calibration records were not supplied.')
    figure(FIG/'khu_image10.png','Figure S24. Original KHU experimental schematic, chemical sequence, and batch/flow comparison for sulfoxide 4a. Source references are Park et al. [4] for the first two rows and the supplied KHU measurements for the FlowPilot rows. The source\'s “181 min” for R2 is a rounded inlet-reference index, not operating-pressure residence time. Its 0.04 in tubing ID equals 1.016 mm; the displayed 1.00 mm is a nominal label. Literature batch timing (180/360 min) differs from the frozen design input (240/360 min). Source artwork is preserved to keep these distinctions auditable; it is not silently normalized or represented as new batch data.')
    figure(OUT/'sources/khu/image11.png','Figure S25. Actual KHU photochemical flow apparatus, as supplied. The labeled photograph identifies the E-series system, UV-150 reactor, downstream coil, MFC, T-mixer, check valve, BPR, cooling module, carrier solvent, and sample loop. The image is not a pressure calibration or a record of startup dynamics; exact check-valve location/orientation and a pressure trace remain to be confirmed.',width=5.5)
    heading('DPDTC-mediated amidation: preparation and operation')
    para('The sequence converts 3-methyl-4-nitrobenzoic acid (1b) to its 2-pyridyl thioester and then to N-benzyl-3-methyl-4-nitrobenzamide (3b), following the chemistry of Saunders et al. [5]. The intermediate is not isolated. Figure S26 preserves the supplied scheme and reference comparison; Figure S27 shows the actual apparatus.')
    for i in (73,74):para(khu[i]['text'].strip())
    para('Nominal stock concentrations in Feed A are 0.500 M acid, 0.525 M DPDTC, and 0.0500 M DMAP. A 2.0 mL loop contains 1.00 mmol acid and is displaced in 12.50 or 7.14 min at the two stated flow rates. The benzylamine stock has a nominal concentration of 2.10 M; the specified 1.05 mmol quantity corresponds to 0.500 mL of that stock. The required total feed volume for priming, overlap with the dispersed sample plug, and collection cannot be reconstructed from this nominal reagent amount alone. The new methods do not specify the second-feed start/stop timing or collection window, which are important for interpreting finite-loop stoichiometry.')
    para('At Set 1 flows, the acid and benzylamine nominal molar feeds are 0.0800 and 0.0840 mmol min−1. At Sets 2/3 flows, they are 0.140 and 0.147 mmol min−1. Both give 1.05 equivalents of benzylamine. The combined stream has acid-equivalent concentration 0.400 M and amine-feed-equivalent concentration 0.420 M before allowing for reaction; these are mixing calculations, not measured thioester or product concentrations. R1 times are 31.25 and 35.7143 min; R2 times are 25.00 and 14.2857 min. Actual intermediate conversion was not measured in the supplied data.')
    para('An aliquot of the collected effluent was analyzed by 1H NMR spectroscopy using 1,3-benzodioxole as an internal standard. The remaining effluent was concentrated by rotary evaporation and purified by flash column chromatography on silica gel. Set 1 gave 86% NMR yield and 84% isolated yield (227.0 mg); Sets 2/3 share a reported NMR yield of 68%. The recorded system pressures were 5.2–5.3 and 5.5–5.8 bar, respectively. Both coils were described as operating at 95 °C without an intervening cooling operation. Individual-run identifiers and repeat statistics were not supplied.')
    figure(FIG/'khu_image12.png','Figure S26. Original KHU reaction/setup schematic and batch/flow comparison for amide 3b. The first two rows reproduce the cited Saunders et al. [5] comparison; the FlowPilot rows report new KHU measurements. The source\'s 0.093 in ID equals 2.3622 mm, whereas 2.40 mm is a nominal inventory label. The literature batch comparison lists 1.10 equivalents of DPDTC; the supplied design input and implemented FlowPilot conditions use 1.05 equivalents. These distinct conditions are retained rather than treated as a matched yield or speed-up comparison. Parenthesized yields in the source are isolated yields; “quant” is the source\'s qualitative quantitative-yield label.')
    figure(OUT/'sources/khu/image13.png','Figure S27. Actual KHU two-stage amidation apparatus, as supplied. Two E-series pump channels deliver the sample-loop/carrier stream and the benzylamine stream. The photograph identifies the sample loop, T-mixer, two reactor coils, BPR, and solution containers. It documents the assembled apparatus, not a complete time-resolved collection or calibration record.',width=6.0)
    heading('Product characterization and supplied spectra')
    para('The characterization below is transcribed from KHU\'s supplied records with punctuation and section-reference corrections only. Reported masses, yields, shifts, couplings, and observed LRMS values are retained. Figures S28–S31 are product-characterization spectra; they are not presented as the missing raw quantitative-NMR yield calculations or as independent repeat measurements.')
    nodes.append(s.image(OUT/'sources/khu/image14.png',width=1.6))
    t=khu[87]['text'].strip().replace('(ESI section 1)','(Section 12.3)')
    t=re.sub(r'\.1$', '. [4]',t)
    para(t)
    nodes.append(s.image(OUT/'sources/khu/image15.png',width=1.6))
    t=khu[89]['text'].strip().replace('produre','procedure').replace('(ESI section 1)','(Section 12.4)').replace('(s. 1H)','(s, 1H)').replace('131.85. 129.02','131.85, 129.02').replace('Hz,1H','Hz, 1H')
    t=re.sub(r'\.2, 3$', '. [5,13]',t)
    para(t)
    para('Mass/yield arithmetic agrees with the supplied starting scales: 42.0 mg of 4a (approximately 223.29 g mol−1) from 0.200 mmol corresponds to 94.1%, and 227.0 mg of 3b (approximately 270.29 g mol−1) from 1.00 mmol corresponds to 84.0%. LRMS observed values are reproduced as reported, not recalibrated; original instrument exports and author approval of these assignments remain part of the analytical record to obtain.')
    for n,stem,label in [(28,'image17','4a: 1H NMR (500 MHz, CDCl3)'),(29,'image18','4a: 13C{1H} NMR (126 MHz, CDCl3)'),(30,'image19','3b: 1H NMR (500 MHz, CDCl3)'),(31,'image20','3b: 13C{1H} NMR (126 MHz, CDCl3)')]:
        figure(FIG/f'khu_{stem}.png',f'Figure S{n}. {label}. Supplied KHU product-characterization spectrum, reproduced without peak reassignment, baseline correction, or deletion of signals. The original EMF vector artwork is retained in the revision evidence package.')
    heading('Implementation reconciliation and calculation basis')
    table('Table S38. Proposal-to-implementation reconciliation. Agreement of selected settings does not imply that unreported operational details were verified.',
      ['Feature','FlowPilot proposal / input','KHU implementation / remaining distinction'],[
      ['Pump configuration','Use compatible channels, avoid unnecessary mixing of automated systems.','Both cases used E-series pumps; Figure 6 used two channels of the same system.'],
      ['Photochemical reactor/light pairing','UV-150-compatible PFA first coil; separately illuminated FEP second coil.','PFA 2/5 mL with Vapourtec 450 nm LED; FEP 20 mL with blue strip.'],
      ['Oxygen placement and quantity','Pure O2 only after Stage 1; 0.090 mL/min at proposal STP convention.','O2 introduced at the interstage T-mixer; same controller setpoint. MFC reference conditions/calibration not supplied.'],
      ['Backflow protection','Gas-branch check valve included in revised proposal.','Photo labels a check valve; exact branch, orientation, and startup pressure trace need confirmation. No empirical failure probability is claimed.'],
      ['Pressure','7 bar(g) design/BPR target.','7 bar cartridge installed; reported pressure 5.1–5.6 bar (Figure 5), 5.2–5.8 bar (Figure 6). Sensor basis/location unreported.'],
      ['Feed introduction','Nominal component concentrations and pump settings.','Finite 2.0 mL sample-loop injection. Pump rates do not establish a constant concentration throughout the dispersed plug.'],
      ['Amidation interstage operation','Direct amine addition, no intermediate isolation.','Two heated coils and interstage T-mixer, both at 95 °C. Water bath reported; no cooling operation described.'],
      ['Tubing dimensions','Nominal inventory IDs: 1.00 mm and 2.40 mm.','Specified 0.04 in = 1.016 mm and 0.093 in = 2.3622 mm. Use actual tubing dimensions for future hydraulic calculations.'],
      ['Performance assessment','Three response sets per chemistry, some identical conditions.','Two distinct implemented settings per chemistry. Shared-set rows are not documented independent repeats.']],[1.3,2.2,3.0])
    table('Table S39. Nominal component feeds from reported stock concentrations and pump settings. Molar feed is C × Q (M × mL/min = mmol/min). Values describe a stock-filled portion of the sample plug; no constant concentration or intermediate conversion is assumed outside that interval.',
      ['Case / set','Feed component','C (M)','Q (mL/min)','Molar feed (mmol/min)'],[
      ['Figure 5, all','Substrate 1a','0.100','0.020','0.00200'],['Figure 5, all','Acrylonitrile 2a','0.200','0.020','0.00400'],['Figure 5, all','Ir photocatalyst','0.000500','0.020','0.0000100'],
      ['Figure 5, all','O2 (STP calculation)','Gas','0.090','0.004015'],
      ['Figure 6, 1','Acid 1b','0.500','0.160','0.0800'],['Figure 6, 1','DPDTC','0.525','0.160','0.0840'],['Figure 6, 1','DMAP','0.0500','0.160','0.00800'],['Figure 6, 1','Benzylamine 2b','2.10','0.040','0.0840'],
      ['Figure 6, 2/3','Acid 1b','0.500','0.280','0.140'],['Figure 6, 2/3','DPDTC','0.525','0.280','0.147'],['Figure 6, 2/3','DMAP','0.0500','0.280','0.0140'],['Figure 6, 2/3','Benzylamine 2b','2.10','0.070','0.147']],[1.3,1.6,1.05,1.1,1.45])
    para('The supplied reference tables also compare the current runs with earlier batch and flow procedures (Figures S24 and S26). They are contextual literature comparisons, not controlled matched experiments: substrate concentration, oxygen feed, irradiation hardware, reactor size, and reagent ratios differ. No intensification factor, productivity advantage, or causal yield improvement is inferred from those comparisons. Quantitative interpretation of the gas–liquid stage requires the gas reference conditions and actual phase-holdup/transport information; the nominal inlet index alone cannot provide it.')
    heading('Outstanding experimental metadata and release status')
    para('Before submission, obtain the raw quantitative-NMR files and integration worksheets (including internal-standard amount/purity), run/sample identifiers and independent repeat counts, exact collection windows and second-feed timing, the gas-controller reference conditions and calibration, pressure-sensor reference/location, actual oxygen-free feed preparation, and check-valve branch/orientation plus startup observations. Confirm the specified tubing IDs, 95 °C bath description, and the distinction between the supplied design-input timings and literature-comparison rows. Raw NMR and LRMS instrument exports should accompany the characterization figures. These items are not replaced by inferred values or synthetic uncertainty estimates.')
    para(AVAIL)
    s.before(anchor,nodes)


def export_experiment_csvs():
    rows=[
      dict(figure=5,sets='1/3',V1_mL=2,V2_mL=20,Q1_mL_min=.020,Q2_mL_min=0,Qgas_reference_mL_min=.090,t1_min=100,t2_index_min=20/.11,temperature_C=25,pressure_bar_min=5.1,pressure_bar_max=5.4,NMR_yield_pct=97,isolated_yield_pct=94,isolated_mass_mg=42,repeat_count='not reported'),
      dict(figure=5,sets='2',V1_mL=5,V2_mL=20,Q1_mL_min=.020,Q2_mL_min=0,Qgas_reference_mL_min=.090,t1_min=250,t2_index_min=20/.11,temperature_C=25,pressure_bar_min=5.2,pressure_bar_max=5.6,NMR_yield_pct=98,isolated_yield_pct='',isolated_mass_mg='',repeat_count='not reported'),
      dict(figure=6,sets='1',V1_mL=5,V2_mL=5,Q1_mL_min=.160,Q2_mL_min=.040,Qgas_reference_mL_min=0,t1_min=5/.16,t2_index_min=5/.20,temperature_C=95,pressure_bar_min=5.2,pressure_bar_max=5.3,NMR_yield_pct=86,isolated_yield_pct=84,isolated_mass_mg=227,repeat_count='not reported'),
      dict(figure=6,sets='2/3',V1_mL=10,V2_mL=5,Q1_mL_min=.280,Q2_mL_min=.070,Qgas_reference_mL_min=0,t1_min=10/.28,t2_index_min=5/.35,temperature_C=95,pressure_bar_min=5.5,pressure_bar_max=5.8,NMR_yield_pct=68,isolated_yield_pct='',isolated_mass_mg='',repeat_count='not reported')]
    with (RAW/'khu_reported_experimental_results.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    (RAW/'experimental_data_dictionary.json').write_text(json.dumps({'source_file':'Supplementary Information (KRICT)_final (1).docx','source_images':['word/media/image10.emf','word/media/image12.emf'],'sets':'KHU groups identical response-set configurations; these are not claimed independent experimental repeats.','pressure':'Reported system pressure; gauge/absolute basis not supplied. Cartridge BPR is 7 bar.','Qgas_reference_mL_min':'MFC setpoint. STP conversion is the proposal convention; MFC reference conditions require confirmation.','t2_index_min':'Figure 5: inlet-reference index, NOT operating-pressure residence. Figure 6: nominal liquid V/(QA+QB).','yield':'Reported NMR or isolated yield. No SD has been inferred.'},indent=2)+'\n')


def main():
    export_experiment_csvs()
    m=main_document();s=esi_document()
    (OUT/'revision_manifest.json').write_text(json.dumps({'inputs':json.loads((OUT/'source_manifest.json').read_text()),'changed_main_nodes':len(m.changed),'changed_esi_nodes':len(s.changed),'new_esi_figures':'S23–S31','new_esi_tables':'S37–S39','main_table':'Table 1','references':{'main':62,'esi':13},'new_experiments_run':False},indent=2)+'\n')
    print('Saved clean and highlighted manuscript/ESI copies in Submission.')

if __name__=='__main__':main()
