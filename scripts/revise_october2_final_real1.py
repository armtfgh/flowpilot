"""Meeting revision preserving document packages and archived evidence."""
from copy import deepcopy
from hashlib import sha256
from pathlib import Path
import json
import posixpath
import re
from zipfile import ZipFile

from lxml import etree as E
from PIL import Image
from revise_manuscript_cases_20260922 import Package,W,NS,text,MILESTONES,R,P
from remove_esi_section32_20261002 import patch_text,REF
from revise_postmeeting_20261001 import table,restrained_bold
from revise_layout_discussion_20260928 import order_properties
from typeset_publication_notation_20261002 import typeset_paragraph,upright_math_descriptors

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'manuscript'
OUT=BASE/'october2_final_review'
FIG=OUT/'figures'
LOG=[]

def change(pkg,node,value,why):
    LOG.append({'document':pkg.source.name,'before':text(node),'after':value,'reason':why})
    pkg.revise(node,value)
    return node

def remove(pkg,nodes,why):
    for n in nodes:
        if n.getparent() is pkg.body:
            LOG.append({'document':pkg.source.name,'removed':text(n),'reason':why})
            pkg.body.remove(n)

def para(pkg,value,style='Normal'):
    return pkg.paragraph(value,style=style)

def image(pkg,path,maxheight=8.0):
    im=Image.open(path);width=min(6.5,maxheight*im.width/im.height)
    n=pkg.image(path,width,page_break=True)
    E.SubElement(n.find(W+'pPr'),W+'jc',{W+'val':'center'})
    return n

def cap(pkg,value):
    n=para(pkg,value)
    pp=n.find(W+'pPr')
    E.SubElement(pp,W+'keepLines')
    E.SubElement(pp,W+'keepNext',{W+'val':'0'})
    return n

def replace_image(pkg,node,path,maxheight=8.0):
    new=image(pkg,path,maxheight)
    pkg.body.replace(node,new)
    LOG.append({'document':pkg.source.name,'image_replacement':str(path.relative_to(ROOT))})

def import_original_scheme(pkg,source_index):
    source=Package(BASE/'Supplementary Information (KRICT)_final (1).docx')
    n=deepcopy(list(source.body)[source_index])
    # Import the original EMF display and embedded ChemDraw object together.
    relmap={r.get('Id'):r for r in source.rels}
    for key in sorted(set(n.xpath('.//@r:id',namespaces=NS))):
        rel=deepcopy(relmap[key]);target=rel.get('Target')
        srcpart=posixpath.normpath('word/'+target)
        name='meeting_original_'+str(pkg.next_id)+'_'+Path(target).name
        folder='embeddings' if 'embeddings/' in target else 'media'
        dest=f'word/{folder}/{name}'
        pkg.parts[dest]=source.parts[srcpart]
        rid='rIdMeetingOriginal'+str(pkg.next_id);pkg.next_id+=1
        rel.set('Id',rid);rel.set('Target',folder+'/'+name);pkg.rels.append(rel)
        for e in n.xpath('.//*[@r:id=$rid]',namespaces=NS,rid=key):e.set('{'+R+'}id',rid)
        LOG.append({'restored_original_part':srcpart,'target':dest,'sha256':sha256(pkg.parts[dest]).hexdigest()})
    # Avoid conflicts with existing VML shape and OLE object identifiers.
    for e in n.iter():
        if E.QName(e).localname=='shape':e.set('id',f'_meeting_shape_{source_index}')
        if E.QName(e).localname=='OLEObject':
            e.set('ShapeID',f'_meeting_shape_{source_index}');e.set('ObjectID',str(1950000000+source_index))
    pp=n.find(W+'pPr')
    if pp is None:pp=E.Element(W+'pPr');n.insert(0,pp)
    E.SubElement(pp,W+'pageBreakBefore');E.SubElement(pp,W+'keepNext')
    target_ct=E.fromstring(pkg.parts['[Content_Types].xml'])
    source_ct=E.fromstring(source.parts['[Content_Types].xml'])
    for item in source_ct:
        if E.QName(item).localname=='Default' and item.get('Extension') in {'emf','bin'}:
            if not any(c.get('Extension')==item.get('Extension') for c in target_ct):target_ct.append(deepcopy(item))
    pkg.parts['[Content_Types].xml']=E.tostring(target_ct,xml_declaration=True,encoding='UTF-8',standalone=True)
    return n

def main_revision(p):
    n=list(p.body)
    edits={
      27:('The complementary handbook-derived resource contains 2,537 structured rules covering reactor design, heat transfer, catalysts, residence time, photochemistry, materials, scale-up, pressure, mixing and safety (Figure 2e–f). These rules provide engineering context when a close chemical analogue is unavailable or when a candidate must be evaluated beyond reaction identity. Coverage is uneven: heat-transfer rules are densely associated with thermal synthesis and cross-coupling, whereas photochemistry rules concentrate in photoredox and photocycloaddition classes. The rule store informs interpretation and review; only the checks implemented in the calculation layer act as deterministic constraints. Figure S3 reports the literature-corpus composition and field-specific data availability.'),
      29:('Figure 2. Literature corpus and engineering rule-base coverage. (a–d) Reaction classes, reactor types, reported materials and inlet-stream configurations in the curated flow-chemistry corpus; explicit inlet-stream counts are available for 154 records. (e) Category and severity distribution within the 2,537-rule engineering knowledge base. (f) Recorded associations between rule categories and chemistry classes. Figure S3 provides the full corpus composition and field-specific denominators. Associations indicate knowledge coverage rather than independent validation of every rule.'),
      43:('Figure 4. Ablation benchmarking and cost efficiency. (a) Architecture comparison under shared inputs and fixed evaluation. (b) Benchmark scores and (c) raw judge critical flags: mean and sample SD across three repeats, each containing three chemistry cases. The six-model display includes an archived GPT-5.4 generator comparison outside the five-model primary cohort. (d) Seven selected Qwen3.8-27B ablation conditions, matching Table S7: mean and between-case SD with one generation per case-condition. The complete 15-condition screen is retained in the source archive. (e) Mean FlowPilot score divided by mean generation cost under the recorded price schedule, excluding judging and deployment. Critical flags are judge findings, not independently confirmed physical errors. ESI Section 4 provides evaluation details.'),
      48:('To examine the role of coordinated review, we separately tested 15 Qwen3.8-27B configurations over the same three cases. Figure 4d and Table S7 show the seven conditions that compare the full pipeline with one-shot generation, removal of the council and removal of individual domain reviewers; the other eight conditions remain in the archived screen. Removing the council reduced the mean score from 0.928 to 0.670, below the 0.743 one-shot result. The initial sequence of agent calls without council review therefore did not recover the quality of the complete configuration. This is consistent with a benefit from reviewing whether chemical and engineering choices form a coherent process, although one generation per case cannot isolate that mechanism from generation variability.'),
      49:('The domain removals help interpret that result. Without the Chemistry reviewer, the mean was 0.902; without Fluidics, Safety or Kinetics it was 0.913, 0.915 or 0.924, respectively. These reviewers examine coupled decisions such as whether a supported catalyst remains in the intended phase, whether gas and liquid feeds support the stated contact time, and whether thermal and pressure assumptions are compatible with the solvent and hardware. Their overlapping responsibilities allow another reviewer or a deterministic check to catch some omissions, so the smaller changes do not establish that a removed specialty is unimportant. Each condition was generated once per chemistry; the bars show between-case variation, not replicate uncertainty, and the screen cannot resolve a causal ranking of individual agents.'),
      52:('We next examined two connected-stage reactions that were separate from the architecture-benchmark cases. These experiments tested whether a researcher’s design brief could be developed into a coherent proposal, reviewed by the agents, assembled with the available equipment and evaluated as a complete reaction sequence. For each chemistry, three standardized response sets retained the same batch protocol and inventory while varying the objective and hypotheses. Set 1 emphasized integrated final yield without unnecessarily long holds, Set 2 emphasized high conversion with justified starting times, and Set 3 emphasized throughput and operating balance. The inputs were related priorities rather than distinct reactions or replicates; Sets 1 and 3 were both interpreted as balanced by the recorded objective policy. Tables S13–S14 retain the actual responses. The council reviewed 12 candidates per run, with the six selected topologies in Figures S12–S17 and detailed assessments in ESI Section 7.3. Some choices converged, producing two distinct implemented operating points per chemistry. This allows the experimental outcomes to be traced to the proposed decisions without treating each response set as an independent validation.'),
      54:('The first case couples photoredox carbon–carbon bond formation to selective oxidation of the resulting sulfide (Figure 5). Photoinduced electron transfer generates an alpha-thiomethyl radical from an alpha-silyl sulfide; addition to acrylonitrile forms the sulfide intermediate, which is then oxidized to sulfoxide 4a.[[57]] The batch procedure separates these transformations in time, with 4 h under argon followed by 6 h exposed to air. FlowPilot had to represent this change as a connected topology: an oxygen-free photoreactor followed by interstage gas addition and an illuminated oxidation reactor. The confirmed inputs required oxygen only at the second-stage inlet and at least two equivalents relative to substrate. This made atmosphere placement, reactor–light compatibility and gas stoichiometry explicit design requirements rather than assumptions left to the experimentalist.'),
      60:('The second case tests whether the agents can coordinate formation and downstream consumption of a reactive intermediate (Figure 6). 3-Methyl-4-nitrobenzoic acid is activated by DPDTC in the presence of DMAP to form a 2-pyridyl thioester; benzylamine then undergoes aminolysis to give the amide.[[58]] The batch procedure uses two 30 min periods at 95 °C, separated by cooling and amine addition. The chemistry interpretation preserved activation before aminolysis, while the council examined whether the cooling interval was a handling step that could be omitted. Candidate assessment therefore linked feed placement to stage-specific exposure: adequate upstream activation would not compensate for an insufficient downstream hold, and adding benzylamine would change both composition and volumetric flow.'),
      62:('Set 1 gave 86% NMR yield. Sets 2/3 shared a second implemented configuration: a 10 mL first coil and a 5 mL second coil, with feed rates of 0.280 and 0.0700 mL min−1, stage times of 35.71 and 14.29 min, and 68% NMR yield. The literature batch reference is 98% NMR yield.[[58]] The second configuration increased nominal substrate delivery from 0.0800 to 0.140 mmol min−1 and reduced total nominal time from 56.25 to 50.00 min, but redistributed exposure toward activation and away from amidation. The lower yield makes the council’s unresolved concern about downstream conversion experimentally relevant; it does not prove a rate-limiting step because several variables changed together. For the agentic workflow, this is actionable feedback: a subsequent design should use the measured outcome to reassess the stage allocation instead of assuming that a numerically consistent, higher-feed-rate candidate preserves yield. Figure S104 and Figure S20 retain the original experimental scheme, condition table and apparatus photograph; ESI Section 8 records feed preparation and implementation differences. Together with the photochemical case, this tests FlowPilot as a decision-support workflow whose proposals and assumptions can be checked against an implemented connected process.'),
      78:('The literature resource comprised 464 classified flow-chemistry records annotated with available reaction, reactor, material, operating-condition, yield and inlet-stream information. Missing metadata were retained as missing rather than filled from model inference. A complementary handbook-derived store contained 2,537 engineering rules with topic, severity, source and applicability information.[[3–4,21]] Relevant rules supplied context to chemistry interpretation; only checks implemented in the calculation and validation routines acted as deterministic constraints. Corpus field availability is reported in Figure S3; rule-base coverage is summarized in Figure 2e–f.'),
      94:('The internal ablation screen fixed the generator to Qwen3.8-27B and tested 15 conditions over the same three cases, producing 45 outcomes with one generation per condition-case pair. Figure 4d and Table S7 report the seven conditions focused on council and domain-reviewer removal; all 15 conditions remain in the archived module summary. Disabled specialists received no substitute score, and active council weights were renormalized. Standard deviations describe between-case variation, not repeatability. No experimental yield uncertainty was inferred from these computational repeats.'),
    }
    for index,value in edits.items():change(p,n[index],value,'Meeting review: scope, agentic interpretation and corrected figure/table links')
    p.after(n[49],[para(p,'The interface changes how that design brief is assembled and how the output is inspected. A worked Claude Opus 4.6 example separates researcher-supplied protocol text from software-assembled context (Figure S101) and compares the returned process diagram with the direct model response (Figure S102). It illustrates an interaction and artifact-management difference, not a token-saving claim: FlowPilot used more internal computation. The historical heuristic QA values displayed in this example are distinct from the fixed-criteria benchmark scores reported above.')])
    p.after(n[54],[para(p,'Council review connected these requirements to competing realizations. Chemistry and Kinetics assessed the two exposures together, while Fluidics and Safety examined the gas-injection junction and available equipment. Under the conversion-focused brief, the Chief selected the 5 mL first-stage coil; the balanced briefs selected 2 mL. Both choices retained the same liquid feed and 20 mL oxidation coil, so the smaller first-stage reactor represented a more compact screen, not increased substrate throughput. The recorded deliberations also retained the unresolved reverse-flow pathway and the need to establish oxygen-service compatibility (ESI Section 7.3). Thus the agent contribution was an inspectable choice among coupled process configurations, with its uncertainties carried forward to implementation.')])
    p.after(n[60],[para(p,'The council compared 12 inventory-constrained candidates for each response set rather than prescribing a common acceleration factor to both steps. For Set 1, it selected a near-batch screening reference with two 5 mL coils. Sets 2/3 converged on a 10 mL activation coil and a 5 mL amidation coil at higher feed rates. The latter retained more upstream exposure but shortened aminolysis; the reviewers explicitly left downstream conversion as an experimental question. Numerical reconciliation then checked the cumulative second-stage flow and the concentration–flow products used to preserve the intended amine equivalents. ESI Section 7.3 retains the specialist assessments and Chief selections, including their limitations; these are rationales for screening designs, not fitted kinetic or yield predictions.')])
    # Exact local replacements leave citations and existing scientific run styles intact.
    replacements={
      '(Figure 4b; Table S8)':'(Figure 4b)',
      '(Figure 4c; Table S9)':'(Figure 4c)',
      'the five-model aggregates in Tables S8–S9':'the five-model primary aggregates',
      ' or Tables S8–S9':'',
      'in Tables S8–S9':'reported here',
      'Table S15 and Figures S18–S19 document the conditions, irradiation and apparatus.':'Figure S18, Figure S103 and Figure S19 document the irradiation, original experimental scheme and condition table, and apparatus.',
      'ESI Section 8 and Tables S15–S18':'ESI Section 8, Figure S103 and Figure S104',
      'the companion version-history record':'ESI Section 9 and Table S101',
      'and an archived discussion example in ESI Section 6.4':'and the experimental-case council records in ESI Section 7.3',
    }
    for node in p.doc.iter(W+'p'):
        for old,new in replacements.items():patch_text(node,re.compile(re.escape(old)),lambda m,v=new:v)

    # These panels retain their layout and data; only institution-specific labels change.
    for index,num in [(57,5),(63,6)]:replace_image(p,n[index],FIG/f'figure{num}_neutral_labels.png',7.55)

def esi_revision(p):
    n=list(p.body)
    deleted_figure_ids=n[64].xpath('.//a:blip/@r:embed',namespaces=NS)
    # Numbers below refer to the user's notation-corrected source, not the output.
    remove(p,n[61:66],'Requested removal of Table S4 and Figure S4 in their entirety')
    remove(p,n[57:61],'Remove empty spacer paragraphs left between the retained corpus figure and deleted content')
    for rid in deleted_figure_ids:
        assert not p.doc.xpath('.//*[@r:embed=$rid]',namespaces=NS,rid=rid)
        rel=next(r for r in p.rels if r.get('Id')==rid)
        part=posixpath.normpath('word/'+rel.get('Target'))
        p.rels.remove(rel);p.parts.pop(part)
        LOG.append({'removed_embedded_figure':part,'reason':'Requested complete removal of old Figure S4'})
    remove(p,[n[184],n[185],n[186],n[187],n[188],n[189],n[190]],'Requested removal of Tables S8 and S9; source values remain archived')
    remove(p,[n[1043],n[1044],n[1055],n[1056]],'Restore original experimental scheme/tables instead of duplicate reconstructed tables')
    remove(p,[n[1075],n[1076],n[1077],n[1078]],'Requested removal of Tables S17 and S18')
    tb=n[182]; rows=list(tb.findall(W+'tr')); wanted=['No council','One-shot','Without Chemistry agent','Without Fluidics agent','Without Safety agent','Without Kinetics agent','Full FlowPilot']
    byname={text(r.findall(W+'tc')[0]):r for r in rows[1:]}
    for row in rows[1:]:tb.remove(row)
    for name in wanted:tb.append(byname[name])
    edits={
      24:'FlowPilot combines problem definition, chemical interpretation, retrieval, candidate review, quantitative calculations and equipment assignment around a laboratory-specific design. The external studies in Table S1 were evaluated using different tasks and are not included in the matched architecture comparison.',
      26:'The workflow returns an inventory-assigned screening proposal or explicit unresolved requirements. Measured observations and confirmed protocol facts are distinguished from hypotheses and model inference; hard safety and equipment limits remain mandatory. The component responsibilities and input-resolution rules are specified below.',
      28:'A run passes through intake, upstream chemistry interpretation, retrieval, engineering calculations, proposal generation, council review, inventory reconciliation and final publication (Figure S1a). Chemistry interpretation establishes the transformation, stage sequence, phase behavior and incompatible feeds. The council assesses candidate choices and proposes revisions; deterministic calculations reconcile dependent quantities after those revisions. Published screening parameters require a closed final contract. Missing capabilities produce an inventory-confirmation state; failed requirements produce a blocked state with diagnostic information rather than executable instructions.',
      53:'The frozen corpus contains 464 classified flow-chemistry literature records. Figure S3 gives the distributions of reaction class, reactor type, material, transformation label, inlet-stream count and reported yield. Field completeness varies: inlet-stream counts are available for 154 records, batch yields for 135 and optimized-flow yields for 285. Unknown, other and long-tail classifications remain in the stated denominators.',
      54:'Yield distributions describe the available records and are not a paired estimate of the benefit of flow: batch and flow values are missing for different subsets, and publication, optimization and chemistry differ across records. Each panel reports its own denominator.',
      68:'The matched architecture comparison assesses the delivered flow-process design under a common protocol, objective, inventory and output contract. It comprises five generator models, two architectures, three cases and three generation repeats (90 outcomes). A separate 15-condition Qwen3.8-27B screen examines internal modules; the seven conditions shown in main-text Figure 4d are tabulated here, with the complete screen retained in the archive. Only normalized final outcomes are scored, not internal discussion length or agent count.',
      72:'Generator models: Qwen3.6-27B, Qwen3.8-27B, GPT-4o, Claude Sonnet 4.6 and Claude Opus 4.6.',
      78:'Each generator was used with both architectures for three repeats of each chemistry, yielding 90 outcomes.',
      94:'Evaluator configuration',
      109:'For candidate i, criterion k and judge j, the criterion mean was:',
      180:'The module screen used Qwen3.8-27B and the same three cases, with one generation per case-condition. Table S7 contains exactly the seven conditions displayed in main-text Figure 4d, in the same order; the complete 15-condition, 45-outcome record is retained in the companion module summary. Disabled specialists received no substitute rating, and active council weights were renormalized. SD describes between-case variation rather than generation repeatability.',
      181:'Table S7. Module conditions shown in main-text Figure 4d. Means and sample SDs summarize three chemistry cases, with one generation per condition-case pair. Calls are mean generation-model calls per case; flags are total judge findings across the three cases. The complete 15-condition screen is archived separately.',
      183:'The selected screen contrasts removal of the whole council with removal of each domain reviewer. Its between-case SD and single generation per case do not support a causal ranking of reviewers. All additional screened conditions remain available in the source record.',
      218:'The GUI provides separate upstream chemistry and downstream proposal/council routes. Runtime provenance records the provider and model identifiers used. Reconstructing a run also requires the frozen inputs, inventory, generation settings, software revision, raw model events, deterministic audit and final record. Historical time conventions remain attached to their original campaigns.',
      221:'The browser figures document protocol entry, fixed-ID questions, inventory editing and JSON export. The front-page capture was repeated on 2 October 2026 using an unsaved copy of the study inventory with neutral display labels; equipment identifiers and numerical capabilities were unchanged. The inventory-form excerpts retain the 1 October capture. No design-generation call or laboratory experiment was performed for these screenshots.',
      222:'The GUI example uses the two-stage DPDTC-mediated amidation protocol described in Section 7.2. It demonstrates input handling rather than adding another benchmark chemistry.',
      224:'Figure S9 shows protocol entry, pending standardized questions, model routing, inventory binding and the readiness gate in one view.',
      227:'Figure S9. Annotated FlowPilot design studio. (1) Workspace navigation; (2) batch protocol; (3) fixed-ID follow-up questions; (4) upstream and council model routes; (5) the bound study inventory; and (6) the readiness gate. The actual 2 October browser capture uses a neutrally named, unsaved copy of the archived version-4 profile. Numbered annotations were added outside the unchanged screenshot; no design was generated for the capture. Figure S10 shows the equipment forms.',
      230:'Forms retain equipment identifiers, quantities, system membership, compatibility, operating limits and notes. Minimum pump flow and setting increment are separate fields; unknown optional ratings remain unspecified. The Constraints section records unavailable functions and permitted alternatives. The version-4 profile in this interface example is an archived inventory, not the later configuration used for the laboratory experiments (Table S11).',
      232:'A browser round-trip test retained the seven reactor records, three pump records, compatibility fields, constraints and provenance after editing, validation, export and re-import. The saved inventory was not overwritten. Direct JSON import bypassed LLM extraction; desktop and mobile tests confirmed that invalid or unvalidated edits could not be exported or used for design.',
      233:'Table S11. Used inventory: selected records from the archived version-4 profile and their design implications.',
      236:'Figure S10. Form-based inventory management. (a) Equipment categories and a pump record; (b) pump fields, including distinct minimum-flow and setting-increment entries; (c) unavailable equipment and permitted alternatives; and (d) validation, JSON export, versioned saving and design binding. Panels are excerpts from an actual 1 October GUI capture using the archived version-4 study profile. Blank values remain unknown, and validation warnings remain accessible. Full captures and the exported profile are archived.',
      238:'Table S12 illustrates distinct defects in retained outcomes and one evaluator false positive. Examples were selected to explain failure mechanisms, not estimate a failure rate. Multiple criterion flags may describe one defect; this post hoc interpretation does not change the archived scores or flag counts.',
      303:'For each selected candidate, the specialist assessments, candidate-specific Skeptic comments and Chief selection rationale are reproduced below. The presentation omits decision-counter fields and generalizes institution-specific labels; it is therefore an edited report extract, not a verbatim transcript. Numerical values, substantive criticisms and noted model errors are retained. The unedited all-candidate reports, full prompts, tool requests, unsuccessful responses and source hashes remain in FlowPilot_Figures5_6_full_council_reports.docx and the companion archive. Record identifiers F5-S1 to F6-S3 link the six runs to that archive.',
      304:'Recorded recommendations concern screening proposals, not approval to operate or independently verified conclusions. The interpretation notes identify numerical and physical errors in the reports rather than adopting them as scientific results.',
      305:'Scientific notation in these edited extracts is typeset for publication. Literal original strings are retained in the unedited source logs.',
      1026:'This section records the experimental procedures, original reaction/setup schemes, condition tables, apparatus photographs and analytical data for main-text Figures 5 and 6. The three response sets yielded two distinct reported operating points per chemistry; shared-set entries are not treated as independent repeats.',
      1037:'The sequence converts alpha-silyl sulfide 1a and acrylonitrile 2a through sulfide 3a to sulfoxide 4a [11]. Figure S103 reproduces the original experimental scheme and condition table; Figure S19 shows the assembled apparatus.',
      1042:'An aliquot of collected effluent was analyzed by 1H NMR spectroscopy using 1,3-benzodioxole as internal standard; the remaining material was concentrated and purified by silica-gel chromatography. Grouped Sets 1/3 (2 mL R1) gave 97% NMR yield at recorded system pressures of 5.1–5.4 bar; Set 2 (5 mL R1) gave 98% at 5.2–5.6 bar. Figure S103 retains the original condition table, including separately identified isolated yields.',
      1045:'The literature flow reference in main-text Figure 5 is 90% NMR yield [11, Table 4, entry 5]. It is contextual, not a matched control. In the original comparison (Figure S103), the 95% batch value is explicitly marked as isolated yield, and the batch first-step duration is listed as 180 min; the supplied agent input specifies 4 h. These source differences are retained rather than silently harmonized.',
      1047:'Figure S19. Photograph of the photochemical apparatus, showing the E-series system, UV-150 module, downstream illuminated coil, MFC, interstage T-mixer, check valve, BPR, cooling module and sample loop. Pressure-reference and check-valve details awaiting confirmation are listed in Section 8.7.',
      1049:'The sequence converts 3-methyl-4-nitrobenzoic acid (1b) through a 2-pyridyl thioester to N-benzyl-3-methyl-4-nitrobenzamide (3b) without intermediate isolation [12]. Figure S104 retains the original experimental scheme and condition table; Figure S20 shows the apparatus.',
      1054:'Effluent aliquots were analyzed by 1H NMR spectroscopy with 1,3-benzodioxole as internal standard. Remaining material was concentrated and purified by silica-gel chromatography. Set 1 gave 86% NMR yield at 5.2–5.3 bar; grouped Sets 2/3 gave 68% at 5.5–5.8 bar. Both coils operated at a reported 95 °C with direct interstage amine addition (Figure S104).',
      1057:'The main-text Figure 6 reference is the 98% NMR batch yield [12, Table 1, entry 5]. The original comparison in Figure S104 also records isolated-yield entries and a different DPDTC equivalence for the literature batch experiment. These distinct measurement bases and conditions are not pooled.',
      1059:'Figure S20. Photograph of the two-stage amidation apparatus. Two E-series pump channels supply the sample-loop/carrier stream and benzylamine feed; the image identifies the sample loop, interstage T-mixer, two heated coils, BPR and reservoirs.',
      1061:'Product-characterization data and spectra are reproduced below (Figures S21–S24). Masses, yields, shifts, couplings and observed LRMS values are retained from the experimental record, with punctuation and section-reference corrections. These spectra characterize the products; raw quantitative-NMR yield worksheets remain outstanding.',
      1079:'Implementation retained the proposed feed concentrations and interstage additions but used finite 2.0 mL sample loops. Both experiments used E-series pump channels and a 7 bar cartridge BPR; measured system-pressure ranges are reported separately. The photochemical reactors used 0.04 in tubing (1.016 mm ID), and the amidation coils used 0.093 in tubing (2.3622 mm ID). The original tables use rounded nominal labels of 1.00 and 2.40 mm. Actual dimensions should be used for subsequent hydraulic calculations.',
    }
    for index,value in edits.items():change(p,n[index],value,'Requested consolidation and meeting review')
    remove(p,n[73:78]+[n[79]]+n[96:99]+[n[219]],'Remove redundant model lists, count display and repeated provenance paragraph')
    # Add the supplied comparison figures without presenting the historical QA
    # score as the newer equal-weight LLM-judge benchmark.
    p.before(n[200],[
      para(p,'Worked example: prompt assembly and returned artifacts'),
      para(p,'Figures S101–S102 show one archived Claude Opus 4.6 photochemical-oxidation pair (repeat 01), not another repeated benchmark. The shared protocol plus objective contains 894 characters; the 4,342-character inventory is supplied to both methods. The complete one-shot messages contain 6,799 characters, whereas FlowPilot assembles 112,754 characters across six internal requests. The illustrated “hand-written” and “hand-pasted” labels describe prompt components, not measured human effort: templates and uploads can also be used with a direct LLM. Recorded totals are 50,175 versus 7,430 tokens and approximately 273 versus 100 s. Thus reduced manual prompt assembly does not imply reduced computation.'),
      para(p,'Panel c of Figure S101 retains the historical heuristic QA values of 0.96 and 0.63. Those trace-dependent diagnostics are not the architecture-neutral score used in main-text Figure 4. Under the fixed 14-criterion evaluation, this same pair scored 0.935898 (FlowPilot) and 0.916667 (one-shot). The FlowPilot final structured artifact had SCREEN disposition; its separately generated narrative was not fully synchronized. The archived one-shot parser reported type-validation errors. The visual labels concern these recorded artifacts, not experimentally verified safety or yield.'),
      image(p,FIG/'esi_fig_white.png',7.2),
      cap(p,'Figure S101. Historical prompt-assembly and artifact comparison for one matched Claude Opus 4.6 oxidation run. (a) Protocol/objective text and additional prompt components; inventory content is shared. (b) User-facing and accumulated model-facing characters. (c) Legacy heuristic QA diagnostics, distinct from the fixed-criteria benchmark. (d) Recorded model calls, tokens and runtime. Artwork supplied for this revision is retained; the text above specifies the interpretation and limitations of its labels.'),
      image(p,FIG/'esi_fig2_white.png',7.0),
      cap(p,'Figure S102. Returned artifacts from the same archived oxidation pair. Top: the inventory-assigned FlowPilot diagram. Bottom: an excerpt of the direct model response. The legacy “validated” label means software consistency checking of the structured result, not experimental validation. The icon’s generic syringe-pump artwork does not change the HPLC-pump assignment in its label. The approximately 8 min value is nominal reactor-volume/liquid-flow time, not measured gas–liquid contact time. The full responses and unsynchronized FlowPilot narrative remain in the source archive.'),
    ])
    replace_image(p,n[222+4],FIG/'gui_front_page.png',7.7)
    replace_image(p,n[235],FIG/'gui_inventory.png',7.7)
    # Use the already corrected corpus layout if available.
    corrected=BASE/'Figure_S3_corrected_20261002.png'
    if corrected.exists():replace_image(p,n[55],corrected,7.6)
    p.before(n[1045],[import_original_scheme(p,41),cap(p,'Figure S103. Original experimental reaction scheme, flow setup and batch/flow comparison for sulfoxide 4a. The first two rows refer to the published chemistry [11]; the FlowPilot rows report the present experiments. The original vector artwork is reproduced without redrawing. Its “181 min” second-stage label is a rounded inlet-reference index (181.82 min from the reported flows), not measured operating-pressure residence time. Superscript a marks isolated yield, including the 95% literature batch entry; other entries report NMR yield. Source-specific timing and rounded tubing labels are discussed in Section 8.3.')])
    p.before(n[1057],[import_original_scheme(p,70),cap(p,'Figure S104. Original experimental reaction scheme, flow setup and batch/flow comparison for amide 3b. The first two rows summarize the literature [12]; the FlowPilot rows report the present experiments. Original vector artwork is retained. Superscript a identifies isolated-yield values; unmarked values are NMR yields, and “quant” is the source’s qualitative isolated-yield label. The literature batch uses 1.10 equivalents of DPDTC versus 1.05 in the supplied flow input. The nominal 2.40 mm tubing label corresponds to the separately specified 0.093 in ID (2.3622 mm).')])
    # Restore the previously archived major-feature history, not minor fixes.
    history=[p.paragraph('Major pipeline development and version boundaries',style='Heading1',page_break=True,keep=True),
             para(p,'Table S101 records major changes using dated repository snapshots, reports and archived runs. Dates identify available evidence, not formal release dates or controlled efficacy tests. Later intake, inventory and transient-pressure capabilities must not be retroactively attributed to earlier benchmark campaigns.'),
             cap(p,'Table S101. Major development milestones and their rationale. Snapshot identifiers are evidence anchors; they do not imply an independently measured performance gain from each change.'),
             table(p,['Date / evidence','Major addition','Purpose'],MILESTONES,[1.8,2.25,2.45])]
    p.before(n[1083],history)
    # Remove decision-counter fields, retaining substantive assessments and
    # using explicit editorial omission marks rather than silently rewriting quotes.
    for node in list(p.body):
        v=text(node)
        if v=='Vetoes: []' or v.startswith('Candidate assessments 1 / Cross domain verdict: no_additional_veto'):
            remove(p,[node],'Requested omission of council decision-counter fields')
        elif 'veto' in v.lower():
            revised=v.replace('; the recorded veto list is empty','')
            revised=re.sub(r'No candidate is vetoed\.\s*','[Decision-counter sentence omitted.] ',revised)
            revised=re.sub(r'No veto is warranted\.','[Decision-counter sentence omitted.]',revised)
            revised=re.sub(r'No vetoes are issued\.?','[Decision-counter sentence omitted.]',revised)
            revised=re.sub(r',?\s*not a hard veto,?','; an unresolved screening issue,',revised)
            revised=re.sub(r' or veto\b','',revised)
            revised=re.sub(r', and does not constitute a veto','',revised)
            revised=re.sub(r',?\s*and does not veto any candidate\.?',' [Decision-counter clause omitted.]',revised)
            if revised!=v:change(p,node,revised,'Council extracts abridged with original records preserved')
    # Fix pointers before renumbering so deleted table numbers cannot be reused.
    replacements={
      'Tables S17–S18':'Section 8',
      'Tables S15–S16 report the laboratory pressure ranges separately.':'The experimental descriptions in Sections 8.3 and 8.4 report the measured system-pressure ranges separately.',
      '(Table S4)':'(Figure S3)',
      'No flag is not evidence of experimental success.':'An absence of flags does not establish experimental success.',
      'All four specialists reviewed all 12 candidates; the recorded veto list is empty.':'All four specialists reviewed all 12 candidates.',
    }
    for node in p.doc.iter(W+'p'):
        for old,new in replacements.items():patch_text(node,re.compile(re.escape(old)),lambda m,v=new:v)
    metadata_heading=next(node for node in p.body if text(node)=='Outstanding experimental metadata')
    for br in list(metadata_heading.findall(W+'pPr/'+W+'pageBreakBefore')):
        br.getparent().remove(br)

def neutralize(pkg):
    # Affiliations and author names remain intact. Laboratory shorthand is
    # removed from presentation, while original identifiers stay in the archive.
    patterns=[
      (r"KHU\u2019s apparatus|KHU's apparatus",'the experimental apparatus'),
      (r"KHU\u2019s supplied records|KHU's supplied records",'the experimental records'),
      (r'Actual KHU ', 'Actual '),
      (r'The KHU qualitative', 'The reported qualitative'),
      (r'The KHU observation', 'The reported observation'),
      (r'the KHU experiment', 'the experiment'),
      (r'no measured KHU values', 'no measured laboratory values'),
      (r'Irradiation sources supplied by KHU', 'Irradiation sources used in the experiments'),
      (r'Supplied KHU product-characterization', 'Original product-characterization'),
      (r'earlier KHU backflow report', 'earlier backflow report'),
      (r'KHU reports ', 'The experimental record reports '),
      (r'KHU supplied ', 'Supplied '),
      (r'KHU\u2019s |KHU\x27s ', 'the study '),
      (r'\bKHU\b', 'study laboratory'),
    ]
    for node in pkg.doc.iter(W+'p'):
        if node.xpath('ancestor::w:sdt',namespaces=NS):continue
        for pat,new in patterns:patch_text(node,re.compile(pat),lambda m,v=new:v)
        # Source paths are replaced by case-record labels in the presentation.
        for pattern,label in [(r'outputs/khu_revised_six_20260915/presentation/figure6_set1 through figure6_set3','F6-S1 through F6-S3 (15 September 2026)'),
                              (r'inventory_khu/\S+','the archived inventory source record')]:
            patch_text(node,re.compile(pattern,re.I),lambda m,v=label:v)

def renumber(esi,main):
    maps={'Figure':{},'Table':{}}
    for n in esi.body:
        m=re.match(r'^(Figure|Table) S(\d+)\.',text(n))
        if m:
            assert int(m[2]) not in maps[m[1]],m[0]
            maps[m[1]][int(m[2])]=len(maps[m[1]])+1
    def sub(m):
        prefix,a,sa,b,sb=m.groups();kind='Figure' if prefix.startswith('Figure') else 'Table'
        mapping=maps[kind]
        nums=list(range(int(a),int(b)+1)) if b else [int(a)]
        assert all(i in mapping for i in nums),('Unresolved reference',m[0])
        mapped=[mapping[x] for x in nums]
        # An old span may contain newly inserted figures; referencing the whole
        # new span is appropriate for the original compound-context references.
        return prefix+' S'+str(mapped[0])+sa+('–S'+str(mapped[-1])+(sb or '') if b else '')
    for pkg in (esi,main):
        for node in pkg.doc.iter(W+'p'):
            if not node.xpath('ancestor::w:sdt',namespaces=NS):patch_text(node,REF,sub)
    (OUT/'numbering_map.json').write_text(json.dumps(maps,indent=2))
    return maps

def format_references(node):
    value=text(node)
    caption=re.match(r'^(Figure|Table) S?\d+\.',value)
    spans=[caption.span()] if caption else [m.span() for m in re.finditer(r'\b(?:Figures?|Tables?)\s+S?\d+[a-z]?(?:\s*(?:[–-]|,\s*(?:and\s+)?|and)\s*S?\d+[a-z]?)*',value)]
    if not spans:return
    offset=0
    for run in list(node):
        if run.tag!=W+'r':
            offset+=len(text(run));continue
        rv=text(run);end=offset+len(rv)
        bounds={0,len(rv)}
        for a,b in spans:
            if offset<a<end:bounds.add(a-offset)
            if offset<b<end:bounds.add(b-offset)
        if not rv or len(run.findall(W+'t'))!=1:
            offset=end;continue
        splits=sorted(bounds)
        for lo,hi in zip(splits,splits[1:]):
            part=deepcopy(run);part.find(W+'t').text=rv[lo:hi]
            part.find(W+'t').set('{http://www.w3.org/XML/1998/namespace}space','preserve')
            props=part.find(W+'rPr')
            if props is None:props=E.Element(W+'rPr');part.insert(0,props)
            inside=any(a<=offset+lo<b for a,b in spans)
            if caption or inside:
                for tag in (W+'b',W+'bCs'):
                    for b in list(props.findall(tag)):props.remove(b)
                E.SubElement(props,W+'b',{W+'val':'1' if inside else '0'})
            run.addprevious(part)
        node.remove(run);offset=end

def refresh_figure_descriptions(pkg):
    last_image=None
    for node in pkg.body:
        if node.xpath('.//a:blip | .//w:object',namespaces=NS):last_image=node
        if text(node).startswith('Figure S') and last_image is not None:
            for item in last_image.iter():
                if E.QName(item).localname in {'docPr','cNvPr'}:
                    item.set('descr',text(node))
            last_image=None

def main():
    OUT.mkdir(exist_ok=True)
    m=Package(BASE/'FlowPilot_manuscript_october2_notation_corrected.docx')
    e=Package(BASE/'FlowPilot_ESI_october2_notation_corrected.docx')
    hashes={str(p.source):sha256(p.source.read_bytes()).hexdigest() for p in (m,e)}
    main_revision(m);esi_revision(e)
    for p in (m,e):neutralize(p)
    maps=renumber(e,m)
    refresh_figure_descriptions(e)
    for label,p in [('manuscript',m),('ESI',e)]:
        # Preserve formatting in unchanged paragraphs and type new scientific text.
        for node in p.doc.iter(W+'p'):
            if not node.xpath('ancestor::w:sdt',namespaces=NS):typeset_paragraph(node)
        upright_math_descriptors(p.doc)
        # Caption labels and in-text cross-references, not entire captions.
        for node in p.body.findall(W+'p'):format_references(node)
        order_properties(p.doc)
        path=BASE/f'FlowPilot_{label}_october2_final_real1.docx'
        p.save(path)
    assert all(sha256(Path(path).read_bytes()).hexdigest()==digest for path,digest in hashes.items())
    (OUT/'revision_log.json').write_text(json.dumps(LOG,indent=2,ensure_ascii=False))
    print(json.dumps({'changes':len(LOG),'numbering':maps},indent=2))

if __name__=='__main__':main()
