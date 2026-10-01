"""Source-grounded second review of the manuscript and matching ESI.

No figures, benchmark values, frozen prompts, transcripts or spectra are replaced.
"""
from copy import deepcopy
from hashlib import sha256
from pathlib import Path
import csv
import json
import re

from lxml import etree as E

import revise_manuscript_20260930 as base
from revise_submission_20260929 import science_typography
from revise_layout_discussion_20260928 import order_properties

ROOT, BASE = base.ROOT, base.BASE
OUT = BASE / 'professional_review_20260930'
MAIN = BASE / 'manuscript_submission_revised_20260930.docx'
ESI = BASE / 'esi_submission_revised_20260929.docx'
W, NS, text, Package = base.W, base.NS, base.text, base.Package
changes = []


def revise(pkg, prefix, value, reason, tag='main'):
    matches = [p for p in pkg.body if text(p).startswith(prefix)]
    assert len(matches) == 1, (prefix, len(matches))
    p = matches[0]
    before = text(p)
    pkg.revise(p, value)
    changes.append({'document': tag, 'before': before, 'after': value, 'reason': reason})
    return p


def add_after(pkg, p, value, reason, tag='main'):
    new = base.paragraph_like(pkg, p, value)
    pkg.after(p, [new])
    changes.append({'document': tag, 'before': '', 'after': value, 'reason': reason})


def revise_main():
    m = Package(MAIN)
    revise(m, 'Translating a batch protocol into continuous flow requires',
           'Translating a batch protocol into continuous flow requires chemical interpretation, quantitative engineering, and compatible equipment to be considered together. '
           'FlowPilot combines large language models (LLMs), standardized chemist intake, literature retrieval, deterministic calculation, and specialist council review to generate inventory-constrained process proposals. '
           'A shared design record links feed compositions, stage conditions, equipment assignments, and the process diagram. '
           'The workflow is supported by 464 classified literature records and 2,537 engineering rules. '
           'Across three chemistry cases, five generator models, and three repeats, matched one-shot and FlowPilot generation produced 90 outcomes evaluated using 14 fixed criteria and separate LLM judges. '
           'Mean benchmark scores were 0.794 for one-shot generation and 0.917 for FlowPilot; judge-reported critical flags totaled 98 and 4, respectively. '
           'Laboratory implementations of photoredox Giese addition/oxidation and DPDTC-mediated amidation gave 97–98% and 68–86% NMR yields, respectively, with isolated yields of 94% and 84% at selected operating points. '
           'These results demonstrate a traceable connection between reaction-level instructions, quantitative flow-process design, and laboratory implementation, while keeping computational design assessment distinct from measured chemical performance.',
           'Tighten the abstract; define LLM; identify flags as judge reports, not ground-truth errors.')

    value = base.INTRODUCTION[-1].replace('[[56]]', '').replace('[[57,58]]', '')
    revise(m, 'Here we present FlowPilot,', value,
           'Remove the misplaced isoxazole reference and reserve chemistry citations for their specific case studies.')
    revise(m, 'The downstream council supplies chemical and cross-domain review',
           'The downstream council reviews alternatives across chemical and engineering domains (Figure 1d). '
           'The Designer proposes candidate operating points; Chemistry, Kinetics, Fluidics, and Safety specialists assess transformation fidelity, reaction-time plausibility, transport, equipment compatibility, and hazards. '
           'The Skeptic examines assumptions and contradictions, and the Orchestrator selects among the reviewed candidates. '
           'The retained record includes revisions, rejected alternatives, and selection rationales, allowing the final proposal to be inspected alongside the decisions that produced it.',
           'Reduce repetition of the detailed agent responsibilities in Methods and ESI.')
    revise(m, 'For a reconciled result, the output is a structured design package',
           'The selected proposal is reconciled into a common final design record containing stage and stream parameters, equipment assignments, and the process graph. '
           'Numerical views and the diagram are generated from this record, while unresolved requirements and earlier calculations remain available as diagnostics. '
           'This arrangement links proposed conditions to specific equipment and preserves the context for later experimental feedback. '
           'Figures S1–S2 and Tables S1–S3 describe the data contracts and provenance mechanisms.',
           'State the implemented record structure without repeating a general validation disclaimer.')
    revise(m, 'Plan-aware retrieval improves analogy quality', 'Plan-aware retrieval and metadata alignment',
           'The measured endpoint is metadata alignment, not independent chemical analogy accuracy.')
    revise(m, 'Coverage alone does not show that a relevant precedent will be selected.',
           'FlowPilot uses the chemistry plan to enrich retrieval queries with reaction class, mechanism, catalyst family, solvent, and wavelength information (Figure 3a). '
           'The operational search progresses from mechanism- and phase-filtered paired records to relaxed filters and, when necessary, the full corpus; a field-aware score ranks the candidates (Figure S5; Table S6). '
           'We examined the contribution of metadata in separate retrospective analyses using stored corpus embeddings. These analyses isolate query enrichment and reranking behavior rather than testing the complete live retrieval workflow.',
           'Correct the mismatch between operational retrieval and the actual offline evaluation scripts.')
    revise(m, 'Figure 3. Plan-aware retrieval.',
           'Figure 3. Retrieval workflow and retrospective metadata-alignment analyses. '
           '(a) Operational query enrichment, metadata-filter relaxation, and field-aware ranking. '
           '(b) Two examples of query enrichment, showing counts of chemically specific terms. '
           '(c) Mean fraction of the top five retrieved records with the same recorded photocatalyst family as the query; n denotes query count. '
           '(d) An illustrative iridium-query ranking. The analyses in (c–d) use stored embeddings and the offline scoring implementation described in Methods and ESI Section 2.4. '
           'Missing family labels are unassigned metadata, not verified chemical mismatches; displayed percentage gains are percentage-point differences.',
           'Align the caption with source scripts and avoid misreading metadata gaps or percentage-point gains.')
    revise(m, 'The enriched query representation approximately doubled',
           'In the two query-enrichment examples, chemically specific term counts increased from 7 to 15 and from 7 to 14 (Figure 3b). '
           'A separate rank analysis evaluated 20 neighbors for each of 80 corpus queries, giving 1,600 query–result pairs. '
           'Adding the metadata score changed 401 ranks (25.1%) relative to embedding-only ordering. These changes quantify reranking activity, not an independent measure of reaction relevance.',
           'Recompute denominators and rank-change count from the archived CSV; avoid conflating changed ranks with improved chemistry.')
    revise(m, 'The strongest retrieval improvement was observed for photocatalyst-family matching.',
           'In the family-alignment analysis, the mean proportion of top-five records sharing the query’s recorded catalyst family increased from 40.0% to 95.0% for iridium, 42.7% to 90.7% for organic dyes, 77.3% to 94.7% for ruthenium, 87.1% to 98.6% for TiO2, and 45.0% to 60.0% for ZnO (Figure 3c). '
           'The corresponding query counts were 8, 15, 15, 14, and 4. Catalyst-family metadata also contribute to the reranking score, so this is a descriptive alignment test, not independent validation of mechanistic transfer or predicted yield. '
           'Figure 3d illustrates the resulting ranking change; Figures S5–S6 and Tables S6–S7 provide the workflow, analysis settings, and applicability limits.',
           'Clarify the endpoint, family-specific sample sizes, and its dependence on the ranking features.')
    revise(m, 'Improved retrieval does not by itself establish',
           'To evaluate the delivered designs, we compared one-shot generation with the full FlowPilot pipeline under the same chemistry, inventory, and requested output contract (Figure 4a). '
           'The cases covered copper(I)-catalyzed azide–alkyne cycloaddition (CuAAC), photochemical oxidation, and hydrogenolysis, with three generation repeats per model–architecture–case combination. '
           'Separate Qwen, OpenAI, and Claude evaluator calls scored normalized final designs against 14 fixed criteria covering chemical fidelity, material and reactor consistency, inventory, operating procedures, and uncertainty. '
           'Applicable integer ratings from 0 to 4 were averaged with equal criterion and judge weights and normalized to a score from 0 to 1; critical flags were recorded separately. '
           'Tables S8–S10 define the rubric and module screen.',
           'Define CuAAC and distinguish evaluator calls from generator architectures.')
    revise(m, 'The earlier council model-matrix and candidate-budget experiment',
           'A separate council model-matrix and candidate-budget study used an integrated isoxazole synthesis case[[56]] (Figure S12; Tables S18–S21). '
           'Across five repeats, the normalized engineering radar-area score changed from 0.31 ± 0.03 before council review to 0.48 ± 0.23 afterward. '
           'This geometric summary of engineering indicators is distinct from the fixed-criteria LLM-judge score. '
           'The archived model matrix, revision counts, and normalization definitions document configuration sensitivity. '
           'Figure S13 and Table S22 retain a worked example in which the structured topology and accompanying narrative disagree, together with the corresponding one-shot output.',
           'Place reference 56 against the actual isoxazole case; remove unsupported generic citation and normalize ± typography.')
    revise(m, 'The ESI also shows how the chemist supplies and inspects',
           'The GUI examples connect these assessments to the chemist’s workflow. Protocol entry and fixed-ID follow-up questions are shown in Figures S14–S15 and Table S23; inventory normalization and warnings are shown in Figure S16 and Table S24. '
           'Figure S17 and Table S25 link stage and feed parameters to the stored topology. '
           'Table S26 records concrete design and evaluation issues, while Figure S18 and Table S27 provide council exchanges and selection decisions. '
           'Figure S19 and Table S28 summarize execution and saved-run provenance. The following laboratory cases examine implementation and chemical outcomes for two connected reaction sequences.',
           'Shorten the figure-by-figure narrative while preserving every ESI citation.')
    revise(m, 'The first connected-process case couples photoredox Giese addition',
           'The first laboratory case couples photoredox Giese addition with oxidation of the resulting sulfide to a sulfoxide (Figure 5). '
           'The reported chemistry uses photoinduced electron transfer to generate an alpha-thiomethyl radical from an alpha-silyl sulfide, followed by addition to acrylonitrile and a separate oxygen-dependent oxidation step.[[57]] '
           'Its process-design requirement is the transition from an oxygen-free first stage to an oxidizing second stage within one connected apparatus. '
           'The supplied batch input specified 4 h under argon followed by 6 h in air; the distinct literature-comparison timings are retained in ESI Section 12.3.',
           'Keep the chemical insight but move provenance minutiae to ESI and avoid treating batch time as a kinetic law.')
    revise(m, 'The revised proposal introduces pure oxygen only between the stages',
           'Kyung Hee University (KHU) implemented the proposal using a 2 or 5 mL PFA coil in a Vapourtec UV-150 module followed by a separately illuminated 20 mL FEP coil, both at 25 ± 1 °C. '
           'A 2.0 mL sample loop introduced substrate 1a (0.100 M), acrylonitrile (0.200 M), and iridium photocatalyst (0.500 mM) into a carrier stream at 0.020 mL min−1. '
           'The nominal first-stage liquid residence times were therefore 100 and 250 min. Oxygen entered at the interstage mixer at a controller setting of 0.090 mL min−1. '
           'Under the proposal’s STP convention (273.15 K, 1 atm), this setting corresponds to 0.00402 mmol min−1 of oxygen, or 2.01 equivalents relative to the nominal substrate feed. '
           'The resulting second-stage inlet-reference index, 20/(0.020 + 0.090), is 181.82 min; it is not a physical residence time at operating pressure. '
           'The laboratory gas-reference conditions and sample-plug collection details are distinguished from these nominal calculations in ESI Section 12.',
           'Define KHU; distinguish sample-loop concentrations, nominal oxygen calculations, and the inlet-reference index.')
    revise(m, 'An earlier high-gas-flow trial showed upstream gas entry',
           'KHU had previously reported backflow toward the first reactor while using pure oxygen at 0.43 mL min−1 with liquid at 0.020 mL min−1. '
           'The revised operating point retained pure oxygen but reduced the gas setting to 0.090 mL min−1. '
           'The proposal included a gas-line check valve; this protects the gas branch against reverse liquid entry and does not, by itself, establish protection of the upstream liquid branch. '
           'The implemented setup used a 7 bar cartridge BPR, with reported system pressures of 5.1–5.6 bar. '
           'The 2 mL first-stage configuration, reported jointly as Sets 1/3, gave 97% NMR and 94% isolated yield of sulfoxide 4a; the 5 mL configuration (Set 2) gave 98% NMR yield. '
           'Both configurations therefore supported productive telescoping, although the one-percentage-point difference cannot establish a residence-time benefit without replicate data. '
           'Figure S20 and Tables S29–S31 document design inputs, proposed settings, and outcomes; Figures S23–S25 and Tables S37–S39 provide irradiation, apparatus, and implementation details.',
           'Correct the actual gas used in the backflow trial and the branch-specific function of a gas-line check valve; retain measured outcomes.')
    revise(m, 'The second case is a thermal two-stage amidation',
           'The second laboratory case is the thermal amidation of 3-methyl-4-nitrobenzoic acid with benzylamine (Figure 6). '
           'Activation with 2,2′-dipyridyldithiocarbonate (DPDTC) in the presence of DMAP generates a 2-pyridyl thioester, which undergoes downstream aminolysis to form the amide.[[58]] '
           'The batch input specifies two 30 min stages at 95 °C, separated by cooling and amine addition. '
           'In the connected process, the intermediate is transferred without isolation, so its formation and downstream consumption must be considered together when choosing feed ratios and stage times.',
           'Define DPDTC and describe coupled intermediate formation without asserting an unmeasured rate-limiting step.')
    revise(m, 'Set 1 afforded amide 3b',
           'Set 1 afforded amide 3b in 86% NMR and 84% isolated yield. '
           'Sets 2/3 were reported together for a 10 mL first reactor and 5 mL second reactor at feed rates of 0.280 and 0.0700 mL min−1, respectively. '
           'These settings give nominal stage times of 35.71 and 14.29 min and a reported NMR yield of 68%. '
           'Relative to Set 1, the total nominal time decreased from 56.25 to 50.00 min while the substrate molar feed increased from 0.0800 to 0.140 mmol min−1. '
           'The higher-flow configuration gave lower yield despite its longer first-stage time, illustrating why the complete sequence must be assessed. The data do not isolate a kinetic cause because the reactor configuration, flow rates, and second-stage time changed together. '
           'Figure S21 and Tables S32–S34 retain the design records and outcomes; Figures S26–S27 and Tables S38–S39 document the apparatus, implementation, and molar-feed calculations.',
           'Avoid presenting a causal yield-throughput trade-off or a measured productivity increase from finite sample-loop data.')
    revise(m, 'The alpha-bromination example from the original preprint',
           'An additional alpha-bromination design based on the chemistry of Lu et al.[[59]] is retained in Figure S22 (ESI Section 9) as a historical computational example, separate from the present laboratory validation.',
           'Reference 59 is the chemistry paper, not the FlowPilot preprint.')

    discussion = [
        'FlowPilot treats batch-to-flow translation as a connected-process design problem. Its principal contribution is to link reaction interpretation to feed compositions, stage calculations, and inventory assignments in a common record. '
        'This makes an operating proposal inspectable across chemistry and engineering domains, rather than relying on agreement between separately generated numerical tables and diagrams. '
        'The task complements microreactor assistants, synthesis-development platforms, and process-simulation agents, while providing a structured starting point for subsequent optimization.[[46-51,60,61]]',
        'The architecture comparison supports this approach within the tested cases: FlowPilot had higher mean scores than matched one-shot generation for each of the five retained generators, although the size of the difference varied by model (Figure 4b; Table S11). '
        'In the Qwen3.8-27B module screen, removing the council reduced the mean score from 0.928 to 0.670, whereas individual specialist removals had smaller effects (Figure 4d; Table S10). '
        'The single generation per case-condition supports descriptive module attribution, not a statistical ranking of every component. '
        'The separate candidate-budget study likewise shows that wider exploration does not automatically improve the selected design (Figure S12; Tables S18–S21).',
        'The laboratory studies examine aspects not resolved by a design score. The Giese/oxidation sequence requires spatial separation of the oxygen-free and oxygen-fed stages; the backflow observation also shows that gas equivalents and volume-flow arithmetic alone are insufficient to establish operability. '
        'The amidation sequence links thioester formation, interstage amine addition, and downstream conversion without isolation. Its different yields at the two operating points demonstrate the importance of evaluating the connected process rather than either reactor in isolation. '
        'These sample-loop experiments establish product formation under the reported settings, while sustained reagent-fed operation and experimental repeatability require separate evaluation.',
        'Retaining design conflicts and experimental deviations is therefore an important part of the workflow. The one-shot and FlowPilot inconsistencies documented in Tables S22 and S26 identify where further checking is needed; the record links such findings to the inputs, calculations, and review decisions that produced them. '
        'Measured outcomes can subsequently be returned with the actual operating conditions to guide revision. Combining this record with Bayesian optimization or language-guided experimental priors offers a route to prospective closed-loop development,[[30-34,62]] rather than treating a numerically consistent first proposal as an optimized reaction.',
    ]
    replace_prose_section(m, 'DISCUSSION', 'CONCLUSION', discussion, 'Condense repetition and keep discussion focused on interpretation, with no new subheadings.')
    conclusion = [
        'FlowPilot connects reaction-level input to a quantitative, inventory-constrained continuous-flow proposal through standardized intake, chemistry-aware retrieval, deterministic calculation, and specialist review. '
        'A common design record makes stream conditions, reactor assignments, process topology, and revision history available for inspection. '
        'In the five-model, three-case comparison, the mean fixed-criteria score was 0.917 for FlowPilot and 0.794 for one-shot generation, with fewer judge-reported critical flags.',
        'Laboratory implementation of two connected-stage reactions afforded a sulfoxide in 94% isolated yield and an amide in 84% isolated yield at selected settings. '
        'The observed backflow issue and the differing amidation yields emphasize that computational consistency and chemical performance are complementary tests. '
        'By retaining proposed conditions, implemented changes, and measured outcomes as distinct records, FlowPilot provides a practical basis for chemist-led flow-process development and subsequent experimental refinement.',
    ]
    replace_prose_section(m, 'CONCLUSION', 'METHODS', conclusion, 'Remove generic extrapolation to all engineering problems and shorten the repeated workflow summary.')

    revise(m, 'Each design began with a batch protocol',
           base.METHODS[0][1][0].replace('Each design began', 'The standardized workflow begins').replace('Follow-up questions used', 'Follow-up questions use').replace('optional evidence could be', 'optional evidence can be').replace('The confirmed package was frozen', 'The confirmed package is frozen').replace('Measured observations, protocol facts, and hypotheses were stored separately, while declared equipment and operating limits constrained design feasibility.', 'Measured observations, protocol facts, and hypotheses are stored separately. Hard safety and equipment constraints remain mandatory even when earlier measurements were obtained under different conditions.'),
           'Do not imply that every historical benchmark used the later intake GUI; clarify non-overridable hard constraints.')
    retrieval = revise(m, 'A chemistry plan was extracted from the intake package',
           'For operational retrieval, the chemistry plan supplies reaction class, mechanism, catalyst, solvent, phase regime, and operating conditions. '
           'The search begins with paired batch-flow records under mechanism and phase filters, relaxes filters when fewer than three candidates remain, and expands to the full corpus when no paired result remains. '
           'The final score combines semantic and field similarity with weights of 0.60 and 0.40. The operational field components and weights are listed in Table S6. '
           'In the architecture benchmark, the designated source record was excluded before final retrieval selection; this restriction does not establish absence from model pretraining.',
           'Separate the production retrieval procedure from the offline figure-generation analysis.')
    add_after(m, retrieval,
              'The retrospective figure analyses used stored corpus embeddings and inverse Euclidean-distance similarity, 1/(1 + d), rather than newly generated query embeddings. '
              'The rank analysis reranked the 20 nearest non-self records for 80 queries using photocatalyst, solvent, and wavelength terms weighted 0.30, 0.20, and 0.20; temperature and concentration were not included in that offline score. '
              'A separate family analysis sampled up to 15 queries per recorded family and measured the fraction of matching-family records in each top-five result set. '
              'Its 56 queries and script-specific similarity functions are documented in ESI Section 2.4 and Table S7. These analyses measure metadata alignment, not the full live retrieval policy or independent chemical correctness.',
              'Document actual scripts, sample units, reduced field score, and separate analysis cohorts.')
    value = base.METHODS[-1][1][0].replace('The substrate/acrylonitrile/iridium-catalyst feed was prepared for the oxygen-free first stage; pure oxygen', 'The first stage was designed to remain oxygen-free, although the exact feed-degassing procedure was not specified in the supplied laboratory methods. Pure oxygen')
    revise(m, 'KHU implemented the two connected-stage cases using', value,
           'Do not report an unprovided oxygen-free feed-preparation procedure as experimentally verified.')
    revise(m, 'Matched architecture comparison and independent judging',
           'Matched architecture comparison and LLM judging',
           'Use a heading consistent with the separate-call, shared-model-family evaluator design.')
    value = base.METHODS[5][1][1].replace('independent judge', 'separate judge')
    value = value.replace('Scoring anchors, criterion definitions, and applicability rules are specified', 'Some evaluator model families also served as generators; separate calls therefore do not establish statistical independence or remove possible model-family preferences. Scoring anchors, criterion definitions, and applicability rules are specified')
    revise(m, 'For judging, final designs were normalized', value,
           'Explicitly distinguish separate evaluator calls from independent human or statistically independent validation.')
    value = base.METHODS[6][1][0] + ' The comparisons are descriptive; no statistical-significance claim or population-wide generalization is made from three repeats and three selected chemistries.'
    revise(m, 'For each generator and architecture, the three case scores', value,
           'Keep the statistical unit and scope explicit without adding a repeated limitations section.')

    # Remove an empty heading, not the unresolved author approvals elsewhere.
    notes = base.find(m, 'Notes')
    assert text(notes.getnext()).strip() == 'Availability of data and code'
    changes.append({'document': 'main', 'before': 'Notes', 'after': '', 'reason': 'Remove an empty heading; do not invent a competing-interests declaration.'})
    m.body.remove(notes)
    abbreviations = next(p for p in m.body if text(p).startswith('BPR, back-pressure regulator;'))
    value = text(abbreviations).replace('BPR, back-pressure regulator;', 'BPR, back-pressure regulator; CuAAC, copper(I)-catalyzed azide–alkyne cycloaddition;').replace('DPDTC, dipyridyldithiocarbonate;', 'DPDTC, 2,2′-dipyridyldithiocarbonate;').replace('LLM, large language model;', 'KHU, Kyung Hee University; LLM, large language model;')
    revise(m, 'BPR, back-pressure regulator;', value, 'Define introduced acronyms consistently.')
    revise(m, 'Yasukawa, T.; Kobayashi, S.',
           'Yasukawa, T.; Kobayashi, S. Translation of Batch to Continuous Flow in Photoredox Reactions. ACS Cent. Sci. 2021, 7 (7), 1099–1101. https://doi.org/10.1021/acscentsci.1c00711.',
           'Normalize the incomplete reference using verified journal, volume, issue and pages.')
    ref = next(p for p in m.body if text(p).startswith('Kim, S.; Choi, J.; Jang, K.;'))
    value = text(ref).replace('arXiv January 22, 2026.', 'arXiv 2026, 2601.15743v1. Preprint.')
    revise(m, 'Kim, S.; Choi, J.; Jang, K.;', value, 'Label the Materealize source as a preprint, consistently with other arXiv references.')
    return m


def replace_prose_section(m, start, end, values, reason):
    old = base.section(m, start, end)
    nodes = [base.paragraph_like(m, old[0], value) for value in values]
    changes.append({'document': 'main', 'before': '\n\n'.join(text(p) for p in old), 'after': '\n\n'.join(values), 'reason': reason})
    base.replace_section(m, start, end, nodes)


def revise_esi():
    s = Package(ESI)
    def edit(prefix, value, why):
        return revise(s, prefix, value, why, 'esi')
    p = next(p for p in s.body if text(p).startswith('FlowPilot separates chemical interpretation,'))
    v = text(p).replace('The workflow is governed by an explicit information-authority order:', 'Interpretation of design evidence follows an explicit information-authority order:')
    v += ' This evidence hierarchy does not authorize overriding a hard safety or equipment constraint: earlier experimental success outside a declared limit does not make a new proposal compliant.'
    edit('FlowPilot separates chemical interpretation,', v, 'Clarify hard constraints separately from evidence precedence.')
    p = next(p for p in s.body if text(p).startswith('The FinalDesignContract is rebuilt'))
    edit('The FinalDesignContract is rebuilt',
         'The FinalDesignContract is rebuilt after engineering realization and inventory reconciliation. A publishable screening result contains canonical parameters, stage definitions, streams, instrument assignments, a process graph, chemistry identity, and the available operating and validation instructions. '
         'The implemented gates check consistency among these records, including arithmetic, declared time basis, equipment allocation, and topology. An unresolved blocking check prevents publication of executable parameters and retains intermediate values as diagnostics. '
         'Passing these checks establishes compliance with the implemented contract; it does not establish chemical completeness, physical safety under all transients, or experimental yield.',
         'Replace an unqualified universal validation guarantee with the implemented gate scope.')
    edit('Three independent judge families evaluated every candidate:',
         'Three separate evaluator model families assessed each candidate:',
         'Keep evaluator terminology consistent with the shared model families described in main-text Methods.')
    p = edit('The frozen retrieval analysis contains 1,600 query-result pairs.',
         'The retrospective rank analysis used the archived implementation in visualization/fig3c_score_decomposition.py. Eighty corpus records were sampled without replacement with seed 42. '
         'Each query was compared with its 20 nearest non-self records, giving 1,600 query–result pairs. Embedding similarity was 1/(1 + d), where d is Euclidean distance. '
         'The offline final score was 0.60 times embedding similarity plus 0.40 times a reduced field score containing photocatalyst, solvent, and wavelength terms weighted 0.30, 0.20, and 0.20. '
         'Temperature and concentration terms were not included, and the reduced weights were not renormalized. The source export records 401 changed ranks (25.1%) and no exact query-ID self-hit. '
         'This is an offline reranking analysis, not a run of the complete operational query-enrichment and tier-filtering pipeline in Table S6.',
         'Correct the actual retrieval metric, sample size, reduced field score, and interpretation.')
    add_after(s, p,
              'The separate family-alignment analysis used visualization/fig3d_rag_quality.py, with seed 42 and up to 15 queries per family: iridium (8), organic dyes (15), ruthenium (15), TiO2 (14), and ZnO (4). '
              'For each query, the metric was the fraction of five retrieved records with the same name-pattern-derived family label; reported rates are means over queries. '
              'The rank and family scripts both used photocatalyst, solvent, and wavelength terms, but their family classifiers and wavelength functions were not identical. In the rank script, wavelength similarity is 1 within 30 nm and decreases linearly to 0 at 100 nm; the family script uses max(0, 1 − |Δλ|/100). '
              'Matching nonempty family labels score 1, nonmatching labels score 0.3, and absent labels score 0. The rank script groups unrecognized nonempty catalyst names as other_pc, whereas the family script leaves unrecognized names unassigned. Solvent scoring uses normalized exact-name matches. '
              'Because family labels enter both ranking and evaluation, these rates describe metadata alignment rather than independent validation of chemical analogy quality.',
              'State separate analysis cohorts and source-specific scoring definitions.', 'esi')
    # Keep leakage controls distinct from the offline family helper's weaker masking.
    p = next(p for p in s.body if text(p).startswith('The separate family-alignment analysis used'))
    add_after(s, p,
              'Source-exclusion controls in the architecture campaigns remove the designated held-out source identifiers before final retrieval selection. The offline rank-pair CSV independently confirms no exact query self-hit. '
              'The family-analysis helper sets the self semantic similarity to zero but does not explicitly mask the self record after adding the metadata score. The archived family aggregates alone therefore do not certify strict self-exclusion for every query. '
              'The illustrative ranking contains no exact query self-hit, but is not a substitute for the missing complete family-query rankings. None of these controls establishes that the source publications were absent from model pretraining.',
              'Disclose the actual self-mask limitation rather than claiming an unverified leakage guarantee.', 'esi')
    edit('Table S7. Frozen retrieval benchmark summary.',
         'Table S7. Retrospective retrieval-analysis summary. Rank changes summarize 80 queries with 20 neighbors each. Family-match rates summarize a separate 56-query sample over five families. The offline scripts use the reduced three-field score described in Section 2.4, not the full operational score in Table S6.',
         'Make the two denominators and offline scope visible at the summary table.')
    edit('Figure S6. Retrieval alignment and source-exclusion controls.',
         'Figure S6. Retrospective metadata alignment and source-exclusion controls. '
         '(a) Mean top-five same-family fractions, with query counts. (b) Rank changes among 1,600 query–result pairs: 401 (25.1%) change rank; positive values indicate promotion. '
         '(c) Weighted component scores and nonzero fractions among pairs with the query field available. (d) Recorded family labels for an illustrative iridium query; ? denotes unassigned metadata. '
         '(e) Source-exclusion procedure used in the architecture campaigns. Panels (a–d) derive from separate offline analyses; Section 2.4 specifies their scoring functions and self-exclusion limitations. The figure does not independently establish reaction relevance or absence from model pretraining.',
         'Keep the schematic exclusion panel distinct from what the offline scripts actually prove.')
    edit('The KHU experimental supplement supplied on 29 September 2026',
         'This section documents the KHU implementation of the photoredox Giese addition/oxidation and DPDTC-mediated amidation cases. '
         'Tables S31 and S34 report the measured outcomes, and Tables S38–S39 connect the implemented equipment and nominal feeds to the design proposals. '
         'The photographs, spectra, experimental procedures, and grouped response-set entries are retained from the laboratory record. Grouped sets are not treated as independent measurements.',
         'Replace document-editing narration with scientific section framing.')
    edit('The 7 bar BPR values above identify the installed cartridge',
         'The 7 bar values identify the installed BPR cartridge, not the measured system pressure. Tables S31 and S34 report the laboratory pressure ranges separately. '
         'The pressure reference (gauge or absolute), sensor position, and calibration were not specified. The reported 95 °C water-bath arrangement is retained pending laboratory confirmation.',
         'Shorten repeated editorial explanations while retaining unresolved metrology.')
    p = next(p for p in s.body if text(p).startswith('At the stated settings, nominal Stage 1 times'))
    v = text(p).replace('The Stage 2 value labeled “181 min” in the supplied artwork is interpreted here as the rounded inlet-reference index 20/(0.020 + 0.090) = 181.82 min.', 'The supplied artwork labels Stage 2 as “181 min”; recalculation from the stated settings gives the nominal inlet-reference index 20/(0.020 + 0.090) = 181.82 min.')
    edit('At the stated settings, nominal Stage 1 times', v, '181 is not the nearest-minute rounding of 181.82; distinguish source label from recalculation.')
    p = next(p for p in s.body if text(p).startswith('Figure S24. Original KHU experimental schematic,'))
    v = text(p).replace('The source\'s “181 min” for R2 is a rounded inlet-reference index, not operating-pressure residence time.', 'The source labels R2 as “181 min”; the stated settings give 181.82 min on the inlet-reference basis, not operating-pressure residence time.')
    edit('Figure S24. Original KHU experimental schematic,', v, 'Correct the rounding explanation without changing original KHU artwork.')
    p = next(p for p in s.body if text(p).startswith('The supplied reference tables also compare'))
    add_after(s, p,
              'The earlier KHU backflow report concerned pure oxygen at 0.43 mL min−1 and liquid at 0.020 mL min−1, despite the original proposal’s air-based feed calculation. '
              'The revised point retained pure oxygen at a reduced controller setting of 0.090 mL min−1. A gas-branch check valve resists reverse liquid entry toward the MFC; it does not necessarily prevent gas entering the upstream liquid branch. '
              'The assembled photograph identifies a check valve, but the exact branch, orientation, and startup-pressure history remain unconfirmed. No backflow probability or verified hydraulic protection is inferred from the reduced gas setting.',
              'Correct the experiment history and check-valve scope in the experimental reconciliation.', 'esi')
    edit('Before submission, obtain the raw quantitative-NMR files',
         'The available record does not include raw quantitative-NMR files and yield-calculation worksheets (including internal-standard amount and purity), independent repeat identifiers, exact collection windows, or second-feed timing. '
         'Gas-controller reference conditions and calibration, pressure-sensor reference and location, the implemented oxygen-free feed preparation, and check-valve branch and orientation remain to be confirmed. '
         'Original NMR and LRMS instrument exports, the specified tubing dimensions, and the reported 95 °C bath arrangement also require final laboratory confirmation. These gaps are retained explicitly; no measurement or uncertainty has been inferred to fill them.',
         'Replace instructions to the author with a concise account of data availability.')
    for prefix in ('Figure S28.', 'Figure S29.', 'Figure S30.', 'Figure S31.'):
        p = next(p for p in s.body if text(p).startswith(prefix))
        v = text(p).replace(' The original EMF vector artwork is retained in the revision evidence package.', '')
        edit(prefix, v, 'Remove revision-package administration from scientific captions; preserve spectral information.')
    for number in (23, 24):
        caption = next(p for p in s.body if text(p).startswith(f'Table S{number}.'))
        table = caption.getnext()
        assert table.tag == W + 'tbl'
        paragraphs = list(table.iter(W + 'p'))
        for p in [caption] + paragraphs[:-1]:
            props = p.find(W + 'pPr')
            if props is None:
                props = E.Element(W + 'pPr')
                p.insert(0, props)
            if props.find(W + 'keepNext') is None:
                E.SubElement(props, W + 'keepNext')
        changes.append({'document': 'esi', 'before': f'Table S{number}: split over two pages.',
                        'after': f'Table S{number}: keep the short table with its caption on one page.',
                        'reason': 'Pagination only: prevent a stranded final row without changing cell text, font or shading.'})
    return s


def finish(pkg, stem):
    science_typography(pkg)
    base.prior.bold_references(pkg.doc)
    order_properties(pkg.doc)
    clean = BASE / f'{stem}_submission_reviewed_20260930.docx'
    pkg.save(clean)
    pkg.save(BASE / f'{stem}_submission_reviewed_20260930_marked.docx', marked=True)


def main():
    OUT.mkdir(exist_ok=True)
    manifest = {str(p.relative_to(ROOT)): sha256(p.read_bytes()).hexdigest() for p in (MAIN, ESI)}
    main_pkg, esi_pkg = revise_main(), revise_esi()
    finish(main_pkg, 'manuscript')
    finish(esi_pkg, 'esi')
    assert all(sha256((ROOT / p).read_bytes()).hexdigest() == h for p, h in manifest.items())
    (OUT / 'source_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    (OUT / 'paragraph_changes.json').write_text(json.dumps(changes, indent=2, ensure_ascii=False) + '\n')
    with (OUT / 'changes.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['document', 'reason', 'before', 'after'])
        writer.writeheader(); writer.writerows(changes)
    print(f'Saved reviewed clean and marked manuscript/ESI: {len(changes)} recorded edits.')


if __name__ == '__main__':
    main()
