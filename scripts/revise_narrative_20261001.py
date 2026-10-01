"""Revise the scientific narrative without changing artwork or archived evidence."""
from copy import deepcopy
from hashlib import sha256
from pathlib import Path
import csv
import json
import re

from lxml import etree as E

from revise_manuscript_cases_20260922 import Package, W, NS, text
from revise_manuscript_20260930 import paragraph_like
from revise_prior_work_20260928 import bold_references
from revise_submission_20260929 import science_typography
from revise_layout_discussion_20260928 import order_properties
from overhaul_submission_20260930 import with_cites

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'manuscript/Submission'
OUT = BASE / 'narrative_revision_20261001'
INPUTS = {name: BASE / f'{name}_submission_inventory_gui_20261001.docx'
          for name in ('manuscript', 'esi')}
TITLE = 'FlowPilot: An Agentic Research Assistant for End-to-End Flow Process Design'
CHANGES = []


def edit(pkg, index, value, reason):
    p = pkg.ps[index]
    assert not p.xpath('.//a:blip | .//w:fldChar', namespaces=NS), index
    CHANGES.append({'document': pkg.source.name, 'source_paragraph': index,
                    'before': with_cites(p), 'after': value, 'reason': reason})
    pkg.revise(p, value)
    return p


def after(pkg, index, values, reason, heading=False):
    anchor = pkg.ps[index]
    template = pkg.ps[22] if heading else pkg.ps[index]
    nodes = [paragraph_like(pkg, template, value, heading=heading) for value in values]
    pkg.after(anchor, nodes)
    for value in values:
        CHANGES.append({'document': pkg.source.name, 'source_paragraph': index,
                        'before': '', 'after': value, 'reason': reason})
    return nodes


def main_revision():
    m = Package(INPUTS['manuscript'])
    edit(m, 0, TITLE, 'Express the researcher-facing design purpose in the title.')
    edit(m, 6,
         'Designing a continuous-flow synthesis requires chemical objectives, transport, operating conditions and available equipment to be considered together. '
         'Here we present FlowPilot, an agentic research assistant that supports this end-to-end design task from a batch protocol and the researcher’s laboratory constraints. '
         'Standardized dialogue establishes the design brief; chemistry-aware retrieval and a handbook-derived rule base inform candidate generation; and a specialist council reviews alternatives alongside deterministic engineering calculations. '
         'The resulting process configuration connects feed compositions, stage conditions, equipment assignments and an inspectable topology, with records retained for experimental feedback. '
         'Across three chemistry cases, five generator models and three repeats, a matched architecture comparison produced 90 designs evaluated against 14 fixed criteria by separate LLM judges. '
         'Mean benchmark scores increased from 0.794 for one-shot generation to 0.917 for FlowPilot, while judge-reported critical flags decreased from 98 to 4. '
         'Laboratory implementation of photoredox Giese addition/oxidation and telescoped DPDTC-mediated amidation gave 97–98% and 68–86% NMR yields, respectively. '
         'Together, these studies demonstrate how coordinated chemical reasoning, quantitative checks and equipment-aware design can support researcher-led development of connected flow processes.',
         'Frame the whole design workflow, retain all outcomes, and avoid equating a design score with measured performance.')

    intro = {
        8: 'Continuous-flow chemistry allows synthesis to be organized around controlled mixing, heating, irradiation and contact between phases.[[1-4]] '
           'These capabilities have broadened access to photochemical, electrochemical and gas–liquid transformations and enabled routes to pharmaceutical intermediates.[[5-10]] '
           'Connecting reaction stages can further reduce intermediate handling and support multistep or on-demand production.[[11-14]] '
           'For the researcher, however, the opportunity is accompanied by a design problem: choosing an arrangement of streams, reactors and controls that realizes the chemistry with the instruments actually available.[[15-16]] '
           'A batch protocol establishes a starting point for this task, but does not specify the continuous process that should be built.',
        9: 'The difficulty lies in the coupling between decisions. Reactor geometry affects irradiation, heat and mass transfer, and pressure loss, while adding a reagent between reactors changes both composition and downstream flow.[[17-21]] '
           'Gas delivery couples stoichiometry to phase behavior and makes the reference conditions used for volumetric flow and residence time consequential.[[22]] '
           'In a telescoped synthesis, incomplete conversion upstream also changes the material available for the next transformation. Integrated apixaban synthesis and photochemical routes to sartan intermediates illustrate the coordination required across reaction stages and hardware.[[23-24]] '
           'An effective design assistant must therefore reason about the connected process, rather than recommend each operating parameter in isolation.',
        10: 'Digital methods address complementary parts of this challenge. Computer-aided synthesis planning identifies reaction routes,[[25-27]] process simulation evaluates connected unit operations,[[28-29]] '
            'and Bayesian or active-learning methods guide experimental optimization.[[30-34]] AI-informed robotic flow synthesis links planning to physical execution,[[35]] while modular platforms such as RoboChem-Flex and LLM-assisted UV-visible auto-reactometry extend automated optimization and screening.[[36-37]] '
            'Alongside these developments, researchers need support in formulating the initial process itself: specifying feeds, selecting compatible equipment and defining a defensible operating region before an experimental campaign can refine it.',
        11: 'Large language models (LLMs) offer a way to connect that design intent to specialist tools through natural-language interaction. ChemCrow and Coscientist combine chemical reasoning with tool use and experimentation, '
            'whereas El Agente coordinates computational chemistry workflows.[[38-40]] Co-Scientist and Paper2Agent extend collaborative scientific reasoning and the reuse of computational methods.[[41-42]] '
            'Closer to laboratory execution, ACRA links literature procedures to executable chemical descriptions and available hardware; AutoLabs combines clarification and self-correction for liquid handling; and PRISM incorporates simulation feedback before robotic execution.[[43-45]] '
            'These studies establish the value of agents as coordinators of scientific work, not merely generators of chemical text.',
        12: 'Flow chemistry and process development are already part of this emerging scope. Chat-microreactor uses literature-derived information for flow-pattern prediction and microreactor design guidance,[[46]] '
            'and SapoMind combines recommendations with experimental optimization of continuous-flow lanolin saponification.[[47]] '
            'LLM-RDF addresses end-to-end synthesis development through agents for literature search, experimental design, execution, analysis and interpretation, including reactor-design tasks.[[48]] '
            'Multi-agent process-design systems provide complementary capabilities: Tian et al. generate topologies and parameters for simulation flowsheets,[[49]] and the CeProAgents preprint integrates knowledge extraction, conceptual design and parameter optimization.[[50]] '
            'Together, these studies show how chemical agents increasingly connect reactor and process design to broader synthesis-development workflows.',
        13: 'The relevant question is how such capabilities are organized around a laboratory-specific process. The CAAF preprint explores deterministic constraint checking and candidate revision, including a constructed flow-reactor problem,[[51]] '
            'while El Agente Gráfico and La Agente Óptima investigate structured scientific execution and agent-supervised optimization.[[52-53]] '
            'For researcher-led flow development, this direction calls for a common representation linking the intended chemistry, the quantitative conditions of each stage and the equipment that can realize them. '
            'It should also allow the researcher to express priorities, inspect competing designs and return experimental observations without losing the assumptions behind an earlier proposal. '
            'The scope and reported validation of directly related systems are compared in Table S1.',
        14: 'Here we introduce FlowPilot as an agentic research assistant for end-to-end flow-process design. The researcher supplies a batch procedure, available instruments and a design objective; '
            'standardized follow-up questions establish missing requirements, experimental evidence and hypotheses. FlowPilot then develops and reviews alternative connected configurations using chemistry-aware retrieval, handbook-derived engineering knowledge, quantitative calculations and a specialist council. '
            'Its design task extends from deciding where reagents and gases enter to selecting reactor conditions and assigning equipment. End-to-end denotes this integrated design workflow, with the researcher retaining responsibility for experimental implementation and interpretation.',
        15: 'We evaluate FlowPilot at three complementary levels: the alignment of retrieved precedents with the chemistry plan, the quality of delivered designs relative to matched one-shot generation, '
            'and the implementation of connected processes in the laboratory. The architecture comparison uses several model families, a fixed outcome-based rubric and separate evaluator calls; internal ablations examine the contribution of council review and candidate exploration. '
            'Two experimental sequences then probe distinct design challenges. Photoredox Giese addition followed by oxidation requires a controlled change from an oxygen-free to an oxygen-fed environment, '
            'whereas DPDTC-mediated amidation requires coordinated activation, interstage reagent addition and downstream conversion without intermediate isolation. '
            'Together, these assessments examine whether the proposed architecture supports coherent process development, beyond producing plausible reaction instructions.',
    }
    for i, value in intro.items():
        edit(m, i, value, 'Build a connected introduction around process design, related work, researcher interaction and the evaluation argument.')
    after(m, 14, [
        'The organizing principle is that chemical interpretation, engineering consistency and equipment feasibility must inform one another. '
        'Retrieved precedents guide candidate choices but do not establish transferable kinetics; engineering tools recalculate dependent quantities; and council specialists examine chemical, kinetic, fluidic and safety implications before selection. '
        'A shared final record connects stage and stream conditions to the equipment list and diagram, making the proposal available for inspection and subsequent revision. '
        'This structure is important because fluent chemical reasoning can coexist with elementary errors or overconfidence,[[54]] and model-guided optimization can still produce invalid suggestions within a defined search space.[[55]] '
        'For a design assistant, success must therefore be assessed at the level of the complete proposed process.'
    ], 'Give FlowPilot a substantive second introductory paragraph and connect its architecture to a testable evaluation requirement.')

    after(m, 16, ['Coordinating chemistry, engineering and laboratory equipment'], 'Give the Results opening an intentional scientific role.', heading=True)
    edit(m, 17,
         'FlowPilot begins by making the researcher’s process-development problem explicit (Figure 1). A protocol supplies chemical facts, whereas the objective, inventory and follow-up answers define what the proposed process should achieve and what the laboratory can support. '
         'The confirmed input package keeps measured evidence separate from hypotheses and missing information. The upstream chemistry module identifies the transformations, stage boundaries, incompatible feeds and sensitivities that shape the design space. '
         'Literature retrieval and handbook-derived knowledge then inform an initial quantitative proposal. In this way, a desired chemical outcome is developed into a set of connected operations rather than treated as a request to rewrite the batch procedure.',
         'Explain the design problem and information flow rather than enumerate software steps.')
    edit(m, 18,
         'The council evaluates alternatives where those operations interact (Figure 1d). A Designer proposes candidate configurations; Chemistry, Kinetics, Fluidics and Safety specialists examine transformation fidelity, reaction-time plausibility, transport, hardware and hazards. '
         'The Skeptic examines contradictions and unsupported assumptions, and the Orchestrator selects among the reviewed candidates. '
         'A shorter residence time, for example, is not considered independently of the reactor volume, achievable feed rate or chemical evidence supporting that exposure. '
         'When operating choices change, deterministic calculations update the dependent quantities before the revised candidate is assessed. '
         'The review record retains disagreement, rejected alternatives and selection rationales, allowing the researcher to distinguish a constraint-driven decision from an exploratory hypothesis.',
         'Explain why council deliberation and recalculation matter to scientific decisions; do not claim that every decision is correct.')
    edit(m, 19,
         'Equipment assignment is part of this design process rather than a final list appended to it. Available pump channels, reactor dimensions, thermal limits and compatible irradiation modules constrain which operating points can be realized. '
         'The selected proposal is reconciled into a common final record containing the streams, stages, equipment assignments and process graph. '
         'Numerical views and the diagram derive from that record; unresolved requirements and earlier calculations remain identifiable as diagnostics. '
         'This links what the researcher would assemble to the quantities used to assess it, and preserves a basis for incorporating later experimental observations. '
         'Figures S1–S2 and Tables S2–S3 describe the data contracts and provenance mechanisms.',
         'Promote inventory-aware realization from display detail to a design responsibility.')
    edit(m, 22, 'Grounding design choices in literature and engineering knowledge', 'Link corpus coverage to the design argument.')
    edit(m, 23,
         'Candidate design requires both chemical precedents and general engineering knowledge. FlowPilot’s curated literature resource contains 464 classified records spanning thermal synthesis, photoredox catalysis, heterogeneous photocatalysis, cross-coupling, oxidation/reduction and other reaction classes (Figure 2). '
         'The recorded hardware includes capillary and coil reactors, microreactors, photoreactors and packed beds, with fluoropolymers prominent among reported tubing materials. '
         'Explicit inlet-stream counts are available for 154 processes; two- and three-stream configurations account for 91 and 30 records, respectively. '
         'These annotations help relate a chemical plan to alternative ways of supplying reagents and arranging reactors, without prescribing a default topology or treating an absent annotation as evidence that an operation is unnecessary.',
         'Interpret the corpus as design evidence and retain its actual denominators.')
    edit(m, 24,
         'The complementary handbook-derived resource contains 2,537 structured rules covering reactor design, heat transfer, catalysts, residence time, photochemistry, materials, scale-up, pressure, mixing and safety (Figure 2e–f). '
         'These rules provide engineering context when a close chemical analogue is unavailable or when a candidate must be evaluated beyond reaction identity. '
         'Coverage is uneven: heat-transfer rules are densely associated with thermal synthesis and cross-coupling, whereas photochemistry rules concentrate in photoredox and photocycloaddition classes. '
         'The rule store therefore informs interpretation and review rather than functioning as a universally validated predictive model; only the checks implemented in the calculation layer act as deterministic constraints. '
         'Figures S3–S4 and Tables S4–S5 report the frozen corpus and computed rule-base coverage.',
         'Explain the distinct roles of handbook knowledge and implemented engineering checks.')
    edit(m, 27, 'Retrieving precedents that reflect the chemistry plan', 'Connect retrieval evaluation to the researcher’s design task.')
    edit(m, 28,
         'A broad corpus is useful only if the retrieved records reflect the process under consideration. FlowPilot therefore enriches the search with reaction class, mechanism, catalyst family, solvent and wavelength from the chemistry plan (Figure 3a). '
         'Search progresses from mechanism- and phase-filtered paired records to relaxed filters and, when necessary, the full corpus; field-aware ranking then orders the candidates (Figure S5; Table S6). '
         'This allows the assistant to seek closely related evidence while retaining a route to broader analogies when exact matches are sparse. '
         'We examined the contribution of query enrichment and metadata reranking in separate retrospective analyses using stored corpus embeddings, rather than treating corpus size alone as evidence of retrieval quality.',
         'State the reason for retrieval and the actual scope of its evaluation.')
    edit(m, 30,
         'In a separate family-alignment analysis, the mean proportion of top-five records sharing the query’s recorded catalyst family increased from 40.0% to 95.0% for iridium, '
         '42.7% to 90.7% for organic dyes, 77.3% to 94.7% for ruthenium, 87.1% to 98.6% for TiO2, and 45.0% to 60.0% for ZnO (Figure 3c); the corresponding query counts were 8, 15, 15, 14 and 4. '
         'The effect was therefore not uniform across families, but the ranking more often retained the chemistry descriptors requested by the plan. '
         'Because catalyst-family metadata contribute to the score, this measures alignment with those descriptors, not independent proof that reaction times or yields transfer between systems. '
         'Retrieved records remain evidence for council assessment rather than ready-made operating instructions. Figure 3d illustrates the ranking change; Figures S5–S6 and Tables S6–S7 give the analyses and their scope.',
         'Interpret the alignment outcome and transition from retrieving evidence to evaluating complete designs.')

    edit(m, 33, 'Architecture effects on flow-process design quality', 'Describe the scientific question rather than a test-log category.')
    edit(m, 34,
         'We next asked whether coordinating these capabilities improves the delivered process design beyond what the same LLM can produce in one response (Figure 4a). '
         'Three cases exposed different design demands: Cu/C-catalyzed azide–alkyne cycloaddition (CuAAC) required catalyst placement and pressurized liquid operation; photochemical oxidation required irradiation and gas bookkeeping; '
         'and hydrogenolysis combined a liquid feed, hydrogen and a solid catalyst. '
         'For each case, one-shot generation and FlowPilot received the same protocol, objective, authoritative inventory and requested final-output structure. '
         'The one-shot baseline thus received a complete design brief, not merely a short chemistry question. Source records were excluded from FlowPilot retrieval, although this does not establish their absence from model pretraining.',
         'Explain the architecture question, complementary cases and information-matched baseline.')
    after(m, 34, [
        'The five-model comparison included Qwen3.6-27B and Qwen3.8-27B alongside GPT-4o, Claude Sonnet 4.6 and Claude Opus 4.6. '
        'This spans locally served 27B models and commercial model families, allowing the architecture effect to be examined separately from choosing a particular generator. '
        'Each model was tested in both architectures for three repeats of each case, giving 90 outcomes. '
        'Three separate Qwen, OpenAI and Claude evaluator calls assessed each normalized final design using the same 14 criteria, covering chemical fidelity, feed chemistry, numerical and topology consistency, inventory, physical plausibility, operating procedures and uncertainty. '
        'The rubric rewards the delivered design rather than the presence of agents or a long discussion. Applicable integer ratings from 0 to 4 were averaged with equal judge and criterion weights and divided by four; critical flags were retained separately. '
        'Tables S8–S9 define the rubric, and Table S10 specifies the complementary module-ablation screen.',
        'The comparison holds task information constant, not computation: FlowPilot can retrieve evidence, call tools and revise candidates, whereas one-shot generation uses one model call. '
        'It therefore tests the utility of the complete design architecture, not whether agent dialogue alone causes an improvement at equal token cost. '
        'For each model and architecture, the three case scores were averaged within each repeat; the reported mean and sample standard deviation summarize those three repeat-level values. '
        'Generator and architecture labels were removed from evaluator packets, but judging remained an LLM-based assessment, with shared model families and potentially recognizable output styles. '
        'The exact inputs, evaluator configuration, applicability rules and repeat structure are retained in ESI Section 4 and Methods.'
    ], 'Explain model choice, scoring and experimental matching before interpreting the plotted outcomes.')
    edit(m, 37,
         'FlowPilot increased the mean score for each of the five generators (Figure 4b; Table S11). Across this cohort, the architecture means were 0.917 for FlowPilot and 0.794 for one-shot generation. '
         'The gain was largest for GPT-4o (+0.226), followed by Qwen3.8-27B (+0.168) and Qwen3.6-27B (+0.139); Claude Sonnet 4.6 and Opus 4.6 showed smaller gains of +0.034 and +0.049. '
         'The one-shot means spanned 0.684–0.890, whereas FlowPilot means occupied the narrower range of 0.904–0.932. '
         'This pattern suggests that the design architecture was particularly useful when direct generation left more requirements unresolved. '
         'The two 27B models within FlowPilot also exceeded the commercial one-shot means in this cohort, supporting the practical value of workflow structure without establishing that a smaller model is intrinsically more capable. '
         'The smaller Claude gains are equally informative: stronger direct outputs leave less room for improvement. Figure 4 additionally retains an archived GPT-5.4 comparison, which is not pooled into the five-model summaries.',
         'Interpret model-dependent gains and local-model relevance while explicitly distinguishing the preserved six-model artwork.')
    after(m, 37, [
        'The critical-flag analysis provides a complementary view of these averages (Figure 4c; Table S12). '
        'Judges recorded 98 flags across 45 one-shot outcomes and 4 across 45 FlowPilot outcomes; 18 and 43 outcomes, respectively, had no flag. '
        'Thus the higher mean score was accompanied by fewer flagged defects, rather than only more complete descriptions. '
        'These counts represent evaluator findings: several flags can refer to one underlying problem, and a flag is not an independently confirmed physical failure. '
        'Accordingly, the computational results support improved design quality under the fixed assessment, not a corresponding numerical probability of successful or safe experimental execution.'
    ], 'Interpret critical findings separately from scores and avoid presenting judge output as physical ground truth.')
    edit(m, 38,
         'The case-specific analysis helps locate the improvement (Figure S7). Hydrogenolysis showed the largest mean gain for both Qwen models and for Opus 4.6, '
         'consistent with the demands of coordinating gas delivery, catalyst contact and pressure control. '
         'At criterion level, the displayed gains in executable topology were positive for every model–case mean, whereas transformation-fidelity changes were small in comparison (Figure S8). '
         'This pattern points to process integration, not simply recognition of the named reaction, as an important source of the observed benefit. '
         'Not every criterion improved, and the run-level flag map retains the remaining failures (Figure S9). '
         'Resource use must also be considered: Figure S10 reports tokens, runtime and generation cost, while Figure 4e relates score to cost under the archived price schedule. '
         'The Qwen configurations offered higher score-per-generation-cost ratios under that schedule; this is a deployment-relevant trade-off, not a measurement of local hardware costs or experimental productivity. '
         'Table S13 defines the numerical closure checks underlying the engineering assessment.',
         'Use case and criterion evidence to explain what improved, and separate cost efficiency from chemical productivity.')
    after(m, 38, [
        'To examine which parts of the architecture contributed, we separately tested 15 Qwen3.8-27B configurations over the same three cases (Figure 4d; Table S10). '
        'Removing the council reduced the mean score from 0.928 to 0.670, below the 0.743 one-shot result in this screen. '
        'An initial staged workflow without cross-domain review was therefore not sufficient to outperform direct generation. '
        'Removing individual Chemistry, Fluidics, Safety or Kinetics reviewers produced smaller changes, with means between 0.902 and 0.924. '
        'However, one-pass council review (0.932) and omission of preselection refinement (0.935) slightly exceeded the full configuration. '
        'The evidence favors coordinated review over the absence of review, but not the assumption that every extra agent call or revision cycle is beneficial. '
        'Because each condition was generated once per chemistry, these standard deviations describe between-case variation; they do not establish a statistically resolved ranking of individual modules.'
    ], 'Discuss the internal ablation fully, including simpler configurations with slightly higher means and the correct replication limit.')
    edit(m, 39,
         'A separate historical study examined council configuration and candidate budgets for integrated isoxazole synthesis[[56]] (Figure S11; Tables S14–S17). '
         'Across five repeats, the normalized engineering radar-area score changed from 0.31 ± 0.03 before council review to 0.48 ± 0.23 afterward. '
         'The higher mean and larger dispersion show why candidate exploration should be assessed together with selection consistency, rather than by the best individual run. '
         'This engineering-profile statistic is distinct from the fixed-criterion judge score; the archived model matrix, revision counts and metric definitions allow the two analyses to be interpreted separately.',
         'Use the historical study to interpret exploration and variability without conflating its metric with the architecture benchmark.')
    edit(m, 40,
         'These assessments also need to remain useful to the researcher making an experimental decision. '
         'The interface retains the original protocol and answered follow-up questions (Figure S12), the categorized inventory and constraints (Figure S13; Table S18), '
         'and linked stage conditions, topology and council review (Figure S14; Table S19). '
         'The audit examples include pressure-control and residence-time inconsistencies in one-shot designs, a catalyst-placement defect in a FlowPilot output, and an evaluator false positive (Table S20). '
         'Recorded council exchanges show how conflicting assessments and candidate selection can be inspected rather than hidden behind a final recommendation (Table S21). '
         'This connection between design, evidence and review provides the starting point for testing the proposed process in the laboratory.',
         'Turn the GUI and failure evidence into a bridge from benchmark assessment to experimental research.')
    after(m, 40, [
        'We next examined two connected-stage reactions that were separate from the architecture-benchmark cases. '
        'The purpose was to assess how the design workflow accommodates chemical sequence, equipment and researcher priorities when preparing an experiment, and what becomes apparent only after implementation. '
        'For each chemistry, three standardized response sets retained the same batch protocol and laboratory inventory while varying the design brief. '
        'Set 1 prioritized integrated final yield without unnecessarily long residence times; Set 2 emphasized high conversion with the shortest reasonably justified starting times; '
        'and Set 3 gave greater emphasis to throughput, compactness and overall operating balance. '
        'These were related design priorities, not three different reactions or replicate experiments. Full objectives, hypotheses and fixed-ID answers are reproduced in ESI Section 8. '
        'Some sets converged to the same implemented conditions, giving two distinct reported operating points for each chemistry. '
        'The following analysis connects those conditions to chemical outcomes, while distinguishing the initial proposal from changes made during implementation.'
    ], 'Introduce the experimental purpose and shared response-set logic before either case, without overclaiming controlled objective sensitivity.')

    edit(m, 41, 'Staged oxygen delivery in photoredox Giese addition and oxidation', 'Make the heading identify the chemical process-design question.')
    edit(m, 42,
         'The first case couples photoredox carbon–carbon bond formation to selective oxidation of the resulting sulfide (Figure 5). '
         'Photoinduced electron transfer generates an alpha-thiomethyl radical from an alpha-silyl sulfide; addition to acrylonitrile forms the sulfide intermediate, which is then oxidized to sulfoxide 4a.[[57]] '
         'The batch procedure separates these transformations in time, with 4 h under argon followed by 6 h exposed to air. '
         'The connected flow design instead separates the oxygen-free and oxygen-fed environments spatially. '
         'All three response sets therefore required oxygen introduction only at the second-stage inlet and at least two equivalents relative to the substrate. '
         'The process-design question was how to combine sufficient upstream sulfide formation with downstream oxygen delivery, irradiation and gas–liquid operation, not simply how far the batch time could be shortened.',
         'Connect mechanism, stage boundaries and the shared design constraints.')
    edit(m, 43,
         'We implemented this arrangement using a 2 or 5 mL PFA coil in a Vapourtec UV-150 module followed by a separately illuminated 20 mL FEP coil, both at 25 ± 1 °C. '
         'The reactor–light pairing preserved the compatibility of the first-stage module while allowing the second coil to use its own irradiation source. '
         'A 2.0 mL sample loop introduced substrate 1a (0.100 M), acrylonitrile (0.200 M) and iridium photocatalyst (0.500 mM) into the carrier stream at 0.020 mL min−1. '
         'The first-stage nominal liquid times were consequently 100 and 250 min. '
         'Pure oxygen entered at the interstage mixer at a controller setting of 0.090 mL min−1. '
         'Under the proposal’s STP convention (273.15 K, 1 atm), this corresponds to 0.00402 mmol min−1 of oxygen, or 2.01 equivalents against the nominal substrate feed. '
         'The displayed second-stage index of 181.82 min uses inlet-reference gas volume, 20/(0.020 + 0.090); it is not the residence time at reactor pressure. '
         'ESI Section 9 distinguishes these nominal calculations from the experimental gas-reference and sample-collection metadata.',
         'Use a collective author voice and explain why the particular equipment/flow decisions matter, retaining time-basis accuracy.')
    edit(m, 44,
         'Implementation revealed an additional constraint that stoichiometry alone did not capture. '
         'An initial trial with pure oxygen at 0.43 mL min−1 and liquid at 0.020 mL min−1 showed gas backflow toward the first reactor. '
         'We specified pure oxygen in the revised design inputs and reduced the gas setting to 0.090 mL min−1; the gas choice was researcher-directed, not an autonomous discovery by the model. '
         'The proposal included a gas-line check valve, although that component alone does not establish protection of the upstream liquid branch. '
         'With a 7 bar cartridge BPR and recorded system pressures of 5.1–5.6 bar, the implemented 2 mL first-stage configuration, reported jointly for Sets 1/3, gave 97% yield, '
         'whereas the 5 mL configuration associated with Set 2 gave 98%. '
         'These are NMR yields, compared contextually with the published flow yield of 90%.[[57]] '
         'Both operating points supported the connected transformation; the one-percentage-point difference does not resolve a residence-time benefit without replicates. '
         'Table S22 and Figures S15–S16 document the conditions, irradiation and apparatus. '
         'The case demonstrates why spatial reaction design, quantitative gas delivery and observations from the assembled apparatus must be considered together.',
         'Integrate the actual feedback, set mapping and yield interpretation; do not claim the valve proves the backflow risk solved.')
    edit(m, 46, with_cites(m.ps[46]).replace('KHU operating points', 'Implemented operating points').replace('Literature and KHU experiments', 'Literature and present experiments'),
         'Remove institutional separation from the main-text caption without altering artwork or measurements.')

    edit(m, 47, 'Coupled feed and residence-time design for telescoped amidation', 'Make the second case answer a complementary process-design question.')
    edit(m, 48,
         'The second case tests a different form of stage coupling: formation and downstream consumption of a reactive intermediate (Figure 6). '
         '3-Methyl-4-nitrobenzoic acid is activated by 2,2′-dipyridyldithiocarbonate (DPDTC) in the presence of DMAP to form a 2-pyridyl thioester; benzylamine then undergoes aminolysis to give the amide.[[58]] '
         'The batch procedure uses two 30 min periods at 95 °C, separated by cooling and amine addition. '
         'For continuous operation, FlowPilot must distinguish a necessary chemical sequence from a batch-handling operation: benzylamine must enter after activation, whereas cooling need not be reproduced automatically if direct transfer can be implemented. '
         'The three response sets emphasized integrated yield, conversion-focused timing and throughput, respectively, while preserving this chemistry and the intended feed stoichiometry (ESI Section 8).',
         'Explain chemical sequencing and the function of the response sets without claiming measured intermediate kinetics.')
    edit(m, 49,
         'We implemented separate activation and amine feeds using two channels of the same Vapourtec E-series system, with direct interstage mixing and no intermediate isolation. '
         'Feed A contained acid (0.500 M), DPDTC (0.525 M) and DMAP (0.0500 M) in 2-MeTHF; Feed B contained benzylamine (2.10 M). '
         'For Set 1, liquid rates of 0.160 and 0.0400 mL min−1 delivered nominal acid and amine molar flows of 0.0800 and 0.0840 mmol min−1, preserving 1.05 equivalents of amine. '
         'Two 5 mL ETFE coils gave nominal times of 31.25 and 25.00 min, because the downstream coil receives both liquid feeds. '
         'This distinction is central to the design: specifying two reactor times without updating the downstream material balance would not preserve the proposed stoichiometry and exposure. '
         'The implementation used a 2.0 mL sample loop for Feed A, a 95 °C water bath for both coils and a 7 bar cartridge BPR; recorded pressure for Set 1 was 5.2–5.3 bar.',
         'Connect equipment selection, concentrations and stage-wise calculation to the chemical design argument.')
    edit(m, 50,
         'Set 1 gave 86% yield. Sets 2/3 shared a second implemented configuration: a 10 mL first coil and a 5 mL second coil, with feed rates of 0.280 and 0.0700 mL min−1, '
         'stage times of 35.71 and 14.29 min, and 68% yield. '
         'Both outcomes and the 98% literature batch reference are NMR yields.[[58]] '
         'The second configuration increased nominal substrate delivery from 0.0800 to 0.140 mmol min−1 and reduced total nominal time from 56.25 to 50.00 min, '
         'but redistributed exposure toward the activation stage and away from amidation. '
         'Its lower final yield despite the longer first-stage time illustrates why upstream residence time or total process time alone is an inadequate design objective. '
         'The observations motivate further examination of the downstream exposure and intermediate composition, but do not identify a rate-limiting step because several variables changed together. '
         'Table S23 and Figure S17 report the conditions and apparatus; Tables S24–S25 connect the proposals to implementation changes and component molar feeds. '
         'Together with the photochemical case, this experiment shows the role of FlowPilot in formulating testable, connected process choices whose chemical performance remains subject to measurement.',
         'Interpret the real yield difference as a connected-process finding, not an isolated kinetic proof or a successful optimization claim.')
    edit(m, 52, with_cites(m.ps[52]).replace('KHU operating points', 'Implemented operating points'), 'Use a unified author voice in the second case caption.')

    discussion = {
        54: 'The central contribution of FlowPilot is to make the connected flow process, rather than an isolated reaction condition, the object of agentic design. '
            'A researcher can begin with a conventional procedure and a laboratory inventory, express a design priority, and inspect alternative configurations in terms of chemistry, quantitative operation and realizable equipment. '
            'This complements microreactor assistants, synthesis-development platforms and process-simulation agents by organizing their related capabilities around a laboratory-specific design task.[[46-51,59-60]] '
            'Linking these steps allows a proposed operating point to be examined together with its chemical rationale and hardware requirements, rather than as a detached recommendation.',
        55: 'The benchmark helps explain why this organization matters. The same generator produced higher mean scores within FlowPilot for all five models, '
            'and improvements extended to topology, material balance and inventory rather than being confined to chemical nomenclature. '
            'In the module screen, simply separating the initial workflow into stages without council review performed poorly, whereas coordinated review recovered much of the design quality. '
            'At the same time, simpler council settings occasionally matched or exceeded the complete configuration. '
            'These findings argue for selective, accountable review of coupled decisions rather than maximizing agent count or discussion length. '
            'They also identify a practical opportunity for locally served models: the tested 27B generators delivered competitive assessed designs when supported by the architecture, although additional computation is part of that benefit.',
        56: 'The experiments address a different question from the benchmark: what happens when a proposed configuration becomes an apparatus and the complete reaction sequence is tested? '
            'For Giese addition/oxidation, separating oxygen exposure spatially enabled the connected chemistry, but the backflow observation exposed a pressure-interaction problem not settled by gas-equivalent arithmetic. '
            'For amidation, feed stoichiometry and stage allocation were coupled, and the higher nominal substrate feed gave lower yield despite longer upstream exposure. '
            'The response sets thus provided alternative starting priorities rather than a guarantee of distinct or improved experimental outcomes. '
            'The sample-loop implementations establish product formation at the reported conditions; they should not be interpreted as measurements of sustained production or as a controlled comparison of all three objectives.',
        57: 'A useful design assistant must consequently retain both the rationale for a proposal and the evidence that later challenges it. '
            'FlowPilot’s saved inputs, stage calculations, equipment assignments and council records support this relationship between researcher and model, while the remaining design defects and evaluator false positive show why neither model confidence nor a benchmark score should substitute for scientific judgment. '
            'Experimental observations can be returned with the conditions actually used, so that subsequent proposals respond to measured process behavior rather than repeated assumptions. '
            'Coupling this record to Bayesian optimization or language-guided experimental priors offers a route to more systematic prospective refinement.[[30-34,61]] '
            'The next test of that workflow is whether it reduces experimental effort across additional chemistries and inventories, not merely whether it produces a more persuasive initial answer.',
    }
    for i, value in discussion.items():
        edit(m, i, value, 'Synthesize architecture evidence and experiments in continuous prose without repeating a report-style list of results.')
    edit(m, 59,
         'FlowPilot provides an agentic design environment in which researchers can develop continuous-flow processes from reaction-level information and the equipment available in their laboratory. '
         'Standardized interaction defines the problem; literature and handbook knowledge inform the alternatives; and council review with engineering calculations connects chemical intent to feeds, stage conditions and a realizable process configuration. '
         'The shared design record makes these decisions inspectable and provides continuity between a proposal and the next experimental cycle.',
         'Conclude with the research-assistant contribution rather than returning to a translator identity.')
    edit(m, 60,
         'Across the five-model architecture comparison, the mean benchmark score increased from 0.794 for one-shot generation to 0.917 for FlowPilot, with fewer judge-reported critical flags. '
         'The internal screen supports the role of coordinated review while showing that additional refinement is not uniformly beneficial. '
         'Laboratory implementation then demonstrated the connected Giese addition/oxidation and amidation sequences at 97–98% and 68–86% NMR yield, respectively. '
         'The gas-flow observations and yield differences between operating points reveal the importance of testing the complete process, beyond checking the consistency of its description.',
         'Link the two types of evidence without overstating their scope.')
    edit(m, 61,
         'These results position FlowPilot as a partner in researcher-led process development: it organizes the design decisions that must agree, exposes assumptions for review and retains measured outcomes for further refinement. '
         'Its value is not to replace experimentation with a generated protocol, but to connect chemical knowledge, engineering design and laboratory implementation within one accountable workflow.',
         'End on the intended practical scientific contribution, not an apology or an optimization claim.')
    edit(m, 84, with_cites(m.ps[84]).replace('KHU implemented the two connected-stage cases', 'We implemented the two connected-stage cases'), 'Use collective authorship consistently in Methods.')
    edit(m, 96, with_cites(m.ps[96]).replace(' KHU, Kyung Hee University;', ''), 'Remove the unused institutional acronym from main-text abbreviations; affiliations remain unchanged.')
    return m


def esi_revision():
    s = Package(INPUTS['esi'])
    edit(s, 0, TITLE, 'Keep the manuscript and ESI titles identical.')
    edit(s, 22,
         'The scope comparison distinguishes design tasks rather than ranking systems evaluated under different conditions. '
         'FlowPilot assists researchers through a laboratory-constrained, end-to-end flow-process design workflow: defining the brief, interpreting chemistry, retrieving evidence, reviewing candidate configurations, calculating operating conditions and assigning equipment. '
         'Batch protocols provide the reaction information for this workflow. The contribution is the coordinated development of a connected process, not text conversion alone. '
         'The matched architecture test compares the recorded one-shot and FlowPilot configurations, not the external systems in Table S1.',
         'Align the ESI scope statement with the revised scientific positioning.')
    edit(s, 24,
         'FlowPilot supports researcher-led process design by coordinating chemical interpretation, engineering calculation, equipment realization and review of alternative configurations. '
         'The workflow begins with standardized intake of a batch protocol, design objectives and laboratory constraints, and returns an inventory-assigned design or explicit unresolved requirements. '
         'Here, end-to-end describes the design workflow, not autonomous laboratory execution or proof of optimized reaction performance. '
         'Interpretation follows an information-authority order: measured evidence, hard safety and inventory constraints, confirmed protocol facts, chemist hypotheses and model inference. '
         'Hypotheses remain ideas to test rather than observations. Earlier experimental success outside a declared operating limit does not authorize a new proposal to override that limit.',
         'Clarify scope and authority without changing the typed contracts or implying guaranteed execution.')
    edit(s, 87, text(s.ps[87]).replace('final batch-to-flow designs', 'final inventory-constrained flow-process designs'),
         'Align narrative terminology while preserving the exact historical prompts below.')
    edit(s, 114,
         'The archived judge configuration for the matched example was Qwen /models/Qwen3.6-27B, OpenAI gpt-5.4-2026-03-05 and Anthropic claude-sonnet-4-6, at temperature 0.0. '
         'These evaluator calls were separate from candidate generation. GPT-5.4 is not one of the five generators pooled in the primary aggregate, although an archived additional GPT-5.4 generator comparison remains displayed in main-text Figure 4b–c. '
         'Exact prompt schemas, seeds, responses and telemetry are retained in the companion source records. A recorded seed does not guarantee that a provider honors deterministic sampling.',
         'Resolve the ambiguous evaluator-only description against the unchanged six-model main figure.')
    edit(s, 275,
         'These two laboratory-design cases test how FlowPilot supports a researcher in specifying a connected process under real equipment constraints. '
         'They are separate from the three architecture-benchmark cases in Section 4. For each chemistry, the same batch protocol and laboratory inventory were supplied with three standardized response sets: '
         'Set 1 emphasizes integrated final yield without unnecessary reaction time; Set 2 emphasizes conversion with the shortest reasonably justified initial exposure; '
         'and Set 3 places greater emphasis on throughput, compactness and overall operating balance. '
         'The corresponding hypotheses explain the stage interactions to be considered; they are not measured kinetic evidence. '
         'These sets are alternative design briefs, not a factorial experiment or independent replicates. The implemented conditions converge to two reported operating points per chemistry. '
         'Full run packages and all six archived topologies remain in the companion records; Section 9 gives the actual configurations, procedures and yields.',
         'Make the response-set purpose explicit without rewriting any verbatim answers.')
    replacements = {
        277: [('collaborator-requested constraint', 'researcher-specified constraint')],
        278: [('verbatim collaborator transcription', 'verbatim supplied transcription')],
        300: [('earlier collaborator incident', 'earlier experimental backflow observation')],
        303: [('verbatim collaborator transcription', 'verbatim supplied transcription')],
    }
    for i, pairs in replacements.items():
        value = with_cites(s.ps[i])
        for before, replacement in pairs:
            value = value.replace(before, replacement)
        edit(s, i, value, 'Use collective research-team language while preserving source file and inventory provenance.')
    return s


def finish(pkg, name):
    science_typography(pkg)
    # Limit reference formatting to revised paragraphs; preserve all unchanged
    # tables, captions, archived inputs and image containers byte-for-byte.
    for p in pkg.changed:
        if p.getroottree().getroot() is not pkg.doc:
            continue
        bold_references(p)
    if name == 'manuscript':
        for p in pkg.body:
            if (p.xpath('./w:pPr/w:pStyle[@w:val="Heading2" or @w:val="Heading3"]', namespaces=NS)
                    or text(p) in {'Availability of data and code', 'Corresponding Authors', 'Author Contributions'}):
                props = p.find(W + 'pPr')
                keep = props.find(W + 'keepNext')
                if keep is None:
                    keep = E.SubElement(props, W + 'keepNext')
                keep.set(W + 'val', '1')
            if re.match(r'^Figure \d+\.', text(p)):
                props = p.find(W + 'pPr')
                if props is None:
                    props = E.SubElement(p, W + 'pPr')
                if props.find(W + 'keepLines') is None:
                    E.SubElement(props, W + 'keepLines')
    order_properties(pkg.doc)
    for marked in (False, True):
        suffix = '_marked' if marked else ''
        pkg.save(BASE / f'{name}_submission_narrative_20261001{suffix}.docx', marked=marked)


def main():
    OUT.mkdir(exist_ok=True)
    hashes = {str(p.relative_to(ROOT)): sha256(p.read_bytes()).hexdigest() for p in INPUTS.values()}
    for name, pkg in [('manuscript', main_revision()), ('esi', esi_revision())]:
        finish(pkg, name)
    (OUT / 'source_manifest.json').write_text(json.dumps(hashes, indent=2) + '\n')
    (OUT / 'paragraph_changes.json').write_text(json.dumps(CHANGES, ensure_ascii=False, indent=2) + '\n')
    with (OUT / 'paragraph_changes.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(CHANGES[0]))
        writer.writeheader()
        writer.writerows(CHANGES)
    print(json.dumps({'changes': len(CHANGES), 'output_directory': str(OUT)}, indent=2))


if __name__ == '__main__':
    main()
