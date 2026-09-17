"""Edit manuscript prose in the original Word package; preserve existing artwork/styles."""
from copy import deepcopy
from hashlib import sha256
from pathlib import Path
from zipfile import ZipFile
import csv
import difflib
import json
import re
import shutil
import sys

from lxml import etree as E

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "manuscript"
SOURCE = BASE / "manuscript.docx"
ESI = BASE / "esi.docx"
DEST = BASE / "manuscript_text_revised.docx"
CHECK = BASE / "text_revision_checks"
NS = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main",
      "a": "http://schemas.openxmlformats.org/drawingml/2006/main"}
W = "{" + NS["w"] + "}"

# Keys refer to paragraphs in the unchanged original, not the discarded revision.
# Double brackets mark citations using the original superscript run properties.
TEXT = {
3: "Designing a continuous flow process demands coordinated reasoning across fluid dynamics, heat and mass transfer, materials compatibility, and safety. We present FlowPilot, a multi-agent AI system that connects chemical interpretation with deterministic engineering calculation and specialist review. Standardized intake captures the batch protocol, chemist objectives, and laboratory constraints; a structured literature corpus and 2,537 handbook-derived rules support the design process. A seven-agent council evaluates candidate operating conditions, which are reconciled with equipment capabilities before a process diagram and traceable design record are returned. Matched architecture comparisons use three chemistry cases, repeated generation, and fixed-criteria evaluation by independent LLM judges informed by deterministic checks. Across the five-model cohort documented in the ESI, FlowPilot achieves higher mean benchmark scores and fewer critical flags than matched one-shot generation. Internal ablations examine the contribution of council review, while archived model-matrix experiments characterize configuration sensitivity. Two multistage chemistries, photoredox Giese addition followed by aerobic oxidation and DPDTC-mediated amidation, are selected for prospective laboratory evaluation. FlowPilot provides an auditable starting point for experimental development rather than a substitute for chemical judgment or measured validation.",
9: "We introduce FlowPilot, a multi-agent AI pipeline purpose-built for continuous flow chemistry process design. Given a batch protocol, the system returns a flow-process proposal accompanied by a process diagram, traceable engineering calculations, literature analogies, and explicit uncertainty flags. The workflow couples five operations: (i) standardized intake that records the protocol, objective, historical evidence, inventory, and chemist hypotheses; (ii) chemistry interpretation and plan-aware retrieval from a curated flow-chemistry corpus; (iii) deterministic engineering calculations supported by handbook-derived rules; (iv) a seven-agent council that reviews and revises candidate designs under bounded domain authority; and (v) inventory reconciliation and rendering from a common final design record. Candidate selection considers the chemist's objective and process constraints. Intensification is evaluated where justified, rather than treated as evidence that any shorter residence time will preserve reaction performance.",
10: "This work presents the hybrid FlowPilot architecture and evaluates the contribution of its components. Literature and rule-base analyses establish the knowledge coverage, and a retrieval benchmark tests whether chemistry-aware ranking improves the relevance of selected analogies. Matched one-shot versus full-pipeline comparisons then assess delivered designs using a fixed rubric across local and commercial models. Internal module ablations and an earlier isoxazole cycloaddition model-matrix study[[41]] provide complementary evidence about council behavior. Finally, photoredox Giese addition followed by aerobic oxidation[[42]] and telescoped DPDTC-mediated amidation[[43]] provide two connected-process cases for prospective laboratory evaluation. The main figures follow this progression from architecture and knowledge resources to computational benchmarking and experimental application.",
12: "FlowPilot was designed to translate a batch chemistry protocol into a continuous-flow process proposal while keeping chemical interpretation, deterministic engineering calculation, and expert review as separate operations (Figure 1). The input layer accepts the protocol, design objective, and laboratory inventory; standardized follow-up questions resolve missing chemistry, constraints, and relevant experimental history. The upstream module converts the confirmed input into a chemistry plan, including species identification, reaction classification, stream-separation logic, sensitivity flags, and possible flow-specific advantages. Retrieval and engineering calculations then support an initial proposal containing reactor geometry, stream assignments, residence-time assumptions, and operating conditions. These initial values remain subject to council review and equipment reconciliation.",
14: "Figure 1. FlowPilot system architecture. User inputs are converted into a chemistry plan and an initial engineering proposal, supported by flow-chemistry literature and handbook knowledge. The seven-agent council generates and reviews candidate designs, records structured verdicts, and proposes bounded revisions. The selected candidate is reconciled with engineering calculations and laboratory inventory before a process-flow diagram, protocol, report, confidence assessment, and supporting citations are returned. The illustrated recipe card is schematic; computational validation does not establish experimental performance.",
16: "The output of the workflow is a structured design package rather than only a text answer. It contains a process-flow diagram, stage and stream parameters, equipment assignments, confidence and safety flags, supporting analogies, and the review record. The numerical views and diagram are derived from the same final design contract, while unresolved requirements remain separately visible. This exposes both the proposed operating point and how it was selected without treating a model's confidence as experimental validation. Detailed architecture, authority boundaries, intake contracts, and run provenance are provided in Figures S1-S2 and Tables S1-S3.",
18: "The literature dataset covers a broad range of flow-chemistry applications (Figure 2). The curated export contains 464 classified literature records spanning thermal synthesis, photoredox catalysis, heterogeneous photocatalysis, cross-coupling, oxidation/reduction, and other reaction classes. Reactor types range from capillary/coil and microreactor/chip systems to photoreactors and packed beds, with fluoropolymer tubing prominent among the reported materials. Explicit inlet-stream counts are available for 154 processes; two- and three-stream configurations account for 91 and 30 of these records, respectively. These annotations supply precedents rather than prescribing a default number of streams. The diversity supports retrieval across photochemical and non-photochemical processes with different materials and operating regimes.",
20: "Figure 2. Literature corpus and engineering rule-base coverage. (a-d) Reaction classes, reactor types, reported materials, and inlet-stream configurations in the curated flow-chemistry corpus; explicit inlet-stream counts are available for 154 records. (e) Category and severity distribution within the 2,537-rule engineering knowledge base. (f) Recorded associations between rule categories and chemistry classes. Full categorical exports and field-specific availability are reported in Figures S3-S4 and Tables S4-S5; associations indicate knowledge coverage rather than independent validation of every rule.",
23: "The retrieval system was evaluated against a semantic-search baseline (Figure 3). A standard RAG pipeline embeds the batch protocol or query summary and ranks records by cosine similarity. FlowPilot enriches the query using the chemistry plan, including reaction class, mechanism, photocatalyst family, solvent, and wavelength. The filtering sequence starts with paired records under mechanism and phase constraints, relaxes filters when fewer than three matches remain, and expands to all records if no paired match remains. A field-aware score then reranks the retrieved candidates. Figure 3a summarizes this workflow, while the precise tier transitions and scoring weights are specified in Figure S5 and Table S6.",
27: "The strongest retrieval improvement was observed for photocatalyst-family matching. For iridium systems, top-five family matching increased from 40.0% under semantic search to 95.0% after reranking. Organic-dye systems improved from 42.7% to 90.7%, ruthenium systems from 77.3% to 94.7%, TiO2 systems from 87.1% to 98.6%, and ZnO systems from 45.0% to 60.0% (Figure 3c). The demonstration in Figure 3d illustrates how field-aware ranking promotes records with a matching photocatalyst family and relevant wavelength context. These results measure metadata alignment, not the accuracy of a transferred kinetic estimate or the eventual reaction yield. Retrieval tiers, scoring weights, benchmark data, and source-exclusion controls are detailed in Figures S5-S6 and Tables S6-S7.",
29: "FlowPilot was evaluated through matched architecture comparisons that held the chemistry, inventory, and requested output contract constant while changing whether the generator operated as a one-shot model or within the full pipeline (Figure 4a). The three cases were CuAAC, photochemical oxidation, and hydrogenolysis, with three independent generation repeats per model-architecture-case combination. Independent Qwen, OpenAI, and Claude judge families rated the normalized final designs against 14 fixed criteria, using the same source context and deterministic verification evidence. Applicable criteria received equally weighted integer ratings from 0 to 4, which were averaged across judges and normalized to a benchmark score from 0 to 1. Critical flags were recorded separately from the score.",
31: "Figure 4. Ablation benchmarking and cost efficiency. (a) Architecture comparison under shared inputs and fixed evaluation. (b) Benchmark scores and (c) raw judge critical flags: mean and sample SD across three three-case repeats. Flags are not independently confirmed physical errors. These panels retain the archived GPT-5.4 comparison, whereas the ESI summary cohort contains five models. (d) Selected Qwen3.8-27B module ablations: mean and between-case SD, with one generation per case-condition. (e) FlowPilot score per generation cost under the recorded price schedule (log scale), excluding deployment costs. Full definitions and case-level results are in ESI Section 3.",
32: "Across the five-model cohort in Table S11, full-pipeline mean scores ranged from 0.904 to 0.932, compared with 0.684 to 0.890 for matched one-shot generation (Figure 4b). The mean improvement was largest for GPT-4o (+0.226) and smaller for the Claude models (+0.034 and +0.049). Across 45 outcomes per architecture, the judges recorded 98 critical flags for one-shot and 4 for FlowPilot; 18 and 43 outcomes, respectively, had no critical flag (Figure 4c; Table S12). Thus, the observed advantage concerns consistency and completeness under the specified evaluation, not a universal ranking of models or proof of experimental success. In the internal screen, removing the council produced the largest score decrease among the configurations plotted in Figure 4d, while specialist removals had smaller and chemistry-dependent effects.",
33: "Figure 4e reports the complementary resource trade-off: the Qwen configurations achieved higher score-per-generation-cost values under the recorded token-price schedule. Full-pipeline computation nevertheless includes multiple calls, so user-facing simplicity should not be confused with lower total token use. Figures S7-S10 and Tables S8-S12 provide case-level scores, scoring criteria, module results, critical-flag maps, and resource use. Tables S13-S16 define engineering closure, inventory realization, feedback records, and computational replay. Figure S11 and Table S17 illustrate user-composed input, system-assembled prompts, and returned artifacts for one matched run; the legacy heuristic QA values in that example are distinct from the fixed-criteria multi-LLM score used in Figure 4.",
34: "The earlier council model-matrix and candidate-budget experiment is reported separately in Figure S12 and Tables S18-S21. In the five-repeat comparison, the normalized radar-area score changed from 0.31 +/- 0.03 before council review to 0.48 +/- 0.23 afterward. This engineering-profile measure is not the multi-LLM benchmark score in Figure 4. The ESI defines Need revision, New versions, Combined pool, and Disqualified, and reports runtime, token use, geometry, and radar normalization. Figure S13 and Table S22 distinguish the final structured FlowPilot topology from its contemporaneous generated narrative; the complete one-shot response is retained alongside this worked example.",
36: "The first laboratory case couples photoredox Giese addition with aerobic oxidation to a sulfoxide[[42]] (Figure 5). The supplied batch sequence uses an oxygen-free first stage followed by oxygen exposure during the second irradiation stage. The revised KHU input therefore requires oxygen introduction only at Stage 2 and equipment selection from the confirmed inventory, without an inline degassing device. This case will assess whether the translated stage configuration, gas delivery, and residence-time choices support the complete reaction sequence. The updated topology and measured outcomes will be added when the experiments are complete.",
38: "Figure 5. Placeholder for the experimental evaluation of photoredox Giese addition followed by aerobic oxidation. The diagram indicates the required oxygen-free first stage and oxygen introduction at Stage 2. The final figure will contain the updated equipment topology, operating conditions, and wet-lab results.",
42: "Case study 2: telescoped DPDTC-mediated amidation",
43: "The second laboratory case is a two-stage DPDTC-mediated amidation through a thioester intermediate[[43]] (Figure 6). The supplied batch sequence activates 3-methyl-4-nitrobenzoic acid before benzylamine addition, without isolating the intermediate. This case tests interstage feed placement, concentration and molar-flow consistency, and the allocation of reactor volume and residence time across the connected process. The revised KHU inventory and response sets define the available equipment and design objectives. The updated topology, operating conditions, and measured amide yield will be inserted after laboratory evaluation.",
45: "Figure 6. Placeholder for the experimental evaluation of telescoped DPDTC-mediated amidation. Thioester formation is followed by benzylamine addition and amidation without intermediate isolation. The final figure will contain the updated equipment topology, stream concentrations and flow rates, stage conditions, and wet-lab results.",
51: "The results show how separating chemical interpretation from engineering arithmetic can improve batch-to-flow proposals under a shared evaluation. A model may identify a plausible transformation yet report incompatible stream concentrations, reactor volumes, or residence times. FlowPilot addresses these coupled quantities through calculation and reconciliation, while retaining model-based assessment for chemistry and operating choices. The matched comparisons in Figure 4 test the delivered outcome of this combination, rather than awarding points for the presence of additional agents or longer reasoning traces.",
52: "The engineering calculator derives dependent quantities from a common set of inputs, and candidate revisions trigger recalculation before publication. This reduces opportunities for a narrative recommendation to contradict its own volume, flow, and time relationships. Council review adds cross-domain scrutiny, but neither calculation nor deliberation guarantees correct input assumptions. The examples in Tables S22 and S26 make this distinction concrete: a structured result can differ from an accompanying generated narrative, and a judge can misidentify an error. Traceable final parameters and inspectable review records therefore matter as much as a favorable aggregate score.",
55: "The council contribution depends on model and configuration",
56: "The matched architecture comparison provides evidence of higher mean scores with FlowPilot for each of the five retained generators, with the size of the improvement depending on the model (Figure 4b). The internal Qwen3.8-27B screen provides a complementary comparison: removing the council lowered the mean score from 0.928 to 0.670 across the three cases (Figure 4d; Table S10). Individual specialist removals produced smaller differences. Because each internal condition-case cell was generated only once, these results support descriptive component analysis rather than a statistically powered claim that every specialist is necessary or that the full configuration is always optimal.",
57: "The earlier model-matrix and candidate-budget experiments reveal additional configuration sensitivity (Figure S12). Some model combinations completed only a screening pathway rather than the validated pathway; all Gemma-upstream and GPT-4o-mini-upstream entries in Table S18 are marked screen required. Increasing the candidate budget broadened the explored design families and revision activity without producing monotonically better outcomes (Tables S19-S21). These observations qualify the role of the council: it can revise and compare proposals, but cannot be assumed to rescue every weak upstream result. They also distinguish historical development experiments from the matched model comparison in Figure 4.",
59: "Several limitations should be noted. First, computational benchmark performance is not a measurement of reaction yield or laboratory safety. The three benchmark chemistries sample only part of flow-chemistry design space, and three generation repeats provide limited evidence of variability. The laboratory cases in Figures 5 and 6 are intended to test selected proposals against measured outcomes. Integration with experimental platforms[[44]] could extend this process, but chemist review and experimental validation remain essential.",
60: "Second, gas-flow and residence-time definitions require careful interpretation. Gas equivalents are calculated from inlet molar flow at a declared reference temperature and pressure, not from a gas-to-liquid volumetric ratio. The nominal inlet/STP time V/(Q_L + Q_g,STP) is also not a measured in-channel contact time. Gas compression, dissolution, phase holdup, and slip can separate these quantities.[[45]] The ESI retains the distinct time bases used by archived runs (Table S13); comparisons should use a common definition rather than silently relabeling old results.",
62: "Fourth, the outcome benchmark does not establish performance on long multistep telescopes. Figures 5 and 6 extend the evaluation toward connected two-stage chemistry, where an upstream conversion deficit changes the composition entering the next reactor. Independent stage yields, kinetic information, and operational observations may be needed to interpret final performance and guide subsequent refinement.",
63: "Fifth, LLM-based evaluation remains fallible even with fixed criteria and deterministic evidence. Shared model families, incomplete source information, or differences in response presentation may affect judgments. Identity masking is not complete architectural blinding, and multiple critical flags can concern the same underlying defect. The checked false-positive example in Table S26 illustrates why these judgments should remain auditable. Excluding a source paper from retrieval also does not establish that it was absent from model pretraining.",
65: "The immediate next step is to use the existing feedback interface to connect proposed designs with measured process behavior, preserving the input, equipment profile, actual settings, and observed response at each iteration. This would support evidence-based refinement and future integration with Bayesian optimization tools[[24-28]]. Broader chemistry coverage and longer telescoped sequences are further directions for evaluation. Sustainability assessment would require measured or fully specified material and energy balances: reactor volume, nominal time, and temperature alone do not establish an E-factor or process mass intensity. These extensions build on the same auditable design record rather than assuming that a shorter or smaller process is necessarily better.",
67: "We have presented FlowPilot as a hybrid multi-agent system for translating a chemistry brief into a traceable continuous-flow process proposal. Standardized intake captures chemist intent and laboratory constraints; chemistry interpretation and literature retrieval inform the design; deterministic calculations reconcile dependent engineering quantities; and a specialist council reviews candidate operating points. Inventory assignment and a shared final design record connect the selected parameters to the process diagram and supporting report.",
68: "Matched comparisons across the five-model cohort show higher mean fixed-criteria scores and fewer judge critical flags for FlowPilot than for one-shot generation on the three evaluated chemistries. Module ablations indicate a substantial contribution from council review, while the historical model-matrix and budget studies reveal configuration-dependent limitations. The proposed Giese/oxidation and DPDTC-amidation laboratory evaluations will test whether these computational advantages translate into useful experimental starting conditions.",
69: "The practical contribution is to organize the transition from a batch protocol to an inspectable first flow design: the chemist supplies the protocol, answers targeted questions, and binds an inventory, while the system assembles the calculation, review, equipment, and provenance records. This reduces manual information assembly without removing experimental responsibility. FlowPilot is therefore positioned as a design co-pilot whose proposals can be checked, tested, and refined, not as an autonomous guarantee of successful synthesis.",
72: "A literature corpus containing 464 classified flow-chemistry records was annotated with available reaction, reactor, material, operating-condition, yield, and stream-count metadata. Field completeness was reported separately (Table S4). A complementary handbook-derived engineering knowledge base[[20,21,32]] contained 2,537 rules categorized by topic and severity. The rule-base analysis describes coverage and recorded associations, while the deterministic calculator implements the numerical checks used during design.",
74: "The translation workflow comprises standardized intake, chemistry interpretation, retrieval, engineering calculation, candidate generation, council review, inventory realization, and topology rendering. LLMs propose chemical interpretations and independent design choices; deterministic routines recalculate dependent quantities and check their consistency. Chemist objectives and available evidence guide residence-time and throughput choices. A nominal intensification factor is not treated as proof of conversion, and an inventory-compatible operating point is not itself an experimentally validated procedure. Stage parameters, stream assignments, and topology are reconciled before final publication.",
76: "The matched architecture comparison used the same frozen case input and final-output contract for one-shot generation and FlowPilot. Five retained generator models, two architectures, three chemistry cases, and three generation repeats yielded 90 outcomes in the ESI cohort. Three judge families received identity-masked final designs, source context, deterministic verification evidence, and the fixed rubric in Tables S8-S9. Each applicable criterion was rated from 0 to 4; ratings were averaged across judges and criteria, then divided by four. Gas bookkeeping and multistage closure were omitted only when inapplicable. Per-model error bars summarize the sample SD of three repeat-level means, each averaging the three cases. Critical flags were recorded separately. Exact inputs, model records, applicability rules, and archived evaluation limitations are given in ESI Section 3.",
}

AFTER = {
34: "The operational examples connect these evaluations to the chemist-facing workflow. Figures S14-S15 and Table S23 show batch input and fixed-ID follow-up questions; Figure S16 and Table S24 show inventory import, normalization, and remaining equipment warnings. Figure S17 and Table S25 show stage-resolved topology and stream parameters from a reopened run. Table S26 provides concrete one-shot and FlowPilot issues, including an evaluator false positive. Figure S18 and Table S27 retain actual council disagreement and selection records, while Figure S19 and Table S28 place these views in the full GUI and saved-run context. These examples document human interaction and provenance rather than additional wet-lab outcomes.",
76: "The internal module screen fixed the generator to Qwen3.8-27B and generated each condition-case combination once; its SD therefore measures variation between chemistries, not repeatability. Figure 4d shows selected conditions and Table S10 provides the displayed module summaries. The separate historical model-matrix experiment crossed four upstream and four council endpoints, and the candidate-budget study tested B = 1, 6, 12, and 24 with five repeats. Their validation dispositions, numerical data, and metric definitions are reported in ESI Section 6.1 and Tables S18-S21. These studies are kept distinct from the independent-judge outcome benchmark.",
}

TEXT[29] += " Scoring anchors, the criterion definitions, and module-screen summaries are provided in Tables S8-S10."

REMOVE = {39, 40, 41, 46, 47, 48}


def text(p):
    return "".join(p.xpath(".//w:t/text()", namespaces=NS))


def marked_text(p):
    parts=[]
    for r in p.findall(W+"r"):
        prop=r.find(W+"rPr")
        align=prop.find(W+"vertAlign") if prop is not None else None
        value=text(r)
        parts.append("[["+value+"]]" if align is not None and align.get(W+"val")=="superscript" else value)
    return "".join(parts)


def first_body_properties(p):
    runs = p.findall(W + "r")
    for r in runs:
        prop = r.find(W + "rPr")
        if prop is None:
            if text(r).strip():
                return None
            continue
        if prop.find(W + "vertAlign") is None and prop.find(W + "b") is None and prop.find(W + "rStyle") is None and text(r).strip():
            return deepcopy(prop)
    return deepcopy(runs[0].find(W + "rPr")) if runs else None


def append_run(p, value, props):
    if not value:
        return
    r = E.SubElement(p, W + "r")
    if props is not None:
        r.append(deepcopy(props))
    t = E.SubElement(r, W + "t")
    t.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")
    t.text = value


def revise(p, value, citation_props):
    """Retain original character styles on matching spans; inherit body style for new text."""
    assert not p.xpath(".//w:drawing|.//w:fldChar|.//w:instrText", namespaces=NS)
    old = text(p)
    new = re.sub(r"\[\[([^]]+)\]\]", r"\1", value)
    default = first_body_properties(p)
    props = []
    for r in p.findall(W + "r"):
        props.extend([r.find(W + "rPr")] * len(text(r)))
    assert len(props) == len(old)
    new_props = [default] * len(new)
    for match in difflib.SequenceMatcher(None, old, new, autojunk=False).get_matching_blocks():
        for k in range(match.size):
            # Do not let an unrelated number inherit a citation/subscript by coincidence.
            prop = props[match.a + k]
            if prop is None or prop.find(W + "vertAlign") is None:
                new_props[match.b + k] = prop
    offset = 0
    for m in re.finditer(r"\[\[([^]]+)\]\]", value):
        start = m.start() - offset
        new_props[start:start + len(m[1])] = [citation_props] * len(m[1])
        offset += 4
    for child in list(p):
        if child.tag != W + "pPr":
            p.remove(child)
    for i, ch in enumerate(new):
        if i and new_props[i] is new_props[i-1]:
            p[-1][-1].text += ch
        else:
            append_run(p, ch, new_props[i])


def placeholder(number):
    """New vector-layout placeholders only; no edits to Figures 1-4."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    fig, ax = plt.subplots(figsize=(8, 3.5))
    fig.subplots_adjust(left=.01, right=.99, bottom=.01, top=.99)
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.axis("off")
    labels = (["Stage 1\nGiese addition\nOxygen-free", "Stage 2\nAerobic oxidation", "Product analysis\nResults pending"] if number == 5 else
              ["Stage 1\nThioester formation", "Stage 2\nAmidation", "Product analysis\nResults pending"])
    for x, label in zip([.025, .365, .705], labels):
        ax.add_patch(Rectangle((x, .36), .27, .32, facecolor="#F5F7F8", edgecolor="#516679", linewidth=1.3))
        ax.text(x+.135, .52, label, ha="center", va="center", fontsize=12, linespacing=1.5, color="#253A4B")
    for x1,x2 in [(.295,.365),(.635,.705)]:
        ax.annotate("", xy=(x2,.52), xytext=(x1,.52), arrowprops={"arrowstyle":"->","lw":1.4,"color":"#516679"})
    ax.text(.33, .85, "Oxygen feed" if number == 5 else "Benzylamine feed", ha="center", fontsize=11)
    ax.annotate("", xy=(.33,.52), xytext=(.33,.79), arrowprops={"arrowstyle":"->","lw":1.4,"color":"#516679"})
    ax.text(.5, .15, "Placeholder: updated topology and wet-lab data to be inserted", ha="center", fontsize=11, color="#516679")
    # Keep the original Word picture width but use the compact diagram's aspect ratio.
    target = CHECK / f"figure_{number}_placeholder.png"
    fig.savefig(target, dpi=300, facecolor="white")
    fig.savefig(CHECK / f"figure_{number}_placeholder.svg", facecolor="white")
    plt.close(fig)
    return target.read_bytes()


def references(document, esi):
    captions = {}
    for p in esi.xpath("/w:document/w:body/w:p", namespaces=NS):
        s = text(p)
        m = re.match(r"(Figure|Table) S(\d+)\.", s)
        if m:
            captions[(m[1], int(m[2]))] = s
    rows = []
    for i,p in enumerate(document.xpath("/w:document/w:body/w:p", namespaces=NS)):
        s = text(p)
        pattern = r"\b(Figures?|Tables?) S(\d+(?:\s*[-\u2013]\s*S?\d+)?(?:\s*(?:,|and)\s*S?\d+(?:\s*[-\u2013]\s*S?\d+)?)*)"
        for m in re.finditer(pattern, s):
            kind = m[1].rstrip("s")
            for a,b in re.findall(r"(\d+)(?:\s*[-\u2013]\s*S?(\d+))?", m[2]):
                for n in range(int(a), int(b or a)+1):
                    assert (kind,n) in captions, (kind,n,s)
                    rows.append({"paragraph": i, "reference": f"{kind} S{n}", "caption_in_unchanged_esi": captions[kind,n], "manuscript_text": s})
    with (CHECK / "esi_cross_references.csv").open("w", newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    assert set(captions).issubset({(r["reference"].split()[0],int(r["reference"].split(" S")[1])) for r in rows})
    orders={}
    for kind in ["Figure", "Table"]:
        seen=[]
        for row in rows:
            if row["reference"].startswith(kind):
                n=int(row["reference"].split(" S")[1])
                if n not in seen:seen.append(n)
        assert seen==sorted(seen), (kind,seen)
        orders[kind]=seen
    return {"all_targets_exist": True, "all_19_figures_and_28_tables_cited": True,
            "esi_first_citation_order":orders}


def main():
    CHECK.mkdir(exist_ok=True)
    original_hash = sha256(SOURCE.read_bytes()).hexdigest()
    esi_hash = sha256(ESI.read_bytes()).hexdigest()
    assert original_hash == "d8515d9002f841080f9c1d1d79a7e3d8d7b8ed7e924eb72a7a5ae9d2c6801a18"
    assert esi_hash == "8db59e26eb0568c0821ae40d9b4a42e15f69e35afa3c9f4501ba6897a03605fb"
    with ZipFile(SOURCE) as z:
        content={name:z.read(name) for name in z.namelist()}
        infos=z.infolist()
    doc=E.fromstring(content["word/document.xml"])
    old_doc=deepcopy(doc)
    ps=doc.xpath("/w:document/w:body/w:p",namespaces=NS)
    original_ps=old_doc.xpath("/w:document/w:body/w:p",namespaces=NS)
    citation=next(deepcopy(p) for p in ps[5].xpath(".//w:rPr",namespaces=NS) if p.find(W+"vertAlign") is not None)

    # Small local edits preserve the surrounding original argument and citation order.
    TEXT[5]=marked_text(ps[5]).replace("These intensified conditions translate directly into reduced solvent consumption, lower energy input, and smaller equipment footprints, aligning flow synthesis with the resource-efficiency goals of sustainable manufacturing.","These features can reduce solvent use, energy demand, and equipment footprint when the process is appropriately designed, supporting the resource-efficiency goals of sustainable manufacturing.")
    TEXT[6]=marked_text(ps[6]).replace("No current tool designs a flow process from a chemistry brief while honoring these handbook-level fundamentals.","The remaining challenge is to connect a chemistry brief to equipment-aware flow design while making these handbook-level fundamentals available for systematic review.")
    TEXT[7]=marked_text(ps[7]).replace("with no mechanism to enforce cross-domain consistency. The predictable outcome is internally inconsistent outputs and no traceable audit of which constraints were evaluated.","without a separate calculation and reconciliation workflow. A fluent answer therefore does not by itself establish cross-domain consistency or show which constraints were checked.")
    TEXT[21]=marked_text(ps[21]).replace("This structured rule landscape constrains model-generated designs with flow-chemistry fundamentals instead of relying only on model priors.","This structured rule landscape supplies domain context for evaluating model-generated designs alongside deterministic checks, rather than relying only on model priors.")
    TEXT[54]=marked_text(ps[54]).replace("Zeng et al.[[34]]", "Zeng et al.[[36]]").replace("Zeng et al.34", "Zeng et al.[[36]]")
    TEXT[88]=text(ps[88]).replace("BPR, back-pressure regulator;", "BPR, back-pressure regulator; DPDTC, dipyridyldithiocarbonate;")
    TEXT[60]=TEXT[60].replace("The nominal inlet/STP time V/(Q_L + Q_g,STP)","The nominal inlet/STP time, calculated from reactor volume divided by the combined liquid and inlet/STP gas flows,")
    TEXT[65]=TEXT[65].replace("[[24-28]]", "[[24\u201328]]")
    changes=[]
    for i,value in sorted(TEXT.items()):
        changes.append({"original_paragraph":i,"before":text(ps[i]),"after":re.sub(r"\[\[([^]]+)\]\]",r"\1",value)})
        revise(ps[i],value,citation)
        assert E.tostring(ps[i].find(W+"pPr"))==E.tostring(original_ps[i].find(W+"pPr"))
    for i,value in AFTER.items():
        new=deepcopy(original_ps[i]);revise(new,value,citation);ps[i].addnext(new)
        changes.append({"inserted_after_original_paragraph":i,"after":value})
    for i in REMOVE:
        changes.append({"original_paragraph":i,"before":text(ps[i]),"after":"", "reason":"Obsolete case-specific results replaced by the brief Figure 5/6 placeholder sections."})
        ps[i].getparent().remove(ps[i])

    # Replace only reference 43, retaining its original journal/year/volume formatting.
    r=ps[132].findall(W+"r")
    segments=["Saunders, J. M.; Oceguera Nava, E.; Li, J.; Wong, M. J.; Freiberg, K. M.; Lipshutz, B. H. Flow-to-Flow Technology: Amide Formation in the Absence of Traditional Coupling Reagents Using DPDTC. ","ACS Sustainable Chem. Eng."," ","2025",", ","13",", 6646\u20136655. https://doi.org/10.1021/acssuschemeng.5c00914."]
    assert len(r)==len(segments)
    old_ref=text(ps[132])
    for run,value in zip(r,segments):
        ts=run.findall(W+"t");assert len(ts)==1;ts[0].text=value
    changes.append({"original_paragraph":132,"before":old_ref,"after":text(ps[132])})

    replacements={}
    for number,idx in [(5,37),(6,44)]:
        replacements[f"word/media/image{number}.png"]=placeholder(number)
        extent=ps[idx].find(".//{http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing}extent")
        width=int(extent.get("cx"));height=round(width*3.5/8)
        extent.set("cy",str(height))
        xfrm=ps[idx].find(".//{http://schemas.openxmlformats.org/drawingml/2006/main}xfrm/{http://schemas.openxmlformats.org/drawingml/2006/main}ext")
        xfrm.set("cy",str(height))
        # A placeholder has no chemistry-specific accessibility metadata from the old figure.
        for prop in ps[idx].xpath(".//*[@descr] | .//*[@title]"):
            if "descr" in prop.attrib:prop.set("descr",f"Figure {number} placeholder: updated topology and wet-lab data pending")
            if "title" in prop.attrib:prop.set("title",f"Figure {number} placeholder")
    replacements["word/document.xml"]=E.tostring(doc,xml_declaration=True,encoding="UTF-8",standalone=True)
    with ZipFile(DEST,"w") as z:
        for info in infos:z.writestr(info,replacements.get(info.filename,content[info.filename]))

    allowed={"word/document.xml","word/media/image5.png","word/media/image6.png"}
    with ZipFile(DEST) as z:
        changed={name for name in content if content[name]!=z.read(name)}
        assert changed==allowed,changed
    for tag in ["rFonts","sz","szCs"]:
        before={tuple(sorted(e.attrib.items())) for e in old_doc.xpath("//w:"+tag,namespaces=NS)}
        after={tuple(sorted(e.attrib.items())) for e in doc.xpath("//w:"+tag,namespaces=NS)}
        assert after.issubset(before), (tag,after-before)
    assert "[[" not in text(doc)
    superscripts=doc.xpath("//w:r[w:rPr/w:vertAlign[@w:val='superscript']]/w:t/text()",namespaces=NS)
    cited=set()
    for s in superscripts:
        for a,b in re.findall(r"(\d+)(?:[-\u2013](\d+))?",s):
            cited.update(range(int(a),int(b or a)+1))
    assert cited==set(range(1,46)),cited
    for i in range(len(ps)):
        if i not in set(TEXT)|REMOVE|{37,44,132}:
            assert E.tostring(ps[i])==E.tostring(original_ps[i]),i
    assert len(doc.xpath("//w:drawing",namespaces=NS))==6
    with ZipFile(ESI) as z:esidoc=E.fromstring(z.read("word/document.xml"))
    xrefs=references(doc,esidoc)
    assert sha256(SOURCE.read_bytes()).hexdigest()==original_hash
    assert sha256(ESI.read_bytes()).hexdigest()==esi_hash
    assert not (BASE/"revision_20260917").exists()
    checks={"source_manuscript_unchanged":True,"esi_unchanged":True,
            "figures_1_to_4_identical_bytes":True,"figures_1_to_4_drawing_xml_unchanged":True,
            "styles_numbering_themes_headers_footers_identical_bytes":True,
            "section_page_settings_unchanged":E.tostring(doc.find(".//"+W+"sectPr"))==E.tostring(old_doc.find(".//"+W+"sectPr")),
            "existing_paragraph_properties_unchanged":True,"font_sizes_not_reset":True,
            "changed_docx_parts":sorted(changed),"old_revision_removed":True,**xrefs}
    (CHECK/"preservation_checks.json").write_text(json.dumps(checks,indent=2)+"\n")
    (CHECK/"text_changes.json").write_text(json.dumps(changes,indent=2,ensure_ascii=False)+"\n")
    print(json.dumps(checks,indent=2));print(DEST)


def check_layout():
    import fitz
    from PIL import Image, ImageDraw
    results={}
    for stem in ["manuscript", "manuscript_text_revised"]:
        pdf_path=Path("/tmp/flowpilot_text_only_review")/(stem+".pdf")
        shutil.copy2(pdf_path,CHECK/(stem+".pdf"))
        pdf=fitz.open(pdf_path)
        folder=CHECK/stem;folder.mkdir(exist_ok=True)
        outside=[];pages=[]
        for i,page in enumerate(pdf):
            pix=page.get_pixmap(matrix=fitz.Matrix(1.3,1.3),alpha=False)
            path=folder/f"page_{i+1:02d}.png";pix.save(path);pages.append(path)
            for word in page.get_text("words"):
                if word[0]<-1 or word[1]<-1 or word[2]>page.rect.width+1 or word[3]>page.rect.height+1:
                    outside.append({"page":i+1,"text":word[4]})
        for start in range(0,len(pages),6):
            sheet=Image.new("RGB",(1250,1800),"#d9dde0");draw=ImageDraw.Draw(sheet)
            for j,path in enumerate(pages[start:start+6]):
                tile=Image.open(path);tile.thumbnail((615,565))
                x=(j%2)*625;y=(j//2)*600
                draw.text((x+8,y+8),f"{stem}: page {start+j+1}",fill="black")
                sheet.paste(tile,(x+(625-tile.width)//2,y+30))
            sheet.save(folder/f"contact_{start+1:02d}.png")
        results[stem]={"pages":len(pdf),"out_of_page_words":outside}
    (CHECK/"layout_checks.json").write_text(json.dumps(results,indent=2)+"\n")
    print(json.dumps(results,indent=2))


if __name__=="__main__":
    check_layout() if "--check-layout" in sys.argv else main()
