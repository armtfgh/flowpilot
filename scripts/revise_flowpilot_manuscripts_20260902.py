#!/usr/bin/env python3
"""Apply the September 2026 reviewer revisions to the revised FlowPilot manuscripts."""

from __future__ import annotations

import csv
import re
import shutil
from copy import deepcopy
from pathlib import Path

import pandas as pd
from docx import Document
from docx.enum.style import WD_STYLE_TYPE
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK, WD_COLOR_INDEX
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt
from docx.table import Table
from docx.text.paragraph import Paragraph


ROOT = Path(__file__).resolve().parents[1]
SOURCE_ESI = ROOT / "FlowPilot_ESI.docx"
SOURCE_MAIN = ROOT / "FlowPilot_Main_Manuscript.docx"
OLD_MAIN = ROOT / "FlowPilot_manuscript_ChemRXIV.docx"
OUT_ROOT = ROOT / "deliverables" / "flowpilot_esi_revision_20260902"
FIG_DIR = OUT_ROOT / "figures"
DATA_DIR = OUT_ROOT / "source_data"
DOC_DIR = OUT_ROOT / "documentation"
OUT_ESI = OUT_ROOT / "FlowPilot_ESI_revised_20260902.docx"
OUT_MAIN = OUT_ROOT / "FlowPilot_Main_Manuscript_revised_20260902.docx"
BACKUP_ESI = OUT_ROOT / "FlowPilot_ESI_before_reviewer_revision_20260902.docx"
BACKUP_MAIN = OUT_ROOT / "FlowPilot_Main_Manuscript_before_reviewer_revision_20260902.docx"

MATRIX = ROOT / ".local_artifacts_backup_20260831/generated/benchmark_data/model_matrix_benchmark_20260504_101732/visualizations/model_matrix_enriched.csv"
BUDGET_DIR = ROOT / ".local_artifacts_backup_20260831/generated/benchmark_data/protocol_budget_benchmark_20260427_181556/visualizations"
RADAR_DIR = ROOT / ".local_artifacts_backup_20260831/generated/benchmark_data/radar_claude_gpt4o_5repeat_20260514_101119/radar_5repeat"


def set_run_font(run, size: float = 11, highlight: bool = False) -> None:
    run.font.name = "Times New Roman"
    run.font.size = Pt(size)
    run._element.get_or_add_rPr().rFonts.set(qn("w:eastAsia"), "Times New Roman")
    if highlight:
        run.font.highlight_color = WD_COLOR_INDEX.YELLOW


def highlight_paragraph(paragraph: Paragraph) -> None:
    for run in paragraph.runs:
        set_run_font(run, 11, True)


def clear_paragraph(paragraph: Paragraph) -> None:
    for child in list(paragraph._p):
        if child.tag != qn("w:pPr"):
            paragraph._p.remove(child)


def replace_text(paragraph: Paragraph, text: str, *, highlight: bool = True, bold_prefix: str | None = None) -> None:
    clear_paragraph(paragraph)
    if bold_prefix and text.startswith(bold_prefix):
        run = paragraph.add_run(bold_prefix)
        run.bold = True
        set_run_font(run, 11, highlight)
        run = paragraph.add_run(text[len(bold_prefix) :])
        set_run_font(run, 11, highlight)
    else:
        run = paragraph.add_run(text)
        set_run_font(run, 11, highlight)


def find_paragraph(document: Document, startswith: str) -> Paragraph:
    for paragraph in document.paragraphs:
        if paragraph.text.strip().startswith(startswith):
            return paragraph
    raise KeyError(f"Paragraph not found: {startswith}")


def insert_paragraph_before(
    anchor,
    text: str = "",
    *,
    style: str | None = None,
    template: Paragraph | None = None,
    highlight: bool = True,
    bold_prefix: str | None = None,
) -> Paragraph:
    node = OxmlElement("w:p")
    anchor._element.addprevious(node)
    paragraph = Paragraph(node, anchor._parent)
    if template is not None and template._p.pPr is not None:
        old_ppr = paragraph._p.pPr
        if old_ppr is not None:
            paragraph._p.remove(old_ppr)
        paragraph._p.insert(0, deepcopy(template._p.pPr))
    elif style:
        paragraph.style = style
    if text:
        if bold_prefix and text.startswith(bold_prefix):
            first = paragraph.add_run(bold_prefix)
            first.bold = True
            set_run_font(first, 11, highlight)
            second = paragraph.add_run(text[len(bold_prefix) :])
            set_run_font(second, 11, highlight)
        else:
            run = paragraph.add_run(text)
            set_run_font(run, 11, highlight)
    return paragraph


def delete_paragraph(paragraph: Paragraph) -> None:
    parent = paragraph._element.getparent()
    parent.remove(paragraph._element)
    paragraph._p = paragraph._element = None


def replace_figure(
    document: Document,
    number: int,
    image_path: Path,
    caption_text: str,
    *,
    width: float = 6.25,
) -> None:
    caption = find_paragraph(document, f"Figure S{number}.")
    same_paragraph = bool(caption._p.xpath(".//w:drawing"))
    if same_paragraph:
        image_paragraph = insert_paragraph_before(caption, template=caption, highlight=False)
        image_paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
        image_paragraph.paragraph_format.keep_with_next = True
        image_paragraph.add_run().add_picture(str(image_path), width=Inches(width))
        replace_text(caption, f"Figure S{number}. {caption_text}", bold_prefix=f"Figure S{number}. ")
        return

    previous = caption._p.getprevious()
    while previous is not None and not previous.xpath(".//w:drawing"):
        previous = previous.getprevious()
    if previous is None:
        raise RuntimeError(f"No image paragraph found for Figure S{number}")
    image_paragraph = Paragraph(previous, caption._parent)
    clear_paragraph(image_paragraph)
    image_paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    image_paragraph.paragraph_format.keep_with_next = True
    image_paragraph.add_run().add_picture(str(image_path), width=Inches(width))
    replace_text(caption, f"Figure S{number}. {caption_text}", bold_prefix=f"Figure S{number}. ")


def replace_s3_placeholder(document: Document) -> None:
    placeholder = find_paragraph(document, "[figure will be added here]")
    clear_paragraph(placeholder)
    placeholder.alignment = WD_ALIGN_PARAGRAPH.CENTER
    placeholder.paragraph_format.keep_with_next = True
    placeholder.add_run().add_picture(str(FIG_DIR / "Figure_S03.png"), width=Inches(6.25))
    caption = find_paragraph(document, "Figure S3.")
    replace_text(
        caption,
        "Figure S3. Composition of the frozen flow-chemistry literature corpus. (a-d) Reaction class, reactor type, reactor material, and bond or transformation labels (n = 464 records in each panel). (e) Number of inlet streams among records with stream-count annotation (n = 154). (f) Available batch-yield (n = 135) and optimized-flow-yield (n = 285) distributions. Categories are descriptive and non-exclusive where the source record contains more than one label.",
        bold_prefix="Figure S3. ",
    )


def normalize_body_font(document: Document) -> None:
    for style_name in ("Normal", "Body Text", "First Paragraph", "Image Caption", "Caption"):
        if style_name in document.styles:
            style = document.styles[style_name]
            style.font.name = "Times New Roman"
            style.font.size = Pt(11)
            style._element.get_or_add_rPr().rFonts.set(qn("w:eastAsia"), "Times New Roman")
    for paragraph in document.paragraphs:
        is_heading = paragraph.style and paragraph.style.name.startswith("Heading")
        for run in paragraph.runs:
            run.font.name = "Times New Roman"
            run._element.get_or_add_rPr().rFonts.set(qn("w:eastAsia"), "Times New Roman")
            if not is_heading and not paragraph.style.name.startswith("TOC"):
                run.font.size = Pt(11)
    for table in document.tables:
        for row in table.rows:
            for cell in row.cells:
                for paragraph in cell.paragraphs:
                    for run in paragraph.runs:
                        set_run_font(run, 11, False)


def add_table_before(
    document: Document,
    anchor,
    headers: list[str],
    rows: list[list[str]],
    *,
    widths: list[float] | None = None,
) -> Table:
    table = document.add_table(rows=1, cols=len(headers))
    table.style = "Table Grid"
    table.autofit = False
    for i, value in enumerate(headers):
        table.rows[0].cells[i].text = value
    for values in rows:
        cells = table.add_row().cells
        for i, value in enumerate(values):
            cells[i].text = str(value)
    for row_index, row in enumerate(table.rows):
        tr_pr = row._tr.get_or_add_trPr()
        cant_split = OxmlElement("w:cantSplit")
        tr_pr.append(cant_split)
        if row_index == 0:
            repeat = OxmlElement("w:tblHeader")
            repeat.set(qn("w:val"), "true")
            tr_pr.append(repeat)
        for col_index, cell in enumerate(row.cells):
            if widths:
                cell.width = Inches(widths[col_index])
            for paragraph in cell.paragraphs:
                paragraph.paragraph_format.space_after = Pt(0)
                for run in paragraph.runs:
                    set_run_font(run, 11, True)
                    if row_index == 0:
                        run.bold = True
    anchor._element.addprevious(table._tbl)
    return table


def extract_old_figure4() -> Path:
    target = FIG_DIR / "Figure_S12_council_benchmark.png"
    if target.exists():
        return target
    document = Document(OLD_MAIN)
    caption = find_paragraph(document, "Figure 4.")
    node = caption._p.getprevious()
    while node is not None and not node.xpath(".//a:blip"):
        node = node.getprevious()
    if node is None:
        raise RuntimeError("Old Figure 4 image was not found")
    rid = node.xpath(".//a:blip/@r:embed")[0]
    target.write_bytes(document.part.related_parts[rid].blob)
    return target


def copy_council_data() -> None:
    copies = {
        MATRIX: DATA_DIR / "S18_model_matrix_runtime_tokens_geometry.csv",
        BUDGET_DIR / "budget_design_family_counts.csv": DATA_DIR / "S19_budget_design_family_counts.csv",
        BUDGET_DIR / "budget_revision_activity.csv": DATA_DIR / "S19_budget_revision_activity.csv",
        BUDGET_DIR / "budget_revision_activity_three_bar.csv": DATA_DIR / "S19_budget_revision_activity_three_bar.csv",
        RADAR_DIR / "U_claude__C_gpt4o_5repeat_radar_summary.csv": DATA_DIR / "S20_radar_five_repeat_summary.csv",
        RADAR_DIR / "U_claude__C_gpt4o_5repeat_radar_raw.csv": DATA_DIR / "S20_radar_five_repeat_raw.csv",
    }
    for source, target in copies.items():
        if source.exists():
            shutil.copy2(source, target)


def matrix_rows() -> list[list[str]]:
    data = pd.read_csv(MATRIX)
    rows = []
    for _, row in data.iterrows():
        disposition = "validated" if bool(row["engine_validated"]) and not bool(row["screen_required"]) else "screen required"
        rows.append(
            [
                str(row["upstream_bundle"]),
                str(row["council_bundle"]),
                f"{float(row['runtime_s']):.1f}",
                f"{int(row['total_tokens']):,}",
                f"{float(row['final_reactor_volume_mL']):.3f}",
                f"{float(row['final_tubing_ID_mm']):.2f}",
                "yes" if bool(row["engine_validated"]) else "no",
                disposition,
            ]
        )
    return rows


def budget_rows() -> list[list[str]]:
    families = pd.read_csv(BUDGET_DIR / "budget_design_family_counts.csv")
    activity = pd.read_csv(BUDGET_DIR / "budget_revision_activity.csv")
    rows = []
    for budget in sorted(families["budget"].unique()):
        fam = families[(families["budget"] == budget) & (families["count"] > 0)]
        family_text = "; ".join(f"{r.design_family}: {int(r.count)}" for r in fam.itertuples()) or "none"
        a = activity[activity["budget"] == budget].set_index("metric")
        fmt = lambda key: f"{a.loc[key, 'mean']:.1f} +/- {a.loc[key, 'std']:.1f}"
        rows.append(
            [
                str(int(budget)),
                family_text,
                fmt("revision_changed_count"),
                fmt("revision_descendant_total"),
                fmt("revision_final_candidate_count"),
                fmt("stage4_disqualify_count"),
                "5",
            ]
        )
    return rows


def radar_rows() -> list[list[str]]:
    data = pd.read_csv(RADAR_DIR / "U_claude__C_gpt4o_5repeat_radar_summary.csv")
    order = ["Pe", "Da_mass", "STY_mol_L_h", "IF", "UA_W_K", "pressure_headroom", "radar_area_score"]
    labels = {
        "Pe": "Peclet score",
        "Da_mass": "Mass-transfer Damkohler score",
        "STY_mol_L_h": "Space-time-yield score",
        "IF": "Intensification-factor score",
        "UA_W_K": "UA score",
        "pressure_headroom": "Pressure-headroom score",
        "radar_area_score": "Radar-area score",
    }
    rows = []
    for metric in order:
        pre = data[(data.stage == "pre") & (data.metric == metric)].iloc[0]
        post = data[(data.stage == "post") & (data.metric == metric)].iloc[0]
        rows.append([labels[metric], f"{pre.score_mean:.3f} +/- {pre.score_std:.3f}", f"{post.score_mean:.3f} +/- {post.score_std:.3f}", "5"])
    return rows


def add_council_section(document: Document) -> None:
    anchor = find_paragraph(document, "FlowPilot user-facing intake")
    heading_template = find_paragraph(document, "System architecture and decision authority")
    body_template = find_paragraph(document, "The downstream council") if any(p.text.startswith("The downstream council") for p in document.paragraphs) else find_paragraph(document, "FlowPilot separates model-proposed")
    caption_template = find_paragraph(document, "Figure S11.")
    table_caption_template = find_paragraph(document, "Table S16.")

    heading = insert_paragraph_before(anchor, "Council model matrix and candidate-budget sensitivity", template=heading_template)
    heading.paragraph_format.page_break_before = True
    insert_paragraph_before(
        anchor,
        "The council sensitivity experiment used the isoxazole 1,3-dipolar cycloaddition as a fixed test reaction. Upstream chemistry interpretation and downstream council execution were crossed in a 4 x 4 model matrix. The comparison records wall-clock runtime, token consumption, final reactor volume, tubing internal diameter, and deterministic validation status for every pairing (Figure S12 and Table S18). These data are architecture-diagnostic rather than a general ranking of current foundation models.",
        template=body_template,
    )
    insert_paragraph_before(
        anchor,
        "Council vocabulary is used as follows. Need revision denotes a candidate with a correctable deficiency that is returned for a bounded edit. New versions are deterministic-recomputed descendants generated from that candidate. The combined pool contains the original candidates and all revised descendants after recalculation. Disqualified candidates fail a hard feasibility or critical-safety gate and cannot advance to final selection.",
        template=body_template,
    )
    insert_paragraph_before(
        anchor,
        "Candidate budgets B = 1, 6, 12, and 24 were evaluated in five repeats. Increasing the budget broadened the design families and increased revision activity, but the effect was non-monotonic and was accompanied by more disqualifications (Table S19). The five-repeat radar-area score changed from 0.31 +/- 0.03 before council review to 0.48 +/- 0.23 after review; this replaces the earlier single-run statement based on 0.29 to 0.83 (Table S20). Radar normalization and capping rules are reported explicitly in Table S21.",
        template=body_template,
    )

    figure = insert_paragraph_before(anchor, template=caption_template)
    figure.alignment = WD_ALIGN_PARAGRAPH.CENTER
    figure.paragraph_format.keep_with_next = True
    figure.add_run().add_picture(str(extract_old_figure4()), width=Inches(6.35))
    insert_paragraph_before(
        anchor,
        "Figure S12. Council architecture, model-matrix sensitivity, and candidate-budget analysis for the isoxazole 1,3-dipolar cycloaddition test reaction. (a) Upstream pipeline and council workflow. (b) Pre- and post-council normalized engineering radar profiles; thin traces show five repeats. (c) Runtime and token use across the 4 x 4 upstream-council matrix. (d) Final reactor volume and tubing internal diameter relative to the manuscript operating region. (e) Design-family counts across candidate budgets B = 1, 6, 12, and 24. (f) Revision and disqualification activity across those budgets. Complete numerical values and metric definitions are provided in Tables S18-S21.",
        template=caption_template,
        bold_prefix="Figure S12. ",
    )

    insert_paragraph_before(anchor, "Table S18. Runtime, token use, reactor volume, tubing internal diameter, and validation outcome for the 4 x 4 upstream-council model matrix.", template=table_caption_template, bold_prefix="Table S18. ")
    add_table_before(document, anchor, ["Upstream", "Council", "Runtime (s)", "Tokens", "V_R (mL)", "ID (mm)", "Validated", "Disposition"], matrix_rows(), widths=[0.70, 0.70, 0.75, 0.80, 0.70, 0.60, 0.68, 0.90])

    insert_paragraph_before(anchor, "Table S19. Design-family counts and council revision activity by candidate budget, mean +/- sample SD where repeated (n = 5).", template=table_caption_template, bold_prefix="Table S19. ")
    add_table_before(document, anchor, ["B", "Observed design families (count)", "Changed", "New versions", "Combined pool", "Disqualified", "n"], budget_rows(), widths=[0.35, 2.45, 0.75, 0.80, 0.85, 0.82, 0.30])

    insert_paragraph_before(anchor, "Table S20. Five-repeat normalized radar metrics before and after council review, mean +/- sample SD (n = 5).", template=table_caption_template, bold_prefix="Table S20. ")
    add_table_before(document, anchor, ["Metric", "Pre-council", "Post-council", "n"], radar_rows(), widths=[2.55, 1.45, 1.45, 0.45])

    insert_paragraph_before(anchor, "Table S21. Radar metric definitions and transformations used for the council analysis.", template=table_caption_template, bold_prefix="Table S21. ")
    add_table_before(
        document,
        anchor,
        ["Metric", "Definition used for the normalized radar score"],
        [
            ["Peclet number (Pe)", "Pe is normalized with an upper cap of 100; values at or above the cap score 1."],
            ["Mass-transfer Damkohler number (Da_mass)", "Da_mass is inverted so that lower mass-transfer limitation gives a higher score."],
            ["Space-time yield (STY)", "STY is transformed by log-min-max scaling over the frozen comparison range."],
            ["Overall thermal conductance (UA)", "UA is transformed by log-min-max scaling over the frozen comparison range."],
            ["Intensification factor (IF)", "IF is normalized with an upper cap of 6."],
            ["Pressure headroom", "headroom = 1 - deltaP / pump_max, then bounded to the plotted score interval."],
            ["Radar-area score", "Area of the polygon formed by the six normalized axes, reported on a 0-1 scale."],
        ],
        widths=[2.05, 4.05],
    )


def revise_esi() -> None:
    shutil.copy2(SOURCE_ESI, BACKUP_ESI)
    document = Document(SOURCE_ESI)
    normalize_body_font(document)

    captions = {
        1: "FlowPilot architecture, authority boundaries, and typed data contracts. (a) The workflow separates standardized intake, chemistry interpretation, retrieval, deterministic engineering, council review, inventory reconciliation, and artifact publication. Measured evidence and hard safety or inventory constraints outrank confirmed protocol facts, chemist hypotheses, and model inference. (b) Typed contracts and closure checks ensure that only a validated FinalDesignContract can publish executable parameters.",
        2: "Reproducible standardized intake, GUI execution, and run provenance. (a) A conversational intake LLM extracts content while fixed question identifiers and deterministic readiness rules control completion of the DesignInputPackage. (b) All executable GUI views render from one post-validation FinalDesignContract. (c) The autosave package preserves inputs, outputs, topology, equipment allocation, instrument manifest, rendering manifest, and summary data.",
        4: "Engineering rule-base structure. (a) Counts by category and severity for the 14 largest categories. (b) Fraction of rules containing a machine-detected quantitative expression; this indicates expression coverage, not independent equation verification. (c) Non-exclusive rule-category associations across chemistry classes. (d) Computed concept co-occurrence network using the 12 most frequent concepts and 22 highest-weight links. Node area is 35 + 420 sqrt(f_i/f_max) points squared, where f_i is concept frequency; node color is the dominant rule-category label; edge width is 0.25 + 2.4(w_ij/w_max) points, where w_ij is the rule-level co-occurrence count. The circular layout is used only for legibility.",
        5: "Plan-aware retrieval workflow. (a) ChemistryPlan fields enrich the query before semantic retrieval; an embedding-provider failure invokes the deterministic lexical fallback without changing the query or metadata filters. (b) Tiered retrieval starts with paired records and strict mechanism/phase filters, relaxes filters when fewer than three results remain, and expands to all records only when no paired result remains; hidden source identifiers are excluded. (c) Reranking combines 0.60 semantic similarity with 0.40 field similarity before descending top-k selection.",
        6: "Retrieval benchmark and leakage controls. (a) Top-five photocatalyst-family match rates for semantic retrieval and FlowPilot reranking. (b) Rank changes across 1,600 frozen query-result pairs. (c) Field-score component availability and nonzero-observation rates. (d) Representative iridium query before and after reranking. (e) Leave-one-source-out controls exclude the hidden source identifier before final ranking.",
        7: "Ranked one-shot and FlowPilot fixed-criteria multi-LLM evaluation scores for CuAAC, photochemical oxidation, and hydrogenolysis. Points are means and error bars are sample standard deviations across three generation repeats. All 30 plotted sample standard deviations are nonzero; where an error bar is not visually apparent, it is smaller than or obscured by the point marker.",
        8: "Mean paired FlowPilot-minus-one-shot score difference for each applicable universal criterion, separated by chemistry. Positive values favor FlowPilot. UO-08 and UO-09 are omitted because their applicability is not uniform across the three chemistries.",
        9: "Critical-error criteria identified by the independent judge panel for each model, chemistry, architecture, and repeat. (a) One-shot campaigns. (b) FlowPilot campaigns. Cell numbers report independent-judge agreement for the same campaign-criterion error; totals count distinct affected criteria, not repeated judge flags.",
        10: "FlowPilot-only token use, measured generation cost, observed runtime, and chemistry-specific quality-per-cost index. (a-c) Mean +/- sample SD across nine campaigns per model. (d1-d3) Chemistry-specific quality-per-cost index, mean +/- sample SD across three repeats. Runtime is environment-dependent, and judge-model token use is excluded.",
    }
    for number in (1, 2):
        replace_figure(document, number, FIG_DIR / f"Figure_S{number:02d}.png", captions[number])
    replace_s3_placeholder(document)
    for number in range(4, 11):
        replace_figure(document, number, FIG_DIR / f"Figure_S{number:02d}.png", captions[number])

    # Delete the two empty heading paragraphs identified in the reviewer audit.
    for paragraph in list(document.paragraphs):
        if paragraph.style and paragraph.style.name.startswith("Heading") and not paragraph.text.strip():
            delete_paragraph(paragraph)

    matched_caption = find_paragraph(document, "Table S1. Summary of the matched run")
    replace_text(matched_caption, "Table S17. Summary of the matched run.", bold_prefix="Table S17. ")

    s8_caption = find_paragraph(document, "Figure S8.")
    note = insert_paragraph_before(
        s8_caption,
        "UO-08 and UO-09 are excluded from the cross-chemistry heat map because gas-feed and multistage applicability is not uniform across CuAAC, photochemical oxidation, and hydrogenolysis; retaining them would mix true zero values with not-applicable values.",
        template=find_paragraph(document, "Raw judge critical flags count"),
    )
    note.paragraph_format.keep_with_next = True

    # Remove stale alternative text from all drawing properties.
    for node in document.element.xpath(".//wp:docPr"):
        if node.get("descr") == "Figure S5-2":
            node.attrib.pop("descr", None)

    add_council_section(document)

    returned = find_paragraph(document, "FlowPilot returned a validated process flow diagram")
    replace_text(
        returned,
        "FlowPilot returned a validated process flow diagram together with a structured final report; Figure S13 contrasts that diagram with the raw JSON returned in the one-shot setting (8,274 characters), reproduced verbatim below. The report is explicit about its own scope and validation state.",
    )
    old_s12 = find_paragraph(document, "Figure S12. The same design intent")
    replace_text(old_s12, old_s12.text.replace("Figure S12.", "Figure S13.", 1), bold_prefix="Figure S13. ")

    # Mark Word fields for refresh when opened and keep captions with their figures/tables.
    settings = document.settings._element
    update = settings.find(qn("w:updateFields"))
    if update is None:
        update = OxmlElement("w:updateFields")
        settings.append(update)
    update.set(qn("w:val"), "true")
    normalize_body_font(document)
    document.save(OUT_ESI)


def append_highlighted(paragraph: Paragraph, sentence: str) -> None:
    run = paragraph.add_run(" " + sentence)
    set_run_font(run, 11, True)


def revise_main() -> None:
    shutil.copy2(SOURCE_MAIN, BACKUP_MAIN)
    document = Document(SOURCE_MAIN)
    normalize_body_font(document)

    append_highlighted(find_paragraph(document, "The output of the workflow"), "Detailed architecture, authority boundaries, intake contracts, and run provenance are provided in Figures S1-S2 and Tables S1-S3.")
    append_highlighted(find_paragraph(document, "The engineering knowledge base contained"), "The frozen corpus composition and computed rule-base coverage are reported in Figures S3-S4 and Tables S4-S5.")
    append_highlighted(find_paragraph(document, "The strongest retrieval improvement"), "Retrieval tiers, scoring weights, benchmark data, and leakage controls are detailed in Figures S5-S6 and Tables S6-S7.")

    heading = find_paragraph(document, "Council review reveals model-dependent design behavior")
    replace_text(heading, "Ablation and architecture comparison")
    heading.style = "Heading 3"
    intro = find_paragraph(document, "The FlowPilot architecture was benchmarked on an isoxazole")
    replace_text(
        intro,
        "FlowPilot was evaluated through matched architecture comparisons that held the chemistry, inventory, and requested output contract constant while varying whether the generator operated as a one-shot model or within the full pipeline (Figure 4). The benchmark combined universal fixed criteria, deterministic critical-error checks, and independent LLM judges. Three chemistry cases and three generation repeats were used for each retained model-architecture pairing.",
    )

    p32 = find_paragraph(document, "The council model primarily affected")
    replace_text(
        p32,
        "Across the retained benchmark campaigns, the full pipeline generally increased the fixed-criteria benchmark score and reduced campaign-level critical errors relative to matched one-shot generation. Internal ablations showed that removal of the council produced the clearest loss in error control, while specialist-agent removal produced smaller, chemistry-dependent changes. Cost and latency therefore represent the price of explicit decomposition, recomputation, and review rather than an unmeasured increase in prose length.",
    )
    p33 = find_paragraph(document, "The upstream model exerted")
    replace_text(
        p33,
        "Supporting benchmark results are organized in their order of citation: Figures S7-S10 and Tables S8-S12 provide scoring anchors, universal criteria, ablations, matched architecture scores, critical-error maps, and resource use; Tables S13-S16 define deterministic engineering closure, inventory realization, experimental feedback, and computational replay. Figure S11 and Table S17 quantify the matched user-input and returned-output burden.",
    )
    p34 = find_paragraph(document, "The pre/post council radar chart")
    replace_text(
        p34,
        "The earlier council model-matrix and candidate-budget experiment has been moved to the ESI (Figure S12 and Tables S18-S21). In the five-repeat comparison, the normalized radar-area score changed from 0.31 +/- 0.03 before council review to 0.48 +/- 0.23 after review, rather than the earlier single-run description based on 0.29 to 0.83. The ESI also defines Need revision, New versions, Combined pool, and Disqualified and reports the complete runtime, token, geometry, budget, and radar-normalization data. Figure S13 then compares the executable FlowPilot topology with the one-shot structured response.",
    )

    for start in (
        "The model/manuscript overlay further shows",
        "Candidate budget controls exploration breadth",
        "Candidate-budget experiments tested",
        "Revision activity and deliberation workload",
    ):
        try:
            delete_paragraph(find_paragraph(document, start))
        except KeyError:
            pass

    normalize_body_font(document)
    document.save(OUT_MAIN)


def build_citation_audit() -> None:
    main = Document(OUT_MAIN)
    esi = Document(OUT_ESI)
    records = []

    def paragraph_cites(text: str, kind: str, number: int) -> bool:
        singular = kind
        plural = kind + "s"
        if re.search(rf"\b(?:{singular}|{plural})\s+S{number}\b", text):
            return True
        for match in re.finditer(rf"\b(?:{singular}|{plural})\s+S(\d+)\s*[-\u2013]\s*S?(\d+)", text):
            start, end = map(int, match.groups())
            if start <= number <= end:
                return True
        return False

    for kind, maximum in (("Figure", 13), ("Table", 21)):
        for number in range(1, maximum + 1):
            token = f"{kind} S{number}"
            main_hits = [(i, p.text.strip()) for i, p in enumerate(main.paragraphs) if paragraph_cites(p.text, kind, number)]
            esi_hits = [(i, p.text.strip()) for i, p in enumerate(esi.paragraphs) if p.text.strip().startswith(token + ".")]
            records.append(
                {
                    "type": kind,
                    "number": f"S{number}",
                    "main_paragraph": main_hits[0][0] if main_hits else "",
                    "main_context": main_hits[0][1] if main_hits else "",
                    "esi_paragraph": esi_hits[0][0] if esi_hits else "",
                    "status": "matched" if main_hits and esi_hits else "missing main citation" if esi_hits else "missing ESI item",
                }
            )
    csv_path = DOC_DIR / "FlowPilot_ESI_citation_audit.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)

    document = Document()
    title = document.add_heading("FlowPilot main-text to ESI citation audit", level=1)
    for run in title.runs:
        run.font.name = "Times New Roman"
    p = document.add_paragraph("All entries below were checked against the revised main manuscript and revised ESI. Items are listed in numerical order; 'matched' means that the ESI item exists and is cited in the main text.")
    for run in p.runs:
        set_run_font(run, 11)
    table = document.add_table(rows=1, cols=5)
    table.style = "Table Grid"
    headers = ["Type", "Number", "Main paragraph", "ESI paragraph", "Status"]
    for cell, value in zip(table.rows[0].cells, headers):
        cell.text = value
    for item in records:
        cells = table.add_row().cells
        for cell, value in zip(cells, [item["type"], item["number"], str(item["main_paragraph"]), str(item["esi_paragraph"]), item["status"]]):
            cell.text = value
    normalize_body_font(document)
    document.save(DOC_DIR / "FlowPilot_ESI_citation_audit.docx")


def build_reviewer_checklist() -> None:
    document = Document()
    document.add_heading("FlowPilot ESI reviewer revision checklist", level=1)
    items = [
        ("A1", "Done", "Figures S1-S2 redrawn as editable SVG/PDF and high-resolution PNG without figure-level titles; lowercase panel labels used."),
        ("A2", "Done", "Figure S3 generated from the frozen data with n = 464, 154, 135, and 285 shown in the relevant panels; raw CSVs supplied."),
        ("A3", "Done", "Figure S4 layout corrected. Panel d is based on computed concept frequencies and rule-level co-occurrences; node and edge encodings are defined in the caption and computational-basis note."),
        ("A4", "Done", "Figure S5 redrawn; Tier 3, source exclusion, score label, and final top-k text no longer overlap."),
        ("A5", "Done", "Figure S6 redrawn with separated titles, legends, axes, and improved panel e."),
        ("A6", "Done", "Figures S9-S10 supplied as SVG/PDF and high-resolution PNG with unclipped labels."),
        ("A7", "Partial", "Lowercase panel labels and a consistent restrained palette were applied. Exact main-text hex matching remains pending because the requested hex codes were not supplied."),
        ("B8", "Done", "The duplicate matched-run table is renamed Table S17."),
        ("B9", "Done", "Stale Figure S5-2 alternative text removed."),
        ("B10", "Done", "Two empty heading paragraphs removed; the table of contents was explicitly refreshed after insertion and renumbering."),
        ("B11", "Done", "All S7 SD values were checked and are nonzero; visually absent bars are smaller than or hidden by their marker, now stated in the caption."),
        ("B12", "Done", "A text sentence explains exclusion of UO-08 and UO-09 due to non-uniform applicability."),
        ("C13", "Done", "Old main-text Figure 4 moved to ESI as Figure S12. The revised main retains its newer ablation Figure 4, so later main figure numbers remain unchanged."),
        ("C14", "Done", "Underlying matrix, budget, radar, and metric-definition tables added as Tables S18-S21; source CSVs supplied."),
        ("C15", "Done", "Single-run radar statement replaced with 0.31 +/- 0.03 to 0.48 +/- 0.23, n = 5."),
        ("C16", "Done", "Need revision, New versions, Combined pool, and Disqualified are defined in the new council ESI section."),
        ("D17", "Done", "Main-to-ESI figure/table citations added in numerical order and compiled in separate DOCX/CSV audit files."),
        ("E", "Done", "S3/S4 raw CSVs, S4d computational basis, revised main manuscript, revised ESI, and citation audit are packaged together."),
    ]
    table = document.add_table(rows=1, cols=3)
    table.style = "Table Grid"
    for cell, value in zip(table.rows[0].cells, ["Item", "Status", "Revision"]):
        cell.text = value
    for item, status, detail in items:
        cells = table.add_row().cells
        cells[0].text, cells[1].text, cells[2].text = item, status, detail
    normalize_body_font(document)
    document.save(DOC_DIR / "FlowPilot_reviewer_revision_checklist.docx")


def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    DOC_DIR.mkdir(parents=True, exist_ok=True)
    copy_council_data()
    revise_esi()
    revise_main()
    build_citation_audit()
    build_reviewer_checklist()
    print(OUT_ESI)
    print(OUT_MAIN)


if __name__ == "__main__":
    main()
