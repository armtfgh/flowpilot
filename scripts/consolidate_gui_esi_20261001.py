"""Consolidate actual GUI captures while preserving unrelated Word content."""
from copy import deepcopy
from hashlib import sha256
from pathlib import Path
import csv
import json
import re

from lxml import etree as E
from PIL import Image, ImageDraw, ImageFont
from matplotlib.font_manager import FontProperties, findfont

import overhaul_submission_20260930 as prior
from revise_layout_discussion_20260928 import order_properties

ROOT, BASE, W, NS = prior.ROOT, prior.BASE, prior.W, prior.NS
Package, text = prior.Package, prior.text
OUT = BASE / "inventory_gui_revision_20261001"
FIG = OUT / "figures"
SHOTS = OUT / "screenshots"
SOURCES = {stem: BASE / f"{stem}_submission_overhauled_20260930.docx" for stem in ["esi", "manuscript"]}
FIG_MAP = {**{n: n for n in range(1, 13)}, 13: 12, 14: 13, 15: 14, 16: 14, **{n: n - 2 for n in range(17, 24)}}
log = []


def panel_stack(filename, specs, width=1900):
    margin, gap, header = 22, 22, 76
    panels = []
    for label, path, crop in specs:
        image = Image.open(path).convert("RGB")
        if crop:
            image = image.crop(crop)
        h = round(image.height * (width - 2 * margin) / image.width)
        image = image.resize((width - 2 * margin, h), Image.Resampling.LANCZOS)
        panels.append((label, image))
    canvas = Image.new("RGB", (width, margin * 2 + sum(header + p.height + gap for _, p in panels) - gap), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.truetype(findfont(FontProperties(family="DejaVu Sans", weight="bold")), 41)
    y = margin
    for label, image in panels:
        draw.text((margin, y + 5), label, fill="#243934", font=font)
        y += header
        canvas.paste(image, (margin, y))
        draw.rectangle((margin, y, width - margin - 1, y + image.height - 1), outline="#b9c9c5", width=2)
        y += image.height + gap
    path = FIG / filename
    canvas.save(path, dpi=(300, 300), optimize=True)
    return path


def figures():
    FIG.mkdir(exist_ok=True)
    # These are crops of real captures, never regenerated UI text or data.
    ready = Image.open(SHOTS / "03_ready.png")
    ready_crop = (0, 75, ready.width, ready.height - 35)
    f12 = panel_stack("figureS12_intake_workflow.png", [
        ("a  Batch protocol", SHOTS / "01_protocol.png", None),
        ("b  Standardized follow-up and answer", SHOTS / "02_questions.png", None),
        ("c  Answered package ready for design", SHOTS / "03_ready.png", ready_crop),
    ])
    f13 = panel_stack("figureS13_inventory_forms.png", [
        ("a  Categorized equipment records", SHOTS / "04_inventory_categories.png", None),
        ("b  Pump specifications in prepared fields", SHOTS / "05_equipment_fields.png", None),
        ("c  Equipment availability and constraints", SHOTS / "06_constraints.png", None),
        ("d  Validate, export, save and bind", SHOTS / "07_validation.png", None),
    ])
    # The topology screenshot includes a unit-operation list below the canvas.
    # Crop at the end of the canvas; the full unchanged screenshot is archived.
    topology = Image.open(SHOTS / "11_topology.png")
    f14 = panel_stack("figureS14_results_and_council.png", [
        ("a  Archived process topology", SHOTS / "11_topology.png", (0, 0, topology.width, 1335)),
        ("b  Final stage parameters", SHOTS / "10_stage_table.png", None),
        ("c  Expanded council assessment (excerpt)", SHOTS / "12_council.png", None),
    ])
    return f12, f13, f14


def remap(value):
    def sub(m):
        if m[1].startswith("Table"):
            return m[0]
        numbers = []
        for bit in re.split(r"\s*(?:,\s*(?:and\s+)?|and)\s*", m[2]):
            ns = [int(n) for n in re.findall(r"\d+", bit)]
            numbers.extend(range(ns[0], ns[1] + 1) if len(ns) == 2 else ns)
        mapped = list(dict.fromkeys(FIG_MAP[n] for n in numbers))
        suffix = re.search(r"S\d+([a-z])$", m[2])
        return ("Figures " if len(mapped) > 1 else "Figure ") + prior.compact(mapped) + (suffix[1] if suffix and len(mapped) == 1 else "")
    return prior.LABEL.sub(sub, value)


def caption(pkg, value):
    p = pkg.paragraph(value)
    ppr = p.find(W + "pPr")
    E.SubElement(ppr, W + "keepLines")
    E.SubElement(ppr, W + "keepNext", {W + "val": "0"})
    prefix = re.match(r"(?:Figure|Table) S\d+\.", value).group()
    for r in list(p.findall(W + "r")):
        p.remove(r)
    for chunk, bold in [(prefix, True), (value[len(prefix):], False)]:
        r = E.SubElement(p, W + "r")
        rp = E.SubElement(r, W + "rPr")
        pkg.font(rp)
        E.SubElement(rp, W + "b", {W + "val": "1" if bold else "0"})
        E.SubElement(rp, W + "bCs", {W + "val": "1" if bold else "0"})
        E.SubElement(r, W + "t", {"{http://www.w3.org/XML/1998/namespace}space": "preserve"}).text = chunk
    return p


def figure(pkg, path, width):
    node = pkg.image(path, width=width, page_break=True)
    pp = node.find(W + "pPr")
    E.SubElement(pp, W + "jc", {W + "val": "center"})
    return node


def replace_para(pkg, prefix, new):
    p = next(p for p in pkg.body if text(p).startswith(prefix))
    log.append({"document": pkg.source.name, "before": text(p), "after": new})
    pkg.revise(p, new)


def revise_esi(pkg, paths):
    nodes = list(pkg.body)
    start = next(i for i, p in enumerate(nodes) if text(p) == "GUI operation, inventory management, and auditable examples")
    end = next(i for i, p in enumerate(nodes) if text(p) == "Connected-process case studies supporting main-text Figures 5 and 6")
    old_section = nodes[start:end]
    (OUT / "previous_gui_section.txt").write_text("\n\n".join(text(p) for p in old_section), encoding="utf-8")

    def existing(prefix):
        return deepcopy(next(p for p in old_section if text(p).startswith(prefix)))

    def table(number):
        index = next(i for i, p in enumerate(old_section) if text(p).startswith(f"Table S{number}."))
        content = next(p for p in old_section[index + 1:] if p.tag == W + "tbl")
        return [deepcopy(old_section[index]), deepcopy(content)]

    p = pkg.paragraph
    f12, f13, f14 = paths
    new = [deepcopy(old_section[0]),
        p("Actual browser captures made on 1 October 2026 document protocol entry, fixed-ID questions, form-based inventory editing and JSON export. Result and council views reopen the archived DPDTC run 20260907_175308_three_protocol_scientific. These interface demonstrations do not constitute new benchmark outcomes or chemical experiments. Full-screen captures, structured input/output records and the browser trace are retained in the companion archive."),
        existing("The GUI example is the two-stage"),
        existing("Batch entry, follow-up questions"),
        p("Figure S12 combines the input, follow-up and completion states of one intake demonstration. The chemist enters the batch protocol, binds a laboratory profile and answers questions selected from the versioned question bank. LLM-assisted extraction was switched off for the capture, so the question-selection check did not require a provider call. This demonstrates repeatability of this input mode, not reproducibility of model-generated chemistry or design performance."),
        existing("Three fresh requests with the same protocol"),
        figure(pkg, f12, 4.50),
        caption(pkg, "Figure S12. Reproducible intake in the live FlowPilot webapp. (a) Batch-protocol entry; the textarea contains the full protocol although only its opening is visible. (b) A fixed-ID objective question and the entered response, with question-bank version and hash. (c) Completion after saving the required answers. The panels are cropped views from different moments in one session; complete questions and answers are retained in the accompanying JSON. The frozen state denotes submission-ready input, not chemical validation."),
        existing("Inventory input, normalization"),
        p("The inventory workspace supports direct entry in categorized forms as well as import from documents or an existing JSON profile (Figure S13). Its 15 categories cover pumps, reactors, tubing, light sources, gas hardware, mixers, pressure controllers, temperature controllers, degassers, filters, separators, connectors, collectors, reactor trains and safety accessories. Prepared fields are derived from the same typed equipment models used by the pipeline. Equipment can be added, edited or deleted without rebuilding the remaining inventory."),
        p("Forms retain equipment identifiers, quantities, system membership and compatibility, together with operating limits and notes. For pumps, the minimum supported flow and the setting increment are separate fields. Unknown optional ratings remain unspecified rather than becoming zero or false. The Constraints section records unavailable functions, permitted alternatives and additional operating limits. The KHU version-4 profile is used here to match the archived example, not to redefine the later laboratory configuration. Table S18 retains the corresponding source values."),
        p("Any committed edit invalidates the previous validation state. The current draft must pass profile validation before Export JSON, Save profile or Use in design is enabled. Export produces the complete InventoryProfile, including equipment, constraints and source provenance; importing this file restores the same structured data. Save profile writes a new version, while Use in design binds the profile and its interpreted operating limits to the intake package. A valid profile can still contain warnings and does not certify hardware compatibility or laboratory safety; design-specific checks are performed separately."),
        p("In the browser check, opening and applying the prepared pump form, validating, exporting and re-importing the KHU profile retained all seven reactor and three pump records, compatibility fields, operating constraints and source provenance. The saved laboratory profile was not overwritten. Dedicated JSON import bypasses LLM extraction. Separate desktop and mobile tests also created a new profile using the equipment forms and confirmed that invalid or unvalidated edits could not be exported or used for design."),
        *table(18),
        figure(pkg, f13, 5.85),
        caption(pkg, "Figure S13. Form-based inventory management in the live webapp. (a) Equipment-category navigation and a pump record. (b) Prepared pump fields distinguish flow limits from the setting increment; a blank increment remains unknown. (c) Declared unavailability of an inline degasser, permitted alternatives and other constraints. (d) Profile validation and the JSON export, versioned-save and design-binding controls. Cropped excerpts use the archived KHU version-4 profile. Warnings remain accessible in the collapsed validation details; schema validity is not laboratory approval. Full views and the exported JSON accompany the figure."),
        existing("Stage-resolved output and run provenance"),
        p("Figure S14 combines the archived topology, final stage table and an expanded council record. Table S19 gives the numerical values at manuscript text size. The second liquid feed enters between the reactors, so Stage 2 flow is the sum of the upstream liquid and the amine feed. Both stages are liquid-only here. For gas-containing designs, an inlet/STP time index is a reporting convention and must not be presented as measured in-channel contact time. The result is reopened by its autosaved run identifier; no new design is generated for these screenshots."),
        *table(19),
        p("The approximately 30.19 min combined nominal time is a proposed screen, not a measured conversion time. The source batch used 95 C; the archived profile restricts the selected PFA coils to 80 C. The result reports LOW confidence and requires verification of the generic interstage mixer. Neither numerical closure nor council agreement establishes that the lower temperature and shorter holds preserve yield. Generic pump pictograms identify delivery operations; the equipment names specify the peristaltic and HPLC pumps. This historical example is retained as recorded and is distinct from the later KHU experimental implementation in Section 9."),
        figure(pkg, f14, 6.15),
        caption(pkg, "Figure S14. Linked views of one archived screening result. (a) The stored icon-based process topology displayed in the GUI. (b) Stage-resolved parameters from the final result; readable numerical values are reproduced in Table S19. (c) An expanded DrChemistry review excerpt, retaining its WARNING status and model-authored uncertainty statements. The screenshot panels and machine-readable records refer to the same run, not separate experiments. Complete feed tables, full council records and uncropped browser views remain in the companion archive. Selected discussion excerpts are transcribed below."),
        existing("Examples of critical findings"), existing("Table S20 gives traceable"), *table(20),
        existing("Unit audit for N2-E85D424E"),
        existing("Recorded council discussion"),
        p("Figure S14c shows an actual expanded Council record for the same DPDTC run. Four domain reviewers assess the candidates, followed by the Skeptic and Chief cross-domain assessment and selection. Table S21 reproduces exact saved excerpts showing disagreement about candidate 6 and the eventual selection of candidate 1. These are explicit model-authored review outputs, not hidden reasoning or reconstructed dialogue. The two display rounds group the recorded contributions and must not be interpreted as two complete optimization cycles."),
        *table(21), existing("The discussion exposes uncertainty"),
    ]
    # Renumber references outside the new section once, including later photographs/spectra.
    for node in list(pkg.body):
        if node in old_section or node.tag != W + "p":
            continue
        value = prior.with_cites(node)
        updated = remap(value)
        if value != updated:
            pkg.revise(node, updated)
    anchor = old_section[0]
    for node in new:
        anchor.addprevious(node)
    for node in old_section:
        pkg.body.remove(node)


def main():
    OUT.mkdir(exist_ok=True)
    manifest = {str(p.relative_to(ROOT)): sha256(p.read_bytes()).hexdigest() for p in SOURCES.values()}
    paths = figures()
    esi, main_doc = Package(SOURCES["esi"]), Package(SOURCES["manuscript"])
    revise_esi(esi, paths)
    for p in list(main_doc.body):
        if p.tag != W + "p":
            continue
        value = prior.with_cites(p)
        if remap(value) != value:
            main_doc.revise(p, remap(value))
    replace_para(main_doc, "The interface links these assessments", "The interface links these assessments to the chemist's workflow. Figure S12 combines protocol entry, fixed-ID follow-up questions and the answered input package. Figure S13 and Table S18 document categorized inventory forms, constraint entry and JSON export; Figure S14 and Table S19 show linked stage-resolved output and council review. Table S20 retains concrete one-shot and FlowPilot problems, including an evaluator false positive, and Table S21 documents council disagreement and selection. These records make assumptions and unresolved requirements inspectable before laboratory implementation.")
    replace_para(main_doc, "Inventory profiles were imported as structured data", "Inventory profiles were entered through categorized equipment forms, imported as structured JSON, or normalized from laboratory documents into typed records with stable identifiers. Available quantities, system membership, material, reactor volume and diameter, pump range and setting increment, pressure and temperature limits, and compatible light modules were recorded where supplied. The form fields share the pipeline's equipment schema, and edited profiles require validation before export or design use. Realization assigned required operations to available items and checked the declared capability and connectivity constraints. Calculations were repeated when equipment selection changed a dependent quantity. An unresolved essential requirement was retained as a diagnostic requirement, not silently replaced with invented hardware (Figure S13; Table S18).")
    for stem, pkg in [("esi", esi), ("manuscript", main_doc)]:
        if stem == "manuscript":
            prior.old.base.prior.bold_references(pkg.doc)
        else:
            for node in pkg.body:
                if node.tag != W + "sdt" and not re.match(r"^(Figure|Table) S\d+\.", text(node)):
                    prior.old.base.prior.bold_references(node)
        if stem == "esi":
            # Restore caption bodies to normal weight after bolding in-text references.
            for node in list(pkg.body):
                if re.match(r"^(Figure|Table) S\d+\.", text(node)) and node in pkg.changed:
                    new = caption(pkg, text(node))
                    props = deepcopy(node.find(W + "pPr"))
                    if props is not None:
                        new.replace(new.find(W + "pPr"), props)
                    pkg.body.replace(node, new)
        order_properties(pkg.doc)
        for suffix, marked in [("", False), ("_marked", True)]:
            pkg.save(BASE / f"{stem}_submission_inventory_gui_20261001{suffix}.docx", marked=marked)
    (OUT / "source_manifest.json").write_text(json.dumps(manifest, indent=2))
    (OUT / "text_changes.json").write_text(json.dumps(log, indent=2))
    with (OUT / "figure_numbering_map.csv").open("w", newline="") as f:
        writer = csv.writer(f); writer.writerow(["previous_figure", "current_figure"])
        writer.writerows((f"S{a}", f"S{b}") for a, b in FIG_MAP.items())
    assert all(sha256((ROOT / k).read_bytes()).hexdigest() == v for k, v in manifest.items())
    print("Created clean and marked ESI/manuscript, with three consolidated GUI figures.")


if __name__ == "__main__":
    main()
