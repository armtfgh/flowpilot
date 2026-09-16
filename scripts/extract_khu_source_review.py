"""Export KHU sources for human review; do not create an executable inventory."""

import hashlib
import json
from pathlib import Path

from openpyxl import load_workbook
from pptx import Presentation


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "inventory_khu"
OUT = SOURCE / "source_review_20260914"
PPT = SOURCE / "Figure 5, 6 batch protocol_results (KHU revised V2)_final.pptx"
XLSX = SOURCE / "Inventory_final.xlsx"
SELECTED = {1: "Figure 5 batch protocol", 5: "Figure 5 revised set 1",
            8: "Figure 5 revised set 2", 11: "Figure 5 revised set 3",
            12: "Figure 6 batch protocol", 16: "Figure 6 revised set 1",
            19: "Figure 6 revised set 2", 22: "Figure 6 revised set 3"}


def shape_text(shapes):
    result = []
    for shape in shapes:
        if hasattr(shape, "shapes"):
            result.extend(shape_text(shape.shapes))
        elif shape.has_text_frame:
            if shape.text.strip():
                result.append(shape.text.replace("\v", "\n"))
        elif shape.has_table:
            result.extend(" | ".join(c.text for c in row.cells)
                          for row in shape.table.rows)
        elif hasattr(shape, "image"):
            result.append("[Embedded image: not transcribed; inspect original slide.]")
    return result


def main():
    OUT.mkdir(exist_ok=True)
    workbook = load_workbook(XLSX, data_only=True)
    sheets = []
    lines = ["# KHU workbook: source-cell review", "",
             "This is a source extraction, NOT a validated FlowPilot inventory JSON.",
             "Blank cells remain unspecified. Merged values are resolved to their anchor",
             "and retain the anchor address; repeated merged stock values are not new stock.",
             "Korean notes and ambiguous limits are preserved verbatim.", ""]
    for sheet in workbook:
        records = []
        lines.extend([f"## {sheet.title}", ""])
        for row in sheet:
            cells = []
            for cell in row:
                anchor = cell.coordinate
                for merged in sheet.merged_cells.ranges:
                    if cell.coordinate in merged:
                        anchor = sheet.cell(merged.min_row, merged.min_col).coordinate
                        break
                value = sheet[anchor].value
                if value is not None:
                    cells.append({"cell": cell.coordinate, "source_cell": anchor,
                                  "value": value})
            if cells:
                records.append({"row": row[0].row, "cells": cells})
                lines.append(f"### Row {row[0].row}")
                for entry in cells:
                    ref = entry['cell']
                    if ref != entry['source_cell']:
                        ref += f" (merged from {entry['source_cell']})"
                    text = str(entry['value']).replace("\n", " / ")
                    lines.append(f"- **{ref}:** {text}")
                lines.append("")
        sheets.append({"sheet": sheet.title, "rows": records})
    (OUT / "inventory_source_cells.md").write_text("\n".join(lines), encoding="utf-8")
    slides = []
    for index, slide in enumerate(Presentation(PPT).slides, 1):
        slides.append({"slide": index, "selected_as": SELECTED.get(index),
                       "text": shape_text(slide.shapes)})
    for filename, selected_only in [("selected_protocols_and_responses.md", True),
                                    ("all_slides_reference.md", False)]:
        lines = ["# KHU protocols and responses", "",
                 "Exact extracted text, not rewritten by an LLM. Images are flagged, not OCR-transcribed.",
                 "Only explicitly revised sets are selected; older designs are comparison material,",
                 "not wet-lab evidence or new constraints. No design run has been performed.", ""]
        for slide in slides:
            if selected_only and not slide['selected_as']:
                continue
            lines.extend([f"## Slide {slide['slide']}: {slide['selected_as'] or 'Reference only'}",
                          "", *slide['text'], ""])
        (OUT / filename).write_text("\n\n".join(lines), encoding="utf-8")
    sources = [{"filename": p.name, "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
               for p in [XLSX, PPT, SOURCE / "Vapourtec Peristaltic pump reagent list.pdf"]]
    audit = {"purpose": "source review, not executable inventory", "sources": sources,
             "sheets": sheets, "slides": slides}
    (OUT / "source_extraction.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False),
                                               encoding="utf-8")
    assert len(sheets) == 9 and len(slides) == 22
    assert sum(s['selected_as'] is not None for s in slides) == 8
    print(f"Exported {len(sheets)} sheets, {len(slides)} slides; 2 protocols + 6 revised sets.")
    print(OUT)


if __name__ == "__main__":
    main()
