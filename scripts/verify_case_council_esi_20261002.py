#!/usr/bin/env python3
"""Render and verify the council documentation and its unchanged source assets."""
from collections import Counter
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
import json
import re
import subprocess

import fitz
from docx import Document
from lxml import etree as E
from PIL import Image, ImageDraw

from add_case_council_esi_20261002 import ROOT, BASE, OUT, SOURCE, TARGET, APPENDIX, FIGTEXT, digest, dump
from revise_manuscript_cases_20260922 import Package, W, NS, text
from refresh_revision_toc_20260922 import refresh


def render(paths):
    subprocess.run([
        "libreoffice", "-env:UserInstallation=file:///tmp/flowpilot-council-final-20261002",
        "--headless", "--convert-to", "pdf", "--outdir", str(OUT / "rendered"),
        *map(str, paths),
    ], check=True, timeout=240)


def main():
    render([TARGET, APPENDIX, FIGTEXT])
    previous = None
    toc_audits = []
    for _ in range(4):
        with fitz.open(OUT / "rendered" / TARGET.with_suffix(".pdf").name) as pdf:
            toc = refresh(TARGET, pdf)
        toc_audits.append(toc)
        render([TARGET])
        if toc == previous:
            break
        previous = toc
    else:
        raise RuntimeError("Contents page numbers did not stabilize")
    dump(OUT / "contents_verification.json", toc_audits)
    assert next(r for r in toc if r["heading"].startswith("Recorded council deliberation"))["section"] == "7.3"
    old, new = Package(SOURCE), Package(TARGET)
    before = [text(p) for p in old.body if p.tag != W + "sdt"]
    after = [text(p) for p in new.body if p.tag != W + "sdt"]
    start = after.index("Recorded council deliberation for the two connected-process cases")
    end = after.index("Laboratory implementation, experimental procedures, and analytical data")
    checks = {
        "original_body_text_preserved": before == after[:start] + after[end:],
        "all_original_tables_preserved": [E.tostring(t) for t in old.body.findall(W + "tbl")] == [E.tostring(t) for t in new.body.findall(W + "tbl")],
        "images_preserved": {k: v for k, v in old.parts.items() if k.startswith("word/media/")} == {k: v for k, v in new.parts.items() if k.startswith("word/media/")},
        "image_placement_preserved": [E.tostring(t) for t in old.doc.iter(W + "drawing")] == [E.tostring(t) for t in new.doc.iter(W + "drawing")],
        "style_and_numbering_parts_preserved": all(old.parts[k] == new.parts[k] for k in ["word/styles.xml", "word/numbering.xml", "word/fontTable.xml"]),
        "page_setup_preserved": E.tostring(old.body[-1]) == E.tostring(new.body[-1]),
        "section_7_3_added_and_contents_updated": True,
    }
    captions = lambda p: [text(n) for n in p.body if re.match(r"^(Figure|Table) S\d+\.", text(n))]
    checks["all_captions_and_numbers_preserved"] = captions(old) == captions(new)
    for doc in [TARGET, APPENDIX, FIGTEXT]:
        with ZipFile(doc) as z:
            for name in z.namelist():
                if name.endswith((".xml", ".rels")):
                    E.fromstring(z.read(name))
    checks["all_document_XML_valid"] = True
    manifest = json.loads((OUT / "source_manifest.json").read_text())
    checks["all_original_sources_unchanged"] = all(digest(ROOT / p) == h for p, h in manifest["source_sha256"].items())
    archived_calls = 0
    for run in manifest["runs"]:
        raw = OUT / "raw" / run["id"]
        data = json.loads((raw / "result.json").read_text())
        calls = data["scientific_assessment"]["calls"]
        llm = [json.loads(line) for line in (raw / "llm_calls.jsonl").read_text().splitlines() if line.strip()]
        observed = Counter(c["response_text"].strip() for c in llm)
        for i, call in enumerate(calls, 1):
            response = call["raw_response"]
            assert observed[response.strip()] > 0, (run["id"], i, "missing model-response provenance")
            observed[response.strip()] -= 1
            fn = raw / f"call_{i:02d}_{call['role']}_attempt{call['attempt']}.txt"
            assert fn.read_text(encoding="utf-8") == response
            archived_calls += 1
    checks["46_council_calls_match_original_LLM_response_logs"] = archived_calls == 46
    bad_fonts = []
    for doc in [APPENDIX, FIGTEXT]:
        for p in Document(doc).paragraphs:
            for run in p.runs:
                if run.text.strip() and (run.font.name != "Times New Roman" or run.font.size.pt != 11):
                    bad_fonts.append(run.text[:60])
    checks["companion_documents_Times_New_Roman_11"] = not bad_fonts
    visual = OUT / "visual_check"
    visual.mkdir(exist_ok=True)
    bounds = []
    pages_meta = []
    for path in [TARGET, APPENDIX, FIGTEXT]:
        with fitz.open(OUT / "rendered" / path.with_suffix(".pdf").name) as pdf:
            subset = []
            for i, page in enumerate(pdf):
                t = page.get_text()
                if path == FIGTEXT or (path == TARGET and any(s in t for s in [
                    "Recorded council deliberation", "Interpretation notes for the unedited",
                    "(F5-S1): selected", "(F5-S2): selected", "(F5-S3): selected",
                    "(F6-S1): selected", "(F6-S2): selected", "(F6-S3): selected",
                    "What these records demonstrate",
                ])) or (path == APPENDIX and (i == 0 or i == len(pdf) - 1 or ("Call 1:" in t and "Status:" in t))):
                    subset.append(i)
                for b in page.get_text("dict")["blocks"]:
                    for line in b.get("lines", []):
                        for span in line["spans"]:
                            if span["text"].strip():
                                x0, y0, x1, y1 = span["bbox"]
                                if x0 < -1 or y0 < -1 or x1 > page.rect.width + 1 or y1 > page.rect.height + 1:
                                    bounds.append({"file": path.name, "page": i + 1, "text": span["text"]})
            thumbs = []
            for i in subset:
                png = visual / f"{path.stem}_page_{i+1:03d}.png"
                pdf[i].get_pixmap(matrix=fitz.Matrix(1.35, 1.35)).save(png)
                with Image.open(png) as im:
                    image = im.copy(); image.thumbnail((330, 460))
                    tile = Image.new("RGB", (350, 490), "#dddddd")
                    tile.paste(image, ((350-image.width)//2, 20))
                    ImageDraw.Draw(tile).text((10, 468), f"Page {i+1}", fill="black")
                    thumbs.append(tile)
            if thumbs:
                sheet = Image.new("RGB", (350*3, 490*((len(thumbs)+2)//3)), "white")
                for j, tile in enumerate(thumbs):
                    sheet.paste(tile, ((j%3)*350, (j//3)*490))
                sheet.save(visual / (path.stem + "_contact.png"))
            pages_meta.append({"file": path.name, "pages": len(pdf), "review_pages": [i+1 for i in subset]})
    checks["no_off_page_text_in_any_rendered_document"] = not bounds
    dump(OUT / "render_verification.json", {"checks": checks, "pages": pages_meta, "outside_page": bounds})
    assert all(checks.values()), checks
    bundle = BASE / "FlowPilot_Figures5_6_council_package_20261002.zip"
    with ZipFile(bundle, "w", ZIP_DEFLATED) as z:
        for p in [TARGET, FIGTEXT, APPENDIX]:
            z.write(p, p.name)
        for p in sorted((OUT / "raw").rglob("*")):
            if p.is_file():
                z.write(p, p.relative_to(OUT))
        for p in OUT.iterdir():
            if p.is_file() and p.suffix in {".json", ".csv", ".md"}:
                z.write(p, p.name)
        for p in sorted((OUT / "rendered").glob("*.pdf")):
            z.write(p, "pdf/" + p.name)
        for name in ["add_case_council_esi_20261002.py", "verify_case_council_esi_20261002.py"]:
            z.write(ROOT / "scripts" / name, "source/" + name)
    print(json.dumps({"checks": checks, "pages": pages_meta, "bundle": str(bundle)}, indent=2))


if __name__ == "__main__":
    main()
