"""Independently inspect source preservation, citations and rendered Word layout."""
import io
import json
import re
from hashlib import sha256
from pathlib import Path
from zipfile import ZipFile

import fitz
from lxml import etree as E
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "manuscript"
OUT = BASE / "introduction_revision_20260923"
NS = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main", "a": "http://schemas.openxmlformats.org/drawingml/2006/main", "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships"}
W = "{" + NS["w"] + "}"
checks = []


def check(name, passed, detail):
    checks.append({"check": name, "passed": bool(passed), "detail": detail})


def read_doc(path):
    with ZipFile(path) as z:
        parts = {n: z.read(n) for n in z.namelist()}
    doc = E.fromstring(parts["word/document.xml"])
    return parts, doc


def text(node):
    return "".join(node.xpath(".//w:t/text()", namespaces=NS))


def section(doc, first, last=None):
    nodes = list(doc.find(W + "body"))
    start = next(i for i, n in enumerate(nodes) if text(n) == first)
    end = next(i for i, n in enumerate(nodes) if text(n) == last) if last else len(nodes)
    return nodes[start + 1:end]


def without_citations(node):
    strings = []
    for run in node.iter(W + "r"):
        value = text(run)
        superscript = run.xpath("./w:rPr/w:vertAlign[@w:val='superscript']", namespaces=NS)
        if superscript and re.fullmatch(r"[\d,\s\-\u2013]+", value):
            continue
        strings.append(value)
    return "".join(strings)


def no_highlight_structure(node):
    return (node.tag, sorted(node.attrib.items()), node.text, node.tail, [no_highlight_structure(c) for c in node if c.tag != W + "highlight"])


def main():
    src_parts, src = read_doc(BASE / "manuscript_revised_20260922.docx")
    dst_parts, dst = read_doc(BASE / "manuscript_introduction_revised_20260923.docx")
    marked_parts, marked = read_doc(BASE / "manuscript_introduction_revised_20260923_marked.docx")
    check("All media and non-document package members preserved", set(src_parts) == set(dst_parts) and all(src_parts[k] == dst_parts[k] for k in src_parts if k != "word/document.xml"), "Includes styles, numbering, relationships, headers, footers and embedded figures")
    check("Image relationship order preserved", src.xpath(".//a:blip/@r:embed", namespaces=NS) == dst.xpath(".//a:blip/@r:embed", namespaces=NS), "All existing main-text figures retain their exact image assignments")
    check("Results through Methods prose preserved", [without_citations(p) for p in section(src, "RESULTS", "REFERENCES")] == [without_citations(p) for p in section(dst, "RESULTS", "REFERENCES")], "Only numerical literature citations change outside the introduction")
    check("Clean and marked text/layout match", no_highlight_structure(dst) == no_highlight_structure(marked), "Yellow highlighting is the sole difference")
    original_intro = section(src, "INTRODUCTION", "RESULTS")
    intro = section(dst, "INTRODUCTION", "RESULTS")
    marked_intro = section(marked, "INTRODUCTION", "RESULTS")
    check("Introduction fully replaced", len(intro) == 9 and all(text(p) not in {text(q) for q in original_intro} for p in intro), "Nine newly written paragraphs")
    for i, p in enumerate(intro, 1):
        runs = list(p.iter(W + "r"))
        check(f"Introduction paragraph {i} typography", all(r.xpath("./w:rPr/w:rFonts[@w:ascii='Times New Roman']", namespaces=NS) and r.xpath("./w:rPr/w:sz[@w:val='22']", namespaces=NS) for r in runs), "Times New Roman 11 pt; superscript literature citations")
        check(f"Introduction paragraph {i} yellow marks", all(r.xpath("./w:rPr/w:highlight[@w:val='yellow']", namespaces=NS) for r in marked_intro[i-1].iter(W + "r") if text(r).strip()), "Every revised introduction run is highlighted in marked copy")
    references = [p for p in section(dst, "REFERENCES") if p.tag == W + "p" and text(p).strip()]
    ids = []
    before_refs = list(dst.find(W + "body"))
    before_refs = before_refs[:next(i for i, p in enumerate(before_refs) if text(p) == "REFERENCES")]
    for p in before_refs:
        for r in p.xpath(".//w:r[w:rPr/w:vertAlign[@w:val='superscript']]", namespaces=NS):
            s = text(r)
            if not re.fullmatch(r"\d+(?:\s*[-\u2013,]\s*\d+)*", s):
                continue
            for bit in s.split(","):
                a, *b = [int(v) for v in re.split(r"[-\u2013]", bit)]
                ids.extend(range(a, (b[0] if b else a) + 1))
    unique = list(dict.fromkeys(ids))
    check("References complete and first-citation ordered", unique == list(range(1, len(references) + 1)), f"{len(references)} bibliography entries; no uncited or dangling references")
    check("No generated double periods in new reference authors", not any(".. " in text(p) for p in references), "Author initials and following sentence punctuation checked")
    check("Version-specific preprints explicitly labeled", all(any(term in text(p) and "Preprint" in text(p) for p in references) for term in ["2602.17902v2", "2609.04564v1"]), "Preprints are not presented as journal publications")
    check("No draft citation delimiters remain", "[[" not in text(dst) and "]]" not in text(dst), "All draft citation tokens are superscript Word runs")
    for kind in ["Figure", "Table", "Section"]:
        pattern = rf"{kind}s? S?\d+(?:[a-z0-9.,\-\u2013 ]*)"
        old = re.findall(pattern, " ".join(text(p) for p in section(src, "RESULTS", "REFERENCES")))
        new = re.findall(pattern, " ".join(text(p) for p in section(dst, "RESULTS", "REFERENCES")))
        check(kind + " cross-references unchanged", old == new, f"{len(old)} literal cross-reference strings preserved")
    report = json.loads((OUT / "verification.json").read_text())
    for path, expected in report["source_hashes_unchanged"].items():
        check("Original file preserved: " + path, sha256((ROOT / path).read_bytes()).hexdigest() == expected, "Source manuscript and matching ESI remain untouched")
    snapshots = []
    for suffix in ["", "_marked"]:
        pdf_path = OUT / f"rendered/manuscript_introduction_revised_20260923{suffix}.pdf"
        pdf = fitz.open(pdf_path)
        outside = []
        empty = []
        for i, page in enumerate(pdf):
            if not page.get_text().strip() and not page.get_images():
                empty.append(i + 1)
            for b in page.get_text("dict")["blocks"]:
                for line in b.get("lines", []):
                    for span in line["spans"]:
                        rect = fitz.Rect(span["bbox"])
                        if rect.x0 < -1 or rect.y0 < -1 or rect.x1 > page.rect.width + 1 or rect.y1 > page.rect.height + 1:
                            outside.append([i + 1, span["text"]])
        check("Rendered text within page: " + (suffix or "clean"), not outside, outside or f"{len(pdf)} pages; no clipped text")
        check("No blank rendered pages: " + (suffix or "clean"), not empty, empty or "None")
        if not suffix:
            for i in [0, 1, 2, 3, len(pdf)-3, len(pdf)-2, len(pdf)-1]:
                filename = OUT / f"rendered/page_{i+1:02d}.png"
                pdf[i].get_pixmap(matrix=fitz.Matrix(1.6, 1.6)).save(filename)
                snapshots.append(filename)
        else:
            pdf[2].get_pixmap(matrix=fitz.Matrix(1.6, 1.6)).save(OUT / "rendered/marked_introduction_page_03.png")
    thumbs = []
    for path in snapshots:
        im = Image.open(path).convert("RGB")
        im.thumbnail((510, 690))
        canvas = Image.new("RGB", (530, 730), "#dddddd")
        canvas.paste(im, ((530-im.width)//2, 24))
        ImageDraw.Draw(canvas).text((12, 710), path.name, fill="black")
        thumbs.append(canvas)
    for tag, items in [("introduction_contact_sheet", thumbs[:4]), ("bibliography_contact_sheet", thumbs[4:])]:
        canvas = Image.new("RGB", (530 * len(items), 730), "white")
        for i, im in enumerate(items):
            canvas.paste(im, (530*i, 0))
        canvas.save(OUT / f"rendered/{tag}.png")
    output = {"passed": sum(c["passed"] for c in checks), "total": len(checks), "checks": checks}
    (OUT / "independent_verification.json").write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"passed": output["passed"], "total": output["total"], "failed": [c for c in checks if not c["passed"]]}, indent=2))
    assert all(c["passed"] for c in checks)


if __name__ == "__main__":
    main()
