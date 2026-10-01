"""Audit revised Word files, cross-reference formatting and rendered page bounds."""
from copy import deepcopy
from hashlib import sha256
import json
import re
from zipfile import ZipFile

import fitz
from lxml import etree as E

import revise_prior_work_20260928 as revision
from verify_introduction_20260923 import no_highlight_structure, without_citations
from refresh_revision_toc_20260922 import normalized

W, NS = revision.W, revision.NS
OUT, BASE = revision.OUT, revision.BASE
checks = []


def check(name, passed, detail):
    checks.append({"check": name, "passed": bool(passed), "detail": detail})


def read(path):
    with ZipFile(path) as z:
        parts = {n: z.read(n) for n in z.namelist()}
    return parts, E.fromstring(parts["word/document.xml"])


def paras(doc):
    return doc.find(W + "body").findall(W + "p")


def subsequence(old, new):
    iterator = iter(new)
    return all(any(a == b for b in iterator) for a in old)


def bold_flags(p):
    value, flags = "", []
    for t in revision.paragraph_text_nodes(p):
        run = t.getparent()
        setting = run.find(W + "rPr/" + W + "b")
        bold = setting is not None and setting.get(W + "val", "true") not in {"0", "false", "off"}
        s = t.text or ""
        value += s
        flags.extend([bold] * len(s))
    return value, flags


def verify_bold(doc, name):
    failed, total = [], 0
    for p in doc.iter(W + "p"):
        value, flags = bold_flags(p)
        for m in revision.LABEL.finditer(value):
            total += 1
            if not all(flags[m.start():m.end()]):
                failed.append(m[0])
    check(name + ": all figure/table labels bold", not failed, {"labels": total, "failed": failed})


def unit_tests():
    p = E.Element(W + "p", nsmap={"w": NS["w"]})
    # Reference spans cross run/hyperlink boundaries and coexist with a field.
    for value in ["See Fig", "ure 4", "b and Tables ", "S8", "-S10; Figures 5 and 6. Plain text."]:
        r = E.SubElement(p, W + "r")
        E.SubElement(r, W + "t").text = value
    link = E.SubElement(p, W + "hyperlink", {W + "anchor": "test_target"})
    r = E.SubElement(link, W + "r")
    E.SubElement(r, W + "instrText").text = " REF test_target "
    E.SubElement(r, W + "t").text = " Figure S7(a-c)"
    drawing = E.SubElement(r, W + "drawing", {"test-id": "preserve"})
    before = revision.text(p)
    revision.bold_references(p)
    value, flags = bold_flags(p)
    expected = [False] * len(value)
    for label in ["Figure 4b", "Tables S8-S10", "Figures 5 and 6", "Figure S7(a-c)"]:
        start = value.index(label)
        expected[start:start+len(label)] = [True] * len(label)
    check("Bold formatter isolates labels across runs", value == before and flags == expected, "Ordinary sentence text remains unbolded")
    check("Bold formatter preserves field/hyperlink/drawing", p.xpath(".//w:instrText/text()", namespaces=NS) == [" REF test_target "] and len(p.findall(".//" + W + "drawing")) == 1 and link.get(W + "anchor") == "test_target", "Field instruction, anchor and drawing retained")
    saved = E.tostring(p)
    revision.bold_references(p)
    check("Bold formatter is idempotent", saved == E.tostring(p), "Repeated formatting does not accumulate runs or change text")


def source_preservation(stem, source, audit):
    src_parts, src = read(source)
    dst_parts, dst = read(BASE / f"{stem}_revised_20260928.docx")
    marked_parts, marked = read(BASE / f"{stem}_revised_20260928_marked.docx")
    for suffix, parts in [("clean", dst_parts), ("marked", marked_parts)]:
        check(f"{stem}/{suffix}: package parts preserved", set(src_parts) == set(parts) and all(src_parts[k] == parts[k] for k in src_parts if k != "word/document.xml"), "Every embedded image, style, relationship, numbering definition, header and footer is byte-identical")
    graphics = lambda doc: [E.tostring(n) for n in doc.iter() if E.QName(n).localname in {"drawing", "pict"}]
    check(stem + ": original figure placements preserved", graphics(src) == graphics(dst), {"graphics": len(graphics(src))})
    check(stem + ": clean/marked match apart from highlighting", no_highlight_structure(dst) == no_highlight_structure(marked), "Text, layout and citation numbering match")
    check(stem + ": no draft citation delimiters", "[[" not in revision.text(dst) and "]]" not in revision.text(dst), "Citation placeholders fully resolved")
    for label, doc in [("clean", dst), ("marked", marked)]:
        verify_bold(doc, f"{stem}/{label}")
    table_texts = lambda doc: [revision.text(t) for t in doc.findall(".//" + W + "tbl")]
    check(stem + ": existing table text preserved", subsequence(table_texts(src), table_texts(dst)), {"original_tables": len(table_texts(src)), "revised_tables": len(table_texts(dst))})
    if stem == "esi":
        check("ESI: all original body paragraphs preserved", subsequence([revision.text(p) for p in paras(src)], [revision.text(p) for p in paras(dst)]), "Existing body prose, captions and headings remain in their original order")
        check("ESI: new comparison table and section present", any(revision.text(p).startswith("Table S36.") for p in paras(dst)) and any(revision.text(p) == "Related flow-chemistry and process-design agents" for p in paras(dst)), "Section 11, Table S36 (two panels)")
    else:
        old, new = paras(src), paras(dst)
        source_ref_index = next(i for i, p in enumerate(old) if revision.text(p) == "REFERENCES")
        new_ref_index = next(i for i, p in enumerate(new) if revision.text(p) == "REFERENCES")
        old_results = next(i for i, p in enumerate(old) if revision.text(p) == "RESULTS")
        old_intro = next(i for i, p in enumerate(old) if revision.text(p) == "INTRODUCTION")
        exclude = {"Related AI systems address complementary parts", "The first connected-process case couples", "The second case is a thermal, two-stage amidation", "The translation workflow comprises standardized intake"}
        kept = [without_citations(p) for i, p in enumerate(old[:source_ref_index]) if not (old_intro + 3 <= i < old_results) and not any(revision.text(p).startswith(s) for s in exclude)]
        check("Main: out-of-scope prose preserved", subsequence(kept, [without_citations(p) for p in new[:new_ref_index]]), "Only specified introduction/discussion paragraphs and three scoped amendments change outside reference numbering")
        refs = [p for p in new[new_ref_index+1:] if revision.text(p).strip()]
        first_ids = []
        for p in new[:new_ref_index]:
            for _, ids in revision.intro_tools.citation_runs(p):
                for item in ids:
                    if item not in first_ids:
                        first_ids.append(item)
        check("Main: all bibliography entries cited in order", first_ids == list(range(1, len(refs)+1)), {"references": len(refs)})
        for entry in json.loads((OUT / "new_references.json").read_text()):
            check(f"Main: new source {entry['id']} present", any(normalized(entry["title"]) in normalized(revision.text(p)) for p in refs), entry["title"])
        new_start = next(i for i, p in enumerate(new) if revision.text(p) == "INTRODUCTION")
        new_end = next(i for i, p in enumerate(new) if revision.text(p) == "RESULTS")
        for i, p in enumerate(new[new_start+3:new_end], 3):
            rs = [r for r in p.iter(W + "r") if revision.text(r).strip()]
            good = all(r.xpath("./w:rPr/w:rFonts[@w:ascii='Times New Roman']", namespaces=NS) and r.xpath("./w:rPr/w:sz[@w:val='22']", namespaces=NS) for r in rs)
            check(f"Main: introduction paragraph {i} typography", good, "Times New Roman 11 pt")
        order_src = re.findall(r"\bS\d+\b", " ".join(revision.text(p) for p in old[:source_ref_index]))
        order_dst = re.findall(r"\bS\d+\b", " ".join(revision.text(p) for p in new[:new_ref_index]))
        check("Main: all old literal ESI references retained", subsequence(order_src, order_dst), "New Table S36 is added after Table S35; no old figure/table numbers are removed")
    return dst


def render_checks(stem):
    pages = {}
    for suffix in ["", "_marked"]:
        path = OUT / "rendered" / f"{stem}_revised_20260928{suffix}.pdf"
        with fitz.open(path) as pdf:
            out, empty = [], []
            pages[suffix or "clean"] = len(pdf)
            for i, page in enumerate(pdf):
                if not page.get_text().strip() and not page.get_images():
                    empty.append(i+1)
                for block in page.get_text("dict")["blocks"]:
                    for line in block.get("lines", []):
                        for span in line["spans"]:
                            box = fitz.Rect(span["bbox"])
                            if box.x0 < -1 or box.y0 < -1 or box.x1 > page.rect.width + 1 or box.y1 > page.rect.height + 1:
                                out.append([i+1, span["text"]])
            check(f"{stem}{suffix}: rendered text within pages", not out, out or f"{len(pdf)} pages; no out-of-page text")
            check(f"{stem}{suffix}: no empty pages", not empty, empty)
            if not suffix:
                page_ids = [1, 2, 3, 4, 11] if stem == "manuscript" else [0, 1] + list(range(len(pdf)-7, len(pdf)))
                for i in page_ids:
                    pdf[i].get_pixmap(matrix=fitz.Matrix(1.5, 1.5)).save(OUT / "rendered" / f"{stem}_check_{i+1:03d}.png")
    check(stem + ": clean/marked pagination equal", pages["clean"] == pages["_marked"], pages)


def main():
    audit = json.loads((OUT / "revision_audit.json").read_text())
    for path, expected in audit["source_hashes"].items():
        check("Original unchanged: " + path, sha256((revision.ROOT / path).read_bytes()).hexdigest() == expected, "SHA-256 unchanged")
    unit_tests()
    source_preservation("manuscript", revision.MAIN_SOURCE, audit)
    source_preservation("esi", revision.ESI_SOURCE, audit)
    for stem in ["manuscript", "esi"]:
        render_checks(stem)
    contents = json.loads((OUT / "contents_page_audit.json").read_text())
    for suffix, rows in contents.items():
        ending = "" if suffix == "clean" else suffix
        with fitz.open(OUT / "rendered" / f"esi_revised_20260928{ending}.pdf") as pdf:
            mismatched = [r for r in rows if normalized(r["heading"]) not in normalized(pdf[r["page"]-1].get_text())]
            check("ESI contents pagination: " + suffix, not mismatched, mismatched or f"All {len(rows)} entries resolve to rendered headings")
    result = {"passed": sum(c["passed"] for c in checks), "total": len(checks), "checks": checks}
    (OUT / "independent_verification.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"passed": result["passed"], "total": result["total"], "failures": [c for c in checks if not c["passed"]]}, indent=2))
    assert all(c["passed"] for c in checks)


if __name__ == "__main__":
    main()
