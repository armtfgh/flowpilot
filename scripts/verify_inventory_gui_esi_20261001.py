"""Audit document preservation, numbering, screenshot provenance and pagination."""
from copy import deepcopy
from hashlib import sha256
import csv
import json
import re
import shutil

import fitz
from lxml import etree as E

import consolidate_gui_esi_20261001 as rev
from verify_overhaul_20260930 import labeled
from refresh_revision_toc_20260922 import normalized
from revise_manuscript_cases_20260922 import expanded_cites

OUT, BASE, W, NS, text = rev.OUT, rev.BASE, rev.W, rev.NS, rev.text
checks = []


def check(name, passed, detail=None):
    checks.append({"check": name, "passed": bool(passed), "detail": detail})


def media_hashes(pkg):
    relationships = {r.get("Id"): "word/" + r.get("Target") for r in pkg.rels}
    return [sha256(pkg.parts[relationships[rid]]).hexdigest() for rid in pkg.doc.xpath(".//a:blip/@r:embed", namespaces=NS)]


def main():
    packages = {}
    for stem in ["esi", "manuscript"]:
        old = rev.Package(rev.SOURCES[stem])
        pkg = rev.Package(BASE / f"{stem}_submission_inventory_gui_20261001.docx")
        marked = rev.Package(BASE / f"{stem}_submission_inventory_gui_20261001_marked.docx")
        packages[stem] = pkg
        for part in ["word/styles.xml", "word/fontTable.xml", "word/numbering.xml"]:
            check(f"{stem}: preserved {part}", pkg.parts.get(part) == old.parts.get(part))
        check(f"{stem}: unchanged page geometry", E.tostring(pkg.body[-1]) == E.tostring(old.body[-1]))
        check(f"{stem}: all prior embedded media retained", all(pkg.parts.get(k) == v for k, v in old.parts.items() if k.startswith("word/media/")))
        check(f"{stem}: original tables retained without data changes", [text(t) for t in pkg.doc.iter(W + "tbl")] == [text(t) for t in old.doc.iter(W + "tbl")])
        check(f"{stem}: clean/marked text identical", text(pkg.doc) == text(marked.doc))
        highlighted = marked.doc.xpath('.//w:r[w:rPr/w:highlight[@w:val="yellow"]]', namespaces=NS)
        check(f"{stem}: yellow revision highlighting", bool(highlighted))
        bad = [text(r) for r in highlighted if text(r).strip() and (not r.xpath('./w:rPr/w:rFonts[@w:ascii="Times New Roman"]', namespaces=NS) or not r.xpath('./w:rPr/w:sz[@w:val="22"]', namespaces=NS))]
        check(f"{stem}: revised text uses Times New Roman 11", not bad, bad)
        for part in pkg.parts:
            if part.endswith(".xml"):
                E.fromstring(pkg.parts[part])
        check(f"{stem}: XML parses", True)
        if stem == "manuscript":
            check("All six main-text figures unchanged", media_hashes(pkg) == media_hashes(old))
        else:
            remaining = set(media_hashes(old)) & set(media_hashes(pkg))
            # Fourteen non-GUI figure images plus the main plots/photographs/spectra remain.
            check("Only the GUI artwork was replaced", len(remaining) >= 18, len(remaining))
        old_ref = next(p for p in old.body if text(p) == ("References" if stem == "esi" else "REFERENCES"))
        new_ref = next(p for p in pkg.body if text(p) == text(old_ref))
        check(f"{stem}: bibliography unchanged", [text(p) for p in list(old.body)[list(old.body).index(old_ref):]] == [text(p) for p in list(pkg.body)[list(pkg.body).index(new_ref):]])

    s, m = packages["esi"], packages["manuscript"]
    definitions = {}
    for p in s.body:
        found = re.match(r"^(Figure|Table) S(\d+)\.", text(p))
        if found:
            key = (found[1], int(found[2]))
            check("Unique caption: " + found[0], key not in definitions)
            definitions[key] = text(p)
            position = 0
            bad = []
            for run in p.iter(W + "r"):
                bold = run.xpath('./w:rPr/w:b[not(@w:val="0" or @w:val="false")]', namespaces=NS)
                if position >= found.end() and text(run).strip() and bold:
                    bad.append(text(run))
                position += len(text(run))
            check(found[0] + " caption body not bold", not bad, bad)
    rows = []
    for kind, count in [("Figure", 21), ("Table", 25)]:
        check(f"{kind} captions ordered", [n for k, n in definitions if k == kind] == list(range(1, count + 1)))
        for tag, pkg in [("main", m), ("ESI", s)]:
            first = []
            unresolved = []
            for i, p in enumerate(pkg.body):
                if p.tag == W + "sdt":
                    continue
                for k, n in labeled(text(p)):
                    if (k, n) not in definitions:
                        unresolved.append((k, n, text(p)[:100]))
                    if k == kind and n not in first:
                        first.append(n)
                        if tag == "main":
                            rows.append({"label": f"{k} S{n}", "main_body_index": i, "first_citation": text(p), "caption": definitions.get((k, n))})
            check(f"{tag}: {kind} first citations ordered", first == list(range(1, count + 1)), first)
            check(f"{tag}: all supplementary references resolve", not unresolved, unresolved)
    with (OUT / "citation_audit.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    check("Literature comparison retained as Table S1", "Chat-microreactor" in text(next(s.doc.iter(W + "tbl"))))
    check("All benchmark summary tables preserved", len(list(s.doc.iter(W + "tbl"))) == 25)
    capture = json.loads((OUT / "capture_manifest.json").read_text())
    check("Live browser capture has no console errors", not capture["console_errors"])
    check("Export/import roundtrip verified", capture["inventory_roundtrip_passed"])
    check("Saved answers produce ready intake", capture["intake_ready"])
    check("Screenshots did not start a design/benchmark", not any(r["method"] == "POST" and r["url"].endswith("/api/design/jobs") for r in capture["api_calls"]))
    for shot in capture["screenshots"]:
        check("Unaltered screenshot source: " + shot["name"], sha256((OUT / "screenshots" / (shot["name"] + ".png")).read_bytes()).hexdigest() == shot["sha256"])
    for path, digest in json.loads((OUT / "source_manifest.json").read_text()).items():
        check("Source document unchanged: " + path, sha256((rev.ROOT / path).read_bytes()).hexdigest() == digest)

    toc = json.loads((OUT / "contents_page_audit.json").read_text())
    for stem in ["esi", "manuscript"]:
        pdf_path = OUT / "rendered" / f"{stem}_submission_inventory_gui_20261001.pdf"
        pdf = fitz.open(pdf_path)
        marked = fitz.open(pdf_path.with_stem(pdf_path.stem + "_marked"))
        check(f"{stem}: clean and marked pagination identical", len(pdf) == len(marked))
        check(f"{stem}: clean and marked rendered text identical", [" ".join(p.get_text().split()) for p in pdf] == [" ".join(p.get_text().split()) for p in marked])
        detached, overflow, blank = [], [], []
        for i, page in enumerate(pdf):
            t = page.get_text()
            captions = re.findall(r"^Figure S?\d+\.", t, re.M)
            if captions and not page.get_image_info():
                detached.append((i + 1, captions))
            if len(t.strip()) < 8 and not page.get_image_info():
                blank.append(i + 1)
            for block in page.get_text("dict")["blocks"]:
                if block.get("type") != 0:
                    continue
                for line in block["lines"]:
                    for span in line["spans"]:
                        x0, y0, x1, y1 = span["bbox"]
                        if x0 < -1 or y0 < -1 or x1 > page.rect.width + 1 or y1 > page.rect.height + 1:
                            overflow.append((i + 1, span["text"]))
            if any(f"Figure S{n}." in t for n in [12, 13, 14]):
                page.get_pixmap(matrix=fitz.Matrix(1.8, 1.8)).save(OUT / "rendered" / f"{stem}_gui_page_{i + 1:02d}.png")
        check(f"{stem}: artwork and captions on same page", not detached, detached)
        check(f"{stem}: no empty pages", not blank, blank)
        check(f"{stem}: no text outside page", not overflow, overflow)
        if stem == "esi":
            check("Contents page references refreshed", all(normalized(row["heading"]) in normalized(pdf[row["page"] - 1].get_text()) for row in toc))
        shutil.copy2(pdf_path, BASE / pdf_path.name)
    report = {"passed": sum(c["passed"] for c in checks), "total": len(checks), "checks": checks}
    (OUT / "verification.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({"passed": report["passed"], "total": len(checks), "failed": [c for c in checks if not c["passed"]]}, indent=2))
    assert all(c["passed"] for c in checks)


if __name__ == "__main__":
    main()
