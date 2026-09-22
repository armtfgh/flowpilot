"""Independent arithmetic, package-preservation and rendered-layout checks."""
import csv
import io
import json
import math
import re
from hashlib import sha256
from pathlib import Path
from zipfile import ZipFile

import fitz
from docx import Document
from lxml import etree as E
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "manuscript"
OUT = BASE / "revision_20260922"
NS = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main", "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships", "a": "http://schemas.openxmlformats.org/drawingml/2006/main"}
checks = []


def check(name, passed, detail):
    checks.append({"check": name, "passed": bool(passed), "detail": detail})


def read_csv(path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def package_checks():
    for stem, src in [("manuscript", "manuscript_text_revised_round2.docx"), ("esi", "esi.docx")]:
        with ZipFile(BASE / src) as a, ZipFile(BASE / f"{stem}_revised_20260922.docx") as b:
            media = [n for n in a.namelist() if n.startswith("word/media/")]
            check(stem + " original media", all(a.read(n) == b.read(n) for n in media), f"{len(media)} original image assets preserved byte-for-byte")
            for part in ["word/styles.xml", "word/numbering.xml"]:
                check(stem + " " + part, a.read(part) == b.read(part), "Original style/numbering definitions")
            old = E.fromstring(a.read("word/document.xml"))
            new = E.fromstring(b.read("word/document.xml"))
            check(stem + " page geometry", [E.tostring(x) for x in old.xpath(".//w:sectPr", namespaces=NS)] == [E.tostring(x) for x in new.xpath(".//w:sectPr", namespaces=NS)], "Section properties unchanged")
            if stem == "manuscript":
                old_rids = old.xpath(".//a:blip/@r:embed", namespaces=NS)[:4]
                new_rids = new.xpath(".//a:blip/@r:embed", namespaces=NS)[:4]
                check("Main Figures 1-4 still assigned", old_rids == new_rids, str(new_rids))
            else:
                rels = E.fromstring(b.read("word/_rels/document.xml.rels"))
                targets = {r.get("Id"): r.get("Target") for r in rels}
                inserted = new.xpath(".//a:blip/@r:embed", namespaces=NS)[-7:]
                expected = [ROOT / f"outputs/figure5_pure_oxygen_physics_20260918/slides/figure5_set{i}/topology.png" for i in range(1, 4)]
                expected += [ROOT / f"outputs/khu_revised_six_20260915/collaborator_slides_20260915_163845/figure6_set{i}/topology.png" for i in range(1, 4)]
                expected += [OUT / "figures/figureS22_preprint_figure6_exact.png"]
                for i, (rid, source) in enumerate(zip(inserted, expected)):
                    actual = b.read("word/" + targets[rid])
                    check(f"ESI inserted image {i+1} assignment", actual == source.read_bytes(), str(source.relative_to(ROOT)))
                check("ESI unique image parts", len({targets[r] for r in inserted}) == 7, "Same basenames do not overwrite distinct saved topologies")
            full = " ".join(new.xpath(".//w:t/text()", namespaces=NS))
            check(stem + " authors", all(n in full[:1800] for n in ["Amirreza Mottafegh", "Mincheol Park", "[Name to be added]", "Dr. Myeong", "Professor Park", "Dr. Gwang-Noh Ahn"]), "Requested order checked visually; names retained as provided")
            check(stem + " placeholders", "XXX" in full and "YYY" in full, "No experimental yields invented")
        clean = Document(BASE / f"{stem}_revised_20260922.docx")
        marked = Document(BASE / f"{stem}_revised_20260922_marked.docx")
        text_of = lambda d: [p.text for p in d.paragraphs] + [[c.text for r in t.rows for c in r.cells] for t in d.tables]
        check(stem + " marked copy content", text_of(clean) == text_of(marked), "Same prose and tables; new edits highlighted in marked copy")
        with ZipFile(BASE / f"{stem}_revised_20260922_marked.docx") as z:
            xml = E.fromstring(z.read("word/document.xml"))
            count = len(xml.xpath(".//w:highlight[@w:val='yellow']", namespaces=NS))
            check(stem + " yellow marking", count > 20, f"{count} yellow-highlighted runs (includes retained original marks)")


def arithmetic_checks():
    for number, folder in [(5, ROOT / "outputs/figure5_pure_oxygen_physics_20260918/slides"), (6, ROOT / "outputs/khu_revised_six_20260915/collaborator_slides_20260915_163845")]:
        stage_rows = [r for r in read_csv(folder / "all_stage_parameters.csv") if r["case"].startswith(f"figure{number}")]
        for r in stage_rows:
            q = float(r["liquid_flow_mL_min"]) + float(r["gas_inlet_STP_mL_min"])
            expected = float(r["volume_mL"]) / q
            check(f"{r['case']} stage {r['stage']} V/Q", math.isclose(expected, float(r["nominal_inlet_residence_min"]), abs_tol=0.0001), f"{r['volume_mL']} / {q} = {expected:.6f} min; inlet/STP index when gas present")
        for k in range(1, 4):
            feed_rows = [r for r in read_csv(folder / "all_feed_parameters.csv") if r["case"] == f"figure{number}_set{k}"]
            for r in feed_rows:
                if r["phase"] == "liquid":
                    expected = float(r["concentration_M"]) * float(r["flow_mL_min"])
                    check(f"{r['case']} {r['component']} C Q", math.isclose(expected, float(r["molar_flow_mmol_min"]), abs_tol=1e-10), f"{expected:.8g} mmol/min")
                else:
                    expected = float(r["flow_mL_min"]) / 22.41272243
                    check(f"{r['case']} gas molar feed", math.isclose(expected, float(r["molar_flow_mmol_min"]), rel_tol=1e-7), f"{expected:.8g} mmol/min, pure O2")
            if number == 6:
                acid = next(r for r in feed_rows if "benzoic acid" in r["component"])
                amine = next(r for r in feed_rows if r["component"].lower() == "benzylamine")
                eq = float(amine["molar_flow_mmol_min"]) / float(acid["molar_flow_mmol_min"])
                check(f"Figure6 Set{k} amine equivalent", math.isclose(eq, 1.05), str(eq))


def figure_and_layout_checks():
    source = fitz.open(BASE / "preprint.pdf")
    original_image = source.extract_image(29)
    expected = Image.open(io.BytesIO(original_image["image"])).convert("RGBA")
    smask = original_image["smask"]
    if smask:
        alpha = Image.open(io.BytesIO(source.extract_image(smask)["image"])).convert("L")
        expected.putalpha(alpha)
    target = Image.open(OUT / "figures/figureS22_preprint_figure6_exact.png").convert("RGBA")
    check("Exact preprint Figure6 extraction", target.tobytes() == expected.tobytes(), f"{target.width} x {target.height}; original straight RGB and separate alpha preserved, without premultiplication")
    for stem in ["manuscript", "esi"]:
        pdf = fitz.open(OUT / f"rendered/{stem}_revised_20260922.pdf")
        outside = []
        for i, page in enumerate(pdf):
            for b in page.get_text("dict")["blocks"]:
                for line in b.get("lines", []):
                    for s in line["spans"]:
                        x0, y0, x1, y1 = s["bbox"]
                        if x0 < -1 or y0 < -1 or x1 > page.rect.width + 1 or y1 > page.rect.height + 1:
                            outside.append({"page": i + 1, "text": s["text"][:90], "bbox": s["bbox"]})
        check(stem + " text within page", not outside, outside or f"{len(pdf)} rendered pages; no text outside page bounds")
        if stem == "manuscript":
            for n in [5, 6]:
                hits = [i for i, p in enumerate(pdf) if re.search(rf"Figure\s+{n}\.\s", p.get_text())]
                expected_tail = f"Section 8.{n-4}."
                complete = any(expected_tail in re.sub(r"\s+", " ", pdf[i].get_text()) for i in hits)
                check(f"Figure{n} caption intact", complete, f"Caption page(s): {[i+1 for i in hits]}")
        else:
            toc = json.loads((OUT / "contents_page_audit.json").read_text())
            norm = lambda t: re.sub(r"[^a-z0-9]", "", t.lower())
            for r in toc:
                page = pdf[r["page"] - 1]
                check("TOC " + r["section"], norm(r["heading"]) in norm(page.get_text()), f"{r['heading']} -> {r['page']}")


if __name__ == "__main__":
    package_checks()
    arithmetic_checks()
    figure_and_layout_checks()
    result = {"checks": checks, "passed": sum(x["passed"] for x in checks), "total": len(checks), "failures": [c for c in checks if not c["passed"]]}
    (OUT / "verification_results.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k != "checks"}, indent=2))
    if result["failures"]:
        raise SystemExit(1)
