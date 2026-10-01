"""Revise only the introduction and literature citations, preserving the Word package."""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import re
import unicodedata
from copy import deepcopy
from datetime import date
from hashlib import sha256
from pathlib import Path
from urllib.parse import quote
from zipfile import ZIP_DEFLATED, ZipFile

import requests
from bs4 import BeautifulSoup
from lxml import etree as E

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "manuscript"
OUT = BASE / "introduction_revision_20260923"
SOURCE = BASE / "manuscript_revised_20260922.docx"
TARGET = BASE / "manuscript_introduction_revised_20260923.docx"
MARKED = BASE / "manuscript_introduction_revised_20260923_marked.docx"
AS_OF = date(2026, 9, 23)
spec = importlib.util.spec_from_file_location("previous_revision", ROOT / "scripts/revise_manuscript_cases_20260922.py")
previous = importlib.util.module_from_spec(spec)
spec.loader.exec_module(previous)
W, NS = previous.W, previous.NS
text = previous.text
expand = previous.expanded_cites
compress = previous.compressed_cites


def normalize(value):
    if "<" in value:
        value = BeautifulSoup(value, "html.parser").get_text()
    value = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]", "", value.lower())


def collect_metadata():
    """Cache publisher-deposited metadata and version-specific preprint records."""
    entries = json.loads((OUT / "new_reference_manifest.json").read_text())
    cache = OUT / "source_metadata"
    cache.mkdir(exist_ok=True)
    for entry in entries:
        assert date.fromisoformat(entry["date"]) <= AS_OF
        saved = cache / f"reference_{entry['id']}.json"
        if saved.exists():
            data = json.loads(saved.read_text())
        elif "doi" in entry:
            url = "https://api.crossref.org/works/" + quote(entry["doi"], safe="")
            response = requests.get(url, timeout=45)
            response.raise_for_status()
            data = response.json()["message"]
            saved.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
        else:
            url = "https://arxiv.org/abs/" + entry["arxiv"]
            response = requests.get(url, timeout=45)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, "html.parser")
            metadata = {}
            for node in soup.select("meta[name^='citation_']"):
                metadata.setdefault(node["name"], []).append(node.get("content", ""))
            data = {"source_url": url, "metadata": metadata}
            saved.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
        actual_title = (data["title"] if "doi" in entry else data["metadata"]["citation_title"])[0]
        assert normalize(actual_title) == normalize(entry["title"]), (entry["id"], actual_title)
        if "doi" in entry:
            actual_date = data.get("published-online", data.get("published", {})).get("date-parts", [[]])[0]
            expected_date = [int(v) for v in entry["date"].split("-")]
            assert actual_date == expected_date[:len(actual_date)], (entry["id"], actual_date, expected_date)
            assert data["DOI"].lower() == entry["doi"].lower()
        else:
            assert data["metadata"]["citation_online_date"][0].replace("/", "-") == entry["date"]
        entry["metadata"] = data
        entry["source_url"] = "https://doi.org/" + entry["doi"] if "doi" in entry else data["source_url"]
        print(f"Verified {entry['id']}: {actual_title}", flush=True)
    (OUT / "verified_references.json").write_text(json.dumps(entries, indent=2, ensure_ascii=False) + "\n")
    return entries


def properties(decoration=None):
    node = E.Element(W + "rPr")
    previous.Package.font(node)
    if decoration:
        E.SubElement(node, W + decoration)
    return node


def initials(given):
    return " ".join("-".join(s[0] + "." for s in part.split("-") if s) for part in given.split())


def author_string(entry):
    data = entry["metadata"]
    if "doi" in entry:
        authors = [a.get("family", a.get("name", "")) + (", " + initials(a["given"]) if a.get("given") else "") for a in data["author"]]
    else:
        authors = []
        for a in data["metadata"]["citation_author"]:
            if "," in a:
                family, given = a.split(",", 1)
                authors.append(family.strip() + ", " + initials(given.strip()))
            else:
                given, family = a.rsplit(" ", 1)
                authors.append(family + ", " + initials(given))
    return "; ".join(authors)


def bibliography_paragraph(entry, template):
    p = deepcopy(template)
    for child in list(p):
        if child.tag != W + "pPr":
            p.remove(child)
    data = entry["metadata"]
    pieces = [(author_string(entry).rstrip(".") + ". " + entry["title"].rstrip(".") + ". ", None)]
    if "doi" in entry:
        journal = data["container-title"][0]
        year = str(entry["date"][:4])
        pieces.extend([(journal, "i"), (" ", None), (year, "b")])
        volume = data.get("volume")
        if volume:
            pieces.extend([( ", ", None), (volume, "i")])
        if data.get("issue"):
            pieces.append((" (" + data["issue"] + ")", None))
        pages = data.get("page") or data.get("article-number")
        if pages:
            pieces.append((", " + pages, None))
        if not volume and not pages:
            pieces.append((", advance online publication", None))
        pieces.append((". https://doi.org/" + entry["doi"] + ".", None))
    else:
        pieces.extend([("arXiv", "i"), (" ", None), (entry["date"][:4], "b"), (", " + entry["arxiv"] + ". Preprint. " + entry["source_url"] + ".", None)])
    for value, decoration in pieces:
        previous.prior.append_run(p, value, properties(decoration))
    return p


def citation_runs(node):
    for r in node.iter(W + "r"):
        if r.xpath("./w:rPr/w:vertAlign[@w:val='superscript']", namespaces=NS):
            ids = expand(text(r))
            if ids:
                yield r, ids


def renumber(node, mapping, changed_runs=None):
    for r, ids in citation_runs(node):
        updated = compress([mapping[i] for i in ids])
        if updated == text(r):
            continue
        ts = r.findall(W + "t")
        assert ts
        ts[0].text = updated
        for t in ts[1:]:
            r.remove(t)
        if changed_runs is not None:
            changed_runs.append(r)


def paragraphs_between(body, start, end):
    nodes = list(body)
    return nodes[nodes.index(start) + 1:nodes.index(end)]


def highlight(node):
    for r in node.iter(W + "r"):
        if not text(r).strip():
            continue
        rp = r.find(W + "rPr")
        if rp is None:
            rp = E.Element(W + "rPr")
            r.insert(0, rp)
        old = rp.find(W + "highlight")
        if old is None:
            old = E.SubElement(rp, W + "highlight")
        old.set(W + "val", "yellow")


def write_package(parts, doc, target):
    with ZipFile(target, "w", ZIP_DEFLATED) as z:
        for name, content in parts.items():
            z.writestr(name, E.tostring(doc, xml_declaration=True, encoding="UTF-8", standalone=True) if name == "word/document.xml" else content)


def structure(node):
    return (node.tag, sorted(node.attrib.items()), node.text, node.tail, [structure(c) for c in node])


def main(fetch_only=False):
    entries = collect_metadata()
    if fetch_only:
        return
    protected = [SOURCE, BASE / "esi_revised_20260922.docx", BASE / "esi_revised_20260922_marked.docx"]
    original_hashes = {str(p.relative_to(ROOT)): sha256(p.read_bytes()).hexdigest() for p in protected}
    with ZipFile(SOURCE) as z:
        parts = {n: z.read(n) for n in z.namelist()}
    doc = E.fromstring(parts["word/document.xml"])
    body = doc.find(W + "body")
    headings = {text(p): p for p in body.findall(W + "p") if text(p) in {"INTRODUCTION", "RESULTS", "REFERENCES"}}
    old_intro = paragraphs_between(body, headings["INTRODUCTION"], headings["RESULTS"])
    original_intro = "\n\n".join(previous.prior.marked_text(p) for p in old_intro)
    ref_nodes = list(body)[list(body).index(headings["REFERENCES"]) + 1:]
    ref_nodes = [p for p in ref_nodes if p.tag == W + "p" and text(p).strip()]
    assert len(ref_nodes) == 50
    bibliography = {i: p for i, p in enumerate(ref_nodes, 1)}
    original_bibliography = {i: text(p) for i, p in bibliography.items()}
    preserved_nodes = [p for p in list(body) if p not in old_intro and p not in ref_nodes]
    preserved_original = [deepcopy(p) for p in preserved_nodes]
    draft = (OUT / "introduction_draft.txt").read_text().strip()
    assert "[ [" not in draft and "] ]" not in draft
    substitutions = {"El Agente Cuantico": "El Agente Cu\u00e1ntico", "El Agente Grafico": "El Agente Gr\u00e1fico", "La Agente Optima": "La Agente \u00d3ptima"}
    for a, b in substitutions.items():
        draft = draft.replace(a, b)
    intro = []
    for value in draft.split("\n\n"):
        p = deepcopy(old_intro[0])
        for child in list(p):
            if child.tag != W + "pPr":
                p.remove(child)
        for bit in re.split(r"(\[\[.*?\]\])", value):
            if not bit:
                continue
            rp = properties()
            if bit.startswith("[["):
                assert expand(bit[2:-2])
                E.SubElement(rp, W + "vertAlign", {W + "val": "superscript"})
                bit = bit[2:-2]
            previous.prior.append_run(p, bit, rp)
        headings["RESULTS"].addprevious(p)
        intro.append(p)
    for p in old_intro:
        body.remove(p)
    for e in entries:
        bibliography[e["id"]] = bibliography_paragraph(e, ref_nodes[0])
    prose = list(body)[:list(body).index(headings["REFERENCES"])]
    order = []
    for p in prose:
        for _, ids in citation_runs(p):
            for i in ids:
                assert i in bibliography, i
                if i not in order:
                    order.append(i)
    mapping = {old: new for new, old in enumerate(order, 1)}
    assert set(e["id"] for e in entries) <= set(mapping)
    removed = sorted(set(original_bibliography) - set(mapping))
    changed_runs = []
    for p in prose:
        renumber(p, mapping, changed_runs)
    for p in ref_nodes:
        body.remove(p)
    cursor = headings["REFERENCES"]
    for i in order:
        cursor.addnext(bibliography[i])
        cursor = bibliography[i]
    # Compare untouched content as XML after applying only the known citation map.
    for old, current in zip(preserved_original, preserved_nodes):
        renumber(old, mapping)
        assert structure(old) == structure(current), text(current)[:100]
    write_package(parts, doc, TARGET)
    marked = deepcopy(doc)
    tree = doc.getroottree()
    changed_paths = {tree.getpath(n) for n in intro + [bibliography[e["id"]] for e in entries] + changed_runs}
    for path in changed_paths:
        found = marked.xpath(path, namespaces=NS)
        assert len(found) == 1, path
        highlight(found[0])
    write_package(parts, marked, MARKED)
    checks = []
    for file in [TARGET, MARKED]:
        with ZipFile(file) as z:
            assert set(z.namelist()) == set(parts)
            for name, original in parts.items():
                if name != "word/document.xml":
                    assert z.read(name) == original, name
            checks.append({"file": file.name, "all_non_document_parts_identical": True, "media_count": sum(n.startswith("word/media/") for n in parts)})
    for p in protected:
        assert sha256(p.read_bytes()).hexdigest() == original_hashes[str(p.relative_to(ROOT))]
    final_order = []
    for p in prose:
        for _, ids in citation_runs(p):
            for i in ids:
                if i not in final_order:
                    final_order.append(i)
    assert final_order == list(range(1, len(order) + 1)), final_order
    for p in intro:
        for r in p.iter(W + "r"):
            assert r.find(W + "rPr/" + W + "sz").get(W + "val") == "22"
            assert r.find(W + "rPr/" + W + "rFonts").get(W + "ascii") == "Times New Roman"
    report = {
        "as_of": AS_OF.isoformat(), "source_hashes_unchanged": original_hashes,
        "outside_introduction_prose_and_layout_unchanged_except_numeric_citations": True,
        "new_reference_count": len(entries), "new_2026_references": sum(e["date"].startswith("2026") for e in entries),
        "new_peer_reviewed_references": sum(e["status"] == "peer-reviewed" for e in entries),
        "new_preprints": sum(e["status"] == "preprint" for e in entries),
        "reference_total": len(order), "removed_uncited_old_references": {i: original_bibliography[i] for i in removed},
        "reference_mapping": mapping, "citation_first_appearance_order_valid": True,
        "introduction_paragraphs": len(intro), "introduction_words": len(re.sub(r"\[\[.*?\]\]", "", draft).split()),
        "introduction_font": "Times New Roman, 11 pt", "word_package_preservation": checks,
    }
    (OUT / "verification.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    (OUT / "original_introduction.txt").write_text(original_intro + "\n")
    final_intro = "\n\n".join(previous.prior.marked_text(p) for p in intro)
    (OUT / "revised_introduction.txt").write_text(final_intro + "\n")
    with (OUT / "new_references_and_claims.csv").open("w", newline="") as f:
        fields = ["reference_number", "title", "status", "date", "source_url", "claim", "boundary"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for e in entries:
            writer.writerow({"reference_number": mapping[e["id"]], **{k: e[k] for k in fields if k != "reference_number"}})
    with (OUT / "reference_renumbering.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["source_reference_or_new_id", "revised_reference", "reference"])
        for i in sorted(bibliography):
            writer.writerow([i, mapping.get(i, "uncited after introduction rewrite"), text(bibliography[i])])
    print(json.dumps({k: report[k] for k in ["new_reference_count", "new_2026_references", "reference_total", "introduction_words"]}, indent=2))
    print(TARGET)
    print(MARKED)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fetch-only", action="store_true")
    main(parser.parse_args().fetch_only)
