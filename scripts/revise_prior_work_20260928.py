"""Update the direct-prior-work comparison and bold figure/table cross-references."""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import re
from copy import deepcopy
from datetime import date
from hashlib import sha256
from pathlib import Path
from urllib.parse import quote
from zipfile import ZipFile

import requests
from bs4 import BeautifulSoup
from lxml import etree as E

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "manuscript"
OUT = BASE / "prior_work_revision_20260928"
MAIN_SOURCE = BASE / "manuscript_introduction_revised_20260923.docx"
ESI_SOURCE = BASE / "esi_revised_20260922.docx"
spec = importlib.util.spec_from_file_location("intro_tools", ROOT / "scripts/revise_introduction_20260923.py")
intro_tools = importlib.util.module_from_spec(spec)
spec.loader.exec_module(intro_tools)
previous = intro_tools.previous
W, NS = intro_tools.W, intro_tools.NS
text = intro_tools.text

ITEM = r"S?\d+(?:[a-z](?:\s*[-\u2013]\s*[a-z])?|\([a-z](?:\s*[-\u2013]\s*[a-z])?\))?"
LABEL = re.compile(r"\b(?:Figures?|Figs?\.?|Tables?)\s+" + ITEM + r"(?:(?:\s*[-\u2013]\s*|\s*,\s*(?:and\s+)?|\s+and\s+)" + ITEM + r")*", re.IGNORECASE)


def fetch_metadata():
    entries = json.loads((OUT / "new_references.json").read_text())
    cache = OUT / "source_metadata"
    cache.mkdir(exist_ok=True)
    for entry in entries:
        path = cache / f"reference_{entry['id']}.json"
        if path.exists():
            data = json.loads(path.read_text())
        elif "doi" in entry:
            response = requests.get("https://api.crossref.org/works/" + quote(entry["doi"], safe=""), timeout=45)
            response.raise_for_status()
            payload = response.json()["message"]
            keys = ["DOI", "title", "author", "container-title", "volume", "issue", "page", "article-number", "published", "published-online", "published-print", "publisher", "URL", "type", "license"]
            data = {k: payload[k] for k in keys if k in payload}
            path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
        else:
            url = "https://arxiv.org/abs/" + entry["arxiv"]
            response = requests.get(url, timeout=45)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, "html.parser")
            values = {}
            for meta in soup.select("meta[name^='citation_']"):
                if meta["name"] != "citation_abstract":
                    values.setdefault(meta["name"], []).append(meta.get("content", ""))
            data = {"source_url": url, "metadata": values}
            path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
        actual_title = (data["title"] if "doi" in entry else data["metadata"]["citation_title"])[0]
        assert intro_tools.normalize(actual_title) == intro_tools.normalize(entry["title"]), actual_title
        if "doi" in entry:
            dates = data.get("published-online", data.get("published", {}))["date-parts"][0]
            assert dates[0] == entry["year"]
            assert date(*(dates + [1] * (3 - len(dates)))) <= date(2026, 9, 28)
            entry["date"] = "-".join(str(v).zfill(2) for v in dates)
            entry["source_url"] = "https://doi.org/" + entry["doi"]
        else:
            entry["date"] = data["metadata"]["citation_online_date"][0].replace("/", "-")
            assert date.fromisoformat(entry["date"]) <= date(2026, 9, 28)
            entry["source_url"] = data["source_url"]
        entry["metadata"] = data
        print(f"Verified {entry['id']}: {actual_title}", flush=True)
    (OUT / "verified_sources.json").write_text(json.dumps(entries, indent=2, ensure_ascii=False) + "\n")
    return entries


def paragraph_text_nodes(p):
    return [n for n in p.iter(W + "t") if next(n.iterancestors(W + "p"), None) is p]


def bold_references(doc):
    """Split only affected text runs; retain hyperlinks, field codes and drawings."""
    audit = []
    for p_index, p in enumerate(doc.iter(W + "p")):
        ts = paragraph_text_nodes(p)
        value = "".join(t.text or "" for t in ts)
        matches = list(LABEL.finditer(value))
        if not matches:
            continue
        mask = [False] * len(value)
        for m in matches:
            mask[m.start():m.end()] = [True] * (m.end() - m.start())
            audit.append({"paragraph": p_index, "label": m[0], "paragraph_text": value})
        positions = {}
        offset = 0
        for t in ts:
            positions[t] = offset
            offset += len(t.text or "")
        runs = list(dict.fromkeys(t.getparent() for t in ts))
        for r in runs:
            assert r.tag == W + "r"
            affected = any(any(mask[positions[t]:positions[t]+len(t.text or "")]) for t in r.findall(W + "t") if t in positions)
            if not affected:
                continue
            replacements = []
            for child in r:
                if child.tag == W + "rPr":
                    continue
                chunks = []
                if child.tag == W + "t" and child in positions:
                    s = child.text or ""
                    flags = mask[positions[child]:positions[child] + len(s)]
                    start = 0
                    for stop in range(1, len(s) + 1):
                        if stop == len(s) or flags[stop] != flags[start]:
                            node = deepcopy(child)
                            node.text = s[start:stop]
                            node.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")
                            chunks.append((node, flags[start]))
                            start = stop
                    if not s:
                        chunks.append((deepcopy(child), False))
                else:
                    chunks.append((deepcopy(child), False))
                for node, make_bold in chunks:
                    new = E.Element(W + "r", dict(r.attrib))
                    rp = deepcopy(r.find(W + "rPr"))
                    if rp is None and make_bold:
                        rp = E.Element(W + "rPr")
                    if rp is not None:
                        if make_bold:
                            for tag in ["b", "bCs"]:
                                setting = rp.find(W + tag)
                                if setting is None:
                                    setting = E.SubElement(rp, W + tag)
                                setting.set(W + "val", "1")
                        new.append(rp)
                    new.append(node)
                    replacements.append(new)
            if replacements:
                replacements[-1].tail = r.tail
                for new in replacements:
                    r.addprevious(new)
                r.getparent().remove(r)
        assert value == "".join(t.text or "" for t in paragraph_text_nodes(p))
    return audit


def clean_paragraph(template, value):
    p = deepcopy(template)
    for c in list(p):
        if c.tag != W + "pPr":
            p.remove(c)
    for part in re.split(r"(\[\[.*?\]\])", value):
        if not part:
            continue
        rp = intro_tools.properties()
        if part.startswith("[["):
            part = part[2:-2]
            assert intro_tools.expand(part)
            E.SubElement(rp, W + "vertAlign", {W + "val": "superscript"})
        previous.prior.append_run(p, part, rp)
    return p


def save_pair(package, stem, changed_nodes, citation_changes=()):
    clean = BASE / f"{stem}_revised_20260928.docx"
    marked = BASE / f"{stem}_revised_20260928_marked.docx"
    audit = bold_references(package.doc)
    intro_tools.write_package(package.parts, package.doc, clean)
    doc = deepcopy(package.doc)
    tree = package.doc.getroottree()
    for node in changed_nodes:
        if node.getroottree().getroot() is not package.doc:
            continue
        matches = doc.xpath(tree.getpath(node), namespaces=NS)
        assert len(matches) == 1
        intro_tools.highlight(matches[0])
    # Superscript citation runs are not split by the figure/table label formatter.
    for run in citation_changes:
        if run.getroottree().getroot() is package.doc:
            found = doc.xpath(tree.getpath(run), namespaces=NS)
            if found:
                intro_tools.highlight(found[0])
    intro_tools.write_package(package.parts, doc, marked)
    with (OUT / f"{stem}_bold_references.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["paragraph", "label", "paragraph_text"])
        writer.writeheader()
        writer.writerows(audit)
    return {"clean": str(clean.relative_to(ROOT)), "marked": str(marked.relative_to(ROOT)), "bold_labels": len(audit)}


def manuscript(entries):
    m = previous.Package(MAIN_SOURCE)
    headings = {text(p): p for p in m.ps if text(p) in {"INTRODUCTION", "RESULTS", "REFERENCES"}}
    old_intro = intro_tools.paragraphs_between(m.body, headings["INTRODUCTION"], headings["RESULTS"])
    changed = []
    draft = (OUT / "introduction_replacement.txt").read_text().strip().replace("El Agente Grafico", "El Agente Gr\u00e1fico").replace("La Agente Optima", "La Agente \u00d3ptima")
    for value in draft.split("\n\n"):
        p = clean_paragraph(old_intro[0], value)
        headings["RESULTS"].addprevious(p)
        changed.append(p)
    for p in old_intro[2:]:
        m.body.remove(p)
    original_ref_nodes = [p for p in list(m.body)[list(m.body).index(headings["REFERENCES"]) + 1:] if p.tag == W + "p" and text(p).strip()]
    assert len(original_ref_nodes) == 59
    bibliography = {i: p for i, p in enumerate(original_ref_nodes, 1)}
    original_refs = {i: text(p) for i, p in bibliography.items()}
    for entry in entries:
        bibliography[entry["id"]] = intro_tools.bibliography_paragraph(entry, original_ref_nodes[0])
        changed.append(bibliography[entry["id"]])
    old_positioning = next(p for p in m.ps if text(p).startswith("Related AI systems address complementary parts"))
    positioning = [
        "The relevant prior art includes flow-pattern-guided microreactor design, experimentally supported synthesis development, continuous-flow optimization, simulator configuration and deterministic constraint checking.[[60-65]] Against this background, FlowPilot addresses a reaction-to-configuration task: connecting the chemistry sequence to explicit stream compositions, molar feeds, stage calculations and laboratory equipment assignments in one record. This complements process-optimization and materials-development agents[[57,58]] through its emphasis on coupled stream, stage and equipment consistency. ESI Section 11 provides a source-linked comparison of task scope, outputs and validation evidence.",
        "The one-shot comparisons in Figure 4 do not constitute head-to-head tests against Chat-microreactor, LLM-RDF, SapoMind, the Text-to-Simulation workflow, CeProAgents or CAAF. Their benchmarks and physical platforms differ, and an unreported feature cannot be treated as an absent capability. Figures 5 and 6 will instead assess whether FlowPilot's proposals can be implemented as connected processes and produce useful chemical outcomes. Inventory assignment and numerical closure support that assessment but do not replace commissioning, kinetic measurements or experimental safety review."
    ]
    for value in positioning:
        p = clean_paragraph(old_positioning, value)
        old_positioning.addprevious(p)
        changed.append(p)
    m.body.remove(old_positioning)
    amendments = [
        ("The first connected-process case couples", " The implementation test therefore links the position of oxygen addition and reactor-light compatibility to the actual connected train, rather than judging a proposed residence time in isolation."),
        ("The second case is a thermal, two-stage amidation", " This case links the proposed amine-addition point and feed stoichiometry to the changed flow entering Stage 2, allowing configuration consistency and final chemical performance to be assessed separately."),
        ("The translation workflow comprises standardized intake", " Table S36 (ESI Section 11) compares the input, output, engineering scope and validation evidence of six directly related systems with FlowPilot. It is a literature-based scope comparison, not an additional benchmark or evidence of performance superiority.")
    ]
    for prefix, addition in amendments:
        p = next(p for p in m.ps if text(p).startswith(prefix))
        m.revise(p, previous.prior.marked_text(p) + addition)
        changed.append(p)
    prose = list(m.body)[:list(m.body).index(headings["REFERENCES"])]
    order = []
    for p in prose:
        for _, ids in intro_tools.citation_runs(p):
            for i in ids:
                assert i in bibliography
                if i not in order:
                    order.append(i)
    mapping = {old: new for new, old in enumerate(order, 1)}
    updated_citations = []
    for p in prose:
        intro_tools.renumber(p, mapping, updated_citations)
    for p in original_ref_nodes:
        m.body.remove(p)
    cursor = headings["REFERENCES"]
    for i in order:
        cursor.addnext(bibliography[i])
        cursor = bibliography[i]
    info = save_pair(m, "manuscript", changed, updated_citations)
    info.update(reference_count=len(order), reference_mapping=mapping, removed_uncited_references={i: original_refs[i] for i in original_refs if i not in mapping}, changed_paragraphs=[text(p) for p in changed if p.tag == W + "p"])
    with (OUT / "reference_mapping.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["old_or_new_id", "revised_number", "reference"])
        for i in sorted(bibliography):
            writer.writerow([i, mapping.get(i, "no longer cited"), text(bibliography[i])])
    return info


def esi(entries):
    m = previous.Package(ESI_SOURCE)
    refs = next(p for p in m.ps if text(p) == "References")
    rows = json.loads((OUT / "comparison_rows.json").read_text())
    additions = [
        m.paragraph("Related flow-chemistry and process-design agents", "Heading1", page_break=True, keep=True),
        m.paragraph("This section compares FlowPilot with six studies selected for their direct relevance to flow-chemistry or chemical-process agents. Table S36 records the task, delivered output, quantitative engineering scope, equipment context and type of validation reported in the checked sources [7-12]. Journal articles, a conference paper and preprints are identified separately. The comparison is descriptive: it does not assign performance scores, infer absent capabilities from silence, or equate different benchmarks. Source versions were checked on 28 September 2026."),
        m.paragraph("Table S36. Task-level comparison of FlowPilot with directly related flow-chemistry and chemical-process agents. Panel I: inputs, delivered outputs and reported validation. Bracketed numbers refer to the ESI reference list.", keep=True),
        m.table(["System / source", "Input and purpose", "Delivered output", "Reported evidence"], [[r["system"], r["input"], r["output"], r["evidence"]] for r in rows], widths=[1.35, 1.65, 1.7, 1.8]),
        m.paragraph("Table S36 (continued). Panel II: engineering and equipment scope, and the appropriate comparison with FlowPilot. These are scope distinctions, not proof that a capability is exclusive to either system.", page_break=True, keep=True),
        m.table(["System / source", "Quantitative engineering", "Equipment / process scope", "Interpretation for comparison"], [[r["system"], r["engineering"], r["equipment"], r["comparison"]] for r in rows], widths=[1.35, 1.65, 1.5, 2.0]),
        m.paragraph("Interpretation and experimental boundary", "Heading2", keep=True),
        m.paragraph("A common input-output description does not establish identical capabilities. FlowPilot's proposed contribution is the reconciliation of reaction sequencing, feed composition, molar flow, stage-specific calculations and available equipment in a shared record. The scope comparison cannot establish priority for every feature or replace a controlled comparison against the cited systems. The architecture benchmark in Section 3 compares the recorded one-shot and FlowPilot configurations only; it should not be read as a ranking of these six external methods."),
        m.paragraph("For the prospective laboratory cases, implementation and chemical response must be reported separately. Figure 5 concerns a connected oxygen-free/oxygen-fed photochemical sequence; records should distinguish nominal gas bookkeeping, the actual assembled train and observed operation. Figure 6 concerns telescoped activation and amidation; feed concentrations, molar ratios, intermediate transfer and the downstream sum of liquid flows are part of the configuration to verify. Tables S31 and S34 retain the corresponding analytical reporting fields. XXX and YYY remain explicit placeholders, and no experimental outcome or new safety assurance is inferred from the literature comparison.")
    ]
    m.before(refs, additions)
    ref_template = m.ps[-1]
    new_refs = []
    for index, entry in enumerate(entries, 7):
        p = intro_tools.bibliography_paragraph(entry, ref_template)
        rp = E.Element(W + "r")
        rp.append(intro_tools.properties())
        t = E.SubElement(rp, W + "t")
        t.text = str(index) + ". "
        t.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")
        p.insert(1 if p.find(W + "pPr") is not None else 0, rp)
        new_refs.append(p)
    m.after(m.ps[-1], new_refs)
    result = save_pair(m, "esi", additions + new_refs)
    result.update(new_section=11, new_table="S36", added_references=list(range(7, 13)))
    return result


def main(fetch_only=False):
    entries = fetch_metadata()
    if fetch_only:
        return
    sources = {str(p.relative_to(ROOT)): sha256(p.read_bytes()).hexdigest() for p in [MAIN_SOURCE, ESI_SOURCE]}
    report = {"source_hashes": sources, "manuscript": manuscript(entries), "esi": esi(entries)}
    for path, expected in sources.items():
        assert sha256((ROOT / path).read_bytes()).hexdigest() == expected
    (OUT / "revision_audit.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"main_references": report["manuscript"]["reference_count"], "main_bold_labels": report["manuscript"]["bold_labels"], "esi_bold_labels": report["esi"]["bold_labels"]}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fetch-only", action="store_true")
    main(parser.parse_args().fetch_only)
