"""Refresh only the cached ESI contents and heading anchors, preserving all other parts."""
from copy import deepcopy
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
import json
import re

import fitz
from lxml import etree as E

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "manuscript"
OUT = BASE / "revision_20260922"
W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
NS = {"w": W[1:-1]}


def text(n):
    return "".join(n.xpath(".//w:t/text()", namespaces=NS))


def normalized(t):
    return re.sub(r"[^a-z0-9]", "", t.lower())


def add_run(parent, value=None, tag="t", **attrs):
    r = E.SubElement(parent, W + "r")
    props = E.SubElement(r, W + "rPr")
    E.SubElement(props, W + "rFonts", {W + "ascii": "Times New Roman", W + "hAnsi": "Times New Roman"})
    E.SubElement(props, W + "sz", {W + "val": "22"})
    child = E.SubElement(r, W + tag, {W + k: v for k, v in attrs.items()})
    if value is not None:
        child.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")
        child.text = value


def refresh(path, pdf):
    with ZipFile(path) as z:
        parts = {name: z.read(name) for name in z.namelist()}
    doc = E.fromstring(parts["word/document.xml"])
    body = doc.find(W + "body")
    sdt = doc.xpath(".//w:sdt[w:sdtContent//w:instrText[contains(text(),' TOC ')]]", namespaces=NS)[0]
    content = sdt.find(W + "sdtContent")
    old_paras = list(content)
    template = {level: next(p for p in old_paras if p.xpath(f"./w:pPr/w:pStyle[@w:val='TOC{level}']", namespaces=NS)) for level in [1, 2]}
    headings = [p for p in body.findall(W + "p") if p.xpath("./w:pPr/w:pStyle[@w:val='Heading1' or @w:val='Heading2']", namespaces=NS)]
    pdf_text = [normalized(p.get_text()) for p in pdf]
    entries = []
    h1 = h2 = 0
    minimum = 2
    first_props = headings[0].find(W + "pPr")
    if first_props.find(W + "pageBreakBefore") is None:
        E.SubElement(first_props, W + "pageBreakBefore")
    bookmark_id = max([int(x.get(W + "id")) for x in doc.findall(".//" + W + "bookmarkStart")] + [0]) + 1
    for i, heading in enumerate(headings):
        level = int(heading.find(W + "pPr").find(W + "pStyle").get(W + "val")[-1])
        if level == 1:
            h1 += 1
            h2 = 0
            prefix = str(h1)
        else:
            h2 += 1
            prefix = f"{h1}.{h2}"
        title = text(heading)
        matches = [j for j, t in enumerate(pdf_text) if j >= minimum and normalized(title) in t]
        if title == "References":
            matches = matches[-1:]
        if not matches:
            raise ValueError(f"Cannot locate heading in rendered document: {title}")
        page_index = matches[0]
        minimum = page_index
        anchor = f"_Revision20260922_{i + 1}"
        for tag in ["bookmarkStart", "bookmarkEnd"]:
            for element in list(heading.findall(W + tag)):
                if element.get(W + "name", "").startswith("_Revision20260922_"):
                    old_id = element.get(W + "id")
                    heading.remove(element)
                    for end in list(heading.findall(W + "bookmarkEnd")):
                        if end.get(W + "id") == old_id:
                            heading.remove(end)
        begin = E.Element(W + "bookmarkStart", {W + "id": str(bookmark_id), W + "name": anchor})
        end = E.Element(W + "bookmarkEnd", {W + "id": str(bookmark_id)})
        heading.insert(1, begin)
        heading.append(end)
        bookmark_id += 1
        entry = E.Element(W + "p")
        entry.append(deepcopy(template[level].find(W + "pPr")))
        if i == 0:
            add_run(entry, tag="fldChar", fldCharType="begin", dirty="false")
            add_run(entry, ' TOC \\o "1-2" \\h \\z \\u ', tag="instrText")
            add_run(entry, tag="fldChar", fldCharType="separate")
        link = E.SubElement(entry, W + "hyperlink", {W + "anchor": anchor, W + "history": "1"})
        add_run(link, prefix)
        add_run(link, tag="tab")
        add_run(link, title)
        add_run(link, tag="tab")
        add_run(link, tag="fldChar", fldCharType="begin")
        add_run(link, f" PAGEREF {anchor} \\h ", tag="instrText")
        add_run(link, tag="fldChar", fldCharType="separate")
        add_run(link, str(page_index + 1))
        add_run(link, tag="fldChar", fldCharType="end")
        if i == len(headings) - 1:
            add_run(entry, tag="fldChar", fldCharType="end")
        entries.append((entry, {"section": prefix, "heading": title, "page": page_index + 1}))
    for p in list(content)[1:]:
        content.remove(p)
    for entry, _ in entries:
        content.append(entry)
    parts["word/document.xml"] = E.tostring(doc, xml_declaration=True, encoding="UTF-8", standalone=True)
    with ZipFile(path, "w", ZIP_DEFLATED) as z:
        for name, value in parts.items():
            z.writestr(name, value)
    return [row for _, row in entries]


if __name__ == "__main__":
    pdf = fitz.open(OUT / "rendered/esi_revised_20260922.pdf")
    rows = refresh(BASE / "esi_revised_20260922.docx", pdf)
    refresh(BASE / "esi_revised_20260922_marked.docx", pdf)
    (OUT / "contents_page_audit.json").write_text(json.dumps(rows, indent=2))
    print(f"Updated {len(rows)} ESI contents entries in clean and marked copies.")
