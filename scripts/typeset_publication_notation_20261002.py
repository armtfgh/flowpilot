#!/usr/bin/env python3
"""Normalize editable scientific notation without rebuilding Word documents."""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re
from zipfile import ZipFile, ZIP_DEFLATED

from lxml import etree as E

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "manuscript"
OUT = BASE / "notation_revision_20261002"
SOURCES = {
    "manuscript": BASE / "FlowPilot_manuscript_october2_corrected.docx",
    "ESI": BASE / "FlowPilot_ESI_october2_council_discussions.docx",
}
W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
M = "{http://schemas.openxmlformats.org/officeDocument/2006/math}"
NS = {"w": W[1:-1], "m": M[1:-1]}
XMLSPACE = "{http://www.w3.org/XML/1998/namespace}space"
BASE_VAR = ("baseline", True)
SUB = ("subscript", False)
SUP = ("superscript", False)
UPRIGHT = ("baseline", False)
ALIASES = {}


def aliases(words, base, suffix):
    for word in words.split():
        ALIASES[word] = [(base, BASE_VAR), (suffix, SUB)]


aliases("Q_liq Q_liquid Qliq Qliquid QL Q_L", "Q", "liq")
aliases("V_liq V_liquid Vliq Vliquid VL V_L", "V", "liq")
aliases("V_R V_reactor VR", "V", "R")
aliases("Q_gas_STP Qg,STP Qgas,STP Q_g,STP", "Q", "gas,STP")
aliases("Q_gas_ch Qg,ch Qgas,ch Q_g,ch", "Q", "gas,ch")
aliases("τ_liq τ_liquid τliq τliquid tau_liq tau_liquid tau_l τ_l", "\u03c4", "liq")
aliases("τ_inlet τinlet tau_inlet", "\u03c4", "inlet")
aliases("t_STP tSTP", "t", "STP")
aliases("T_STP TSTP", "T", "STP")
aliases("P_STP PSTP", "P", "STP")
aliases("P_abs Pabs", "P", "abs")
aliases("T_ch Tch", "T", "ch")
aliases("V_m_STP Vm,STP", "V", "m,STP")
aliases("C_lim Clim", "C", "lim")
aliases("Q_lim Qlim", "Q", "lim")
aliases("y_active yactive", "y", "active")
aliases("A_wall Awall", "A", "wall")
for letter in "AB":
    aliases(f"Q{letter} Q_{letter}", "Q", letter)
for index in (1, 2):
    for symbol in "QV\u03c4":
        aliases(f"{symbol}{index} {symbol}_{index}", symbol, str(index))
    aliases(f"tau{index} tau_{index}", "\u03c4", str(index))
TOKEN = re.compile(r"(?<![\w])(?:" + "|".join(re.escape(k) for k in sorted(ALIASES, key=len, reverse=True)) + r")(?![\w])")
BARE_TAU = re.compile(r"(?<![\w])(?:tau|\u03c4)(?![\w])")
FORMULAS = [
    "C11H14NO2S", "C11H13NO2SK", "C15H15N2O3", "C15H14N2O3Na",
    "CDCl3", "DMSO-d6", "KMnO4", "CF3", "PF6", "TiO2", "Al2O3", "CO2", "H2O",
    "O2", "H2", "N2", "O\u2082", "H\u2082", "N\u2082", "CO\u2082",
]
CHEM = re.compile(r"(?<![\w])(?:" + "|".join(sorted(map(re.escape, FORMULAS), key=len, reverse=True)) + r")(?![\w])")
ISOTOPE = re.compile(r"(?<![\w])(?:1H|13C|19F|31P)(?![\w])")
UNITS = re.compile(r"(?<![\w])(?:mmol|mol|mL|L|min|ms|s|h|kg|mg|g|mW|W|kJ|J|K|bar|MPa|Pa|cm|mm|m)(?:[\-\u2212][123]|[\u207b][\u00b9\u00b2\u00b3]|[23\u00b2\u00b3])(?![\w])")
PROTECTED = re.compile(
    r"https?://\S+|\bN2-[A-F0-9]{8}\b|\b(?:Q-[A-Z]+-\d+|UO-\d+)\b|"
    r"\b[A-Za-z0-9_.-]+\.(?:jsonl?|csv|docx|py|md|pdf|png|svg)\b"
)
UNIFORM_STYLES = re.compile(r"(?<![\w])(?:Pe|Re|Da|UA|\u1e45)(?![\w])")
SCI_NOTE = (
    "Notation convention. Italic symbols denote physical quantities; descriptive subscripts are upright. "
    "Q_liq denotes liquid volumetric flow rate, V_R the reactor volume and V_liq the liquid holdup volume. "
    "The inlet/STP index \u03c4_inlet = V_R/(Q_liq + Q_gas_STP) is distinct from liquid residence time \u03c4_liq. "
    "Liquid-flow and liquid-holdup symbols use a consistent descriptive subscript; reactor volume and liquid holdup are not interchangeable. "
    "Literal software field names, identifiers and paths retain their original spelling."
)
QUOTE_NOTE = (
    "Typography of reproduced reports. Scientific symbols, chemical formulas and unit exponents in the displayed council reports "
    "are normalized for publication. Their numerical values and scientific wording are retained, including the errors identified "
    "in the interpretation notes. Exact original strings remain available in the unchanged raw logs and full-council companion."
)


def plain(node):
    return "".join(node.xpath(".//w:t/text()", namespaces=NS))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(path, obj):
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def normalize_numeric_text(value):
    return value.translate(str.maketrans("\u2080\u2081\u2082\u2083\u2084\u2085\u2086\u2087\u2088\u2089\u207b\u00b9\u00b2\u00b3", "0123456789\u2212123"))


def digits(value):
    # Capture every digit, including subscripts/isotopes, to protect scientific
    # values without confusing a digit with a changed mathematical label.
    return "".join(re.findall(r"[0-9]", normalize_numeric_text(value)))


def properties(run, style):
    rp = run.find(W + "rPr")
    if rp is None:
        rp = E.Element(W + "rPr"); run.insert(0, rp)
    for name, value in [("vertAlign", style[0]), ("i", "1" if style[1] else "0"), ("iCs", "1" if style[1] else "0")]:
        for old in list(rp.findall(W + name)):
            rp.remove(old)
        E.SubElement(rp, W + name, {W + "val": value})
    # Preserve font family, size, bold, citation properties and other run styles.


def operations(value):
    protected = [m.span() for m in PROTECTED.finditer(value)]
    ops = []

    def add(start, end, replacement, kind):
        if any(start < b and end > a for a, b in protected):
            return
        if any(start < b and end > a for a, b, *_ in ops):
            return
        ops.append((start, end, replacement, kind))

    for m in TOKEN.finditer(value):
        add(*m.span(), ALIASES[m[0]], "scientific_symbol")
    for m in BARE_TAU.finditer(value):
        add(*m.span(), [("\u03c4", BASE_VAR)], "greek_tau")
    for m in CHEM.finditer(value):
        ss = normalize_numeric_text(m[0])
        replacement = [(piece, SUB if piece.isdigit() else UPRIGHT) for piece in re.findall(r"\d+|\D+", ss)]
        add(*m.span(), replacement, "chemical_formula")
    for m in ISOTOPE.finditer(value):
        # Peak integrations such as "(d, J = 8.4 Hz, 1H)" count protons;
        # they are not isotope labels like "1H NMR" or "13C{1H}".
        if (m[0] == "1H" and re.search(r",\s*$", value[:m.start()])
                and re.match(r"\s*[);,]", value[m.end():])):
            continue
        a, b = re.fullmatch(r"(\d+)([A-Za-z]+)", m[0]).groups()
        add(*m.span(), [(a, SUP), (b, UPRIGHT)], "isotope")
    for m in UNITS.finditer(value):
        unit, exponent = re.fullmatch(r"(.+?)([\-\u2212][123]|\u207b[\u00b9\u00b2\u00b3]|[23\u00b2\u00b3])", m[0]).groups()
        exponent = normalize_numeric_text(exponent).replace("-", "\u2212")
        add(*m.span(), [(unit, UPRIGHT), (exponent, SUP)], "unit_exponent")
    for m in re.finditer(r"(?<=\))2(?=\(dtbpy\))", value):
        add(*m.span(), [("2", SUB)], "ligand_multiplicity")
    for m in re.finditer(r"(?<=\])\+(?![\w])", value):
        add(*m.span(), [("+", SUP)], "ion_charge")
    for m in UNIFORM_STYLES.finditer(value):
        add(*m.span(), [(m[0], BASE_VAR)], "quantity_italic")
    # Bare Latin symbols are formatted only inside explicit expressions. This
    # avoids changing author initials, degrees C or stream labels in prose.
    for m in re.finditer(r"(?<![\w])(?:V/Q|C\s*[\u00d7 ]\s*Q|U\s+A)(?![\w])", value):
        replacement = [(c, BASE_VAR if c.isalpha() else UPRIGHT) for c in m[0]]
        add(*m.span(), replacement, "expression")
    for m in re.finditer(r"(?<![\w])(?:V|Q|C|U|T|P|n|J)(?=\s*=)", value):
        add(*m.span(), [(m[0], BASE_VAR)], "quantity_italic")
    for m in re.finditer(r"(?<![\w])V(?=/(?:Q|\(\s*Q))", value):
        add(*m.span(), [(m[0], BASE_VAR)], "quantity_italic")
    return sorted(ops)


def text_nodes(p):
    return [t for t in p.xpath(".//w:r/w:t", namespaces=NS)
            if next(t.iterancestors(W + "p"), None) is p
            and not t.xpath("ancestor::w:drawing | ancestor::w:pict", namespaces=NS)]


def typeset_paragraph(p):
    ts = text_nodes(p)
    value = "".join(t.text or "" for t in ts)
    # JSON contracts and literal fenced code remain exact. Normal prose can
    # contain identifiers: the allowlisted notation patterns do not match them.
    if value.lstrip().startswith(("{", "}", '"', "```")):
        return []
    # A superscript reference can directly follow a variable without a space.
    # Mask references for matching, but retain their runs and digits exactly.
    scientific_spans = [m.span() for pattern in (ISOTOPE, UNITS) for m in pattern.finditer(value)]
    match_chars = list(value)
    offset = 0
    for t in ts:
        token = t.text or ""
        end = offset + len(token)
        if (re.fullmatch(r"[0-9]+(?:[,\u2013-][0-9]+)*", token)
                and t.getparent().xpath('./w:rPr/w:vertAlign[@w:val="superscript"]', namespaces=NS)
                and not any(offset < b and end > a for a, b in scientific_spans)):
            match_chars[offset:end] = " " * len(token)
        offset = end
    ops = operations("".join(match_chars))
    if not ops:
        return []
    slots = [[(c, None)] for c in value]
    for start, end, replacement, kind in ops:
        slots[start] = replacement
        for i in range(start + 1, end):
            slots[i] = []
    offset = 0
    for t in ts:
        run = t.getparent()
        old = t.text or ""
        chunk = slots[offset:offset + len(old)]
        offset += len(old)
        if all(x == [(c, None)] for x, c in zip(chunk, old)):
            continue
        if any(c.tag not in {W + "rPr", W + "t"} for c in run) or len(run.findall(W + "t")) != 1:
            raise ValueError("Unsafe mixed run; preserve it for manual review: " + value)
        groups = []
        for cell in chunk:
            for content, style in cell:
                if groups and groups[-1][1] == style:
                    groups[-1] = (groups[-1][0] + content, style)
                else:
                    groups.append((content, style))
        parent = run.getparent()
        pos = parent.index(run)
        for content, style in groups:
            if not content:
                continue
            new = deepcopy(run)
            nt = new.find(W + "t")
            nt.text = content; nt.set(XMLSPACE, "preserve")
            if style is not None:
                properties(new, style)
            parent.insert(pos, new); pos += 1
        parent.remove(run)
    result = "".join(t.text or "" for t in text_nodes(p))
    expected = value
    for start, end, replacement, _ in reversed(ops):
        expected = expected[:start] + "".join(s for s, _ in replacement) + expected[end:]
    assert result == expected
    assert digits(value) == digits(result), (value, result)
    return [{"kind": kind, "before": value[a:b], "after": "".join(s for s, _ in new)} for a, b, new, kind in ops]


def upright_math_descriptors(doc):
    changed = []
    for sub in doc.iter(M + "sub"):
        runs = list(sub.findall(M + "r"))
        values = ["".join(r.xpath("./m:t/text()", namespaces=NS)) for r in runs]
        joined = "".join(values)
        matches = list(re.finditer(r"FlowPilot|one-shot|wall|rem|gen|STP|liq|inlet|abs|lm|th", joined))
        offset = 0
        for run, value in zip(runs, values):
            if value and any(offset >= m.start() and offset + len(value) <= m.end() for m in matches):
                rp = run.find(M + "rPr")
                if rp is None:
                    rp = E.Element(M + "rPr"); run.insert(0, rp)
                for old in list(rp.findall(M + "sty")):
                    rp.remove(old)
                E.SubElement(rp, M + "sty", {M + "val": "p"})
                changed.append(value)
            offset += len(value)
    return changed


def note_like(template, value):
    p = E.Element(W + "p")
    if template.find(W + "pPr") is not None:
        pp = deepcopy(template.find(W + "pPr"))
        for tag in ["keepNext", "pageBreakBefore"]:
            for node in list(pp.findall(W + tag)):
                pp.remove(node)
        p.append(pp)
    r = E.SubElement(p, W + "r")
    rp = E.SubElement(r, W + "rPr")
    E.SubElement(rp, W + "rFonts", {W + "ascii": "Times New Roman", W + "hAnsi": "Times New Roman"})
    E.SubElement(rp, W + "sz", {W + "val": "22"})
    E.SubElement(rp, W + "szCs", {W + "val": "22"})
    E.SubElement(rp, W + "b", {W + "val": "0"})
    E.SubElement(r, W + "t").text = value
    return p


def process(label, source):
    with ZipFile(source) as z:
        parts = {k: z.read(k) for k in z.namelist()}
    original_parts = dict(parts)
    doc = E.fromstring(parts["word/document.xml"])
    body = doc.find(W + "body")
    changes = []
    if label == "ESI":
        t = next(p for p in body.findall(W + "p") if plain(p).startswith("Liquid component molar delivery"))
        t.addprevious(note_like(t, SCI_NOTE))
        t = next(p for p in body.findall(W + "p") if plain(p).startswith("Quoted or transcribed model statements"))
        t.addnext(note_like(t, QUOTE_NOTE))
    for i, p in enumerate(doc.iter(W + "p")):
        if p.xpath("ancestor::w:sdt", namespaces=NS):
            continue
        before = plain(p)
        edits = typeset_paragraph(p)
        if edits:
            changes.append({"paragraph": i, "before": before, "after": plain(p), "edits": edits})
    math_changes = upright_math_descriptors(doc)
    # Keep property order valid while leaving unrelated runs/paragraphs intact.
    from revise_layout_discussion_20260928 import order_properties
    order_properties(doc)
    parts["word/document.xml"] = E.tostring(doc, xml_declaration=True, encoding="UTF-8", standalone=True)
    target = BASE / f"FlowPilot_{label}_october2_notation_corrected.docx"
    with ZipFile(target, "w", ZIP_DEFLATED) as z:
        for name, value in parts.items():
            z.writestr(name, value)
    assert all(parts[k] == v for k, v in original_parts.items() if k != "word/document.xml")
    counts = Counter(e["kind"] for c in changes for e in c["edits"])
    dump(OUT / f"{label}_changes.json", changes)
    return {"source": str(source.relative_to(ROOT)), "source_sha256": sha(source),
            "output": str(target.relative_to(ROOT)), "changed_paragraphs": len(changes),
            "operations": dict(counts), "math_descriptors": math_changes,
            "other_package_parts_byte_identical": True}


def main():
    OUT.mkdir(exist_ok=True)
    reports = [process(label, source) for label, source in SOURCES.items()]
    dump(OUT / "manifest.json", reports)
    print(json.dumps(reports, indent=2))


if __name__ == "__main__":
    main()
