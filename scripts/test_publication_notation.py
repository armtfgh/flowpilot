"""Focused tests for run-preserving scientific typography."""
import unittest
from lxml import etree as E

from typeset_publication_notation_20261002 import (
    W, M, NS, plain, properties, typeset_paragraph, upright_math_descriptors,
)


def paragraph(*chunks):
    p = E.Element(W + "p")
    for value in chunks:
        r = E.SubElement(p, W + "r")
        rp = E.SubElement(r, W + "rPr")
        E.SubElement(rp, W + "rFonts", {W + "ascii": "Times New Roman"})
        E.SubElement(rp, W + "sz", {W + "val": "22"})
        E.SubElement(r, W + "t").text = value
    return p


def formatted(p, content, position, italic):
    return any(plain(r) == content and r.xpath(
        f'./w:rPr/w:vertAlign[@w:val="{position}"]', namespaces=NS
    ) and r.xpath(f'./w:rPr/w:i[@w:val="{1 if italic else 0}"]', namespaces=NS)
        for r in p.findall(W + "r"))


class TypographyTests(unittest.TestCase):
    def test_cross_run_symbol(self):
        p = paragraph("Use Q_", "liquid = 0.02 mL/min.")
        typeset_paragraph(p)
        self.assertEqual(plain(p), "Use Qliq = 0.02 mL/min.")
        self.assertTrue(formatted(p, "Q", "baseline", True))
        self.assertTrue(formatted(p, "liq", "subscript", False))

    def test_tau_and_volume_distinction(self):
        p = paragraph("tau_liq = V_liq/Q_liq; V_R; tau_inlet")
        typeset_paragraph(p)
        self.assertEqual(plain(p), "\u03c4liq = Vliq/Qliq; VR; \u03c4inlet")
        self.assertTrue(formatted(p, "R", "subscript", False))

    def test_software_fields_and_ids(self):
        s = "Q_gas_sccm flow_rate_mL_min pump_e_series_blue Q-GAS-003 N2-D1FC65D8"
        p = paragraph(s); typeset_paragraph(p)
        self.assertEqual(plain(p), s)

    def test_url_and_filename(self):
        s = "https://example.org/Q_liq tau_liq.json"
        p = paragraph(s); typeset_paragraph(p)
        self.assertEqual(plain(p), s)

    def test_literal_json(self):
        s = '{"Q_liq": 0.2, "gas": "O2"}'
        p = paragraph(s); self.assertEqual(typeset_paragraph(p), [])
        self.assertEqual(plain(p), s)

    def test_chemical_and_isotope(self):
        p = paragraph("O2 and 13C NMR; C15H15N2O3; Al2O3")
        typeset_paragraph(p)
        self.assertTrue(formatted(p, "13", "superscript", False))
        self.assertTrue(formatted(p, "2", "subscript", False))
        self.assertEqual(plain(p), "O2 and 13C NMR; C15H15N2O3; Al2O3")

    def test_unicode_unit(self):
        p = paragraph("mmol min\u207b\u00b9 and mW/cm2")
        typeset_paragraph(p)
        self.assertEqual(plain(p), "mmol min\u22121 and mW/cm2")
        self.assertTrue(formatted(p, "\u22121", "superscript", False))

    def test_nmr_integrations_are_not_isotopes(self):
        p = paragraph("1H NMR: 7.95 (d, J = 8.4 Hz, 1H), 7.77 (s, 1H); 13C{1H} NMR.")
        typeset_paragraph(p)
        self.assertIn("Hz, 1H)", plain(p))
        self.assertIn("(s, 1H)", plain(p))
        superscripts = p.xpath('.//w:r[w:rPr/w:vertAlign[@w:val="superscript"]]/w:t/text()', namespaces=NS)
        self.assertEqual(superscripts, ["1", "13", "1"])

    def test_non_scientific_numbers(self):
        s = "Figure 5, Table S10, Qwen3.6 and Candidate C3 at 95 \u00b0C"
        p = paragraph(s); typeset_paragraph(p)
        self.assertEqual(plain(p), s)
        self.assertFalse(p.xpath('.//w:vertAlign', namespaces=NS))

    def test_bold_and_citation_preserved(self):
        p = paragraph("Q_liq", "57")
        E.SubElement(p[0].find(W + "rPr"), W + "b")
        E.SubElement(p[1].find(W + "rPr"), W + "vertAlign", {W + "val": "superscript"})
        typeset_paragraph(p)
        self.assertEqual(plain(p), "Qliq57")
        self.assertTrue(p[-1].xpath('./w:rPr/w:vertAlign[@w:val="superscript"]', namespaces=NS))

    def test_idempotence(self):
        p = paragraph("Q_liquid = 0.02; tau_inlet = 181.82; O2, min-1")
        typeset_paragraph(p); first = E.tostring(p)
        typeset_paragraph(p); second = E.tostring(p)
        self.assertEqual(first, second)

    def test_math_descriptor_not_variable_index(self):
        root = E.Element(M + "oMath")
        for name in ["wall", "i"]:
            sub = E.SubElement(root, M + "sub")
            r = E.SubElement(sub, M + "r")
            E.SubElement(r, M + "t").text = name
        changed = upright_math_descriptors(root)
        self.assertEqual(changed, ["wall"])
        self.assertFalse(root[-1].xpath('.//m:sty', namespaces=NS))


if __name__ == "__main__":
    unittest.main()
