"""
FLORA Visualization — Shared data loader and normalizer.
Import this in any figure script:
    from data_loader import load_records, RECORDS_DIR
"""

import json
import re
from pathlib import Path
from collections import Counter

# ── Path config ──────────────────────────────────────────────────────────────
RECORDS_DIR = Path(__file__).resolve().parent.parent / "flora_translate" / "data" / "records"


# ── Normalization maps ────────────────────────────────────────────────────────
# Each key is a lowercase canonical label; values are substrings to match (case-insensitive).
REACTION_CLASS_MAP = {
    "Photoredox Catalysis":      ["photoredox"],
    "Photocycloaddition":        ["photocycloaddition"],
    "Photochem. Oxidation":      ["photochem", "photocatal", "photo-oxid", "photodegr"],
    "Cross-Coupling":            ["cross-coupl", "cross coupl", "suzuki", "heck", "negishi", "buchwald"],
    "Hydrogenation":             ["hydrogenat"],
    "Oxidation":                 ["oxidation", "oxidative"],
    "Reduction":                 ["reduction", "reductive"],
    "Condensation":              ["condensat"],
    "Polymerization":            ["polymer"],
    "Biocatalysis":              ["biocatal", "enzymat"],
    "Electrochemistry":          ["electrochem", "electrooxid", "electroreduct"],
    "Glycosylation":             ["glycosyl"],
    "Nitration":                 ["nitrat"],
    "Halogenation":              ["halogen", "fluorin", "chlorin", "bromina"],
    "Other / Unknown":           [],  # catch-all
}

REACTOR_TYPE_MAP = {
    "Microreactor":              ["microreact", "microfluid"],
    "Capillary Reactor":         ["capillar"],
    "Coil Reactor":              ["coil"],
    "Packed-Bed Reactor":        ["packed", "packed-bed", "fixed bed"],
    "Tubular Reactor":           ["tubular", "tube reactor"],
    "Continuous-Flow Reactor":   ["continuous flow", "continuous-flow", "flow react", "^flow$"],
    "Photoreactor":              ["photoreact"],
    "Chip / EWOD":               ["chip", "ewod", "pdms"],
    "Other / Unknown":           [],
}

REACTOR_MATERIAL_MAP = {
    "Stainless Steel":           ["stainless"],
    "PTFE":                      ["ptfe", "teflon"],
    "PFA":                       ["^pfa$", "perfluoroalkoxy"],
    "FEP":                       ["^fep$", "fluorinated ethylene"],
    "Glass":                     ["glass", "borosilicate", "quartz"],
    "PDMS":                      ["pdms"],
    "Silicon":                   ["silicon"],
    "PMMA":                      ["pmma", "acrylic"],
    "Other / Unknown":           [],
}

BOND_MAP = {
    "C–C":   ["c-c", "c–c", "c−c", "c\u2013c"],
    "C–N":   ["c-n", "c–n", "c\u2013n"],
    "C–O":   ["c-o", "c–o", "c\u2013o"],
    "C–H":   ["c-h", "c–h", "c\u2013h"],
    "C=C":   ["c=c"],
    "C–S":   ["c-s", "c–s"],
    "N–N":   ["n-n", "n–n"],
    "Amide": ["amide"],
    "Other / Unknown": [],
}

LIGHT_SOURCE_MAP = {
    "LED (blue/violet)":  ["blue led", "violet led", "purple led", "390", "395", "400", "405", "410", "450", "460", "465", "470"],
    "LED (white/other)":  ["white led", "green led", "red led", "led"],
    "UV Lamp":            ["uv lamp", "uv light", "uv irrad", "hg lamp", "mercury"],
    "Xenon Lamp":         ["xenon", "xe lamp"],
    "Solar Simulator":    ["solar"],
    "Microwave":          ["microwave"],
    "None / Dark":        [],
}


# ── Core loader ───────────────────────────────────────────────────────────────

def load_records(records_dir: Path = RECORDS_DIR) -> list[dict]:
    """Load all non-failed JSON records from records_dir."""
    records = []
    for path in sorted(records_dir.glob("*.json")):
        if "_FAILED" in path.name:
            continue
        try:
            d = json.loads(path.read_text())
            if isinstance(d, dict):
                records.append(d)
        except Exception:
            pass
    return records


# ── Field extractors ──────────────────────────────────────────────────────────

def _normalize(value: str, mapping: dict) -> str:
    """Map a raw string to its canonical label using substring matching."""
    v = (value or "").strip().lower()
    if not v:
        return "Other / Unknown"
    for label, patterns in mapping.items():
        if label == "Other / Unknown":
            continue
        for pat in patterns:
            if re.search(pat, v, re.IGNORECASE):
                return label
    return "Other / Unknown"


def get_reaction_classes(records: list[dict]) -> Counter:
    raw = [r.get("chemistry", {}).get("reaction_class", "") or "" for r in records]
    return Counter(_normalize(v, REACTION_CLASS_MAP) for v in raw)


def get_reactor_types(records: list[dict]) -> Counter:
    raw = [r.get("reactor", {}).get("type", "") or "" for r in records]
    return Counter(_normalize(v, REACTOR_TYPE_MAP) for v in raw)


def get_reactor_materials(records: list[dict]) -> Counter:
    raw = [r.get("reactor", {}).get("material", "") or "" for r in records]
    return Counter(_normalize(v, REACTOR_MATERIAL_MAP) for v in raw)


def get_bond_types(records: list[dict]) -> Counter:
    raw = [r.get("chemistry", {}).get("bond_formed", "") or "" for r in records]
    return Counter(_normalize(v, BOND_MAP) for v in raw)


def get_light_sources(records: list[dict]) -> Counter:
    raw = [r.get("light_source", {}).get("type", "") or "" for r in records]
    return Counter(_normalize(v, LIGHT_SOURCE_MAP) for v in raw)


def get_pump_inlets(records: list[dict]) -> Counter:
    out = Counter()
    for r in records:
        val = r.get("pump", {}).get("number_of_inlets")
        try:
            n = int(float(val))
            out[n] += 1
        except (TypeError, ValueError):
            pass
    return out


def get_yields(records: list[dict]) -> list[float]:
    yields = []
    for r in records:
        y = r.get("flow_optimized", {}).get("yield_percent")
        try:
            v = float(y)
            if 0 <= v <= 100:
                yields.append(v)
        except (TypeError, ValueError):
            pass
    return yields


def get_residence_times(records: list[dict]) -> list[float]:
    times = []
    for r in records:
        t = r.get("flow_optimized", {}).get("residence_time_min")
        try:
            v = float(t)
            if 0 < v < 1000:
                times.append(v)
        except (TypeError, ValueError):
            pass
    return times


# ── LLM-classified field accessors ───────────────────────────────────────────
# These require running llm_classifier.classify_all() first.
# They fall back gracefully to raw normalization if cache is missing.

def get_classified_counts(field: str, records: list[dict] | None = None) -> Counter:
    """
    Return a Counter for an LLM-classified field.
    field: 'chemistry_class' | 'reactor_type' | 'reactor_material' | 'bond_type' | 'light_source'
    Requires classification cache (run llm_classifier.classify_all() once).
    """
    try:
        from llm_classifier import load_classifications, _get_record_id, CHEMISTRY_CLASSES
    except ImportError:
        return Counter()

    if records is None:
        records = load_records()

    cache = load_classifications()
    if not cache:
        return Counter()

    values = []
    for r in records:
        rec_id = _get_record_id(r)
        clf = cache.get(rec_id, {})
        values.append(clf.get(field, "Other"))
    return Counter(values)
