"""
FLORA Visualization — LLM-based paper classifier.

Uses GPT-4o-mini to read each JSON record and assign standardized canonical
labels for: chemistry_class, reactor_type, reactor_material, bond_type, light_source.

Results are cached in  visualization/cache/classifications.json
so you only pay for new papers on subsequent runs.

Jupyter usage (run once, then all figure scripts use the cache):
    from llm_classifier import classify_all, load_classifications
    stats = classify_all()          # processes only un-classified papers
    clf = load_classifications()    # dict keyed by source_pdf stem

Cost estimate: ~$0.001 per paper with GPT-4o-mini → ~$0.50 for 464 papers.
"""

import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from data_loader import load_records, RECORDS_DIR

# ── Paths ─────────────────────────────────────────────────────────────────────
CACHE_PATH = Path(__file__).parent / "cache" / "classifications.json"
CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)

# ── Canonical label sets ──────────────────────────────────────────────────────
CHEMISTRY_CLASSES = [
    "Photoredox Catalysis",
    "Heterogeneous Photocatalysis",
    "Photocycloaddition",
    "Thermal Synthesis",
    "Cross-Coupling",
    "Hydrogenation",
    "Oxidation / Reduction",
    "Electrochemistry",
    "Biocatalysis",
    "Polymer Synthesis",
    "Organocatalysis",
    "Precipitation / Crystallization",
    "Other",
]

REACTOR_TYPES = [
    "Capillary / Coil Reactor",
    "Microreactor / Chip",
    "Packed-Bed Reactor",
    "Photoreactor",
    "Tubular Flow Reactor",
    "Continuous-Flow Reactor",
    "Other",
]

REACTOR_MATERIALS = [
    "Fluoropolymer (PTFE / PFA / FEP)",
    "Stainless Steel",
    "Glass / Quartz",
    "PDMS / Silicon",
    "Other",
]

BOND_TYPES = [
    "C–C",
    "C–N",
    "C–O",
    "C–H",
    "C=C",
    "C–S",
    "N–N",
    "Other / Multiple",
]

LIGHT_SOURCES = [
    "Blue / Violet LED",
    "White / Other LED",
    "UV Lamp",
    "Xenon / Solar Lamp",
    "Microwave",
    "None (dark / thermal)",
]


# ── Prompt builder ─────────────────────────────────────────────────────────────

def _build_prompt(record: dict) -> str:
    chem = record.get("chemistry", {})
    reactor = record.get("reactor", {})
    light = record.get("light_source", {})
    flow = record.get("flow_optimized", {})
    notes = record.get("notes", "")

    context = f"""
Reaction name: {chem.get('reaction_name', '')}
Reaction class (raw): {chem.get('reaction_class', '')}
Mechanism type (raw): {chem.get('mechanism_type', '')}
Bond formed: {chem.get('bond_formed', '')}
Catalyst: {chem.get('catalyst', {}).get('name', '')} ({chem.get('catalyst', {}).get('type', '')})
Photocatalyst: {chem.get('photocatalyst', {}).get('name', '')}
Product: {chem.get('product', '')}
Reactor type (raw): {reactor.get('type', '')}
Reactor material (raw): {reactor.get('material', '')}
Light source (raw): {light.get('type', '')} {light.get('color', '')} {light.get('wavelength_nm', '') or ''}nm
Solvent: {flow.get('solvent', '')}
Notes: {notes[:300] if notes else ''}
""".strip()

    classes_str = "\n".join(f"  - {c}" for c in CHEMISTRY_CLASSES)
    reactor_str = "\n".join(f"  - {r}" for r in REACTOR_TYPES)
    material_str = "\n".join(f"  - {m}" for m in REACTOR_MATERIALS)
    bond_str = "\n".join(f"  - {b}" for b in BOND_TYPES)
    light_str = "\n".join(f"  - {l}" for l in LIGHT_SOURCES)

    return f"""You are classifying a flow chemistry paper into standardized categories.
Read the paper summary and return ONLY a JSON object with exactly these 5 keys.
Each value must be EXACTLY one of the allowed options listed.

Paper summary:
{context}

Allowed chemistry_class values:
{classes_str}

Allowed reactor_type values:
{reactor_str}

Allowed reactor_material values:
{material_str}

Allowed bond_type values:
{bond_str}

Allowed light_source values:
{light_str}

Rules:
- If reactor type is unknown/blank/weird → use "Capillary / Coil Reactor" as default
- If material is unknown → use "Fluoropolymer (PTFE / PFA / FEP)" as default
- If no bond is clearly described → use "Other / Multiple"
- If no light source is mentioned → use "None (dark / thermal)"
- For chemistry_class: use your best scientific judgment from the full context

Return ONLY valid JSON, no explanation. Example format:
{{"chemistry_class": "...", "reactor_type": "...", "reactor_material": "...", "bond_type": "...", "light_source": "..."}}"""


# ── LLM call ──────────────────────────────────────────────────────────────────

def _classify_single(client, record: dict) -> dict | None:
    prompt = _build_prompt(record)
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=150,
            temperature=0.0,
            response_format={"type": "json_object"},
        )
        raw = response.choices[0].message.content
        result = json.loads(raw)
        # Validate keys
        required = {"chemistry_class", "reactor_type", "reactor_material", "bond_type", "light_source"}
        if not required.issubset(result.keys()):
            return None
        return result
    except Exception as e:
        print(f"  LLM error: {e}")
        return None


def _get_record_id(record: dict) -> str:
    src = record.get("source_pdf", "")
    return Path(src).stem if src else ""


# ── Public API ────────────────────────────────────────────────────────────────

def load_classifications() -> dict:
    """Load cached classifications. Returns dict keyed by record id (PDF stem)."""
    if CACHE_PATH.exists():
        return json.loads(CACHE_PATH.read_text())
    return {}


def classify_all(
    records_dir=None,
    api_key: str | None = None,
    batch_size: int = 20,
    delay_s: float = 0.05,
    verbose: bool = True,
) -> dict:
    """
    Classify all records using GPT-4o-mini. Skips already-classified papers.

    Parameters
    ----------
    records_dir : path-like, optional
        Override default records directory.
    api_key : str, optional
        OpenAI API key. Falls back to OPENAI_API_KEY env var.
    batch_size : int
        Print progress every N papers.
    delay_s : float
        Small delay between API calls to avoid rate limits.
    verbose : bool
        Print progress.

    Returns
    -------
    dict : All classifications (existing + newly computed), keyed by record id.
    """
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError("pip install openai")

    key = api_key or os.environ.get("OPENAI_API_KEY")
    if not key:
        raise ValueError("Set OPENAI_API_KEY environment variable or pass api_key=")

    client = OpenAI(api_key=key)
    records = load_records(records_dir) if records_dir else load_records()
    cache = load_classifications()

    to_process = [r for r in records if _get_record_id(r) and _get_record_id(r) not in cache]

    if verbose:
        print(f"Total records: {len(records)}")
        print(f"Already classified: {len(cache)}")
        print(f"To classify now: {len(to_process)}")
        if not to_process:
            print("Nothing to do — all records already classified.")
            return cache

    new_count = 0
    fail_count = 0

    for i, record in enumerate(to_process):
        rec_id = _get_record_id(record)
        result = _classify_single(client, record)

        if result:
            cache[rec_id] = result
            new_count += 1
        else:
            # Fallback defaults
            cache[rec_id] = {
                "chemistry_class": "Other",
                "reactor_type": "Capillary / Coil Reactor",
                "reactor_material": "Fluoropolymer (PTFE / PFA / FEP)",
                "bond_type": "Other / Multiple",
                "light_source": "None (dark / thermal)",
            }
            fail_count += 1

        # Save incrementally every batch_size papers
        if (i + 1) % batch_size == 0:
            CACHE_PATH.write_text(json.dumps(cache, indent=2))
            if verbose:
                print(f"  [{i+1}/{len(to_process)}] classified, {fail_count} fallbacks so far...")

        if delay_s > 0:
            time.sleep(delay_s)

    # Final save
    CACHE_PATH.write_text(json.dumps(cache, indent=2))

    if verbose:
        print(f"\nDone. Newly classified: {new_count}, fallbacks: {fail_count}")
        print(f"Cache saved to: {CACHE_PATH}")

    return cache


def get_classified_field(field: str, records_dir=None) -> list[str]:
    """
    Get a list of classified values for one field across all records.
    field: one of 'chemistry_class', 'reactor_type', 'reactor_material', 'bond_type', 'light_source'
    """
    records = load_records(records_dir) if records_dir else load_records()
    cache = load_classifications()
    result = []
    for r in records:
        rec_id = _get_record_id(r)
        clf = cache.get(rec_id, {})
        result.append(clf.get(field, "Other"))
    return result


# ── Notebook quick-run ────────────────────────────────────────────────────────
if __name__ == "__main__":
    clf = classify_all(verbose=True)
    from collections import Counter
    print("\nChemistry classes:", Counter(v["chemistry_class"] for v in clf.values()).most_common())
    print("Reactor types:", Counter(v["reactor_type"] for v in clf.values()).most_common())
