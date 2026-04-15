"""
FLORA Visualization — LLM-based rule classifier for Figure 2.

Uses GPT-4o-mini to read each fundamentals rule and assign:
  - applicable_chemistry_classes : list of chemistry classes this rule is relevant to
  - key_concepts                 : 2-3 engineering terms / dimensionless numbers

Results cached in  visualization/cache/rule_classifications.json
Safe to interrupt and resume — incremental.

Jupyter usage:
    from rule_classifier import classify_all_rules, load_rule_classifications
    classify_all_rules()                    # ~$0.03 for all 1077 rules
    clf = load_rule_classifications()       # dict keyed by rule_id
"""

import json
import os
import sys
import time
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
RULES_PATH  = Path(__file__).parent.parent / "flora_fundamentals" / "data" / "rules.json"
CACHE_PATH  = Path(__file__).parent / "cache" / "rule_classifications.json"
CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)

# ── Canonical labels (must match llm_classifier.py) ──────────────────────────
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
]

# Engineering concepts pool (LLM will pick from these + add its own)
CONCEPT_HINTS = [
    "Reynolds number (Re)", "Damköhler number (Da)", "Péclet number (Pe)",
    "Nusselt number (Nu)", "Thiele modulus (φ)", "Fourier number (Fo)",
    "residence time", "flow rate", "back pressure", "temperature gradient",
    "heat transfer", "mass transfer", "mixing efficiency", "pressure drop",
    "photon flux", "light penetration depth", "Hatta number",
    "slug flow", "Taylor flow", "laminar flow", "turbulent flow",
    "scale-up", "channel geometry", "catalyst loading", "solubility",
    "selectivity", "conversion", "yield", "turnover number (TON)",
]


# ── Data loader ────────────────────────────────────────────────────────────────

def load_rules() -> list[dict]:
    data = json.loads(RULES_PATH.read_text())
    return data.get("rules", [])


def load_rule_classifications() -> dict:
    if CACHE_PATH.exists():
        return json.loads(CACHE_PATH.read_text())
    return {}


# ── Prompt ────────────────────────────────────────────────────────────────────

def _build_prompt(rule: dict) -> str:
    classes_str = "\n".join(f"  - {c}" for c in CHEMISTRY_CLASSES)
    hints_str   = ", ".join(CONCEPT_HINTS[:20])

    return f"""You are a flow chemistry expert. Classify this engineering rule from a flow chemistry textbook.

Rule category: {rule.get('category', '')}
Condition:     {rule.get('condition', '')}
Recommendation:{rule.get('recommendation', '')}
Reasoning:     {rule.get('reasoning', '')}
Formula:       {rule.get('quantitative', '')}

Task 1 — applicable_chemistry_classes:
Which of the following chemistry types is this rule DIRECTLY relevant to?
Return a JSON list with 1-4 items. Only include if genuinely applicable.
{classes_str}

Task 2 — key_concepts:
Extract 2-3 key engineering concepts, dimensionless numbers, or physical quantities
central to this rule. Use short canonical names (e.g. "Reynolds number (Re)",
"residence time", "photon flux", "back pressure").
Hint pool (use these exact names when applicable): {hints_str}

Return ONLY valid JSON:
{{"applicable_chemistry_classes": ["...", "..."], "key_concepts": ["...", "..."]}}"""


# ── Single call ────────────────────────────────────────────────────────────────

def _classify_single(client, rule: dict) -> dict | None:
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": _build_prompt(rule)}],
            max_tokens=120,
            temperature=0.0,
            response_format={"type": "json_object"},
        )
        result = json.loads(resp.choices[0].message.content)
        if "applicable_chemistry_classes" not in result or "key_concepts" not in result:
            return None
        # Validate chemistry classes
        valid = set(CHEMISTRY_CLASSES)
        result["applicable_chemistry_classes"] = [
            c for c in result["applicable_chemistry_classes"] if c in valid
        ]
        result["key_concepts"] = result["key_concepts"][:4]
        return result
    except Exception as e:
        print(f"  LLM error on rule {rule.get('rule_id')}: {e}")
        return None


# ── Public API ────────────────────────────────────────────────────────────────

def classify_all_rules(
    api_key: str | None = None,
    batch_size: int = 50,
    delay_s: float = 0.02,
    verbose: bool = True,
) -> dict:
    """
    Classify all rules using GPT-4o-mini. Skips already-classified rules.

    Cost estimate: ~$0.03 for all 1077 rules.

    Parameters
    ----------
    api_key   : OpenAI API key (falls back to OPENAI_API_KEY env var)
    batch_size: Save cache every N rules
    delay_s   : Small delay between calls
    verbose   : Print progress

    Returns
    -------
    dict : All classifications keyed by rule_id
    """
    from openai import OpenAI

    key = api_key or os.environ.get("OPENAI_API_KEY")
    if not key:
        raise ValueError("Set OPENAI_API_KEY or pass api_key=")

    client   = OpenAI(api_key=key)
    rules    = load_rules()
    cache    = load_rule_classifications()
    todo     = [r for r in rules if r["rule_id"] not in cache]

    if verbose:
        print(f"Total rules   : {len(rules)}")
        print(f"Already cached: {len(cache)}")
        print(f"To classify   : {len(todo)}")
        if not todo:
            print("All rules already classified.")
            return cache

    fails = 0
    for i, rule in enumerate(todo):
        result = _classify_single(client, rule)
        if result:
            cache[rule["rule_id"]] = result
        else:
            cache[rule["rule_id"]] = {
                "applicable_chemistry_classes": [],
                "key_concepts": [],
            }
            fails += 1

        if (i + 1) % batch_size == 0:
            CACHE_PATH.write_text(json.dumps(cache, indent=2))
            if verbose:
                print(f"  [{i+1}/{len(todo)}] done, {fails} fallbacks...")
        if delay_s:
            time.sleep(delay_s)

    CACHE_PATH.write_text(json.dumps(cache, indent=2))
    if verbose:
        print(f"\nDone. Classified: {len(todo)-fails}, fallbacks: {fails}")
        print(f"Saved to: {CACHE_PATH}")
    return cache


if __name__ == "__main__":
    clf = classify_all_rules()
    from collections import Counter
    all_concepts = []
    for v in clf.values():
        all_concepts.extend(v.get("key_concepts", []))
    print("\nTop concepts:", Counter(all_concepts).most_common(15))
