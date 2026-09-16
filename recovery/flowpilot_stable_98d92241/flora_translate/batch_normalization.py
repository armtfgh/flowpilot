"""Deterministic recovery of basic batch quantities from protocol text.

This module is intentionally conservative. It only infers values that are
simple, local, and materially important for downstream engineering.
"""

from __future__ import annotations

import re
from typing import Any

_FLOAT_RE = r"(\d+(?:\.\d+)?)"
_EXPLICIT_CONC_RE = re.compile(rf"{_FLOAT_RE}\s*M\b", re.IGNORECASE)
_MMOL_RE = re.compile(rf"{_FLOAT_RE}\s*mmol\b", re.IGNORECASE)
_ML_RE = re.compile(rf"{_FLOAT_RE}\s*mL\b", re.IGNORECASE)
_TIME_RE = re.compile(rf"\bfor\s+{_FLOAT_RE}\s*(min|mins|minute|minutes|h|hr|hrs|hour|hours)\b", re.IGNORECASE)

_CONTAINER_WORDS = (
    "vial",
    "flask",
    "tube",
    "reactor",
    "schlenk",
    "bottle",
    "autoclave",
)
_VOLUME_HINT_WORDS = (
    "des",
    "solvent",
    "mixture",
    "medium",
    "buffer",
    "electrolyte",
    "water",
    "acetonitrile",
    "methanol",
    "ethanol",
    "dcm",
    "dmso",
    "dmf",
    "tbab",
    "eg",
)


def _coerce_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        match = re.search(_FLOAT_RE, value)
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                return None
    return None


def _context_slice(text: str, start: int, end: int, radius: int = 48) -> str:
    lo = max(0, start - radius)
    hi = min(len(text), end + radius)
    return text[lo:hi].lower()


def infer_scale_mmol(raw_text: str | None, explicit_scale_mmol: Any = None) -> float | None:
    explicit = _coerce_float(explicit_scale_mmol)
    if explicit and explicit > 0:
        return explicit
    text = (raw_text or "").strip()
    if not text:
        return None
    match = _MMOL_RE.search(text)
    if not match:
        return None
    try:
        value = float(match.group(1))
    except ValueError:
        return None
    return value if value > 0 else None


def infer_reaction_time_h(raw_text: str | None, explicit_reaction_time_h: Any = None) -> float | None:
    explicit = _coerce_float(explicit_reaction_time_h)
    if explicit and explicit > 0:
        return explicit
    text = (raw_text or "").strip()
    if not text:
        return None
    match = _TIME_RE.search(text)
    if not match:
        return None
    try:
        value = float(match.group(1))
    except ValueError:
        return None
    unit = match.group(2).lower()
    if unit.startswith("min"):
        return value / 60.0
    return value


def _choose_reaction_volume_candidate(raw_text: str | None) -> tuple[float, str] | None:
    text = (raw_text or "").strip()
    if not text:
        return None

    candidates: list[tuple[int, float, int, str]] = []
    for idx, match in enumerate(_ML_RE.finditer(text)):
        try:
            value = float(match.group(1))
        except ValueError:
            continue
        if value <= 0:
            continue
        context = _context_slice(text, match.start(), match.end())
        score = 0
        if any(word in context for word in _CONTAINER_WORDS):
            score -= 5
        if any(word in context for word in _VOLUME_HINT_WORDS):
            score += 4
        if "added directly to" in context or "in " in context:
            score += 1
        if value <= 5:
            score += 1
        evidence = text[max(0, match.start() - 80) : min(len(text), match.end() + 80)].strip()
        candidates.append((score, value, idx, evidence))

    if not candidates:
        return None
    candidates.sort(key=lambda item: (-item[0], item[2], item[1]))
    return candidates[0][1], candidates[0][3]


def _choose_reaction_volume_mL(raw_text: str | None) -> float | None:
    candidate = _choose_reaction_volume_candidate(raw_text)
    return candidate[0] if candidate else None


def build_batch_quantity_evidence(raw_text: str | None, explicit_data: dict | None = None) -> dict:
    """Return auditable deterministic evidence for engineering-critical fields.

    The values here are intentionally limited to local arithmetic/extraction from
    the protocol text. They are safe to use as authoritative corrections in
    lightweight upstream mode because they are not model-generated design choices.
    """
    text = (raw_text or "").strip()
    explicit_data = explicit_data or {}
    evidence: dict[str, dict] = {}

    scale_match = _MMOL_RE.search(text)
    if scale_match:
        value = float(scale_match.group(1))
        evidence["scale_mmol"] = {
            "value": value,
            "source": "protocol_text",
            "evidence_text": text[max(0, scale_match.start() - 60) : min(len(text), scale_match.end() + 60)].strip(),
        }
    else:
        explicit_scale = _coerce_float(explicit_data.get("scale_mmol"))
        if explicit_scale and explicit_scale > 0:
            evidence["scale_mmol"] = {
                "value": explicit_scale,
                "source": "model_explicit",
                "evidence_text": "",
            }

    time_match = _TIME_RE.search(text)
    if time_match:
        value = float(time_match.group(1))
        unit = time_match.group(2).lower()
        value_h = value / 60.0 if unit.startswith("min") else value
        evidence["reaction_time_h"] = {
            "value": value_h,
            "source": "protocol_text",
            "unit_seen": unit,
            "evidence_text": text[max(0, time_match.start() - 60) : min(len(text), time_match.end() + 60)].strip(),
        }
    else:
        explicit_time = _coerce_float(explicit_data.get("reaction_time_h"))
        if explicit_time and explicit_time > 0:
            evidence["reaction_time_h"] = {
                "value": explicit_time,
                "source": "model_explicit",
                "evidence_text": "",
            }

    explicit_conc_match = _EXPLICIT_CONC_RE.search(text)
    if explicit_conc_match:
        value = float(explicit_conc_match.group(1))
        evidence["concentration_M"] = {
            "value": value,
            "source": "protocol_text_explicit_M",
            "evidence_text": text[
                max(0, explicit_conc_match.start() - 60) : min(len(text), explicit_conc_match.end() + 60)
            ].strip(),
        }
    else:
        volume_candidate = _choose_reaction_volume_candidate(text)
        scale = evidence.get("scale_mmol", {}).get("value")
        if scale and volume_candidate:
            volume_mL, volume_evidence = volume_candidate
            concentration = scale / volume_mL
            evidence["reaction_volume_mL"] = {
                "value": volume_mL,
                "source": "protocol_text",
                "evidence_text": volume_evidence,
            }
            evidence["concentration_M"] = {
                "value": concentration,
                "source": "deterministic_mmol_per_mL",
                "formula": f"{scale:g} mmol / {volume_mL:g} mL = {concentration:g} M",
                "evidence_text": f"{evidence['scale_mmol']['evidence_text']} | {volume_evidence}",
            }
        else:
            explicit_conc = _coerce_float(explicit_data.get("concentration_M"))
            if explicit_conc and explicit_conc > 0:
                evidence["concentration_M"] = {
                    "value": explicit_conc,
                    "source": "model_explicit_no_text_arithmetic",
                    "evidence_text": "",
                }

    return evidence


def apply_authoritative_batch_evidence(data: dict, raw_text: str | None) -> tuple[dict, dict]:
    """Apply protocol-text arithmetic over model guesses and return an audit trail."""
    enriched = dict(data)
    evidence = build_batch_quantity_evidence(raw_text, enriched)
    for field in ("scale_mmol", "reaction_time_h", "concentration_M"):
        item = evidence.get(field)
        if item and item.get("source", "").startswith(("protocol_text", "deterministic")):
            enriched[field] = item["value"]
    return enriched, evidence


def infer_batch_concentration_M(
    raw_text: str | None,
    *,
    explicit_concentration_M: Any = None,
    explicit_scale_mmol: Any = None,
) -> float | None:
    explicit_conc = _coerce_float(explicit_concentration_M)
    if explicit_conc and explicit_conc > 0:
        return explicit_conc

    text = (raw_text or "").strip()
    if not text:
        return None

    conc_match = _EXPLICIT_CONC_RE.search(text)
    if conc_match:
        try:
            value = float(conc_match.group(1))
        except ValueError:
            value = None
        if value and value > 0:
            return value

    scale_mmol = infer_scale_mmol(text, explicit_scale_mmol)
    volume_mL = _choose_reaction_volume_mL(text)
    if not scale_mmol or not volume_mL:
        return None

    concentration_M = scale_mmol / volume_mL
    return concentration_M if concentration_M > 0 else None


def enrich_batch_record_dict(data: dict, raw_text: str | None) -> dict:
    """Fill a few missing engineering-critical fields deterministically."""
    enriched = dict(data)
    text = raw_text or enriched.get("raw_text") or enriched.get("reaction_description") or ""

    if not enriched.get("scale_mmol"):
        inferred_scale = infer_scale_mmol(text, enriched.get("scale_mmol"))
        if inferred_scale is not None:
            enriched["scale_mmol"] = inferred_scale

    if not enriched.get("reaction_time_h"):
        inferred_time_h = infer_reaction_time_h(text, enriched.get("reaction_time_h"))
        if inferred_time_h is not None:
            enriched["reaction_time_h"] = inferred_time_h

    if not enriched.get("concentration_M"):
        inferred_conc = infer_batch_concentration_M(
            text,
            explicit_concentration_M=enriched.get("concentration_M"),
            explicit_scale_mmol=enriched.get("scale_mmol"),
        )
        if inferred_conc is not None:
            enriched["concentration_M"] = inferred_conc

    if text and not enriched.get("raw_text"):
        enriched["raw_text"] = text
    return enriched
