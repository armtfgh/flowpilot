"""Separate supplied batch evidence from design requests and equipment context.

These helpers recognize the application's explicit input envelopes. They do
not infer a reaction identity or manufacture a chemist confirmation.
"""

from __future__ import annotations

import json
import re
from typing import Any


_HEADINGS = {
    "RAW BATCH PROTOCOL": "raw_protocol",
    "BATCH PROCEDURE": "raw_protocol",
    "BATCH PROTOCOL": "raw_protocol",
    "OBJECTIVE": "objective",
    "REQUEST": "objective",
    "HARD CONSTRAINTS": "operating_limits",
    "AVAILABLE INVENTORY (AUTHORITATIVE JSON)": "inventory_constraints",
    "AVAILABLE EQUIPMENT": "inventory_constraints",
    "CHEMIST-CONFIRMED CHEMISTRY IDENTITY": "chemistry_identity_confirmation",
}
_HEADING_RE = re.compile(
    r"^\s*(?:#{1,3}\s*)?(" + "|".join(re.escape(s) for s in _HEADINGS) + r")\s*:?\s*$",
    re.IGNORECASE | re.MULTILINE,
)
_CONTEXT_KEYS = {
    "objective", "request", "inventory", "inventory_constraints", "hard_constraints",
    "operating_limits", "chemistry_identity_confirmation", "historical_data",
    "hypotheses", "output_preferences", "screening_priority", "answers", "question_log",
    "missing_question_ids", "ready_for_design", "authority_order", "schema_version",
    "question_bank_version", "inventory_profile_snapshot", "inventory_review",
    "active_domains", "active_question_ids", "engineering_requirements", "question_set_hash",
}
_DESIGN_CONTEXT_KEYS = _CONTEXT_KEYS - {"schema_version", "question_bank_version"}


def _as_dict(value: str | dict) -> dict | None:
    if isinstance(value, dict):
        return value
    try:
        parsed = json.loads(value)
    except (ValueError, TypeError):
        return None
    return parsed if isinstance(parsed, dict) else None


def design_context_from_input(value: str | dict) -> dict[str, Any] | None:
    """Return explicit envelope fields, or None for an ordinary batch record.

    Text envelopes must start with a recognized protocol heading; a word such
    as 'request' within normal prose is never treated as a section boundary.
    """
    data = _as_dict(value)
    if data is not None:
        # Normalize aliases before either structured-envelope branch. Original
        # canonical fields, including explicit empty values, keep precedence.
        data = dict(data)
        for alias, canonical in (("request", "objective"), ("inventory", "inventory_constraints"),
                                 ("hard_constraints", "operating_limits")):
            if canonical not in data and alias in data:
                data[canonical] = data[alias]
        if "raw_protocol" in data or "extracted_batch_fields" in data:
            # Mixed JSON envelopes can place typed batch values alongside the
            # protocol. Preserve them; explicit nested values take precedence.
            fields = {k: v for k, v in data.items()
                      if k not in _CONTEXT_KEYS and k not in {"raw_protocol", "extracted_batch_fields"}}
            nested = data.get("extracted_batch_fields", {})
            if isinstance(nested, dict) and fields:
                data["extracted_batch_fields"] = {**fields, **nested}
            return data
        if not any(k in data for k in _DESIGN_CONTEXT_KEYS):
            return None
        context = {k: v for k, v in data.items() if k in _CONTEXT_KEYS}
        fields = {k: v for k, v in data.items() if k not in _CONTEXT_KEYS}
        context["raw_protocol"] = extract_protocol_text(fields)
        context["extracted_batch_fields"] = fields
        return context
    text = str(value or "")
    matches = list(_HEADING_RE.finditer(text))
    if not matches or text[:matches[0].start()].strip():
        return None
    if _HEADINGS[matches[0].group(1).upper()] != "raw_protocol":
        return None
    sections: dict[str, Any] = {}
    for i, match in enumerate(matches):
        key = _HEADINGS[match.group(1).upper()]
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        content = text[match.end():end].strip()
        if key in sections:
            raise ValueError(f"Duplicate design-input section: {key}")
        if key in {"inventory_constraints", "chemistry_identity_confirmation"}:
            parsed = _as_dict(content)
            sections[key] = parsed if parsed is not None else content
        else:
            sections[key] = content
    return sections


def extract_protocol_text(value: str | dict) -> str:
    """Return supplied protocol evidence without the surrounding design brief."""
    data = _as_dict(value)
    if data is not None:
        # An original protocol outranks a model-extracted record's raw_text.
        if data.get("raw_protocol"):
            return extract_protocol_text(str(data["raw_protocol"]))
        if data.get("raw_text"):
            return extract_protocol_text(str(data["raw_text"]))
        if isinstance(data.get("extracted_batch_fields"), dict):
            return extract_protocol_text(data["extracted_batch_fields"])
        fields = {k: v for k, v in data.items() if k not in _CONTEXT_KEYS and k != "raw_protocol"}
        return json.dumps(fields, sort_keys=True, default=str)
    text = str(value or "")
    sections = design_context_from_input(text)
    return str(sections["raw_protocol"]) if sections else text


def protocol_batch_input(value: str | dict) -> str | dict:
    """Keep structured batch fields while removing non-protocol authority."""
    data = _as_dict(value)
    if data is None:
        return extract_protocol_text(value)
    if "raw_protocol" in data or "extracted_batch_fields" in data:
        data = design_context_from_input(data)
    fields = data.get("extracted_batch_fields") if "raw_protocol" in data or "extracted_batch_fields" in data else data
    if not isinstance(fields, dict) or not fields:
        return extract_protocol_text(data)
    fields = {k: v for k, v in fields.items() if k not in _CONTEXT_KEYS}
    fields["raw_text"] = extract_protocol_text(data)
    return fields
