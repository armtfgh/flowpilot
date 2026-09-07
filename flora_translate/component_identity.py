"""Shared feed identity and role normalization, independent of chemical names."""

from __future__ import annotations

import re
from collections.abc import Iterable


_QUANTITY = re.compile(
    r"\d+(?:\.\d+)?\s*(?:mM|M|mmol|mol|equiv|eq\.?|mg|g|mL|uL|wt\s*%)(?![A-Za-z])",
    re.I,
)
_MEDIUM_ROLE = re.compile(r"\b(?:co[ -]?solvent|solvent|carrier|diluent)\b", re.I)
_REACTIVE_ROLE = re.compile(
    r"\b(?:base|acid|substrate|reactant|reagent|catalyst|photocatalyst|oxidant|"
    r"reductant|donor|acceptor|nucleophile|electrophile)\b", re.I,
)


def component_name(value: str) -> str:
    text = str(value).strip()
    # Strip a dose/role annotation, never parentheses within a chemical name.
    tail = re.search(r"\s+\(([^()]*)\)\s*$", text)
    if tail and (
        _QUANTITY.search(tail[1]) or _MEDIUM_ROLE.search(tail[1])
        or re.fullmatch(r"[A-Za-z]?\d+[A-Za-z]?", tail[1])
    ):
        return text[:tail.start()].strip() or text
    return text


def component_key(value: str) -> str:
    return re.sub(r"\s+", "", component_name(value)).casefold()


def unique_components(values: Iterable[str]) -> list[str]:
    """Discard bare duplicates, but retain conflicting annotated entries for validation."""
    groups: dict[str, list[str]] = {}
    for value in values:
        text = str(value).strip()
        if not text:
            continue
        group = groups.setdefault(component_key(text), [])
        if text.casefold() not in {item.casefold() for item in group}:
            group.append(text)
    output = []
    for group in groups.values():
        annotated = [item for item in group if component_name(item) != item]
        output.extend(annotated or group[:1])
    return output


def is_solvent_component(source: str, declared_solvent: str, planned_role: str) -> bool:
    """A medium-only annotation is not a stoichiometric reagent requirement.

    A dual role (for example solvent/base) remains reactive. An explicit inline
    medium-only annotation overrides a mistaken upstream role, as before.
    """
    tail = re.search(r"\s+\(([^()]*)\)\s*$", source)
    if tail and _MEDIUM_ROLE.search(tail[1]):
        return not bool(_REACTIVE_ROLE.search(tail[1]))
    if _MEDIUM_ROLE.search(planned_role):
        return not bool(_REACTIVE_ROLE.search(planned_role))
    if _REACTIVE_ROLE.search(planned_role):
        return False
    return bool(declared_solvent) and component_key(source) == component_key(declared_solvent)


def declared_solvent_member(name: str, solvent: str) -> bool:
    """Match only explicitly named solvent-mixture members, without chemical guessing."""
    mixture = re.sub(r"\s*\([^()]*\d[^()]*\)\s*$", "", solvent).strip()
    members = re.split(r"\s*[:/]\s*", mixture)
    return component_key(name) in {component_key(item) for item in [solvent, mixture, *members] if item}


def protocol_component_quantity(name: str, protocol: str) -> str:
    """Read a dose immediately following an exact chemical name in the protocol."""
    doses = []
    for match in re.finditer(r"(?<![A-Za-z0-9])" + re.escape(component_name(name)) + r"\s*\(([^()]*)\)", protocol, re.I):
        dose = re.search(r"\d+(?:\.\d+)?\s*(?:mol\s*%|equiv(?:alents)?|eq\.?)(?![A-Za-z])", match[1], re.I)
        if dose:
            doses.append(dose[0])
    # Multiple different doses require stage-specific interpretation, not a guess.
    unique = list(dict.fromkeys(doses))
    return unique[0] if len(unique) == 1 else ""
