"""Source-declared component aliases, scoped to one protocol and one feed.

Never infer equivalence from a model's abbreviation or molecular similarity.
Concentration/equivalent conflicts and ambiguous alias declarations stay visible.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import re
from typing import Iterable

from flora_translate.component_identity import component_key, component_name
from flora_translate.protocol_input import extract_protocol_text

_ALIAS_TAIL = re.compile(r"^(?P<name>.+?)\s+\((?P<alias>[A-Z][A-Z0-9]{1,15})\)$")
_PORTIONS = re.compile(r"\b(?:portion(?:s|wise)?|aliquots?|split[- ](?:charge|addition)|(?:first|second|third|additional|separate)\s+(?:charge|addition|aliquot)|(?:two|three|\d+)\s+(?:(?:equal|separate|successive)\s+)?(?:parts|charges|additions))\b", re.I)
_CONFLICT = "source_component_alias_conflict: "


@dataclass(frozen=True)
class SourceAliases:
    canonical_by_key: dict[str, str]
    evidence_by_key: dict[str, str]
    conflicts: tuple[str, ...]

    def canonical_name(self, name: str) -> str:
        return self.canonical_by_key.get(component_key(name), name)

    def canonical_text(self, text: str) -> str:
        name = component_name(text)
        canonical = self.canonical_name(name)
        return canonical + text[len(name):] if canonical != name else text


def source_aliases(protocol: str, names: Iterable[str]) -> SourceAliases:
    """Find full-name (ACRONYM) declarations beside known candidate names.

    Candidate names delimit possible full names; only an exact declaration in
    the original protocol grants equivalence. Generated names alone grant none.
    """
    protocol = extract_protocol_text(protocol)
    candidates = list(dict.fromkeys(component_name(str(name)) for name in names))
    declarations: dict[str, dict[str, tuple[str, str]]] = {}
    known_declaration_positions: dict[str, set[int]] = {}
    asserted_pairs = []
    for candidate in candidates:
        tail = _ALIAS_TAIL.fullmatch(candidate)
        full = tail['name'] if tail else candidate
        if tail:
            asserted_pairs.append((full, tail['alias']))
        # Boundaries prevent methyl... and substituted methyl... sharing an
        # accidental substring. Source aliases are uppercase declarations.
        pattern = r"(?<![A-Za-z0-9-])" + re.escape(full) + r"\s*\(\s*(?P<alias>[A-Z][A-Z0-9]{1,15})(?=\s*[;,)])[^)]*\)"
        for match in re.finditer(pattern, protocol, re.I):
            alias = match['alias']
            if not alias.isupper():
                continue
            alias_key = component_key(alias)
            declarations.setdefault(alias_key, {})[component_key(full)] = (full, match.group(0))
            known_declaration_positions.setdefault(alias_key, set()).add(match.start('alias'))
    canonical = {}
    evidence = {}
    conflicts = []
    for alias_key, full_names in declarations.items():
        # Candidate names do not delimit source authority. A second declaration
        # may name a compound the model omitted entirely. Refuse the alias if
        # any occurrence has not been grounded to a known full-name declaration.
        source_positions = {match.start('alias') for match in re.finditer(
            r"\(\s*(?P<alias>" + re.escape(alias_key) + r")(?=\s*[;,)])[^)]*\)",
            protocol, re.I) if match['alias'].isupper()}
        if len(full_names) != 1 or source_positions - known_declaration_positions[alias_key]:
            conflicts.append(f"Protocol acronym {alias_key!r} has conflicting or incompletely resolved source declarations; clarify identity before dispensing.")
            continue
        full, quote = next(iter(full_names.values()))
        for key in (alias_key, component_key(full)):
            canonical[key] = full
            evidence[key] = quote
        # Only exact candidate '(ALIAS)' descriptions inherit this declaration.
        for candidate in candidates:
            tail = _ALIAS_TAIL.fullmatch(candidate)
            if tail and component_key(tail['name']) == component_key(full) and component_key(tail['alias']) == alias_key:
                canonical[component_key(candidate)] = full
                evidence[component_key(candidate)] = quote
    candidate_keys = {component_key(name) for name in candidates}
    for full, alias in asserted_pairs:
        if component_key(alias) in candidate_keys and component_key(alias) not in canonical:
            conflicts.append(f"Feed contains both {alias!r} and {full!r} with an alias claim not unambiguously defined by the original protocol; resolve before dispensing.")
    return SourceAliases(canonical, evidence, tuple(dict.fromkeys(conflicts)))


def normalize_plan_component_aliases(plan, protocol: str) -> None:
    """Normalize a copied chemistry plan before missing-feed restoration."""
    feeds = [*plan.stream_logic, *(feed for stage in plan.stages for feed in stage.feed_streams)]
    names = [item.name for item in plan.reagents] + [name for feed in feeds for name in feed.reagents]
    registry = source_aliases(protocol, names)
    if registry.conflicts or _PORTIONS.search(protocol):
        # Final validation retains/flags uncertainty; do not erase evidence here.
        return
    declarations = {}
    for item in plan.reagents:
        key = component_key(registry.canonical_name(item.name))
        signature = (item.role.strip().casefold(), item.equiv_or_loading.strip().casefold())
        if key in declarations and declarations[key] != signature:
            # Renaming would let downstream name-keyed role maps overwrite
            # contradictory declarations. Keep their original evidence intact.
            return
        declarations[key] = signature
    for item in plan.reagents:
        item.name = registry.canonical_name(item.name)
    for feed in feeds:
        feed.reagents = [registry.canonical_text(name) for name in feed.reagents]


def collapse_source_alias_components(components: list, protocol: str) -> list:
    """Return canonical copies; never change the supplied proposal or sum doses.

    This caller supplies ONE physical stream. Separate feeds/stages therefore
    remain separate. Explicit portions are never silently dropped. Conflicting
    quantities remain duplicate canonical entries for the existing release gate.
    """
    registry = source_aliases(protocol, [item.name for item in components])
    result = []
    seen = {}
    for source in components:
        item = source.model_copy(deep=True)
        original_name = item.name
        key = component_key(original_name)
        item.name = registry.canonical_name(original_name)
        if key in registry.evidence_by_key:
            item.provenance = list(dict.fromkeys([*item.provenance,
                'source_declared_alias: ' + registry.evidence_by_key[key]]))
        for conflict in registry.conflicts:
            item.provenance.append(_CONFLICT + conflict)
        canonical_key = component_key(item.name)
        feed_key = (item.stream_label, canonical_key)
        same = seen.get(feed_key)
        separate = (_PORTIONS.search(protocol) or _PORTIONS.search(source.source_text)
                    or (same is not None and _PORTIONS.search(same.source_text)))
        quantities = ('concentration_M', 'molar_equiv', 'loading_mol_pct')
        def quantities_agree(a, b):
            for field in quantities:
                left, right = getattr(a, field), getattr(b, field)
                if left is None or right is None:
                    if left != right:
                        return False
                elif not math.isclose(left, right, rel_tol=1e-8, abs_tol=1e-12):
                    return False
            return a.quantification_required == b.quantification_required
        # Only source-proven alias groups may collapse; an unrelated repeated
        # name/quantity remains under the existing duplicate conflict policy.
        known_alias = canonical_key in registry.canonical_by_key
        compatible_assignment = (same is None or (
            same.role.strip().casefold() == item.role.strip().casefold()
            and getattr(same, 'phase', None) == getattr(item, 'phase', None)))
        if same is not None and known_alias and not compatible_assignment:
            item.provenance.append(_CONFLICT + f"Source-identical component {item.name!r} has conflicting role or phase assignments in stream {item.stream_label!r}; resolve before dispensing.")
        if same is not None and known_alias and compatible_assignment and not separate and quantities_agree(same, item):
            same.provenance = list(dict.fromkeys([*same.provenance, *item.provenance,
                f'alias_description_collapsed_without_adding_dose: {source.source_text}']))
            continue
        result.append(item)
        seen.setdefault(feed_key, item)
    return result


def source_alias_conflicts(components: list) -> list[str]:
    return list(dict.fromkeys(note[len(_CONFLICT):] for item in components
                             for note in item.provenance if note.startswith(_CONFLICT)))
