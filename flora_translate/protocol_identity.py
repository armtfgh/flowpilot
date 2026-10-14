"""Conservative reaction identity evidence from researcher-supplied protocol text.

These rules recognise a few explicit reactant-to-product changes, not arbitrary
chemistry and not molecular structures. Each accepted inference retains the exact
source spans and is checked against the proposed reactants and product. An LLM's
reaction description, SMILES, inventory, and mechanism are never source evidence.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
import unicodedata
from typing import Any
from flora_translate.component_identity import component_name


@dataclass(frozen=True)
class ProtocolIdentityEvidence:
    family: str
    rule: str
    substrates: tuple[str, ...]
    product: str
    source_evidence: tuple[str, ...]
    substrate_aliases: tuple[tuple[str, ...], ...] = ()


def _plain(text: str) -> str:
    return unicodedata.normalize("NFKC", text).translate(str.maketrans("–—−", "---"))


def chemical_name_key(text: str) -> str:
    """Only spelling/locant normalisation; never remove stereochemistry."""
    value = _plain(text).lower()
    # Positional alcohol nomenclature, e.g. piperidin-4-ol / 4-piperidinol.
    value = re.sub(r"([a-z]+)idin-(\d+)-ol", r"\2-\1idinol", value)
    return re.sub(r"[^a-z0-9]", "", value)


def _identity_keys(text: str) -> set[str]:
    """Strip role/quantity annotations, not chemical substituents."""
    text = component_name(text)
    parts = re.split(r"\s+(?:of|to|over|with|and|using|under)\s+", text, flags=re.I)
    names = list(parts)
    for part in parts:
        # A separate final parenthesis may declare an acronym/name. Embedded
        # chemical parentheses remain part of the molecule, as in Pd(OH)2.
        alias = re.fullmatch(r"(.+?)\s+\(([^()]+)\)", part.strip())
        if alias:
            names.extend(alias.groups())
    return {chemical_name_key(name) for name in names}


def affirmative_protocol_text(text: str) -> str:
    """Do not convert negative or hypothetical statements into source facts."""
    clauses = re.split(r"(?<=[.;])\s+|\n+", _plain(text))
    excluded = re.compile(
        r"\b(?:not|no|without|unobserved|hypothetical|hypothesized|proposed|"
        r"could|might|may|would|expected|anticipated|if|failed)\b", re.I
    )
    return " ".join(clause for clause in clauses if not excluded.search(clause))


def protocol_label_text(text: str) -> str:
    """Equipment-purpose labels are not claims about the performed reaction."""
    text = affirmative_protocol_text(text)
    family_word = r"hydrogenation|hydrogenolysis|oxidation|reduction|nitration|amidation|coupling|substitution|cuaac"
    hardware_modifier = r"(?:batch|flow|glass|steel|packed[- ]bed|stirred[- ]tank|pressure[- ]rated|commercial|jacketed|tubular)"
    return re.sub(
        rf"\b(?:{family_word})(?=[ -]+(?:{hardware_modifier}[ -]+){{0,2}}(?:vessel|skid|apparatus|module|pump|reactor|equipment)\b)",
        "", text, flags=re.I,
    )


def _source_aliases(protocol: str, sources: tuple[str, ...]) -> tuple[tuple[str, ...], ...]:
    """Accept only acronyms explicitly defined beside a source compound name."""
    text = _plain(protocol)
    aliases = []
    for source in sources:
        match = re.search(
            re.escape(source) + r"\s*\(\s*(?P<alias>[A-Z][A-Z0-9]{1,15})\s*(?=[;,)])",
            text,
            # Match source spelling case-insensitively, but require an actual
            # uppercase abbreviation rather than an arbitrary parenthesis.
            re.I,
        )
        alias = match['alias'] if match and match['alias'].isupper() else ''
        aliases.append((alias,) if alias else ())
    return tuple(aliases)


def _outcome(text: str, feature: str) -> tuple[str, str] | None:
    """Return a named observed product and its sentence, never a charged reagent."""
    chemical = r"[A-Za-z0-9αβγ,()\[\]\-]+(?:\s+[A-Za-z0-9αβγ,()\[\]\-]+){0,4}?"
    for sentence in re.split(r"(?<=[.;])\s+|\n+", affirmative_protocol_text(text)):
        if not re.search(feature, sentence, re.I):
            continue
        patterns = (
            rf"(?:conversion|converted)\s+to\s+(?P<name>{chemical})(?=\s+(?:was|is|at|in)\b|[.;]|$)",
            rf"(?P<name>{chemical})\s+(?:was|were|is)\s+(?:formed|isolated|obtained|produced|detected)\b",
            rf"(?:gave|afforded|yielded|furnished|produced)\s+(?P<name>{chemical})(?=\s+(?:in|at|with)\b|[.;]|$)",
        )
        for pattern in patterns:
            match = re.search(pattern, sentence, re.I)
            if not match:
                continue
            name = re.sub(r"^(?:the\s+)?(?:corresponding\s+|product\s+)", "", match["name"], flags=re.I)
            if re.search(feature, name, re.I):
                return name.strip(), sentence.strip()
    return None


def _evidence(text: str, matches: list[re.Match | None], outcome: tuple[str, str]) -> tuple[str, ...]:
    quotes = [match.group(0) for match in matches if match] + [outcome[1]]
    exact = []
    for quote in quotes:
        if quote in text:
            exact.append(quote)
            alias_context = re.search(
                re.escape(quote) + r"\s*\(\s*[A-Z][A-Z0-9]{1,15}(?=\s*[;,)])[^)]*\)",
                text,
            )
            if alias_context:
                exact.append(alias_context.group(0))
        else:
            # Preserve the source's spelling/Unicode even when recognition
            # used canonical dashes or compatibility characters.
            exact.append(next((clause for clause in re.split(r"(?<=[.;])\s+|\n+", text)
                               if quote in _plain(clause)), text))
    return tuple(dict.fromkeys(exact))


def infer_protocol_identity(protocol: str) -> ProtocolIdentityEvidence | None:
    """Require observed product + appropriate reactants/reagent, not a keyword."""
    text = affirmative_protocol_text(protocol)
    # Cleavage of a named N/O/S benzylic substituent by hydrogen. A matching
    # deprotected parent is required: H2/Pd alone could also reduce an alkene.
    protected = re.search(
        r"\b[NOS]-(?:diphenylmethyl|benzhydryl|benzyl)-?(?P<parent>[a-z][a-z0-9,\-]*)",
        text, re.I,
    )
    hydrogen = re.search(r"\b(?:hydrogen|H2)\b", text, re.I)
    catalyst = re.search(r"\b(?:Pd(?:\(OH\)2)?|palladium|Pt|platinum|Raney nickel|Ni)\b", text, re.I)
    if protected and hydrogen and catalyst:
        product = _outcome(text, r"[a-z]")
        if product and chemical_name_key(protected["parent"]) == chemical_name_key(product[0]):
            return ProtocolIdentityEvidence(
                "hydrogenolysis", "benzylic_group_cleavage_to_matching_parent_v1",
                (protected.group(0),), product[0],
                _evidence(protocol, [protected, hydrogen, catalyst], product),
                _source_aliases(protocol, (protected.group(0),)),
            )

    # Sulfur oxidation must link the observed sulfoxide to the starting sulfide
    # skeleton. Methionine is a thioether; stereochemistry may be unspecified in
    # the product name, but the exact starting stereochemical label is retained.
    product = _outcome(text, r"\bsulfoxide\b")
    oxidant = re.search(r"\b(?:air|oxygen|O2|H2O2|hydrogen peroxide|peroxide|mCPBA)\b", text, re.I)
    if product and oxidant:
        stem = re.sub(r"\s*sulfoxide\b.*", "", product[0], flags=re.I).strip()
        source_pattern = re.escape(stem).replace(r"\-", "-")
        if re.search(r"methionine$", stem, re.I):
            source_pattern = source_pattern.replace("methionine", "(?:[LD]-)?methionine")
            source_pattern += r"(?!\s+(?:sulfoxide|sulfone))"
        elif stem:
            source_pattern += r"\s+(?:sulfide|thioether)\b"
        source = re.search(r"\b" + source_pattern, text[:text.find(product[1])], re.I) if stem else None
        if source:
            return ProtocolIdentityEvidence(
                "oxidation", "sulfide_to_matching_sulfoxide_v1", (source.group(0),), product[0],
                _evidence(protocol, [source, oxidant], product),
                _source_aliases(protocol, (source.group(0),)),
            )

    # A copper-mediated azide/alkyne combination and an observed triazole are
    # all required. A triazole already in the starting charge is insufficient.
    product = _outcome(text, r"1,2,3-triazole\b")
    if product:
        charge = text[:text.find(product[1])]
        azide = re.search(r"\b(?:[a-z0-9,\-]+\s+)?azide\b", charge, re.I)
        alkyne = re.search(r"\b[a-z0-9,\-]*(?:acetylene|[1-9]-yne|alkyne)\b", charge, re.I)
        copper = re.search(r"\b(?:Cu(?:/C|\(I\)|SO4|I|Br)?|copper)\b", charge, re.I)
        if azide and alkyne and copper:
            return ProtocolIdentityEvidence(
                "cuaac", "copper_azide_alkyne_to_triazole_v1",
                (azide.group(0), alkyne.group(0)), product[0],
                _evidence(protocol, [azide, alkyne, copper], product),
                _source_aliases(protocol, (azide.group(0), alkyne.group(0))),
            )
    return None


def plan_identity_issues(evidence: ProtocolIdentityEvidence, plan: dict[str, Any]) -> list[str]:
    """Check source anchors against identity-bearing plan fields only.

    This is not a validation of mechanism, SMILES, yield, or complete chemistry.
    Generic product descriptions (e.g. corresponding triazole) constrain the
    product class and regioisomer, not a fabricated molecular identity.
    """
    names = [str(plan.get("reaction_name") or "")]
    reagents = plan.get("reagents") or []
    feed_names = [str(item.get("name") or "") for item in reagents if not re.search(
        r"product|intermediate|solvent|catalyst", str(item.get("role") or ""), re.I
    )]
    feed_keys = set().union(*(_identity_keys(name) for name in (feed_names or names)))
    issues = []
    for index, source in enumerate(evidence.substrates):
        aliases = evidence.substrate_aliases[index] if index < len(evidence.substrate_aliases) else ()
        accepted_keys = {chemical_name_key(name) for name in (source, *aliases)}
        if not accepted_keys.intersection(feed_keys):
            issues.append(f"Protocol substrate {source!r} is not preserved in the plan's reaction name/reactant identities.")
    products = [str(item.get("name") or "") for item in reagents if re.search(
        r"(?:target\s+)?product", str(item.get("role") or ""), re.I
    ) and not re.search(r"by[- ]?product", str(item.get("role") or ""), re.I)]
    output_text = " ".join(products or names)
    if evidence.family == "cuaac":
        if products and not re.search(r"1,2,3-triazole", output_text, re.I):
            issues.append("The plan's stated product does not preserve the protocol's 1,2,3-triazole.")
        if re.search(r"1,2,4-triazole|1,5[- ]disubstituted|\b1-[a-z]+-5-", output_text, re.I):
            issues.append("The plan changes the protocol's 1,2,3-triazole/1,4 product identity.")
    else:
        expected = chemical_name_key(evidence.product)
        # Preserve an explicitly stated stereoisomer; permit the plan to retain
        # starting stereochemistry when the protocol's product omits it.
        output_keys = set().union(*(_identity_keys(name) for name in (products or names)))
        variants = {expected}
        if "methionine" in expected and not re.search(r"[ld]methionine", expected):
            starting_stereo = {match[1] for source in evidence.substrates
                               if (match := re.search(r"([ld])methionine", chemical_name_key(source)))}
            variants |= {expected.replace("methionine", stereo + "methionine") for stereo in starting_stereo}
        if not variants.intersection(output_keys):
            issues.append(f"Protocol product {evidence.product!r} is not preserved in the plan's reaction name/product identities.")
    return issues
