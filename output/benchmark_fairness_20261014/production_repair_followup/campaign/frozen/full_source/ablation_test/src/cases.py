from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN_CASE_TERMS = (
    "tetrahydroquinoline",
    "6-methylquinoline",
    "profpark",
)


@dataclass(frozen=True)
class AblationCase:
    case_id: str
    title: str
    protocol: str
    suite_id: str
    category: str
    difficulty: str = "high"
    tags: tuple[str, ...] = ()
    source_record_id: str = ""
    source_record_aliases: tuple[str, ...] = ()
    reference_quality: str = "not_applicable"
    expected_features: dict[str, Any] = field(default_factory=dict)
    reference_flow: dict[str, Any] = field(default_factory=dict)
    objective: str = (
        "Translate the batch protocol into a feasible, safe, and testable "
        "continuous-flow design."
    )
    hard_constraints: tuple[str, ...] = ()
    inventory: dict[str, Any] = field(default_factory=dict)
    chemistry_identity_confirmation: dict[str, Any] = field(default_factory=dict)
    scenario_id: str = ""
    pair_id: str = ""
    scenario_kind: str = ""
    expected_disposition: str = ""
    controlled_change: str = ""
    oracle_constraints: tuple[dict[str, Any], ...] = ()
    oracle_witness: dict[str, Any] = field(default_factory=dict)

    @property
    def protocol_sha256(self) -> str:
        return hashlib.sha256(self.protocol.encode("utf-8")).hexdigest()

    @property
    def design_input_text(self) -> str:
        """Canonical text delivered to every benchmark condition."""
        if not (
            self.hard_constraints
            or self.inventory
            or self.scenario_id
        ):
            return self.protocol
        sections = [
            "RAW BATCH PROTOCOL:",
            self.protocol.strip(),
            "",
            "OBJECTIVE:",
            self.objective.strip(),
            "",
            "HARD CONSTRAINTS:",
            *[f"- {item}" for item in self.hard_constraints],
            "",
            "AVAILABLE INVENTORY (authoritative JSON):",
            json.dumps(
                self.inventory,
                sort_keys=True,
                ensure_ascii=False,
                separators=(",", ":"),
            ),
        ]
        if self.chemistry_identity_confirmation:
            sections.extend(
                [
                    "",
                    "CHEMIST-CONFIRMED CHEMISTRY IDENTITY:",
                    json.dumps(
                        self.chemistry_identity_confirmation,
                        sort_keys=True,
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ),
                ]
            )
        return "\n".join(sections).strip()

    @property
    def design_input_sha256(self) -> str:
        return hashlib.sha256(self.design_input_text.encode("utf-8")).hexdigest()

    @property
    def excluded_record_ids(self) -> set[str]:
        return {
            value
            for value in (self.source_record_id, *self.source_record_aliases)
            if value
        }

    def public_payload(self) -> dict[str, Any]:
        """Payload allowed to enter an LLM prompt."""
        return {
            "case_id": self.case_id,
            "scenario_id": self.scenario_id or self.case_id,
            "title": self.title,
            "protocol": self.protocol,
            "category": self.category,
            "objective": self.objective,
            "hard_constraints": list(self.hard_constraints),
            "inventory": self.inventory,
            "chemistry_identity_confirmation": self.chemistry_identity_confirmation,
            "design_input_text": self.design_input_text,
            "design_input_sha256": self.design_input_sha256,
        }

    def manifest_payload(self) -> dict[str, Any]:
        return {
            **self.public_payload(),
            "suite_id": self.suite_id,
            "difficulty": self.difficulty,
            "tags": list(self.tags),
            "source_record_id": self.source_record_id,
            "source_record_aliases": list(self.source_record_aliases),
            "reference_quality": self.reference_quality,
            "expected_features": self.expected_features,
            "reference_flow": self.reference_flow,
            "protocol_sha256": self.protocol_sha256,
            "scenario_id": self.scenario_id,
            "pair_id": self.pair_id,
            "scenario_kind": self.scenario_kind,
            "expected_disposition": self.expected_disposition,
            "controlled_change": self.controlled_change,
            "oracle_constraints": list(self.oracle_constraints),
            "oracle_witness": self.oracle_witness,
        }


def _load_file(path: Path) -> list[AblationCase]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    suite_id = payload["suite_id"]
    inventories = payload.get("inventories", {})
    cases: list[AblationCase] = []
    for raw in payload["cases"]:
        inventory = raw.get("inventory", {})
        inventory_id = raw.get("inventory_id")
        if inventory_id:
            if inventory_id not in inventories:
                raise KeyError(f"Unknown inventory_id {inventory_id!r} in {path}")
            inventory = inventories[inventory_id]
        cases.append(
            AblationCase(
                case_id=raw["case_id"],
                title=raw["title"],
                protocol=raw["protocol"],
                suite_id=suite_id,
                category=raw["category"],
                difficulty=raw.get("difficulty", "high"),
                tags=tuple(raw.get("tags", [])),
                source_record_id=raw.get("source_record_id", ""),
                source_record_aliases=tuple(raw.get("source_record_aliases", [])),
                reference_quality=raw.get("reference_quality", "not_applicable"),
                expected_features=raw.get("expected_features", {}),
                reference_flow=raw.get("reference_flow", {}),
                objective=raw.get(
                    "objective",
                    "Translate the batch protocol into a feasible, safe, and "
                    "testable continuous-flow design.",
                ),
                hard_constraints=tuple(raw.get("hard_constraints", [])),
                inventory=inventory,
                chemistry_identity_confirmation=raw.get(
                    "chemistry_identity_confirmation", {}
                ),
                scenario_id=raw.get("scenario_id", ""),
                pair_id=raw.get("pair_id", ""),
                scenario_kind=raw.get("scenario_kind", ""),
                expected_disposition=raw.get("expected_disposition", ""),
                controlled_change=raw.get("controlled_change", ""),
                oracle_constraints=tuple(raw.get("oracle_constraints", [])),
                oracle_witness=raw.get("oracle_witness", {}),
            )
        )
    return cases


def load_cases_from_path(path: Path) -> list[AblationCase]:
    cases = _load_file(path)
    assert_no_forbidden_cases(cases)
    ids = [case.case_id for case in cases]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate case IDs in ablation suite")
    return cases


def load_cases(include_adversarial: bool = False) -> list[AblationCase]:
    cases = _load_file(ROOT / "protocols" / "literature_cases.json")
    if include_adversarial:
        cases.extend(_load_file(ROOT / "protocols" / "adversarial_cases.json"))
    assert_no_forbidden_cases(cases)
    ids = [case.case_id for case in cases]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate case IDs in ablation suite")
    return cases


def assert_no_forbidden_cases(cases: list[AblationCase]) -> None:
    violations: list[str] = []
    for case in cases:
        searchable = f"{case.case_id}\n{case.title}\n{case.protocol}".lower()
        for term in FORBIDDEN_CASE_TERMS:
            if term.lower() in searchable:
                violations.append(f"{case.case_id}: {term}")
    if violations:
        raise ValueError("Forbidden case contamination: " + ", ".join(violations))


def select_cases(
    cases: list[AblationCase],
    requested_ids: list[str],
) -> list[AblationCase]:
    if requested_ids == ["*"]:
        return cases
    by_id = {case.case_id: case for case in cases}
    missing = [case_id for case_id in requested_ids if case_id not in by_id]
    if missing:
        raise KeyError(f"Unknown case IDs: {missing}")
    return [by_id[case_id] for case_id in requested_ids]
