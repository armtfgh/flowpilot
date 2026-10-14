from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .cases import AblationCase


VALID_DISPOSITIONS = {"EXECUTE", "SCREEN", "BLOCK"}


def canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _proposal(result: dict[str, Any]) -> dict[str, Any]:
    proposal = result.get("proposal")
    return proposal if isinstance(proposal, dict) else {}


def _field(payload: dict[str, Any], path: str) -> Any:
    current: Any = payload
    for part in path.split("."):
        if not isinstance(current, dict):
            return None
        current = current.get(part)
    return current


def _reported_disposition(result: dict[str, Any]) -> str:
    direct = str(result.get("reported_disposition") or "").upper()
    if direct in VALID_DISPOSITIONS:
        return direct

    proposal = _proposal(result)
    raw = result.get("raw_proposal")
    raw = raw if isinstance(raw, dict) else {}
    candidate = str(
        raw.get("recommended_disposition")
        or proposal.get("recommended_disposition")
        or ""
    ).upper()
    if candidate in VALID_DISPOSITIONS:
        return candidate

    flags = " ".join(map(str, proposal.get("safety_flags") or [])).lower()
    validation = result.get("final_validation")
    validation = validation if isinstance(validation, dict) else {}
    if any(marker in flags for marker in ("block", "do not execute", "infeasible")):
        return "BLOCK"
    if (
        validation.get("status") == "screen_required"
        or "screen_required" in flags
        or "screen required" in flags
        or str(proposal.get("confidence") or "").upper() == "LOW"
    ):
        return "SCREEN"
    return "EXECUTE" if proposal else "BLOCK"


def evaluate_constraint(
    proposal: dict[str, Any],
    constraint: dict[str, Any],
) -> dict[str, Any]:
    path = str(constraint["field"])
    observed = _field(proposal, path)
    passed = True
    reasons: list[str] = []

    if constraint.get("required") and observed is None:
        passed = False
        reasons.append("missing")

    if observed is not None and "min" in constraint:
        try:
            if float(observed) < float(constraint["min"]):
                passed = False
                reasons.append("below_min")
        except (TypeError, ValueError):
            passed = False
            reasons.append("not_numeric")

    if observed is not None and "max" in constraint:
        try:
            if float(observed) > float(constraint["max"]):
                passed = False
                reasons.append("above_max")
        except (TypeError, ValueError):
            passed = False
            reasons.append("not_numeric")

    if observed is not None and "allowed" in constraint:
        allowed = constraint["allowed"]
        tolerance = float(constraint.get("tolerance", 1e-6))
        if isinstance(observed, (int, float)):
            match = any(abs(float(observed) - float(item)) <= tolerance for item in allowed)
        else:
            match = str(observed).lower() in {str(item).lower() for item in allowed}
        if not match:
            passed = False
            reasons.append("not_allowed")

    return {
        "constraint_id": constraint["constraint_id"],
        "field": path,
        "observed": observed,
        "passed": passed,
        "reasons": reasons,
    }


def score_scenario(case: AblationCase, result: dict[str, Any]) -> dict[str, Any]:
    proposal = _proposal(result)
    disposition = _reported_disposition(result)
    checks = [
        evaluate_constraint(proposal, constraint)
        for constraint in case.oracle_constraints
    ]
    expected = case.expected_disposition.upper()
    disposition_correct = disposition == expected

    # A BLOCK response is not penalized for omitting an intentionally impossible
    # numeric design. For executable/screen designs, every oracle constraint applies.
    if expected == "BLOCK" and disposition == "BLOCK":
        critical_pass = True
        violation_count = 0
    else:
        critical_pass = all(check["passed"] for check in checks)
        violation_count = sum(not check["passed"] for check in checks)

    return {
        "schema_version": "flowpilot_stage1_oracle_v1.0",
        "case_id": case.case_id,
        "scenario_id": case.scenario_id or case.case_id,
        "pair_id": case.pair_id,
        "scenario_kind": case.scenario_kind,
        "expected_disposition": expected,
        "reported_disposition": disposition,
        "disposition_correct": disposition_correct,
        "critical_engineering_pass": critical_pass,
        "critical_violation_count": violation_count,
        "constraint_checks": checks,
        "design_input_sha256": case.design_input_sha256,
    }


def validate_oracle_witness(case: AblationCase) -> dict[str, Any]:
    witness = case.oracle_witness
    result = {
        "proposal": witness.get("proposal", {}),
        "reported_disposition": witness.get("reported_disposition"),
        "schema_valid": True,
    }
    score = score_scenario(case, result)
    return {
        "scenario_id": case.scenario_id or case.case_id,
        "passed": bool(
            score["disposition_correct"]
            and score["critical_engineering_pass"]
            and score["critical_violation_count"] == 0
        ),
        "score": score,
    }


def write_tree_checksums(root: Path) -> Path:
    rows: list[str] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.name == "checksums.sha256":
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        rows.append(f"{digest}  {path.relative_to(root)}")
    destination = root / "checksums.sha256"
    destination.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return destination
