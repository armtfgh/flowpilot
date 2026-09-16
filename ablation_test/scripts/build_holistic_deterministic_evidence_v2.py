"""Export deterministic checks and path-level numerical evidence for the holistic audit."""

from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ablation_test.src.cases import load_cases_from_path
from ablation_test.src.error_audit import audit_result, load_expectations


CAMPAIGN = ROOT / "ablation_results/newgen_benchmark/newgen_holistic_error_audit_v2_20260814"
EXPECTATIONS = ROOT / "ablation_test/benchmarks/newgen_error_expectations_v1.json"
CASE_PATHS = (
    ROOT / "ablation_test/benchmarks/newgen_benchmark_v1_pilot_cuaac/case.json",
    ROOT / "ablation_test/benchmarks/newgen_benchmark_v1_pilot_hydrogenolysis/case.json",
    ROOT / "ablation_test/benchmarks/newgen_benchmark_v1_pilot_multistep/case.json",
)
CASE_IDS = {
    "CuAAC": "cuaac_adsc_200900726",
    "Hydrogenolysis": "hydrogenolysis_oprd_9b00416",
    "Two-stage amidation": "multistep_amidation_c5ra20838f",
}
NUMERIC_TERMS = (
    "residence", "time_min", "flow", "volume", "temperature", "pressure", "bpr",
    "concentration", "equiv", "tubing_id", "diameter", "reynolds", "peclet",
    "damkohler", "productivity", "throughput", "startup_waste", "delta_p", "ua_",
)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def numeric_occurrences(value: Any, path: str = "$") -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if isinstance(value, dict):
        for key, nested in value.items():
            child = f"{path}.{key}"
            key_lower = str(key).lower()
            if isinstance(nested, (int, float)) and not isinstance(nested, bool) and math.isfinite(float(nested)):
                if any(term in key_lower for term in NUMERIC_TERMS):
                    rows.append({"json_path": child, "field": key, "value": nested})
            rows.extend(numeric_occurrences(nested, child))
    elif isinstance(value, list):
        for index, nested in enumerate(value):
            rows.extend(numeric_occurrences(nested, f"{path}[{index}]"))
    return rows


def main() -> None:
    evidence = CAMPAIGN / "deterministic_evidence"
    cases: dict[str, Any] = {}
    for path in CASE_PATHS:
        for case in load_cases_from_path(path):
            cases[case.case_id] = case
    expectations = load_expectations(EXPECTATIONS)
    key = read_json(CAMPAIGN / "frozen/candidate_key_confidential.json")["candidates"]
    criteria_rows: list[dict[str, Any]] = []
    occurrence_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for candidate in key:
        result = read_json(Path(candidate["run_directory"]) / "result.json")
        case_id = CASE_IDS[candidate["case"]]
        audit = audit_result(cases[case_id], result, expectations[case_id])
        write_json(evidence / "audits" / f"{candidate['candidate_id']}.json", audit)
        summary_rows.append({
            "candidate_id": candidate["candidate_id"],
            "generator_model": candidate["generator_model"],
            "architecture": candidate["architecture"],
            "case": candidate["case"],
            "applicable_criteria": audit["applicable_criteria"],
            "deterministic_errors": audit["total_errors"],
            "critical_errors": audit["critical_errors"],
            "failed_criterion_ids": ";".join(audit["failed_criterion_ids"]),
        })
        for finding in audit["criteria"]:
            criteria_rows.append({
                "candidate_id": candidate["candidate_id"],
                "generator_model": candidate["generator_model"],
                "architecture": candidate["architecture"],
                "case": candidate["case"],
                **finding,
            })
        for occurrence in numeric_occurrences(result):
            occurrence_rows.append({
                "candidate_id": candidate["candidate_id"],
                "generator_model": candidate["generator_model"],
                "architecture": candidate["architecture"],
                "case": candidate["case"],
                **occurrence,
            })
    write_csv(evidence / "deterministic_summary.csv", summary_rows)
    write_csv(evidence / "deterministic_criteria.csv", criteria_rows)
    write_csv(evidence / "numeric_occurrences.csv", occurrence_rows)
    write_json(evidence / "manifest.json", {
        "schema_version": "newgen_holistic_deterministic_evidence_v2.0",
        "candidate_count": len(key),
        "criterion_rows": len(criteria_rows),
        "numeric_occurrences": len(occurrence_rows),
        "purpose": "Independent machine-calculated corroboration and model-repair evidence; not an LLM score.",
        "tolerances": {"general_relative": 0.05, "gas_relative": 0.10},
    })
    print(evidence)


if __name__ == "__main__":
    main()
