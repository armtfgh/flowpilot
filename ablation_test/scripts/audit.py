from __future__ import annotations

import csv
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from ablation_test.src.cases import (
    FORBIDDEN_CASE_TERMS,
    ROOT,
    assert_no_forbidden_cases,
    load_cases,
)
from ablation_test.src.paths import RESULTS_ROOT


PROJECT_ROOT = ROOT.parent
RECORDS_DIR = PROJECT_ROOT / "flora_translate" / "data" / "records"


def _record_index() -> dict[str, Path]:
    index: dict[str, Path] = {}
    for path in RECORDS_DIR.glob("*.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if not isinstance(payload, dict):
            continue
        source = str(payload.get("source_pdf", ""))
        if source:
            index[source.lower()] = path
    return index


def main() -> int:
    cases = load_cases(include_adversarial=True)
    assert_no_forbidden_cases(cases)
    index = _record_index()
    rows: list[dict] = []
    missing_sources: list[str] = []

    for case in cases:
        source_path = None
        matched_source = ""
        for source in case.excluded_record_ids:
            if source.lower() in index:
                source_path = index[source.lower()]
                matched_source = source
                break
        if case.source_record_id and source_path is None:
            missing_sources.append(case.case_id)
        source_sha = (
            hashlib.sha256(source_path.read_bytes()).hexdigest()
            if source_path
            else ""
        )
        rows.append(
            {
                "case_id": case.case_id,
                "suite_id": case.suite_id,
                "category": case.category,
                "protocol_sha256": case.protocol_sha256,
                "source_record_id": case.source_record_id,
                "matched_source_id": matched_source,
                "source_record_path": str(source_path or ""),
                "source_record_sha256": source_sha,
                "reference_quality": case.reference_quality,
            }
        )

    manifest_dir = ROOT / "sources"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    with (manifest_dir / "source_manifest.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    protocol_files = sorted((ROOT / "protocols").glob("*.json"))
    forbidden_hits: list[dict] = []
    for path in protocol_files:
        text = path.read_text(encoding="utf-8").lower()
        for term in FORBIDDEN_CASE_TERMS:
            if term.lower() in text:
                forbidden_hits.append({"file": str(path), "term": term})

    report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "case_count": len(cases),
        "literature_case_count": sum(
            case.suite_id.startswith("literature") for case in cases
        ),
        "adversarial_case_count": sum(
            case.suite_id.startswith("adversarial") for case in cases
        ),
        "duplicate_case_ids": len({case.case_id for case in cases}) != len(cases),
        "forbidden_terms": list(FORBIDDEN_CASE_TERMS),
        "forbidden_hits": forbidden_hits,
        "missing_source_records": missing_sources,
        "source_manifest": str(manifest_dir / "source_manifest.csv"),
        "passed": not forbidden_hits and not missing_sources,
    }
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    (RESULTS_ROOT / "audit_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
