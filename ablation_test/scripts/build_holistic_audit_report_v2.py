"""Aggregate the holistic audit into consensus errors and model-repair tickets."""

from __future__ import annotations

import csv
import json
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np

from ablation_test.src.holistic_audit import validate_response

CAMPAIGN = ROOT / "ablation_results/newgen_benchmark/newgen_holistic_error_audit_v2_20260814"
OUTPUT = ROOT / "deliverables/flowpilot_newgen_holistic_error_audit_v2_20260814"
SEVERITY_ORDER = {"NONE": 0, "MINOR": 1, "MAJOR": 2, "CRITICAL": 3}
DETERMINISTIC_MAP = {
    "HC-01": {"NG-01", "NG-05"}, "HC-02": {"NG-02", "NG-03", "NG-12"},
    "HC-03": {"NG-04", "NG-06"}, "HN-01": {"NG-09"},
    "HN-02": {"NG-10", "NG-11", "NG-12", "NG-18"},
    "HN-03": {"NG-15", "NG-16", "NG-17"}, "HN-04": {"NG-13", "NG-14"},
    "HN-05": {"NG-09"}, "HP-01": {"NG-04", "NG-20"},
    "HP-02": {"NG-19"}, "HP-03": {"NG-19"}, "HP-04": {"NG-19"},
    "HS-01": {"NG-20"}, "HS-02": {"NG-20"}, "HS-03": {"NG-20"},
    "HS-04": {"NG-20"}, "HS-05": {"NG-20"}, "HA-01": {"NG-13", "NG-19"},
    "HA-03": {"NG-08", "NG-19", "NG-20"}, "HA-05": {"NG-08"},
}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def resolved_response(
    judge: str, domain: str, candidate_id: str, rubric: dict[str, Any]
) -> dict[str, Any] | None:
    root = CAMPAIGN / "judgments" / judge / domain / candidate_id
    candidates = [(root / "status.json", root / "parsed_response.json")]
    candidates += [
        (path, path.with_name("parsed_response.json"))
        for path in sorted((root / "attempts").glob("attempt_*/status.json"))
    ]
    existing = [pair for pair in candidates if pair[0].is_file() and pair[1].is_file()]
    for status_path, response_path in sorted(existing, key=lambda pair: pair[0].stat().st_mtime_ns, reverse=True):
        response = read_json(response_path)
        if read_json(status_path).get("status") == "valid" and not validate_response(response, rubric, domain, candidate_id):
            return response
    return None


def save_figure(fig: plt.Figure, name: str) -> None:
    for suffix in ("png", "svg", "pdf"):
        fig.savefig(OUTPUT / "figures" / f"{name}.{suffix}", dpi=300 if suffix == "png" else None, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    preserved_docs = {}
    for name in ("BENCHMARK_REVIEW.md", "MODEL_REPAIR_PRIORITIES.md"):
        path = OUTPUT / name
        if path.is_file():
            preserved_docs[name] = path.read_text(encoding="utf-8")
    if OUTPUT.exists():
        shutil.rmtree(OUTPUT)
    (OUTPUT / "figures").mkdir(parents=True)
    (OUTPUT / "tables").mkdir()
    rubric = read_json(CAMPAIGN / "frozen/rubric.json")
    candidates = read_json(CAMPAIGN / "frozen/candidate_key_confidential.json")["candidates"]
    candidate_by_id = {row["candidate_id"]: row for row in candidates}
    judges = ("qwen", "openai", "claude")
    finding_rows: list[dict[str, Any]] = []
    missing_calls: list[dict[str, str]] = []
    for candidate in candidates:
        for domain in rubric["domains"]:
            for judge in judges:
                response = resolved_response(judge, domain, candidate["candidate_id"], rubric)
                if response is None:
                    missing_calls.append({"candidate_id": candidate["candidate_id"], "domain": domain, "judge": judge})
                    continue
                for finding in response["criterion_findings"]:
                    finding_rows.append({
                        "candidate_id": candidate["candidate_id"], "generator_model": candidate["generator_model"],
                        "generator_family": candidate["generator_family"], "architecture": candidate["architecture"],
                        "case": candidate["case"], "judge": judge, "domain": domain,
                        "criterion_id": finding["criterion_id"], "status": finding["status"],
                        "severity": finding["severity"], "error_title": finding["error_title"],
                        "evidence_paths": json.dumps(finding["evidence_paths"], ensure_ascii=False),
                        "observed_values": json.dumps(finding["observed_values"], ensure_ascii=False),
                        "expected_or_correct": finding["expected_or_correct"],
                        "explanation": finding["explanation"], "required_correction": finding["required_correction"],
                        "source_basis": json.dumps(finding["source_basis"]), "confidence": finding["confidence"],
                    })
    write_csv(OUTPUT / "tables/judge_findings.csv", finding_rows)
    write_csv(OUTPUT / "tables/missing_or_invalid_calls.csv", missing_calls)

    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in finding_rows:
        grouped[(row["candidate_id"], row["domain"], row["criterion_id"])].append(row)
    consensus_rows: list[dict[str, Any]] = []
    for (candidate_id, domain, criterion_id), rows in grouped.items():
        candidate = candidate_by_id[candidate_id]
        cross_names = ("openai", "claude") if candidate["generator_family"] == "qwen" else ("qwen", "claude")
        by_judge = {row["judge"]: row for row in rows}
        cross = [by_judge[name] for name in cross_names if name in by_judge]
        statuses = [row["status"] for row in cross]
        if len(cross) < 2:
            consensus = "INCOMPLETE"
        elif statuses == ["ERROR", "ERROR"]:
            consensus = "CONFIRMED_ERROR"
        elif statuses == ["PASS", "PASS"]:
            consensus = "CONFIRMED_PASS"
        elif statuses == ["NOT_ASSESSABLE", "NOT_ASSESSABLE"]:
            consensus = "NOT_ASSESSABLE"
        else:
            consensus = "DISAGREEMENT"
        severity = max((row["severity"] for row in cross), key=lambda x: SEVERITY_ORDER[x], default="NONE")
        counts = Counter(row["status"] for row in rows)
        consensus_rows.append({
            "candidate_id": candidate_id, "generator_model": candidate["generator_model"],
            "architecture": candidate["architecture"], "case": candidate["case"], "domain": domain,
            "criterion_id": criterion_id, "cross_family_judges": "+".join(cross_names),
            "cross_family_consensus": consensus, "consensus_severity": severity,
            "qwen_status": by_judge.get("qwen", {}).get("status", "MISSING"),
            "openai_status": by_judge.get("openai", {}).get("status", "MISSING"),
            "claude_status": by_judge.get("claude", {}).get("status", "MISSING"),
            "all_judges_error_votes": counts["ERROR"], "all_judges_pass_votes": counts["PASS"],
            "all_judges_not_assessable_votes": counts["NOT_ASSESSABLE"],
        })
    write_csv(OUTPUT / "tables/criterion_consensus.csv", consensus_rows)
    confirmed = [row for row in consensus_rows if row["cross_family_consensus"] == "CONFIRMED_ERROR"]
    write_csv(OUTPUT / "tables/confirmed_errors.csv", confirmed)
    write_csv(OUTPUT / "tables/llm_consensus_error_claims.csv", confirmed)

    deterministic_rows = list(csv.DictReader(
        (CAMPAIGN / "deterministic_evidence/deterministic_criteria.csv").open(encoding="utf-8")
    ))
    deterministic_failures: dict[str, set[str]] = defaultdict(set)
    for item in deterministic_rows:
        if item["status"] == "FAIL":
            deterministic_failures[item["candidate_id"]].add(item["criterion_id"])
    tickets: list[dict[str, Any]] = []
    for index, row in enumerate((item for item in confirmed if item["architecture"] == "FlowPilot"), 1):
        source = [item for item in finding_rows if item["candidate_id"] == row["candidate_id"] and item["criterion_id"] == row["criterion_id"] and item["judge"] in row["cross_family_judges"].split("+")]
        evidence_paths = [path for item in source for path in json.loads(item["evidence_paths"])]
        mapped = DETERMINISTIC_MAP.get(row["criterion_id"], set())
        corroborating = sorted(mapped & deterministic_failures[row["candidate_id"]])
        combined_text = " ".join(
            item["error_title"] + " " + item["explanation"] + " " + item["observed_values"]
            for item in source
        ).lower()
        tolerance_review = any(term in combined_text for term in (
            "rounding", "precision", "0.1111 vs 0.11111", "0.11111 vs 0.1111"
        ))
        intermediate_markers = ("pre_council", "raw_proposal", "council_messages", "deliberation_log")
        intermediate_only = bool(evidence_paths) and all(any(marker in path for marker in intermediate_markers) for path in evidence_paths)
        if corroborating:
            triage = "DETERMINISTICALLY_CORROBORATED"
        elif tolerance_review:
            triage = "TOLERANCE_REVIEW_REQUIRED"
        elif intermediate_only:
            triage = "INTERMEDIATE_RECORD_REVIEW_REQUIRED"
        else:
            triage = "LLM_CONSENSUS_REQUIRES_TECHNICAL_REVIEW"
        tickets.append({
            "ticket_id": f"FP-AUDIT-{index:03d}", "status": "NEEDS_TECHNICAL_ADJUDICATION",
            "triage_class": triage, "severity": row["consensus_severity"],
            "deterministic_corroborating_criteria": ";".join(corroborating),
            "possible_tolerance_overcall": tolerance_review,
            "intermediate_only_evidence": intermediate_only,
            "generator_model": row["generator_model"], "case": row["case"], "candidate_id": row["candidate_id"],
            "domain": row["domain"], "criterion_id": row["criterion_id"],
            "error_titles": " | ".join(item["error_title"] for item in source),
            "evidence_paths": " | ".join(item["evidence_paths"] for item in source),
            "observed_values": " | ".join(item["observed_values"] for item in source),
            "required_corrections": " | ".join(item["required_correction"] for item in source),
            "source_result": str(Path(candidate_by_id[row["candidate_id"]]["run_directory"]) / "result.json"),
        })
    write_csv(OUTPUT / "tables/model_fix_tickets.csv", tickets)
    write_json(OUTPUT / "tables/model_fix_tickets.json", tickets)

    # Error burden is shown as raw criterion counts, never collapsed to a quality score.
    cells = [(model, arch) for model in ("Qwen3.6-27B", "GPT-5.4") for arch in ("One-shot", "FlowPilot")]
    domains = list(rubric["domains"])
    count_lookup = Counter((row["generator_model"], row["architecture"], row["domain"]) for row in confirmed)
    matrix = np.array([[count_lookup[(model, arch, domain)] for domain in domains] for model, arch in cells])
    fig, ax = plt.subplots(figsize=(11.5, 5.5))
    image = ax.imshow(matrix, cmap="Reds", aspect="auto")
    ax.set_xticks(range(len(domains)), [item.replace("_", "\n") for item in domains])
    ax.set_yticks(range(len(cells)), [f"{model} | {arch}" for model, arch in cells])
    for y in range(matrix.shape[0]):
        for x in range(matrix.shape[1]):
            ax.text(x, y, str(matrix[y, x]), ha="center", va="center", color="black")
    ax.set_title("Cross-family LLM error claims by audit domain", loc="left", fontweight="bold")
    fig.colorbar(image, ax=ax, label="Consensus error claims")
    save_figure(fig, "confirmed_errors_by_domain")

    status_counts = Counter(row["cross_family_consensus"] for row in consensus_rows)
    fig, ax = plt.subplots(figsize=(7.8, 4.8))
    labels = ["CONFIRMED_ERROR", "CONFIRMED_PASS", "DISAGREEMENT", "NOT_ASSESSABLE", "INCOMPLETE"]
    values = [status_counts[label] for label in labels]
    display_labels = ["LLM_ERROR_CLAIM", "CONSENSUS_PASS", "DISAGREEMENT", "NOT_ASSESSABLE", "INCOMPLETE"]
    bars = ax.bar(display_labels, values, color=["#C94747", "#2F8F62", "#E0A12B", "#8A919C", "#4E6E8E"])
    ax.bar_label(bars, padding=3)
    ax.tick_params(axis="x", rotation=25)
    ax.set_ylabel("Candidate-criterion records")
    ax.set_title("Consensus coverage and disagreement", loc="left", fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    save_figure(fig, "consensus_status_counts")

    deterministic = list(csv.DictReader((CAMPAIGN / "deterministic_evidence/deterministic_summary.csv").open(encoding="utf-8")))
    write_json(OUTPUT / "SUMMARY.json", {
        "schema_version": "newgen_holistic_error_audit_report_v2.0", "composite_score": False,
        "winner_ranking": False, "candidate_count": len(candidates), "planned_judge_calls": len(candidates) * len(domains) * 3,
        "valid_judge_calls": sum(
            1 for c in candidates for d in domains for j in judges
            if resolved_response(j, d, c["candidate_id"], rubric)
        ),
        "missing_or_invalid_calls": len(missing_calls), "criterion_consensus_counts": dict(status_counts),
        "llm_consensus_error_claim_count": len(confirmed),
        "flowpilot_repair_candidate_count": len(tickets),
        "publication_status": "DIAGNOSTIC_ONLY_PENDING_TECHNICAL_ADJUDICATION",
        "deterministic_candidate_summaries": deterministic,
    })
    shutil.copytree(CAMPAIGN / "frozen", OUTPUT / "frozen")
    shutil.copytree(CAMPAIGN / "deterministic_evidence", OUTPUT / "deterministic_evidence")
    (OUTPUT / "README.md").write_text(
        "# NewGen Holistic Error Audit v2\n\n"
        "This package contains no composite quality score and no winner ranking. Each of three independent LLM judges "
        "audits all six domains. The primary result is a fixed-criterion, cross-family unanimous LLM error claim. "
        "`NOT_ASSESSABLE` and disagreement are reported separately. Machine-calculated checks are retained as independent corroboration.\n\n"
        "LLM consensus is not ground truth. Repair candidates remain `NEEDS_TECHNICAL_ADJUDICATION`; rounding-only and "
        "intermediate-record concerns are flagged rather than silently counted as established defects.\n\n"
        "Use `tables/model_fix_tickets.csv` for FlowPilot repair triage, `tables/judge_findings.csv` for exact evidence, "
        "and `deterministic_evidence/numeric_occurrences.csv` to trace numerical values across the complete record.\n",
        encoding="utf-8",
    )
    for name, content in preserved_docs.items():
        (OUTPUT / name).write_text(content, encoding="utf-8")
    print(OUTPUT)


if __name__ == "__main__":
    main()
