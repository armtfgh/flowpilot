from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from ablation_test.src.cases import load_cases
from ablation_test.src.metrics import score_run
from ablation_test.src.paths import resolve_artifact_path
from ablation_test.src.runner import write_checksums


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def finalize_invalid_outputs(experiment: Path) -> list[Path]:
    """Score preserved malformed one-shot outputs without regenerating them."""
    cases = {case.case_id: case for case in load_cases()}
    finalized: list[Path] = []

    for summary_path in sorted(experiment.glob("cross_model/**/run_summary.json")):
        run_dir = summary_path.parent
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if summary.get("status") != "failed":
            continue
        error = json.loads((run_dir / "error.json").read_text(encoding="utf-8"))
        raw_path = run_dir / "raw_response.json"
        if error.get("type") != "JSONDecodeError" or not raw_path.exists():
            continue

        raw = json.loads(raw_path.read_text(encoding="utf-8"))
        parse_error = str(error.get("error", "Malformed JSON"))
        result = {
            "variant": "general_one_shot",
            "proposal": {},
            "schema_valid": False,
            "output_validity": "malformed_json",
            "parse_error": parse_error,
        }
        case_id = run_dir.parent.name
        metrics = score_run(cases[case_id], result, run_dir)
        _write_json(run_dir / "result.json", result)
        _write_json(
            run_dir / "schema_error.json",
            {
                "error": parse_error,
                "classification": "malformed_json",
                "finish_reason": raw.get("finish_reason")
                or raw.get("stop_reason"),
                "preserved_original_response": True,
            },
        )
        summary.update(
            {
                "status": "completed",
                "observed_outcome": "malformed_json",
                "parse_error": parse_error,
                "external_metrics": metrics,
            }
        )
        summary.pop("error", None)
        summary.pop("error_type", None)
        _write_json(run_dir / "metrics.json", metrics)
        _write_json(summary_path, summary)
        write_checksums(run_dir)
        finalized.append(run_dir)

    manifest_path = experiment / "run_manifest.csv"
    with manifest_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    finalized_resolved = {str(path.resolve()) for path in finalized}
    for row in rows:
        if str(resolve_artifact_path(row["run_dir"]).resolve()) in finalized_resolved:
            row["status"] = "completed"
            row["error"] = ""
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    experiment_summary_path = experiment / "experiment_summary.json"
    experiment_summary = json.loads(
        experiment_summary_path.read_text(encoding="utf-8")
    )
    status_counts: dict[str, int] = {}
    for row in rows:
        status_counts[row["status"]] = status_counts.get(row["status"], 0) + 1
    experiment_summary["status_counts"] = status_counts
    experiment_summary["invalid_outputs_scored"] = len(finalized)
    _write_json(experiment_summary_path, experiment_summary)
    return finalized


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Score preserved malformed one-shot outputs as invalid."
    )
    parser.add_argument("--experiment", type=Path, required=True)
    args = parser.parse_args()
    finalized = finalize_invalid_outputs(args.experiment)
    print(
        json.dumps(
            {
                "experiment": str(args.experiment.resolve()),
                "invalid_outputs_scored": len(finalized),
                "run_dirs": [str(path.resolve()) for path in finalized],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
