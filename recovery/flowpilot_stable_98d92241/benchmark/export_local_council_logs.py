from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export full council logs for the local benchmark.")
    parser.add_argument("experiment_dir", help="Path to local_model_benchmark_* directory")
    return parser.parse_args()


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def field_union(rows: list[dict], preferred: list[str] | None = None) -> list[str]:
    ordered: list[str] = []
    seen = set()
    for key in preferred or []:
        if key not in seen:
            ordered.append(key)
            seen.add(key)
    for row in rows:
        for key in row.keys():
            if key not in seen:
                ordered.append(key)
                seen.add(key)
    return ordered


def normalize_row(row: dict, extra_first: list[str] | None = None) -> dict:
    normalized = {}
    for key in extra_first or []:
        if key in row:
            normalized[key] = row[key]
    for key, value in row.items():
        if key in normalized:
            continue
        if isinstance(value, (dict, list)):
            normalized[key] = json.dumps(value, ensure_ascii=False)
        else:
            normalized[key] = value
    return normalized


def flatten_scores(snapshot: dict, stage_label: str) -> list[dict]:
    rows: list[dict] = []
    for domain in ("chemistry", "kinetics", "fluidics", "safety"):
        for item in snapshot.get(f"{domain}_scores", []) or []:
            row = {"stage": stage_label, "domain": domain}
            row.update(item)
            rows.append(normalize_row(row, extra_first=["stage", "domain", "candidate_id"]))
    return rows


def build_markdown(bundle: dict) -> str:
    md: list[str] = []
    meta = bundle["run_summary"]["metadata"]
    final_metrics = bundle["run_summary"].get("final_metrics", {})
    chief = bundle["stage4_chief_selection"].get("chief_data", {})
    stage_events = bundle["stage_events_tail"]

    md.append(f"# Council Log: U_{meta.get('upstream_bundle')} / C_{meta.get('council_bundle')}")
    md.append("")
    md.append("## Final Design")
    md.append(f"- `tau = {final_metrics.get('residence_time_min')} min`")
    md.append(f"- `Q = {final_metrics.get('flow_rate_mL_min')} mL/min`")
    md.append(f"- `d = {final_metrics.get('tubing_ID_mm')} mm`")
    md.append(f"- `V_R = {final_metrics.get('reactor_volume_mL')} mL`")
    md.append(f"- `BPR = {final_metrics.get('BPR_bar')} bar`")
    md.append("")
    md.append("## Chief Selection")
    md.append(f"- Winner id from stage log: `{stage_events.get('winner_id')}`")
    if chief:
        md.append(f"- Selected candidate id in Chief snapshot: `{chief.get('selected_candidate_id')}`")
        md.append(f"- Runner-up ids: `{chief.get('runner_up_ids')}`")
        md.append(f"- Rationale: {chief.get('selection_rationale', '')}")
        resolved = chief.get("resolved_tradeoffs", []) or []
        if resolved:
            md.append("- Resolved tradeoffs:")
            for item in resolved:
                md.append(f"  - {item}")
        uncertainties = chief.get("remaining_uncertainties", []) or []
        if uncertainties:
            md.append("- Remaining uncertainties:")
            for item in uncertainties:
                md.append(f"  - {item}")
    md.append("")
    md.append("## Stage Progress")
    md.append(f"- Stage 2 blocked_by_scoring: `{bundle['stage2_initial_scoring'].get('blocked_by_scoring')}`")
    ref = bundle["stage3_5_refinement_summary"]
    md.append(f"- Stage 3.5 changed_count: `{ref.get('changed_count')}`")
    md.append(f"- Stage 3.5 final_candidate_count: `{ref.get('final_candidate_count')}`")
    md.append(f"- Stage 3.5 dropped_candidate_count: `{ref.get('dropped_candidate_count')}`")
    md.append("")
    md.append("## Final Explanation")
    md.append(bundle["result"].get("formatted_result", {}).get("explanation", ""))
    md.append("")
    md.append("## Files")
    md.append("- `bundle.json`: merged raw data")
    md.append("- `stage1_survivors.csv`: all surviving initial candidates")
    md.append("- `stage1_disqualified.csv`: disqualified initial candidates with reasons")
    md.append("- `stage2_scores_long.csv`: all Stage 2 domain scores and reasoning")
    md.append("- `stage35_final_scores_long.csv`: all final rescoring domain scores and reasoning")
    return "\n".join(md) + "\n"


def main() -> None:
    args = parse_args()
    root = Path(args.experiment_dir).resolve()
    outdir = root / "council_full_logs"
    outdir.mkdir(parents=True, exist_ok=True)

    run_dirs = sorted(root.glob("U_*/C_*/runs/*/budget_*/repeat_01"))
    manifest_rows: list[dict] = []

    for run_dir in run_dirs:
        metadata = load_json(run_dir / "metadata.json")
        meta = metadata["metadata"]
        cell_slug = f"{meta.get('upstream_bundle')}__{meta.get('council_bundle')}"
        cell_dir = outdir / cell_slug
        cell_dir.mkdir(parents=True, exist_ok=True)

        snapshots = run_dir / "snapshots"
        stage1 = load_json(snapshots / "stage1_designer_result.json")
        stage2 = load_json(snapshots / "stage2_initial_scoring.json")
        stage35_summary = load_json(snapshots / "stage3_5_refinement_summary.json")
        stage35_final_scoring = load_json(snapshots / "stage3_5_final_scoring.json")
        stage35_final_audit = load_json(snapshots / "stage3_5_final_audit.json")
        stage4 = load_json(snapshots / "stage4_chief_selection.json")
        stage6 = load_json(snapshots / "stage6_dfmea.json")
        result = load_json(run_dir / "result.json")
        run_summary = load_json(run_dir / "run_summary.json")

        stage_events_tail = {}
        with (run_dir / "stage_events.jsonl").open("r", encoding="utf-8") as handle:
            for line in handle:
                event = json.loads(line)
                if event.get("stage") == "council_stage_4_chief_selection" and event.get("event") == "end":
                    stage_events_tail["winner_id"] = event.get("details", {}).get("winner_id")
                    stage_events_tail["stage4_disqualify_ids"] = event.get("details", {}).get("disqualify_ids")
                if event.get("stage") == "council_stage_7_apply_winner" and event.get("event") == "end":
                    stage_events_tail["applied_changes"] = event.get("details", {}).get("applied_changes")

        bundle = {
            "run_dir": str(run_dir),
            "metadata": metadata,
            "run_summary": run_summary,
            "result": result,
            "stage1_designer_result": stage1,
            "stage2_initial_scoring": stage2,
            "stage3_5_refinement_summary": stage35_summary,
            "stage3_5_final_scoring": stage35_final_scoring,
            "stage3_5_final_audit": stage35_final_audit,
            "stage4_chief_selection": stage4,
            "stage6_dfmea": stage6,
            "stage_events_tail": stage_events_tail,
        }
        write_json(cell_dir / "bundle.json", bundle)

        survivors = [normalize_row(item, extra_first=["id", "tau_min", "Q_mL_min", "d_mm", "V_R_mL", "BPR_bar", "tubing_material", "expected_conversion"]) for item in stage1.get("survivors", [])]
        if survivors:
            write_csv(cell_dir / "stage1_survivors.csv", field_union(survivors), survivors)

        disqualified_rows = []
        for item in stage1.get("disqualified", []):
            row = {"reason": item.get("reason", "")}
            row.update(item.get("candidate", {}))
            disqualified_rows.append(normalize_row(row, extra_first=["id", "tau_min", "Q_mL_min", "d_mm", "V_R_mL", "BPR_bar", "tubing_material", "expected_conversion", "reason"]))
        if disqualified_rows:
            write_csv(cell_dir / "stage1_disqualified.csv", field_union(disqualified_rows), disqualified_rows)

        stage2_rows = flatten_scores(stage2, "stage2_initial")
        if stage2_rows:
            write_csv(cell_dir / "stage2_scores_long.csv", field_union(stage2_rows, preferred=["stage", "domain", "candidate_id", "verdict", "reasoning", "proposed_changes"]), stage2_rows)

        final_rows = flatten_scores(stage35_final_scoring, "stage3_5_final")
        if final_rows:
            write_csv(cell_dir / "stage35_final_scores_long.csv", field_union(final_rows, preferred=["stage", "domain", "candidate_id", "verdict", "reasoning", "proposed_changes"]), final_rows)

        with (cell_dir / "council_log.md").open("w", encoding="utf-8") as handle:
            handle.write(build_markdown(bundle))

        manifest_rows.append(
            {
                "upstream_bundle": meta.get("upstream_bundle"),
                "council_bundle": meta.get("council_bundle"),
                "cell_dir": str(cell_dir),
                "bundle_json": str(cell_dir / "bundle.json"),
                "council_log_md": str(cell_dir / "council_log.md"),
                "stage1_survivors_csv": str(cell_dir / "stage1_survivors.csv"),
                "stage1_disqualified_csv": str(cell_dir / "stage1_disqualified.csv"),
                "stage2_scores_csv": str(cell_dir / "stage2_scores_long.csv"),
                "stage35_final_scores_csv": str(cell_dir / "stage35_final_scores_long.csv"),
            }
        )

    if manifest_rows:
        write_csv(outdir / "manifest.csv", list(manifest_rows[0].keys()), manifest_rows)


if __name__ == "__main__":
    main()
