"""Summarize preserved objective experiments without selecting a desired winner."""
import argparse
import csv
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", type=Path)
    args = parser.parse_args()
    paths = [args.folder / p / "result.json" for p in (
        "controlled_ab_02/A_balanced", "controlled_ab_02/B_yield", "fresh_pipeline_03")]
    labels = ["A: balanced (frozen upstream)", "B: yield-focused (frozen upstream)", "B: yield-focused (fresh full pipeline)"]
    probe = args.folder / "throughput_probe/C_throughput_probe/result.json"
    if probe.exists():
        paths.append(probe)
        labels.append("C: synthetic throughput-priority probe (frozen upstream)")
    rows, candidates, details = [], [], []
    for label, path in zip(labels, paths):
        r = json.loads(path.read_text())
        a = r["scientific_assessment"]
        p = r["proposal"]
        row = {"run": label, "priority": a["objective_policy"]["priority"], "selected_candidate": a["selected_candidate_id"],
               "total_time_min": p["residence_time_min"], "BPR_bar_gauge": p["BPR_bar"],
               "stage_1_min": p["stage_parameters"][0]["residence_time_min"],
               "stage_2_min": p["stage_parameters"][1]["residence_time_min"],
               "stage_1_volume_mL": p["stage_parameters"][0]["reactor_volume_mL"],
               "stage_2_volume_mL": p["stage_parameters"][1]["reactor_volume_mL"],
               "status": r["final_design"]["status"], "candidate_count": a["candidate_count"],
               "domain_reviews": sum(len(v) for v in a["reviews"].values()),
               "selected_design_preserved": a["selected_design_preserved"], "source": str(path)}
        rows.append(row)
        for c in a["candidates"]:
            candidates.append({"run": label, "candidate": c["candidate_id"],
                "target_stage_1_min": c["target_stage_screen_min"][0], "target_stage_2_min": c["target_stage_screen_min"][1],
                "actual_stage_1_min": c["proposal"]["stage_parameters"][0]["residence_time_min"],
                "actual_stage_2_min": c["proposal"]["stage_parameters"][1]["residence_time_min"],
                "selected": c["candidate_id"] == a["selected_candidate_id"]})
        details.append({"run": label, "objective_policy": a["objective_policy"], "chief": a["chief"],
                        "answer_effects": a["answer_effects"], "pool_sha256": a["pool_sha256"]})
    for name, data in [("selected_designs.csv", rows), ("candidate_screens.csv", candidates)]:
        with (args.folder / name).open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(data[0]))
            writer.writeheader()
            writer.writerows(data)
    (args.folder / "comparison.json").write_text(json.dumps({"runs": rows, "decision_details": details,
        "controlled_pools_changed": details[0]["pool_sha256"] != details[1]["pool_sha256"],
        "measured_yield_improvement": None, "independent_repeats_per_condition": 1}, indent=2))
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
