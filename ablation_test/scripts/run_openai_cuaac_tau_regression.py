#!/usr/bin/env python3
"""Run three live CuAAC FlowPilot regressions after the evidence-first tau fix."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ablation_test.scripts.run_manuscript_five_case_benchmark import load_frozen_cases
from ablation_test.src.newgen_2_outcome import stable_seed
from ablation_test.src.runner import execute_cell


OUTPUT = ROOT / "ablation_results/regression/openai_cuaac_tau_policy_20260820"
BUNDLE = {
    "provider": "openai",
    "model": "gpt-5.4-2026-03-05",
    "upstream_mode": "never",
    "family": "openai",
    "display": "GPT-5.4",
}


def main() -> None:
    case = dict(load_frozen_cases())["CuAAC"]
    OUTPUT.mkdir(parents=True, exist_ok=True)
    summaries = []
    for index in range(1, 4):
        repeat_id = f"repeat_{index:02d}"
        run_dir = OUTPUT / repeat_id
        summary_path = run_dir / "run_summary.json"
        if not summary_path.exists():
            execute_cell(
                case=case,
                variant="full",
                bundle_name="openai_flowpilot_tau_regression",
                bundle=BUNDLE,
                run_dir=run_dir,
                candidate_budget=2,
                temperature=0.2,
                seed=stable_seed("openai-cuaac-tau-policy-20260820-v1", repeat_id),
            )
        result = json.loads((run_dir / "result.json").read_text(encoding="utf-8"))
        final = result.get("final_design") or {}
        streams = final.get("streams") or []
        catalyst_in_feed = any(
            re.search(r"copper on carbon|cu/c", str(stream.get("contents") or []), re.I)
            for stream in streams
        )
        parameters = final.get("parameters") or {}
        summaries.append(
            {
                "repeat_id": repeat_id,
                "status": final.get("status"),
                "residence_time_min": parameters.get("residence_time_min"),
                "flow_rate_mL_min": parameters.get("flow_rate_mL_min"),
                "reactor_volume_mL": parameters.get("reactor_volume_mL"),
                "stationary_catalyst_in_feed": catalyst_in_feed,
            }
        )
    payload = {
        "schema_version": "flowpilot_openai_cuaac_tau_regression_v1.0",
        "acceptance": {
            "all_contracts_executable": all(row["status"] == "executable" for row in summaries),
            "no_stationary_catalyst_in_feed": not any(
                row["stationary_catalyst_in_feed"] for row in summaries
            ),
            "no_sub_0_5_min_forced_tau": not any(
                float(row["residence_time_min"] or 0) < 0.5 for row in summaries
            ),
        },
        "runs": summaries,
    }
    (OUTPUT / "summary.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2))
    if not all(payload["acceptance"].values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
