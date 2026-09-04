"""Export current no-council / 1-candidate / 12-candidate ablation results.

Reads any available run JSONs from outputs/ and writes:
  outputs/ablation_results.csv
  outputs/ablation_summary.csv
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

OUT_DIR = Path("outputs")


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _metric_from_arm(arm_name: str, payload: dict) -> dict:
    prop = payload.get(f"{arm_name}_proposal", {})
    calc = payload.get(f"{arm_name}_calculations", {})
    return {
        "residence_time_min": prop.get("residence_time_min", calc.get("residence_time_min")),
        "flow_rate_mL_min": prop.get("flow_rate_mL_min", calc.get("flow_rate_mL_min")),
        "reactor_volume_mL": prop.get("reactor_volume_mL", calc.get("reactor_volume_mL")),
        "tubing_ID_mm": prop.get("tubing_ID_mm", calc.get("tubing_ID_mm")),
        "tubing_material": prop.get("tubing_material"),
        "BPR_bar": prop.get("BPR_bar"),
        "confidence": prop.get("confidence"),
        "reynolds_number": calc.get("reynolds_number"),
        "pressure_drop_bar": calc.get("pressure_drop_bar"),
        "surface_to_volume": calc.get("surface_to_volume"),
        "damkohler_mass": calc.get("damkohler_mass"),
    }


def _pre_metric(payload: dict) -> dict:
    prop = payload.get("pre_council_proposal", {})
    calc = payload.get("pre_council_calculations", {})
    return {
        "residence_time_min": prop.get("residence_time_min", calc.get("residence_time_min")),
        "flow_rate_mL_min": prop.get("flow_rate_mL_min", calc.get("flow_rate_mL_min")),
        "reactor_volume_mL": prop.get("reactor_volume_mL", calc.get("reactor_volume_mL")),
        "tubing_ID_mm": prop.get("tubing_ID_mm", calc.get("tubing_ID_mm")),
        "tubing_material": prop.get("tubing_material"),
        "BPR_bar": prop.get("BPR_bar"),
        "confidence": prop.get("confidence"),
        "reynolds_number": calc.get("reynolds_number"),
        "pressure_drop_bar": calc.get("pressure_drop_bar"),
        "surface_to_volume": calc.get("surface_to_volume"),
        "damkohler_mass": calc.get("damkohler_mass"),
    }


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    rows: list[dict] = []
    summary_rows: list[dict] = []

    case_configs = [
        ("snar", OUT_DIR / "snar_1cand_run.json"),
        ("thermal", OUT_DIR / "thermal_1cand_run.json"),
    ]

    for case_name, path in case_configs:
        if not path.exists():
            continue
        payload = _read_json(path)
        arm_map = {
            "no_council": _pre_metric(payload),
            "one_candidate": _metric_from_arm("one_cand", payload),
            "twelve_candidate": _metric_from_arm("twelve_cand", payload),
        }
        for arm, metrics in arm_map.items():
            rows.append({"case": case_name, "arm": arm, **metrics})

        no_c = arm_map["no_council"]
        one = arm_map["one_candidate"]
        twelve = arm_map["twelve_candidate"]
        summary_rows.append({
            "case": case_name,
            "tau_pre": no_c["residence_time_min"],
            "tau_1cand": one["residence_time_min"],
            "tau_12cand": twelve["residence_time_min"],
            "d_pre": no_c["tubing_ID_mm"],
            "d_1cand": one["tubing_ID_mm"],
            "d_12cand": twelve["tubing_ID_mm"],
            "BPR_pre": no_c["BPR_bar"],
            "BPR_1cand": one["BPR_bar"],
            "BPR_12cand": twelve["BPR_bar"],
            "sv_pre": no_c["surface_to_volume"],
            "sv_1cand": one["surface_to_volume"],
            "sv_12cand": twelve["surface_to_volume"],
        })

    if rows:
        detail_fields = list(rows[0].keys())
        _write_csv(OUT_DIR / "ablation_results.csv", rows, detail_fields)
    if summary_rows:
        summary_fields = list(summary_rows[0].keys())
        _write_csv(OUT_DIR / "ablation_summary.csv", summary_rows, summary_fields)


if __name__ == "__main__":
    main()
