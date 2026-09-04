from __future__ import annotations

import csv
import json
import math
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from flora_translate.config import LAB_INVENTORY_PATH
from flora_translate.design_calculator import DesignCalculator
from flora_translate.schemas import BatchRecord, ChemistryPlan, FlowProposal, LabInventory


ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "matrix_manifest.csv"
OUT = ROOT / "prepost_radar" / "pressure_UA_radar"
FIG_DIR = OUT / "figures"

METRICS = ["Pe", "Da_mass", "STY_mol_L_h", "IF", "UA_W_K", "pressure_headroom"]
RADAR_LABELS = ["Pe", "Da_mass", "STY", "IF", "UA", "Pressure headroom"]


def _safe_float(value, default: float = float("nan")) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _step_num(step) -> int | None:
    return step.get("step") if isinstance(step, dict) else getattr(step, "step", None)


def _step_values(step) -> dict:
    return step.get("values", {}) if isinstance(step, dict) else getattr(step, "values", {}) or {}


def _steps(calc) -> list:
    return calc.get("steps", []) if isinstance(calc, dict) else getattr(calc, "steps", []) or []


def _vals(calc, step_no: int) -> dict:
    for step in _steps(calc):
        if _step_num(step) == step_no:
            return _step_values(step)
    return {}


def _find_context_dir(cell_dir: str, upstream: str) -> Path | None:
    path = Path(cell_dir)
    parts = list(path.parts)
    try:
        idx = next(i for i, part in enumerate(parts) if part.startswith("U_"))
    except StopIteration:
        return None
    benchmark_root = Path(*parts[:idx])
    context_dir = benchmark_root / f"U_{upstream}" / "contexts" / "protocol_isoxazole_des_full"
    if (context_dir / "snapshots" / "proposal_pre_council.json").exists():
        return context_dir
    return None


def _load_json(path: Path) -> dict | list:
    return json.loads(path.read_text())


def _proposal_get(proposal, field: str):
    return getattr(proposal, field) if hasattr(proposal, field) else proposal.get(field)


def _collect_metrics(proposal, calc) -> dict:
    v3 = _vals(calc, 3)
    v4 = _vals(calc, 4)
    v5 = _vals(calc, 5)
    v6 = _vals(calc, 6)
    v7 = _vals(calc, 7)
    v8 = _vals(calc, 8)
    v9 = _vals(calc, 9)

    V_R_mL = _safe_float(_proposal_get(proposal, "reactor_volume_mL"))
    S_V = _safe_float(v7.get("S_V_m_inv"))
    U = _safe_float(v7.get("U_W_m2K"), 300.0)
    heat_area_m2 = S_V * V_R_mL * 1e-6 if not math.isnan(S_V) and not math.isnan(V_R_mL) else float("nan")
    UA = U * heat_area_m2 if not math.isnan(heat_area_m2) else float("nan")

    dP = _safe_float(v5.get("dP_bar"))
    pump_max = _safe_float(v5.get("pump_max_bar"), 400.0)
    pressure_headroom = 1.0 - (dP / pump_max) if pump_max and pump_max > 0 and not math.isnan(dP) else float("nan")
    pressure_headroom = max(0.0, min(1.0, pressure_headroom)) if not math.isnan(pressure_headroom) else pressure_headroom

    return {
        "tau_min": _proposal_get(proposal, "residence_time_min"),
        "Q_mL_min": _proposal_get(proposal, "flow_rate_mL_min"),
        "ID_mm": _proposal_get(proposal, "tubing_ID_mm"),
        "V_R_mL": _proposal_get(proposal, "reactor_volume_mL"),
        "BPR_bar": _proposal_get(proposal, "BPR_bar"),
        "tubing_material": _proposal_get(proposal, "tubing_material"),
        "Pe": v3.get("Pe"),
        "Pe_adequate": v3.get("Pe_adequate"),
        "Re": v4.get("Re"),
        "dP_bar": v5.get("dP_bar"),
        "pump_max_bar": pump_max,
        "pressure_headroom": pressure_headroom,
        "t_mix_s": v6.get("t_mix_s"),
        "Da_mass": v6.get("Da_mass"),
        "Da_th": v7.get("Da_th"),
        "S_V_m_inv": v7.get("S_V_m_inv"),
        "U_W_m2K": U,
        "heat_transfer_area_m2": heat_area_m2,
        "UA_W_K": UA,
        "BPR_calc_bar": v8.get("bpr_bar"),
        "STY_mol_L_h": v9.get("STY_mol_L_h"),
        "productivity_mmol_h": v9.get("productivity_mmol_h"),
        "IF": v9.get("IF"),
    }


def _case_slug(upstream: str, council: str) -> str:
    text = f"U_{upstream}__C_{council}"
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text)


def _minmax_score(value: float, values: list[float]) -> float:
    if math.isnan(value):
        return 0.0
    lo, hi = min(values), max(values)
    if hi <= lo:
        return 1.0
    return max(0.0, min(1.0, (value - lo) / (hi - lo)))


def _normalize_rows(rows: list[dict]) -> list[dict]:
    sty_values = [
        math.log1p(_safe_float(row["raw_STY_mol_L_h"]))
        for row in rows
        if not math.isnan(_safe_float(row["raw_STY_mol_L_h"]))
    ]
    ua_values = [
        math.log1p(_safe_float(row["raw_UA_W_K"]))
        for row in rows
        if not math.isnan(_safe_float(row["raw_UA_W_K"]))
    ]

    for row in rows:
        pe = _safe_float(row["raw_Pe"])
        da = _safe_float(row["raw_Da_mass"])
        sty_log = math.log1p(_safe_float(row["raw_STY_mol_L_h"]))
        if_value = _safe_float(row["raw_IF"])
        ua_log = math.log1p(_safe_float(row["raw_UA_W_K"]))
        headroom = _safe_float(row["raw_pressure_headroom"])

        # All normalized values are oriented as "larger = better".
        row["score_Pe"] = round(max(0.0, min(1.0, pe / 100.0)), 4)
        row["score_Da_mass"] = round(max(0.0, min(1.0, 1.0 / (1.0 + da))), 4)
        row["score_STY_mol_L_h"] = round(_minmax_score(sty_log, sty_values), 4)
        row["score_IF"] = round(max(0.0, min(1.0, if_value / 6.0)), 4)
        row["score_UA_W_K"] = round(_minmax_score(ua_log, ua_values), 4)
        row["score_pressure_headroom"] = round(max(0.0, min(1.0, headroom)), 4)
        row["radar_area_score"] = round(
            sum(row[f"score_{metric}"] for metric in METRICS) / len(METRICS),
            4,
        )
    return rows


def _plot_case(case_rows: list[dict]) -> None:
    pre = next(row for row in case_rows if row["stage"] == "pre")
    post = next(row for row in case_rows if row["stage"] == "post")

    pre_scores = [pre[f"score_{metric}"] for metric in METRICS]
    post_scores = [post[f"score_{metric}"] for metric in METRICS]
    angles = np.linspace(0, 2 * np.pi, len(METRICS), endpoint=False).tolist()
    angles += angles[:1]
    pre_scores += pre_scores[:1]
    post_scores += post_scores[:1]

    fig = plt.figure(figsize=(7.2, 7.2))
    ax = fig.add_subplot(111, polar=True)
    ax.plot(
        angles,
        pre_scores,
        color="#7a869a",
        linewidth=2,
        linestyle="--",
        label=f"Pre (mean={pre['radar_area_score']:.2f})",
    )
    ax.fill(angles, pre_scores, color="#7a869a", alpha=0.12)
    ax.plot(
        angles,
        post_scores,
        color="#0b6b53",
        linewidth=2.6,
        linestyle="-",
        label=f"Post (mean={post['radar_area_score']:.2f})",
    )
    ax.fill(angles, post_scores, color="#0b6b53", alpha=0.20)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(RADAR_LABELS, fontsize=10)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=8)
    ax.grid(color="#d0d7de", linewidth=0.8)
    ax.set_title(
        f"Pre/Post Council Radar: U-{pre['upstream_bundle']} / C-{pre['council_bundle']}",
        fontsize=13,
        pad=24,
    )
    ax.legend(loc="upper right", bbox_to_anchor=(1.26, 1.12), frameon=False)

    footer = (
        "Scores: Pe capped at 100; Da inverted; STY and UA log-minmax; "
        "IF capped at 6; pressure headroom = 1 - ΔP/pump limit."
    )
    fig.text(0.5, 0.02, footer, ha="center", fontsize=8, color="#57606a")
    fig.tight_layout(rect=(0.02, 0.05, 0.98, 0.98))

    slug = _case_slug(pre["upstream_bundle"], pre["council_bundle"])
    fig.savefig(FIG_DIR / f"{slug}_radar.png", dpi=220)
    fig.savefig(FIG_DIR / f"{slug}_radar.svg")
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    inventory = LabInventory.from_json(str(LAB_INVENTORY_PATH))
    rows: list[dict] = []

    with MANIFEST.open() as handle:
        for manifest_row in csv.DictReader(handle):
            if manifest_row.get("status") != "completed":
                continue
            result_path = Path(manifest_row["result_path"])
            if not result_path.exists():
                continue

            upstream = manifest_row["upstream_bundle"]
            council = manifest_row["council_bundle"]
            context_dir = _find_context_dir(manifest_row["cell_dir"], upstream)
            if context_dir is None:
                continue

            snapshots = context_dir / "snapshots"
            batch = BatchRecord(**_load_json(snapshots / "batch_record.json"))
            chemistry = ChemistryPlan(**_load_json(snapshots / "chemistry_plan.json"))
            pre_proposal = FlowProposal(**_load_json(snapshots / "proposal_pre_council.json"))
            analogies_path = snapshots / "analogies.json"
            analogies = _load_json(analogies_path) if analogies_path.exists() else []
            pre_calc = DesignCalculator().run(
                batch,
                chemistry_plan=chemistry,
                proposal=pre_proposal,
                inventory=inventory,
                analogies=analogies,
            )

            result = _load_json(result_path)
            final_candidate = result.get("final_design_candidate", {})
            post_proposal = final_candidate.get("proposal", {})
            final_calc = result.get("final_calculations", {})

            for stage, proposal, calc in (
                ("pre", pre_proposal, pre_calc),
                ("post", post_proposal, final_calc),
            ):
                metrics = _collect_metrics(proposal, calc)
                row = {
                    "upstream_bundle": upstream,
                    "council_bundle": council,
                    "raw_upstream_bundle": manifest_row.get("raw_upstream_bundle", ""),
                    "source_benchmark": manifest_row.get("source_benchmark", ""),
                    "stage": stage,
                    "result_path": str(result_path),
                    "engine_validated": post_proposal.get("engine_validated"),
                    "confidence": post_proposal.get("confidence"),
                    "safety_flags": "; ".join(post_proposal.get("safety_flags") or []),
                }
                for key, value in metrics.items():
                    row[f"raw_{key}"] = value
                rows.append(row)

    rows = _normalize_rows(rows)
    with (OUT / "prepost_radar_pressure_UA_metrics.csv").open("w", newline="") as handle:
        fieldnames = list(rows[0].keys()) if rows else []
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary_rows = []
    grouped: dict[tuple[str, str], list[dict]] = {}
    for row in rows:
        grouped.setdefault((row["upstream_bundle"], row["council_bundle"]), []).append(row)
    for case_rows in grouped.values():
        if len(case_rows) != 2:
            continue
        _plot_case(case_rows)
        pre = next(row for row in case_rows if row["stage"] == "pre")
        post = next(row for row in case_rows if row["stage"] == "post")
        summary_rows.append({
            "upstream_bundle": pre["upstream_bundle"],
            "council_bundle": pre["council_bundle"],
            "pre_radar_area_score": pre["radar_area_score"],
            "post_radar_area_score": post["radar_area_score"],
            "delta_radar_area_score": round(post["radar_area_score"] - pre["radar_area_score"], 4),
            "figure_png": f"figures/{_case_slug(pre['upstream_bundle'], pre['council_bundle'])}_radar.png",
            "figure_svg": f"figures/{_case_slug(pre['upstream_bundle'], pre['council_bundle'])}_radar.svg",
        })

    with (OUT / "prepost_radar_pressure_UA_summary.csv").open("w", newline="") as handle:
        fieldnames = [
            "upstream_bundle",
            "council_bundle",
            "pre_radar_area_score",
            "post_radar_area_score",
            "delta_radar_area_score",
            "figure_png",
            "figure_svg",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"Wrote {OUT / 'prepost_radar_pressure_UA_metrics.csv'}")
    print(f"Wrote {OUT / 'prepost_radar_pressure_UA_summary.csv'}")
    print(f"Wrote {len(summary_rows)} radar figures to {FIG_DIR}")


if __name__ == "__main__":
    main()
