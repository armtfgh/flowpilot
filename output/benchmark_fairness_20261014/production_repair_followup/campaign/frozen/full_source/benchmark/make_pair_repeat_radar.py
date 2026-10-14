from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from flora_translate.config import LAB_INVENTORY_PATH
from flora_translate.design_calculator import DesignCalculator
from flora_translate.schemas import BatchRecord, ChemistryPlan, FlowProposal, LabInventory


METRICS = ["Pe", "Da_mass", "STY_mol_L_h", "IF", "UA_W_K", "pressure_headroom"]
RADAR_LABELS = ["Pe", "Da_mass", "STY", "IF", "UA", "Pressure headroom"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build five-repeat pre/post radar for one model-pair benchmark.")
    parser.add_argument("--experiment-dir", type=Path, default=None)
    parser.add_argument("--output-dir-name", default="radar_5repeat")
    parser.add_argument("--upstream-bundle", default="claude")
    parser.add_argument("--council-bundle", default="gpt4o")
    parser.add_argument("--budget", type=int, default=12)
    return parser.parse_args()


def _latest_experiment() -> Path:
    candidates = sorted(Path("benchmark/data").glob("radar_pair_repeats_*"))
    if not candidates:
        raise FileNotFoundError("No benchmark/data/radar_pair_repeats_* folders found.")
    return candidates[-1]


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


def _proposal_get(proposal, field: str):
    if hasattr(proposal, field):
        return getattr(proposal, field)
    return proposal.get(field)


def _calc_get(calc, field: str, default=None):
    if hasattr(calc, field):
        return getattr(calc, field)
    if isinstance(calc, dict):
        return calc.get(field, default)
    return default


def _collect_metrics(proposal, calc) -> dict:
    v3 = _vals(calc, 3)
    v4 = _vals(calc, 4)
    v5 = _vals(calc, 5)
    v6 = _vals(calc, 6)
    v7 = _vals(calc, 7)
    v8 = _vals(calc, 8)
    v9 = _vals(calc, 9)

    tau_min = _calc_get(calc, "residence_time_min", _proposal_get(proposal, "residence_time_min"))
    flow_rate_ml_min = _calc_get(calc, "flow_rate_mL_min", _proposal_get(proposal, "flow_rate_mL_min"))
    tubing_id_mm = _calc_get(calc, "tubing_ID_mm", _proposal_get(proposal, "tubing_ID_mm"))
    reactor_volume_ml = _safe_float(_calc_get(calc, "reactor_volume_mL", _proposal_get(proposal, "reactor_volume_mL")))
    proposal_volume_ml = _safe_float(_proposal_get(proposal, "reactor_volume_mL"))
    surface_to_volume = _safe_float(v7.get("S_V_m_inv"))
    heat_transfer_coefficient = _safe_float(v7.get("U_W_m2K"), 300.0)
    heat_area_m2 = (
        surface_to_volume * reactor_volume_ml * 1e-6
        if not math.isnan(surface_to_volume) and not math.isnan(reactor_volume_ml)
        else float("nan")
    )
    ua = heat_transfer_coefficient * heat_area_m2 if not math.isnan(heat_area_m2) else float("nan")

    pressure_drop_bar = _safe_float(v5.get("dP_bar"))
    pump_max_bar = _safe_float(v5.get("pump_max_bar"), 400.0)
    pressure_headroom = (
        1.0 - (pressure_drop_bar / pump_max_bar)
        if pump_max_bar and pump_max_bar > 0 and not math.isnan(pressure_drop_bar)
        else float("nan")
    )
    if not math.isnan(pressure_headroom):
        pressure_headroom = max(0.0, min(1.0, pressure_headroom))

    return {
        "tau_min": tau_min,
        "Q_mL_min": flow_rate_ml_min,
        "ID_mm": tubing_id_mm,
        "V_R_mL": reactor_volume_ml,
        "proposal_V_R_mL": proposal_volume_ml,
        "V_R_consistency_error_mL": (
            proposal_volume_ml - reactor_volume_ml
            if not math.isnan(proposal_volume_ml) and not math.isnan(reactor_volume_ml)
            else float("nan")
        ),
        "BPR_bar": _proposal_get(proposal, "BPR_bar"),
        "tubing_material": _proposal_get(proposal, "tubing_material"),
        "Pe": v3.get("Pe"),
        "Re": v4.get("Re"),
        "dP_bar": v5.get("dP_bar"),
        "pump_max_bar": pump_max_bar,
        "pressure_headroom": pressure_headroom,
        "t_mix_s": v6.get("t_mix_s"),
        "Da_mass": v6.get("Da_mass"),
        "Da_th": v7.get("Da_th"),
        "S_V_m_inv": v7.get("S_V_m_inv"),
        "U_W_m2K": heat_transfer_coefficient,
        "heat_transfer_area_m2": heat_area_m2,
        "UA_W_K": ua,
        "BPR_calc_bar": v8.get("bpr_bar"),
        "STY_mol_L_h": v9.get("STY_mol_L_h"),
        "productivity_mmol_h": v9.get("productivity_mmol_h"),
        "IF": v9.get("IF"),
    }


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
    sty_values = sty_values or [0.0]
    ua_values = ua_values or [0.0]

    for row in rows:
        pe = _safe_float(row["raw_Pe"])
        da = _safe_float(row["raw_Da_mass"])
        sty_log = math.log1p(max(0.0, _safe_float(row["raw_STY_mol_L_h"], 0.0)))
        if_value = _safe_float(row["raw_IF"])
        ua_log = math.log1p(max(0.0, _safe_float(row["raw_UA_W_K"], 0.0)))
        headroom = _safe_float(row["raw_pressure_headroom"])

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


def _case_slug(upstream: str, council: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", f"U_{upstream}__C_{council}")


def _load_result_rows(experiment_dir: Path, upstream: str, council: str, budget: int) -> list[dict]:
    inventory = LabInventory.from_json(str(LAB_INVENTORY_PATH))
    result_paths = sorted(
        (experiment_dir / f"U_{upstream}" / f"C_{council}").glob(
            f"runs/*/budget_{budget}/repeat_*/result.json"
        )
    )
    rows: list[dict] = []
    for result_path in result_paths:
        result = json.loads(result_path.read_text())
        repeat_text = result_path.parent.name.replace("repeat_", "")
        repeat_index = int(repeat_text) if repeat_text.isdigit() else len(rows) + 1
        frozen = result.get("frozen_context", {}) or {}
        batch = BatchRecord(**frozen["batch_record"])
        chemistry = ChemistryPlan(**frozen["chemistry_plan"])
        analogies = frozen.get("analogies", []) or []
        pre_proposal = FlowProposal(**(result.get("pre_council_proposal") or frozen["pre_council_proposal"]))
        pre_calc = DesignCalculator().run(
            batch,
            chemistry_plan=chemistry,
            proposal=pre_proposal,
            inventory=inventory,
            analogies=analogies,
        )
        final_candidate = result.get("final_design_candidate", {}) or {}
        post_proposal = FlowProposal(**((final_candidate.get("proposal", {}) or {})))
        post_calc = DesignCalculator().run(
            batch,
            chemistry_plan=chemistry,
            proposal=post_proposal,
            inventory=inventory,
            analogies=analogies,
        )

        for stage, proposal, calc in (
            ("pre", pre_proposal, pre_calc),
            ("post", post_proposal, post_calc),
        ):
            row = {
                "upstream_bundle": upstream,
                "council_bundle": council,
                "repeat_index": repeat_index,
                "stage": stage,
                "result_path": str(result_path),
            }
            for key, value in _collect_metrics(proposal, calc).items():
                row[f"raw_{key}"] = value
            rows.append(row)
    return _normalize_rows(rows)


def _mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else float("nan")


def _std(values: list[float]) -> float:
    return float(np.std(values, ddof=1)) if len(values) > 1 else 0.0


def _summarize(rows: list[dict]) -> list[dict]:
    summary_rows: list[dict] = []
    for stage in ("pre", "post"):
        stage_rows = [row for row in rows if row["stage"] == stage]
        for metric in METRICS:
            raw_values = [_safe_float(row[f"raw_{metric}"]) for row in stage_rows]
            raw_values = [value for value in raw_values if not math.isnan(value)]
            score_values = [_safe_float(row[f"score_{metric}"]) for row in stage_rows]
            score_values = [value for value in score_values if not math.isnan(value)]
            summary_rows.append({
                "stage": stage,
                "metric": metric,
                "raw_mean": round(_mean(raw_values), 6),
                "raw_std": round(_std(raw_values), 6),
                "score_mean": round(_mean(score_values), 6),
                "score_std": round(_std(score_values), 6),
                "n": len(stage_rows),
            })
        area_values = [_safe_float(row["radar_area_score"]) for row in stage_rows]
        summary_rows.append({
            "stage": stage,
            "metric": "radar_area_score",
            "raw_mean": "",
            "raw_std": "",
            "score_mean": round(_mean(area_values), 6),
            "score_std": round(_std(area_values), 6),
            "n": len(stage_rows),
        })
    return summary_rows


def _plot(rows: list[dict], out_dir: Path, upstream: str, council: str) -> None:
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    angles = np.linspace(0, 2 * np.pi, len(METRICS), endpoint=False)
    angles_closed = np.concatenate([angles, angles[:1]])

    colors = {
        "pre": "#7a869a",
        "post": "#0b6b53",
    }
    labels = {
        "pre": "Pre",
        "post": "Post",
    }
    linestyles = {
        "pre": "--",
        "post": "-",
    }

    fig = plt.figure(figsize=(8.0, 8.2))
    ax = fig.add_subplot(111, polar=True)
    for stage in ("pre", "post"):
        stage_rows = sorted([row for row in rows if row["stage"] == stage], key=lambda row: row["repeat_index"])
        score_matrix = np.array([[row[f"score_{metric}"] for metric in METRICS] for row in stage_rows], dtype=float)
        for scores in score_matrix:
            scores_closed = np.concatenate([scores, scores[:1]])
            ax.plot(
                angles_closed,
                scores_closed,
                color=colors[stage],
                linewidth=0.9,
                alpha=0.22,
                linestyle=linestyles[stage],
            )
        mean_scores = score_matrix.mean(axis=0)
        std_scores = score_matrix.std(axis=0, ddof=1) if len(score_matrix) > 1 else np.zeros(len(METRICS))
        lower = np.clip(mean_scores - std_scores, 0, 1)
        upper = np.clip(mean_scores + std_scores, 0, 1)
        ax.fill_between(
            angles_closed,
            np.concatenate([lower, lower[:1]]),
            np.concatenate([upper, upper[:1]]),
            color=colors[stage],
            alpha=0.12 if stage == "pre" else 0.16,
        )
        ax.plot(
            angles_closed,
            np.concatenate([mean_scores, mean_scores[:1]]),
            color=colors[stage],
            linewidth=2.7 if stage == "post" else 2.3,
            linestyle=linestyles[stage],
            label=f"{labels[stage]} mean ± SD (area={score_matrix.mean(axis=1).mean():.2f})",
        )

    ax.set_xticks(angles)
    ax.set_xticklabels(RADAR_LABELS, fontsize=10)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=8)
    ax.grid(color="#d0d7de", linewidth=0.8)
    ax.set_title(f"Five-repeat pre/post radar: U-{upstream} / C-{council} (n=5)", fontsize=13, pad=26)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.19), frameon=False, ncol=1, fontsize=10)
    footer = (
        "Each faint line is one repeat. Scores are oriented larger = better; "
        "bands show ±1 SD across repeats."
    )
    fig.text(0.5, 0.025, footer, ha="center", fontsize=8, color="#57606a")
    fig.tight_layout(rect=(0.02, 0.10, 0.98, 0.98))

    slug = _case_slug(upstream, council)
    fig.savefig(fig_dir / f"{slug}_5repeat_radar.png", dpi=240)
    fig.savefig(fig_dir / f"{slug}_5repeat_radar.svg")
    plt.close(fig)


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    experiment_dir = args.experiment_dir or _latest_experiment()
    out_dir = experiment_dir / args.output_dir_name
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = _load_result_rows(experiment_dir, args.upstream_bundle, args.council_bundle, args.budget)
    if not rows:
        raise RuntimeError(f"No completed result rows found under {experiment_dir}")

    summary_rows = _summarize(rows)
    slug = _case_slug(args.upstream_bundle, args.council_bundle)
    _write_csv(out_dir / f"{slug}_5repeat_radar_raw.csv", rows)
    _write_csv(out_dir / f"{slug}_5repeat_radar_summary.csv", summary_rows)
    _plot(rows, out_dir, args.upstream_bundle, args.council_bundle)

    print(json.dumps({
        "experiment_dir": str(experiment_dir),
        "output_dir": str(out_dir),
        "raw_csv": str(out_dir / f"{slug}_5repeat_radar_raw.csv"),
        "summary_csv": str(out_dir / f"{slug}_5repeat_radar_summary.csv"),
        "figure_png": str(out_dir / "figures" / f"{slug}_5repeat_radar.png"),
        "figure_svg": str(out_dir / "figures" / f"{slug}_5repeat_radar.svg"),
        "repeat_count": len({row["repeat_index"] for row in rows}),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
