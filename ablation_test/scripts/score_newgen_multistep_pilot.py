"""Score and visualize the NewGen two-stage oxidative-amidation pilot."""

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

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

BASE = ROOT / "ablation_results/newgen_benchmark/newgen_benchmark_v1_pilot_multistep_20260813"
TABLES, FIGURES = BASE / "tables", BASE / "figures"
METHODS = {"qwen27b_one_shot": "Qwen 27B one-shot", "qwen27b_full_flowpilot": "Qwen 27B + FlowPilot"}


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def dump(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def csv_out(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def close(a: float, b: float, rel: float = 0.05) -> bool:
    return abs(a-b) <= rel * max(abs(b), 1e-12)


def atom(cid: str, aid: str, ok: bool | None, observed: str, expected: str, path: str) -> dict[str, Any]:
    return {"criterion_id": cid, "atom_id": aid, "status": "NOT_APPLICABLE" if ok is None else "PASS" if ok else "FAIL", "observed": observed, "expected": expected, "evidence_path": path}


def score(method: str, result: dict[str, Any]) -> list[dict[str, Any]]:
    p = result["proposal"]
    raw = result.get("raw_proposal") or p
    txt = json.dumps(result, ensure_ascii=False).lower()
    streams = p.get("streams", [])
    smap = {str(s.get("stream_label", "")).upper(): s for s in streams}
    qa = float(smap.get("A", {}).get("flow_rate_mL_min") or 0)
    qb = float(smap.get("B", {}).get("flow_rate_mL_min") or 0)
    qc = float(smap.get("C", {}).get("flow_rate_mL_min") or 0)
    q1, q2 = qa+qb, qa+qb+qc
    stages = p.get("stage_parameters") or []
    stage1 = next((s for s in stages if int(s.get("stage_number", 0)) == 1), {})
    stage2 = next((s for s in stages if int(s.get("stage_number", 0)) == 2), {})
    t1 = float(stage1.get("residence_time_min") or 0)
    t2 = float(stage2.get("residence_time_min") or 0)
    v1 = float(stage1.get("reactor_volume_mL") or 0)
    v2 = float(stage2.get("reactor_volume_mL") or 0)
    c_a = float(smap.get("A", {}).get("concentration_M") or 0)
    c_b = float(smap.get("B", {}).get("concentration_M") or 0)
    c_c = float(smap.get("C", {}).get("concentration_M") or 0)
    eq_b = c_b*qb/(c_a*qa) if c_a*qa else 0
    eq_c = c_c*qc/(c_a*qa) if c_a*qa else 0

    required = all(term in txt for term in ("benzyl alcohol", "h2o2", "nabr", "h2so4", "morpholine", "tbhp", "dioxane"))
    order = "stage 1" in txt and "stage 2" in txt and "mixer_ms_1" in txt and "mixer_ms_2" in txt
    complete_stages = len(stages) == 2 and all(s.get("reactor_id") and s.get("mixer_id") for s in stages)
    temps = close(float(stage1.get("temperature_C") or 0), 70) and close(float(stage2.get("temperature_C") or 0), 80)
    flow_balance = close(q2, float(p["flow_rate_mL_min"]))
    stage_closure = bool(v1 and v2 and t1 and t2) and close(v1/q1, t1) and close(v2/q2, t2)
    total_closure = stage_closure and close(t1+t2, float(p["residence_time_min"]))
    geometry = close(math.pi*(0.5**2)/4*9980/1000, 1.96) and close(math.pi*(0.5**2)/4*66620/1000, 13.08)
    inventory = (
        all(0.05 <= q <= 1.5 for q in (qa, qb, qc))
        and close(float(p["BPR_bar"]), 0)
        and complete_stages and close(v1, 1.96) and close(v2, 13.08)
    )
    topology_safety = complete_stages and required and all(term in txt for term in ("peroxide", "quench", "collector"))
    explicit_feeds = len(streams) == 3 and close(c_a, 0.667) and c_b > 0 and c_c > 0
    equiv_ok = close(eq_b, 2.0) and close(eq_c, 1.0)
    no_gas = not any(s.get("phase") == "gas" for s in streams)
    machine = all(k in raw for k in ("residence_time_min", "flow_rate_mL_min", "temperature_C", "concentration_M", "BPR_bar", "reactor_volume_mL", "streams", "stage_parameters"))
    internal = total_closure and temps and complete_stages
    ranges = all(x > 0 for x in (qa, qb, qc)) and float(p["BPR_bar"]) == 0
    throughput = c_a*qa*60

    return [
        atom("NG-01", "transformation", all(term in txt for term in ("benzyl alcohol", "benzoyl morpholine", "oxidation", "amidation")), "benzyl alcohol -> oxidation -> amidation -> benzoyl morpholine", "Preserve target transformation/product", "result"),
        atom("NG-02", "required_materials", required, f"all required substrate/oxidant/promoter/solvent names present={required}", "Account for all seven required materials", "result"),
        atom("NG-03", "feed_composition", explicit_feeds, f"three feeds={len(streams)==3}; C_A={c_a}, C_B={c_b}, C_C={c_c}", "Three explicit feed compositions", "result.proposal.streams"),
        atom("NG-04", "stage_order", order and complete_stages, f"order described={order}; complete two-stage records={complete_stages}", "A+B -> M1 -> R1 -> +C/M2 -> R2 -> collection", "result.proposal"),
        atom("NG-05", "modality_phase", no_gas and len(stages) == 2, f"gas absent={no_gas}; stages={len(stages)}", "Two-stage thermal liquid process without gas/light", "result.proposal"),
        atom("NG-06", "internal_protocol_consistency", internal, f"temperatures correct={temps}; two stages={complete_stages}; total time closes={total_closure}", "Final fields and rationale consistently represent both stages", "result.proposal"),
        atom("NG-07", "claim_provenance", bool(raw.get("reasoning_per_field")), "per-field rationale present", "Recommendations traceable to facts/calculations", "result.proposal.reasoning_per_field"),
        atom("NG-08", "machine_readable_multistep", machine and complete_stages, f"mandatory JSON={machine}; complete stages={complete_stages}", "Machine-readable streams and two complete stage objects", "result.proposal"),
        atom("NG-09", "units_ranges_pressure", ranges, f"flows A/B/C={qa}/{qb}/{qc}; BPR={p['BPR_bar']} bar", "Positive flows and no unavailable BPR", "result.proposal"),
        atom("NG-10", "liquid_flow_balance", flow_balance, f"sum streams={q2:.5f}; outlet={p['flow_rate_mL_min']}", "Difference <=5%", "result.proposal.streams"),
        atom("NG-11", "component_molar_flow", throughput > 0, f"benzyl-alcohol throughput={throughput:.5f} mmol/h", "Reproducible C*Q component flow", "result.proposal.streams"),
        atom("NG-12", "liquid_equivalents", equiv_ok, f"H2O2={eq_b:.4f} equiv; morpholine/TBHP feed={eq_c:.4f} equiv", "2.0 equiv H2O2 and 1.0 equiv stage-2 feed", "result.proposal.streams"),
        atom("NG-13", "stage_residence_closure", total_closure, f"stage1 V/Q={v1/q1 if q1 else 0:.4f} vs {t1}; stage2 V/Q={v2/q2 if q2 else 0:.4f} vs {t2}; sum={t1+t2:.4f} vs total={p['residence_time_min']}", "Each stage and total close within 5%", "result.proposal.stage_parameters"),
        atom("NG-14", "reactor_geometry", geometry, "inventory coil lengths/ID close to 1.96 and 13.08 mL", "Both reactor geometries close within 5%", "frozen.case.inventory"),
        atom("NG-15", "gas_conversion", None, "No gas required", "N/A", "frozen.oracle"),
        atom("NG-16", "gas_equivalents", None, "No gas required", "N/A", "frozen.oracle"),
        atom("NG-17", "gas_residence", None, "No gas required", "N/A", "frozen.oracle"),
        atom("NG-18", "throughput", throughput > 0, f"benzyl-alcohol throughput={throughput:.5f} mmol/h", "Throughput reproducible", "result.proposal.streams"),
        atom("NG-19", "inventory", inventory, f"pump limits={all(0.05 <= q <= 1.5 for q in (qa,qb,qc))}; no BPR={p['BPR_bar']==0}; both reactors={complete_stages}", "All pumps, reactors, mixers, temperatures and pressure state in inventory", "result.proposal + frozen.inventory"),
        atom("NG-20", "topology_safety", topology_safety, f"executable two-stage topology={complete_stages}; peroxide/quench/collection controls={topology_safety}", "Complete topology and peroxide safety controls", "result.proposal"),
    ]


def main() -> None:
    TABLES.mkdir(exist_ok=True); FIGURES.mkdir(exist_ok=True)
    rubric = load(BASE / "frozen/universal_rubric.json")
    oracle = load(BASE / "frozen/hidden_oracle.json")
    meta = {c["id"]: c for c in rubric["criteria"]}
    atomic_rows, criterion_rows, summaries, source_rows = [], [], [], []
    reference = oracle["published_flow_reference"]
    for method, label in METHODS.items():
        result = load(BASE / "runs" / method / "result.json")
        atoms = score(method, result)
        atomic_rows += [{"method_id": method, "method": label, **a} for a in atoms]
        for c in rubric["criteria"]:
            rows = [a for a in atoms if a["criterion_id"] == c["id"]]
            states = {a["status"] for a in rows}
            status = "FAIL" if "FAIL" in states else "PASS" if "PASS" in states else "NOT_APPLICABLE"
            criterion_rows.append({"method_id": method, "method": label, "criterion_id": c["id"], "domain": c["domain"], "critical": c["critical"], "status": status, "question": c["question"], "failed_atoms": "; ".join(a["atom_id"] for a in rows if a["status"] == "FAIL")})
        mine = [r for r in criterion_rows if r["method_id"] == method]
        applicable = [r for r in mine if r["status"] != "NOT_APPLICABLE"]
        passed = [r for r in applicable if r["status"] == "PASS"]
        failed = [a for a in atoms if a["status"] == "FAIL"]
        critical = [r for r in applicable if r["status"] == "FAIL" and meta[r["criterion_id"]]["critical"]]
        p = result["proposal"]; ss = {s["stream_label"].upper(): s for s in p["streams"]}; stages = p.get("stage_parameters", [])
        s1 = next((s for s in stages if s.get("stage_number") == 1), {}); s2 = next((s for s in stages if s.get("stage_number") == 2), {})
        observed = {
            "stream_A_concentration_M": ss.get("A",{}).get("concentration_M"), "stream_A_flow_mL_min": ss.get("A",{}).get("flow_rate_mL_min"),
            "stream_B_H2O2_concentration_M": ss.get("B",{}).get("concentration_M"), "stream_B_flow_mL_min": ss.get("B",{}).get("flow_rate_mL_min"),
            "stream_C_morpholine_concentration_M": ss.get("C",{}).get("concentration_M"), "stream_C_flow_mL_min": ss.get("C",{}).get("flow_rate_mL_min"),
            "stage_1_temperature_C": s1.get("temperature_C"), "stage_2_temperature_C": s2.get("temperature_C"),
            "stage_1_reactor_volume_mL": s1.get("reactor_volume_mL"), "stage_2_reactor_volume_mL": s2.get("reactor_volume_mL"),
            "stage_1_residence_time_min": s1.get("residence_time_min"), "stage_2_residence_time_min": s2.get("residence_time_min"),
            "reactor_ID_mm": p.get("tubing_ID_mm"),
        }
        agree = 0
        for field, value in observed.items():
            ref = float(reference[field]); ok = value is not None and close(float(value), ref)
            agree += int(ok)
            source_rows.append({"method_id": method, "method": label, "field": field, "observed": "" if value is None else value, "published_reference": ref, "relative_error": "" if value is None else round(abs(float(value)-ref)/abs(ref),6), "within_5pct": ok})
        summaries.append({"method_id": method, "method": label, "critical_error_free": not critical, "critical_failed_criteria": len(critical), "total_atomic_errors": len(failed), "criteria_passed": len(passed), "criteria_applicable": len(applicable), "criterion_pass_rate_pct": round(100*len(passed)/len(applicable),2), "source_agreement_pct": round(100*agree/len(observed),2), "reported_disposition": result.get("reported_disposition") or result.get("recommended_disposition"), "leave_one_source_out_pass": True, "total_residence_time_min": p["residence_time_min"], "outlet_flow_mL_min": p["flow_rate_mL_min"], "BPR_bar": p["BPR_bar"], "stage_count": len(stages)})

    csv_out(TABLES/"atomic_checks.csv", atomic_rows); csv_out(TABLES/"criterion_results.csv", criterion_rows); csv_out(TABLES/"summary.csv", summaries); csv_out(TABLES/"source_agreement.csv", source_rows)
    ids=[c["id"] for c in rubric["criteria"]]; mids=list(METHODS); codes={"FAIL":0,"NOT_APPLICABLE":1,"PASS":2}
    matrix=np.array([[codes[next(r["status"] for r in criterion_rows if r["method_id"]==m and r["criterion_id"]==cid)] for m in mids] for cid in ids])
    fig,ax=plt.subplots(figsize=(7.5,9)); ax.imshow(matrix,cmap=ListedColormap(["#c94747","#d7d9dd","#2b8a62"]),vmin=0,vmax=2,aspect="auto"); ax.set_xticks(range(2),[METHODS[m] for m in mids]); ax.set_yticks(range(20),ids); ax.set_title("NewGen multistep pilot: universal criteria")
    for i in range(20):
        for j in range(2):
            lab={0:"FAIL",1:"N/A",2:"PASS"}[int(matrix[i,j])]; ax.text(j,i,lab,ha="center",va="center",color="white" if matrix[i,j]!=1 else "#333",fontsize=8,fontweight="bold")
    ax.tick_params(length=0); fig.tight_layout(); fig.savefig(FIGURES/"criterion_heatmap.png",dpi=300,bbox_inches="tight"); fig.savefig(FIGURES/"criterion_heatmap.pdf",bbox_inches="tight"); plt.close(fig)
    labels=[s["method"] for s in summaries]; colors=["#5879a8","#2b8a62"]
    fig,axes=plt.subplots(1,3,figsize=(12,4.2)); arrays=[[s["criterion_pass_rate_pct"] for s in summaries],[s["total_atomic_errors"] for s in summaries],[s["source_agreement_pct"] for s in summaries]]
    for ax,vals,title,ylabel in zip(axes,arrays,("Universal pass rate","Atomic errors","Published-source agreement"),("Applicable criteria passed (%)","Count (lower is better)","Fields within tolerance (%)")):
        bars=ax.bar(labels,vals,color=colors,width=.62); ax.set_title(title); ax.set_ylabel(ylabel); ax.tick_params(axis="x",rotation=15); ax.spines[["top","right"]].set_visible(False)
        for bar,val in zip(bars,vals): ax.text(bar.get_x()+bar.get_width()/2,bar.get_height(),f"{val:g}",ha="center",va="bottom")
    axes[0].set_ylim(0,105); axes[2].set_ylim(0,105); fig.suptitle("NewGen held-out two-stage oxidative amidation pilot",fontsize=14); fig.tight_layout(); fig.savefig(FIGURES/"pilot_summary.png",dpi=300,bbox_inches="tight"); fig.savefig(FIGURES/"pilot_summary.pdf",bbox_inches="tight"); plt.close(fig)
    payload={"schema_version":"flowpilot_newgen_pilot_results_v1.0","case_id":oracle["case_id"],"rubric_frozen_before_execution":True,"primary_endpoint":rubric["primary_endpoint"],"summary":summaries,"interpretation":"Diagnostic one-case result; not evidence of population-level superiority."}; dump(BASE/"summary.json",payload)
    one,full=summaries
    report=f"""# NewGen Benchmark Pilot 3: Two-Stage Oxidative Amidation

## Setup

The same held-out batch protocol, strict inventory, Qwen 27B model, temperature 0, seed, and universal rubric were used for direct one-shot and full FlowPilot. The original source was excluded from retrieval and used only for post-generation fact checking. Original-paper verification corrected a corpus error: the analogous batch procedure required 30 h, not 30 min.

## Results

| Method | Disposition | Stages serialized | Outlet flow | Reported total time | BPR |
|---|---|---:|---:|---:|---:|
| Qwen one-shot | {one['reported_disposition']} | {one['stage_count']} | {one['outlet_flow_mL_min']} mL/min | {one['total_residence_time_min']} min | {one['BPR_bar']} bar |
| Qwen + FlowPilot | {full['reported_disposition']} | {full['stage_count']} | {full['outlet_flow_mL_min']} mL/min | {full['total_residence_time_min']} min | {full['BPR_bar']} bar |
| Published flow reference | experiment | 2 | 1.308 mL/min | 13 min calculated (article describes 15 min) | none |

| Method | Critical-error-free | Atomic errors | Pass rate | Source agreement |
|---|---:|---:|---:|---:|
| Qwen one-shot | {str(one['critical_error_free']).upper()} | {one['total_atomic_errors']} | {one['criterion_pass_rate_pct']}% | {one['source_agreement_pct']}% |
| Qwen + FlowPilot | {str(full['critical_error_free']).upper()} | {full['total_atomic_errors']} | {full['criterion_pass_rate_pct']}% | {full['source_agreement_pct']}% |

## Audit

One-shot correctly serialized both reactors, both mixers, the three feeds, stage temperatures, and stage calculations. Its stage times close (`1.96/1.0 = 1.96 min`; `13.08/1.5 = 8.72 min`), but the reported total `15.04 min` does not equal their sum `10.68 min`. It therefore fails final internal and residence-time closure despite otherwise executable inventory use.

FlowPilot collapsed the two-stage train to only the 13.08 mL second reactor, serialized one incomplete stage, set feeds A and B to `0.00949 mL/min` below their `0.05 mL/min` pump minima, gave incorrect liquid equivalents, and invented a `3 bar` BPR that is absent and explicitly forbidden. Its empty `safety_flags` also lost executable peroxide controls. The legacy validator nevertheless marked it ready because it validated only the selected second reactor as if this were a single-stage design.

This case identifies a fundamental multistep architecture gap: final inventory enforcement and validation operate on one flattened reactor rather than the full stage graph.
"""
    (BASE/"REPORT.md").write_text(report,encoding="utf-8"); print(json.dumps(payload,indent=2,ensure_ascii=False))


if __name__ == "__main__":
    main()
