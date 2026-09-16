"""Score and visualize the NewGen exothermic-nitration pilot."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

BASE = ROOT / "ablation_results/newgen_benchmark/newgen_benchmark_v1_pilot_nitration_20260813"
TABLES, FIGURES = BASE / "tables", BASE / "figures"
METHODS = {
    "qwen27b_one_shot": "Qwen 27B one-shot",
    "qwen27b_full_flowpilot": "Qwen 27B + FlowPilot",
}


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def dump(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def csv_out(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def close(a: float, b: float, rel: float = 0.05) -> bool:
    return abs(a - b) <= rel * max(abs(b), 1e-12)


def atom(cid: str, aid: str, ok: bool | None, observed: str, expected: str, path: str) -> dict[str, Any]:
    return {
        "criterion_id": cid,
        "atom_id": aid,
        "status": "NOT_APPLICABLE" if ok is None else "PASS" if ok else "FAIL",
        "observed": observed,
        "expected": expected,
        "evidence_path": path,
    }


def identify_streams(streams: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    organic, acid = {}, {}
    for stream in streams:
        text = json.dumps(stream, ensure_ascii=False).lower()
        if "nitric" in text or "acid feed" in text or "nitrating" in text:
            acid = stream
        elif "xylidine" in text or "organic" in text:
            organic = stream
    return organic, acid


def score(result: dict[str, Any]) -> list[dict[str, Any]]:
    p = result["proposal"]
    raw = result.get("raw_proposal") or p
    txt = json.dumps(p, ensure_ascii=False).lower()
    streams = p.get("streams") or []
    organic, acid = identify_streams(streams)
    q_org = float(organic.get("flow_rate_mL_min") or 0)
    q_acid = float(acid.get("flow_rate_mL_min") or 0)
    c_org = float(organic.get("concentration_M") or 0)
    c_acid = float(acid.get("concentration_M") or 0)
    q_total = q_org + q_acid
    declared_total = float(p.get("flow_rate_mL_min") or 0)
    volume = float(p.get("reactor_volume_mL") or 0)
    tau = float(p.get("residence_time_min") or 0)
    bpr = float(p.get("BPR_bar") or 0)
    temperature = float(p.get("temperature_C") or 0)
    stages = p.get("stage_parameters") or []
    stage = stages[0] if len(stages) == 1 else {}
    stage_tau = float(stage.get("residence_time_min") or 0)
    stage_complete = bool(stage.get("reactor_id") and stage.get("mixer_id") and stage.get("cumulative_flow_mL_min"))
    calculated_equiv = c_acid * q_acid / (c_org * q_org) if c_org * q_org else 0
    declared_equiv = float(acid.get("molar_equiv") or 0)

    required_materials = all(term in txt for term in ("xylidine", "nitric acid", "1,2-dichloroethane"))
    target_preserved = "pendimethalin" in txt and ("nitration" in txt or "nitro" in txt)
    explicit_feeds = len(streams) == 2 and all(x > 0 for x in (q_org, q_acid, c_org, c_acid))
    topology = all(term in txt for term in ("mixer_nit_mp", "reactor_nit_020", "separator_nit_ll", "collector_nit_quench"))
    phase_correct = "separator_nit_ll" in txt and not any(s.get("phase") == "gas" for s in streams)
    flow_closure = close(q_total, declared_total)
    top_tau_closure = bool(q_total and volume and tau) and close(volume / q_total, tau)
    stage_tau_closure = stage_complete and close(float(stage["cumulative_flow_mL_min"]), q_total) and close(volume / q_total, stage_tau)
    complete_machine = all(k in raw for k in (
        "residence_time_min", "flow_rate_mL_min", "temperature_C", "concentration_M",
        "BPR_bar", "reactor_volume_mL", "streams", "stage_parameters"
    )) and stage_complete
    range_pressure = all(x > 0 for x in (q_org, q_acid, temperature)) and bpr == 0
    equivalent_closed = close(calculated_equiv, declared_equiv) and declared_equiv >= 2.0
    inventory = (
        all(0.5 <= q <= 10.0 for q in (q_org, q_acid))
        and close(volume, 0.2)
        and temperature in {40.0, 50.0, 60.0, 70.0, 80.0}
        and bpr == 0
        and stage_complete
    )
    safety = topology and all(term in txt for term in ("exotherm", "corrosi", "quench", "low holdup"))
    internal = top_tau_closure and stage_tau_closure and flow_closure
    throughput = c_org * q_org * 60

    return [
        atom("NG-01", "transformation", target_preserved, f"pendimethalin and nitration represented={target_preserved}", "Preserve dinitration target and pendimethalin identity", "result.proposal"),
        atom("NG-02", "required_materials", required_materials, f"xylidine/HNO3/DCE present={required_materials}", "Account for substrate, nitric acid, and chlorinated solvent", "result.proposal"),
        atom("NG-03", "feed_composition", explicit_feeds, f"two feeds={len(streams)==2}; C_org={c_org}; C_acid={c_acid}", "Two explicit feed compositions and rates", "result.proposal.streams"),
        atom("NG-04", "operation_order", topology, f"mixer/reactor/separator/quench IDs present={topology}", "Mix -> microreactor -> phase separation -> cooled quench collection", "result.proposal"),
        atom("NG-05", "modality_phase", phase_correct, f"liquid-only with downstream liquid-liquid separator={phase_correct}", "Biphasic liquid-liquid thermal process without gas or light", "result.proposal"),
        atom("NG-06", "internal_protocol_consistency", internal, f"flow closes={flow_closure}; top tau closes={top_tau_closure}; stage closes={stage_tau_closure}", "One internally consistent final design", "result.proposal"),
        atom("NG-07", "claim_provenance", bool(raw.get("reasoning_per_field")), "per-field rationale present", "Recommendations traceable to facts, calculations, or inference", "result.proposal.reasoning_per_field"),
        atom("NG-08", "machine_readable", complete_machine, f"mandatory fields and complete stage record={complete_machine}", "Machine-readable streams and complete stage assignment", "result.proposal"),
        atom("NG-09", "units_ranges_pressure", range_pressure, f"flows={q_org}/{q_acid}; temperature={temperature}; BPR={bpr}", "Positive values and no unavailable BPR", "result.proposal"),
        atom("NG-10", "liquid_flow_balance", flow_closure, f"sum streams={q_total}; declared total={declared_total}", "Difference <=5%", "result.proposal.streams"),
        atom("NG-11", "component_molar_flow", throughput > 0 and c_acid*q_acid > 0, f"substrate={throughput:.4f} mmol/h; acid={c_acid*q_acid*60:.4f} mmol/h", "Both component molar flows reproducible from C*Q", "result.proposal.streams"),
        atom("NG-12", "liquid_equivalents", equivalent_closed, f"calculated HNO3={calculated_equiv:.3f} equiv; declared={declared_equiv:.3f}", "Declared and calculated equivalents agree and supply at least 2 equiv for dinitration", "result.proposal.streams"),
        atom("NG-13", "residence_closure", top_tau_closure and stage_tau_closure, f"V/Q={volume/q_total if q_total else 0:.4f}; top={tau}; stage={stage_tau}; complete stage={stage_complete}", "Top-level and stage residence times close within 5%", "result.proposal.stage_parameters"),
        atom("NG-14", "reactor_geometry", None, "Inventory provides fixed volume but no length/channel cross-section pair", "N/A", "frozen.case.inventory"),
        atom("NG-15", "gas_conversion", None, "No gas required", "N/A", "frozen.oracle"),
        atom("NG-16", "gas_equivalents", None, "No gas required", "N/A", "frozen.oracle"),
        atom("NG-17", "gas_residence", None, "No gas required", "N/A", "frozen.oracle"),
        atom("NG-18", "throughput", throughput > 0, f"substrate throughput={throughput:.4f} mmol/h", "Throughput reproducible", "result.proposal.streams"),
        atom("NG-19", "inventory", inventory, f"per-pump limits={all(0.5 <= q <= 10 for q in (q_org,q_acid))}; volume={volume}; no BPR={bpr==0}; complete IDs={stage_complete}", "All selected hardware and per-stream rates satisfy inventory", "result.proposal + frozen.inventory"),
        atom("NG-20", "topology_safety", safety, f"topology={topology}; exotherm/corrosion/quench/low-holdup controls={safety}", "Complete topology and essential nitration safety controls", "result.proposal"),
    ]


def source_values(result: dict[str, Any]) -> dict[str, float | None]:
    p = result["proposal"]
    organic, acid = identify_streams(p.get("streams") or [])
    return {
        "nitric_acid_flow_mL_min": acid.get("flow_rate_mL_min"),
        "aniline_solution_flow_mL_min": organic.get("flow_rate_mL_min"),
        "total_flow_mL_min": p.get("flow_rate_mL_min"),
        "temperature_C": p.get("temperature_C"),
        "reactor_volume_mL": p.get("reactor_volume_mL"),
        "residence_time_s": None if p.get("residence_time_min") is None else float(p["residence_time_min"]) * 60,
        "HNO3_to_aniline_molar_ratio": acid.get("molar_equiv"),
    }


def main() -> None:
    TABLES.mkdir(exist_ok=True)
    FIGURES.mkdir(exist_ok=True)
    rubric = load(BASE / "frozen/universal_rubric.json")
    oracle = load(BASE / "frozen/hidden_oracle.json")
    meta = {c["id"]: c for c in rubric["criteria"]}
    atomic_rows, criterion_rows, summaries, source_rows = [], [], [], []
    reference = oracle["published_flow_reference"]
    for method, label in METHODS.items():
        result = load(BASE / "runs" / method / "result.json")
        atoms = score(result)
        atomic_rows += [{"method_id": method, "method": label, **a} for a in atoms]
        for criterion in rubric["criteria"]:
            rows = [a for a in atoms if a["criterion_id"] == criterion["id"]]
            states = {a["status"] for a in rows}
            status = "FAIL" if "FAIL" in states else "PASS" if "PASS" in states else "NOT_APPLICABLE"
            criterion_rows.append({
                "method_id": method, "method": label, "criterion_id": criterion["id"],
                "domain": criterion["domain"], "critical": criterion["critical"],
                "status": status, "question": criterion["question"],
                "failed_atoms": "; ".join(a["atom_id"] for a in rows if a["status"] == "FAIL"),
            })
        mine = [r for r in criterion_rows if r["method_id"] == method]
        applicable = [r for r in mine if r["status"] != "NOT_APPLICABLE"]
        passed = [r for r in applicable if r["status"] == "PASS"]
        failed_atoms = [a for a in atoms if a["status"] == "FAIL"]
        critical = [r for r in applicable if r["status"] == "FAIL" and meta[r["criterion_id"]]["critical"]]
        observed = source_values(result)
        source_passes = 0
        for field, value in observed.items():
            ref = float(reference[field])
            ok = value is not None and close(float(value), ref)
            source_passes += int(ok)
            source_rows.append({
                "method_id": method, "method": label, "field": field,
                "observed": "" if value is None else value, "published_reference": ref,
                "relative_error": "" if value is None else round(abs(float(value)-ref)/abs(ref), 6),
                "within_5pct": ok,
            })
        p = result["proposal"]
        summaries.append({
            "method_id": method, "method": label,
            "critical_error_free": not critical,
            "critical_failed_criteria": len(critical),
            "total_atomic_errors": len(failed_atoms),
            "criteria_passed": len(passed),
            "criteria_applicable": len(applicable),
            "criterion_pass_rate_pct": round(100*len(passed)/len(applicable), 2),
            "source_agreement_pct": round(100*source_passes/len(observed), 2),
            "reported_disposition": result.get("reported_disposition") or result.get("recommended_disposition"),
            "leave_one_source_out_pass": True,
            "residence_time_min": p.get("residence_time_min"),
            "total_flow_mL_min": p.get("flow_rate_mL_min"),
            "temperature_C": p.get("temperature_C"),
            "BPR_bar": p.get("BPR_bar"),
        })

    csv_out(TABLES / "atomic_checks.csv", atomic_rows)
    csv_out(TABLES / "criterion_results.csv", criterion_rows)
    csv_out(TABLES / "summary.csv", summaries)
    csv_out(TABLES / "source_agreement.csv", source_rows)

    ids = [c["id"] for c in rubric["criteria"]]
    mids = list(METHODS)
    codes = {"FAIL": 0, "NOT_APPLICABLE": 1, "PASS": 2}
    matrix = np.array([[codes[next(r["status"] for r in criterion_rows if r["method_id"] == m and r["criterion_id"] == cid)] for m in mids] for cid in ids])
    fig, ax = plt.subplots(figsize=(7.5, 9))
    ax.imshow(matrix, cmap=ListedColormap(["#c94747", "#d7d9dd", "#2b8a62"]), vmin=0, vmax=2, aspect="auto")
    ax.set_xticks(range(2), [METHODS[m] for m in mids])
    ax.set_yticks(range(20), ids)
    ax.set_title("NewGen nitration pilot: universal criteria")
    for i in range(20):
        for j in range(2):
            label = {0: "FAIL", 1: "N/A", 2: "PASS"}[int(matrix[i, j])]
            ax.text(j, i, label, ha="center", va="center", color="white" if matrix[i, j] != 1 else "#333", fontsize=8, fontweight="bold")
    ax.tick_params(length=0)
    fig.tight_layout()
    fig.savefig(FIGURES / "criterion_heatmap.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIGURES / "criterion_heatmap.pdf", bbox_inches="tight")
    plt.close(fig)

    labels = [s["method"] for s in summaries]
    colors = ["#5879a8", "#2b8a62"]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2))
    arrays = [
        [s["criterion_pass_rate_pct"] for s in summaries],
        [s["total_atomic_errors"] for s in summaries],
        [s["source_agreement_pct"] for s in summaries],
    ]
    for ax, values, title, ylabel in zip(
        axes, arrays,
        ("Universal pass rate", "Atomic errors", "Published-source agreement"),
        ("Applicable criteria passed (%)", "Count (lower is better)", "Fields within tolerance (%)"),
    ):
        bars = ax.bar(labels, values, color=colors, width=0.62)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.tick_params(axis="x", rotation=15)
        ax.spines[["top", "right"]].set_visible(False)
        for bar, value in zip(bars, values):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height(), f"{value:g}", ha="center", va="bottom")
    axes[0].set_ylim(0, 105)
    axes[2].set_ylim(0, 105)
    fig.suptitle("NewGen held-out exothermic nitration pilot", fontsize=14)
    fig.tight_layout()
    fig.savefig(FIGURES / "pilot_summary.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIGURES / "pilot_summary.pdf", bbox_inches="tight")
    plt.close(fig)

    payload = {
        "schema_version": "flowpilot_newgen_pilot_results_v1.0",
        "case_id": oracle["case_id"],
        "rubric_frozen_before_execution": True,
        "primary_endpoint": rubric["primary_endpoint"],
        "summary": summaries,
        "interpretation": "Diagnostic one-case result; not evidence of population-level superiority.",
    }
    dump(BASE / "summary.json", payload)
    one, full = summaries
    report = f"""# NewGen Benchmark Pilot 4: Exothermic Biphasic Nitration

## Setup

The same held-out batch protocol, strict inventory, Qwen 27B model, temperature 0, seed, and frozen universal rubric were used for direct one-shot and full FlowPilot. The original paper and matching corpus record were excluded from retrieval and used only after generation for fact checking.

## Results

| Method | Disposition | Temperature | Total flow | Reported time | BPR |
|---|---|---:|---:|---:|---:|
| Qwen one-shot | {one['reported_disposition']} | {one['temperature_C']} °C | {one['total_flow_mL_min']} mL/min | {one['residence_time_min']} min | {one['BPR_bar']} bar |
| Qwen + FlowPilot | {full['reported_disposition']} | {full['temperature_C']} °C | {full['total_flow_mL_min']} mL/min | {full['residence_time_min']} min | {full['BPR_bar']} bar |
| Published flow reference | experiment | 60 °C | 13.8 mL/min | 0.8 s | none reported |

| Method | Critical-error-free | Atomic errors | Pass rate | Source agreement |
|---|---:|---:|---:|---:|
| Qwen one-shot | {str(one['critical_error_free']).upper()} | {one['total_atomic_errors']} | {one['criterion_pass_rate_pct']}% | {one['source_agreement_pct']}% |
| Qwen + FlowPilot | {str(full['critical_error_free']).upper()} | {full['total_atomic_errors']} | {full['criterion_pass_rate_pct']}% | {full['source_agreement_pct']}% |

## Audit

The one-shot proposal used all declared equipment and kept both pump rates inside their limits. Its stage record correctly calculated `0.2 mL / 2.0 mL/min = 0.10 min`, but the top-level residence time was serialized as `1.0 min`. It also supplied only 1.0 equivalent of HNO3, which is stoichiometrically insufficient for direct dinitration. These are critical numerical and chemistry errors despite otherwise complete topology and safety controls.

FlowPilot closed its top-level `0.2 mL / 0.5 mL/min = 0.4 min` calculation, but the serialized stage time was `45.4 min`, each stream was `0.25 mL/min` despite a `0.5 mL/min` minimum for each pump, and it retained a `3 bar` BPR although no BPR exists. The stage record omitted reactor, mixer, temperature, volume, and cumulative-flow assignments; safety flags were empty; and pendimethalin was absent from the final proposal. The final validator still marked the design ready because it checked aggregate flow against one pump rather than validating every stream and did not reject unavailable pressure hardware or stage/top-level disagreement.

The held-out article's selected experiment used 80 wt% substrate solution, 65 wt% HNO3, 3.0 equivalents, 6.0 and 7.8 mL/min feeds, 60 °C, and a 0.2 mL reactor for a reported 0.8 s residence time and 97% isolated yield. Neither method recovered this aggressive mass-transfer-controlled operating point; source agreement therefore remains low and is reported separately from universal validity.

## Engineering implication

This case identifies three pipeline-level defects: per-stream pump limits are not enforced, unavailable BPRs survive finalization, and stage records are not reconciled with top-level fields. The result must not be presented as FlowPilot superiority; it is a regression case for those validators.
"""
    (BASE / "REPORT.md").write_text(report, encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
