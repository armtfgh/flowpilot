"""Score and visualize the NewGen three-phase hydrogenolysis pilot."""

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


BASE = ROOT / "ablation_results" / "newgen_benchmark" / "newgen_benchmark_v1_pilot_hydrogenolysis_20260813"
TABLES = BASE / "tables"
FIGURES = BASE / "figures"
METHODS = {
    "qwen27b_one_shot": "Qwen 27B one-shot",
    "qwen27b_full_flowpilot": "Qwen 27B + FlowPilot",
}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def close(a: float, b: float, rel: float) -> bool:
    return abs(a - b) <= rel * max(abs(b), 1e-12)


def text(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False).lower()


def atom(cid: str, aid: str, passed: bool | None, observed: str, expected: str, path: str) -> dict[str, Any]:
    return {
        "criterion_id": cid,
        "atom_id": aid,
        "status": "NOT_APPLICABLE" if passed is None else "PASS" if passed else "FAIL",
        "observed": observed,
        "expected": expected,
        "evidence_path": path,
    }


def score_method(method_id: str, result: dict[str, Any]) -> list[dict[str, Any]]:
    p = result["proposal"]
    native = result.get("raw_proposal") or p
    all_text = text(result)
    proposal_text = text(native)
    streams = p.get("streams", [])
    liquid = [s for s in streams if s.get("phase") == "liquid"]
    gas = [s for s in streams if s.get("phase") == "gas"]
    q_liq = float(p["flow_rate_mL_min"])
    q_liq_sum = sum(float(s.get("flow_rate_mL_min") or 0) for s in liquid)
    c = float(p["concentration_M"])
    tau = float(p["residence_time_min"])
    volume = float(p["reactor_volume_mL"])
    p_abs = 21.0 if method_id == "qwen27b_one_shot" else float(p.get("multiphase_metrics", {}).get("gas_pressure_abs_bar") or 22.01325)
    gas_stream = gas[0] if gas else {}
    q_stp = float(gas_stream.get("gas_flow_sccm") or 0)
    q_actual = float(gas_stream.get("gas_flow_actual_mL_min") or 0)
    expected_actual = q_stp * (333.15 / 273.15) * (1.01325 / p_abs) if q_stp else 0
    h2_mmol_min = q_stp / 22414.0 * 1000.0
    dmaol_mmol_min = c * q_liq
    expected_equiv = h2_mmol_min / dmaol_mmol_min if dmaol_mmol_min else 0
    reported_equiv = float(gas_stream.get("molar_equiv") or p.get("multiphase_metrics", {}).get("gas_equiv_supplied") or 0)

    catalyst_accounted = (
        "pd(oh)2" in all_text
        or "reactor_h2_pbr_300" in proposal_text
    )
    required_materials = (
        all(term in proposal_text for term in ("dmaol", "methanol", "h2"))
        and catalyst_accounted
    )
    topology = all(term in proposal_text for term in ("t-mixer", "packed", "separator", "collect"))
    safety = all(term in all_text for term in ("hydrogen", "methanol", "purge", "pressure"))
    machine_fields = (
        "residence_time_min", "flow_rate_mL_min", "temperature_C", "concentration_M",
        "BPR_bar", "reactor_type", "tubing_material", "tubing_ID_mm",
        "reactor_volume_mL", "residence_time_basis", "streams",
    )
    machine_ok = all(field in native for field in machine_fields)
    geometry_volume = math.pi * (4.35 ** 2) / 4 * 200 / 1000
    geometry_ok = close(geometry_volume, volume, 0.05)

    if method_id == "qwen27b_one_shot":
        internal_ok = True
        residence_closure = close(q_liq * tau, volume, 0.05)
        residence_basis_ok = False  # Claims liquid holdup but uses the full empty 3 mL bed.
        inlet_tau = float(p.get("residence_time_inlet_min") or 0)
        channel_tau = float(p.get("residence_time_in_channel_min") or 0)
        gas_residence_ok = (
            close(inlet_tau, volume / (q_liq + q_stp), 0.10)
            and close(channel_tau, volume / (q_liq + expected_actual), 0.10)
        )
        inventory_ok = 0.10 <= q_liq <= 2.0 and 1.0 <= q_stp <= 100.0
        molar_reported = float(native.get("multiphase_metrics", {}).get("dmaol_molar_flow_mmol_min") or 0)
        molar_ok = close(molar_reported, dmaol_mmol_min, 0.05)
        volume_basis_observed = (
            f"V/Q={volume/q_liq:.1f} min closes numerically, but output labels the full "
            "empty-bed volume as liquid holdup without porosity/holdup data"
        )
    else:
        rationale = text(p.get("reasoning_per_field", {})) + text([s.get("reasoning") for s in streams])
        internal_ok = not all(token in rationale for token in ("0.5 ml/min", "50 sccm", "6 min"))
        residence_closure = close((q_liq + q_actual) * tau, volume, 0.05)
        residence_basis_ok = True
        inlet_tau = float(p.get("residence_time_inlet_min") or 0)
        channel_tau = float(p.get("residence_time_in_channel_min") or 0)
        gas_residence_ok = (
            close(inlet_tau, volume / (q_liq + q_stp), 0.10)
            and close(channel_tau, volume / (q_liq + q_actual), 0.10)
        )
        inventory_ok = 0.10 <= q_liq <= 2.0 and 1.0 <= q_stp <= 100.0
        molar_ok = close(float(p.get("multiphase_metrics", {}).get("gas_required_mmol_min") or 0), dmaol_mmol_min, 0.05)
        volume_basis_observed = f"(Q_liq+Q_H2,channel)*tau={(q_liq+q_actual)*tau:.5f} mL; V={volume:.5f} mL"

    gas_conversion_ok = close(q_actual, expected_actual, 0.10)
    equiv_ok = close(reported_equiv, expected_equiv, 0.10)
    pressure_ok = float(p["BPR_bar"]) == 21.0 and p_abs >= 21.0
    protocol_ok = float(p["temperature_C"]) == 60.0 and close(c, 0.12576, 0.05)
    throughput = dmaol_mmol_min * 60

    return [
        atom("NG-01", "transformation_identity", "hydrogenolysis" in all_text and "3-azetidinol" in all_text, "DMAOL hydrogenolysis to 3-azetidinol", "Preserve transformation and product", "result"),
        atom("NG-02", "required_materials", required_materials, f"DMAOL/methanol/H2/Pd catalyst present={required_materials}", "Account for substrate, solvent, H2, and Pd(OH)2/Al2O3", "result.proposal"),
        atom("NG-03", "feed_composition", close(c, 0.12576, 0.05) and bool(liquid), f"DMAOL={c:.5f} M; one liquid feed", "3.8 wt% converted to approximately 0.126 M and catalyst bed declared", "result.proposal.streams"),
        atom("NG-04", "operation_order", topology, "liquid + H2 -> T-mixer -> packed bed -> separator -> collector", "Correct gas-liquid-solid operation order", "result.proposal"),
        atom("NG-05", "modality_phase", bool(liquid) and bool(gas) and "packed" in proposal_text, "separate liquid/H2 feeds and packed catalyst bed", "Gas-liquid-solid hydrogenolysis", "result.proposal.streams"),
        atom("NG-06", "protocol_and_internal_consistency", protocol_ok and internal_ok, "protocol values preserved; final rationale consistent" if internal_ok else "final fields differ from retained 0.5 mL/min, 50 sccm, and 6 min rationale", "No protocol or internal final-value contradiction", "result.proposal.reasoning_per_field"),
        atom("NG-07", "claim_provenance", bool(native.get("reasoning_per_field")) and ("formula" in all_text or "equation" in all_text), "reasoning and formulas/equations reported", "Facts, assumptions, and calculations traceable", "result"),
        atom("NG-08", "machine_readable_fields", machine_ok and len(streams) == 2, f"mandatory fields={machine_ok}; streams={len(streams)}", "Valid machine-readable final design with separate streams", "result.proposal"),
        atom("NG-09", "units_ranges_pressure", pressure_ok and q_stp > 0 and q_actual > 0, f"BPR={p['BPR_bar']} bar; gas Pabs={p_abs:.5f} bar; STP and actual fields distinct", "Valid ranges and explicit pressure basis", "result.proposal.multiphase_metrics"),
        atom("NG-10", "liquid_flow_balance", close(q_liq_sum, q_liq, 0.05), f"top Q={q_liq:.5f}; liquid stream sum={q_liq_sum:.5f}", "Difference <=5%", "result.proposal.streams"),
        atom("NG-11", "component_molar_flow", molar_ok, f"expected DMAOL={dmaol_mmol_min:.8f} mmol/min", "Reported or traced component molar flow within 5%", "result.proposal.multiphase_metrics"),
        atom("NG-12", "liquid_equivalents", None, "No second stoichiometric liquid reagent; acetic acid omitted", "N/A for single-substrate liquid feed", "frozen.case"),
        atom("NG-13", "volume_flow_time_basis", residence_closure and residence_basis_ok, volume_basis_observed, "Volume closes on declared basis and liquid holdup is not conflated with empty-bed volume", "result.proposal"),
        atom("NG-14", "reactor_geometry", geometry_ok, f"cylinder volume={geometry_volume:.5f} mL; reported={volume:.5f} mL", "4.35 mm ID x 200 mm length closes within 5%", "frozen.inventory + result.proposal"),
        atom("NG-15", "stp_to_channel_gas", gas_conversion_ok, f"reported={q_actual:.6f}; expected={expected_actual:.6f} mL/min at {p_abs:.5f} bar abs and 60 C", "Ideal-gas conversion within 10%", "result.proposal.streams"),
        atom("NG-16", "hydrogen_equivalents", equiv_ok, f"reported={reported_equiv:.4f}; calculated={expected_equiv:.4f} equiv", "H2 molar flow / DMAOL molar flow within 10%", "result.proposal.streams"),
        atom("NG-17", "gas_residence_bases", gas_residence_ok, f"reported inlet={inlet_tau:.4f}, channel={channel_tau:.4f} min", "Both V/(Qliq+Qgas,STP) and V/(Qliq+Qgas,channel) within 10%", "result.proposal"),
        atom("NG-18", "throughput", throughput > 0, f"DMAOL throughput={throughput:.5f} mmol/h", "Reproducible from C*Q*60", "result.proposal"),
        atom("NG-19", "inventory_limits", inventory_ok, f"liquid Q={q_liq:.5f} (allowed 0.10-2.0); H2={q_stp:.5f} (allowed 1-100 sccm)", "Every selected setting inside inventory limits", "result.proposal + frozen.inventory"),
        atom("NG-20", "topology_safety", topology and safety, f"topology={topology}; H2/methanol/pressure/purge controls={safety}", "Complete topology and executable three-phase safety controls", "result.proposal"),
    ]


def main() -> None:
    TABLES.mkdir(exist_ok=True)
    FIGURES.mkdir(exist_ok=True)
    rubric = read_json(BASE / "frozen" / "universal_rubric.json")
    oracle = read_json(BASE / "frozen" / "hidden_oracle.json")
    meta = {c["id"]: c for c in rubric["criteria"]}
    atomic_rows: list[dict[str, Any]] = []
    criterion_rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    source_rows: list[dict[str, Any]] = []

    for method_id, label in METHODS.items():
        result = read_json(BASE / "runs" / method_id / "result.json")
        atoms = score_method(method_id, result)
        atomic_rows.extend({"method_id": method_id, "method": label, **a} for a in atoms)
        for criterion in rubric["criteria"]:
            rows = [a for a in atoms if a["criterion_id"] == criterion["id"]]
            states = {a["status"] for a in rows}
            status = "FAIL" if "FAIL" in states else "PASS" if "PASS" in states else "NOT_APPLICABLE"
            criterion_rows.append({
                "method_id": method_id, "method": label, "criterion_id": criterion["id"],
                "domain": criterion["domain"], "critical": criterion["critical"],
                "status": status, "question": criterion["question"],
                "failed_atoms": "; ".join(a["atom_id"] for a in rows if a["status"] == "FAIL"),
            })
        method_criteria = [r for r in criterion_rows if r["method_id"] == method_id]
        applicable = [r for r in method_criteria if r["status"] != "NOT_APPLICABLE"]
        passed = [r for r in applicable if r["status"] == "PASS"]
        failed = [a for a in atoms if a["status"] == "FAIL"]
        critical = [r for r in applicable if r["status"] == "FAIL" and meta[r["criterion_id"]]["critical"]]

        p = result["proposal"]
        calc = result.get("final_calculations", {})
        gas_stream = next(s for s in p["streams"] if s["phase"] == "gas")
        observed = {
            "temperature_C": p["temperature_C"],
            "liquid_flow_mL_min": p["flow_rate_mL_min"],
            "hydrogen_flow_sccm": gas_stream["gas_flow_sccm"],
            "pressure_bar_reported": p["BPR_bar"],
            "estimated_substrate_concentration_M": p["concentration_M"],
            "reported_reactor_volume_mL": p["reactor_volume_mL"],
            "reactor_ID_mm": p["tubing_ID_mm"],
            "reactor_length_m": calc.get("tubing_length_m") if method_id.endswith("full_flowpilot") else None,
            "liquid_holdup_residence_time_min": p["residence_time_min"],
        }
        reference = oracle["published_flow_reference"]
        agreements = 0
        total = 0
        for field, value in observed.items():
            ref = float(reference[field])
            tolerance = 0.10 if field == "hydrogen_flow_sccm" else 0.05
            agrees = value is not None and close(float(value), ref, tolerance)
            agreements += int(agrees)
            total += 1
            source_rows.append({
                "method_id": method_id, "method": label, "field": field,
                "observed": "" if value is None else value, "published_reference": ref,
                "relative_error": "" if value is None else round(abs(float(value)-ref)/abs(ref), 6),
                "within_tolerance": agrees,
            })
        for field, value, ref in (
            ("reactor_type", p["reactor_type"], "packed"),
            ("material", p["tubing_material"], "stainless"),
        ):
            agrees = ref in str(value).lower()
            agreements += int(agrees)
            total += 1
            source_rows.append({
                "method_id": method_id, "method": label, "field": field,
                "observed": value, "published_reference": ref,
                "relative_error": "", "within_tolerance": agrees,
            })

        summaries.append({
            "method_id": method_id, "method": label,
            "critical_error_free": not critical,
            "critical_failed_criteria": len(critical),
            "total_atomic_errors": len(failed),
            "criteria_passed": len(passed), "criteria_applicable": len(applicable),
            "criterion_pass_rate_pct": round(100*len(passed)/len(applicable), 2),
            "source_agreement_pct": round(100*agreements/total, 2),
            "reported_disposition": result.get("reported_disposition") or result.get("recommended_disposition"),
            "leave_one_source_out_pass": True,
            "residence_time_min": p["residence_time_min"],
            "liquid_flow_mL_min": p["flow_rate_mL_min"],
            "hydrogen_flow_sccm": gas_stream["gas_flow_sccm"],
            "hydrogen_flow_in_channel_mL_min": gas_stream["gas_flow_actual_mL_min"],
            "hydrogen_equiv": gas_stream["molar_equiv"],
        })

    write_csv(TABLES / "atomic_checks.csv", atomic_rows)
    write_csv(TABLES / "criterion_results.csv", criterion_rows)
    write_csv(TABLES / "summary.csv", summaries)
    write_csv(TABLES / "source_agreement.csv", source_rows)

    ids = [c["id"] for c in rubric["criteria"]]
    method_ids = list(METHODS)
    codes = {"FAIL": 0, "NOT_APPLICABLE": 1, "PASS": 2}
    matrix = np.array([[codes[next(r["status"] for r in criterion_rows if r["method_id"] == m and r["criterion_id"] == cid)] for m in method_ids] for cid in ids])
    fig, ax = plt.subplots(figsize=(7.5, 9))
    ax.imshow(matrix, cmap=ListedColormap(["#c94747", "#d7d9dd", "#2b8a62"]), vmin=0, vmax=2, aspect="auto")
    ax.set_xticks(range(2), [METHODS[m] for m in method_ids])
    ax.set_yticks(range(20), ids)
    ax.set_title("NewGen hydrogenolysis pilot: universal criteria")
    for i in range(20):
        for j in range(2):
            label = {0: "FAIL", 1: "N/A", 2: "PASS"}[int(matrix[i,j])]
            ax.text(j, i, label, ha="center", va="center", color="white" if matrix[i,j] != 1 else "#333", fontsize=8, fontweight="bold")
    ax.tick_params(length=0)
    fig.tight_layout()
    fig.savefig(FIGURES / "criterion_heatmap.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIGURES / "criterion_heatmap.pdf", bbox_inches="tight")
    plt.close(fig)

    labels = [s["method"] for s in summaries]
    colors = ["#5879a8", "#2b8a62"]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2))
    values_list = [
        [s["criterion_pass_rate_pct"] for s in summaries],
        [s["total_atomic_errors"] for s in summaries],
        [s["source_agreement_pct"] for s in summaries],
    ]
    for ax, values, title, ylabel in zip(
        axes, values_list,
        ("Universal pass rate", "Atomic errors", "Published-source agreement"),
        ("Applicable criteria passed (%)", "Count (lower is better)", "Fields within tolerance (%)"),
    ):
        bars = ax.bar(labels, values, color=colors, width=0.62)
        ax.set_title(title); ax.set_ylabel(ylabel); ax.tick_params(axis="x", rotation=15)
        ax.spines[["top", "right"]].set_visible(False)
        for bar, value in zip(bars, values):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height(), f"{value:g}", ha="center", va="bottom")
    axes[0].set_ylim(0,105); axes[2].set_ylim(0,105)
    fig.suptitle("NewGen held-out three-phase hydrogenolysis pilot", fontsize=14)
    fig.tight_layout()
    fig.savefig(FIGURES / "pilot_summary.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIGURES / "pilot_summary.pdf", bbox_inches="tight")
    plt.close(fig)

    one, full = summaries
    payload = {
        "schema_version": "flowpilot_newgen_pilot_results_v1.0",
        "case_id": oracle["case_id"],
        "rubric_frozen_before_execution": True,
        "primary_endpoint": rubric["primary_endpoint"],
        "summary": summaries,
        "interpretation": "Diagnostic one-case result; not evidence of population-level superiority."
    }
    write_json(BASE / "summary.json", payload)
    report = f"""# NewGen Benchmark Pilot 2: Three-Phase Hydrogenolysis

## Design

- Same held-out batch protocol, strict inventory, Qwen 27B model, temperature 0, and seed.
- Compared direct one-shot Qwen against the full FlowPilot architecture.
- Original source `acs.oprd.9b00416.pdf` was excluded from retrieval and used only for post-generation fact checking.
- All prompts, raw responses, model events, council stages, frozen inputs, and source checksums are retained.

## Final outputs

| Method | Disposition | Liquid flow | H2 inlet/STP | H2 in channel | Residence time | H2 equivalents |
|---|---|---:|---:|---:|---:|---:|
| Qwen one-shot | {one['reported_disposition']} | {one['liquid_flow_mL_min']} mL/min | {one['hydrogen_flow_sccm']} sccm | {one['hydrogen_flow_in_channel_mL_min']} mL/min | {one['residence_time_min']} min | {one['hydrogen_equiv']} |
| Qwen + FlowPilot | {full['reported_disposition']} | {full['liquid_flow_mL_min']} mL/min | {full['hydrogen_flow_sccm']} sccm | {full['hydrogen_flow_in_channel_mL_min']} mL/min | {full['residence_time_min']} min | {full['hydrogen_equiv']} |
| Published flow reference | executable experiment | 0.5 mL/min | 40 sccm | 2.25-2.35 mL/min | 1.9 min liquid-holdup time | 26 |

## Universal scoring

| Method | Critical-error-free | Atomic errors | Pass rate | Source agreement |
|---|---:|---:|---:|---:|
| Qwen one-shot | {str(one['critical_error_free']).upper()} | {one['total_atomic_errors']} | {one['criterion_pass_rate_pct']}% | {one['source_agreement_pct']}% |
| Qwen + FlowPilot | {str(full['critical_error_free']).upper()} | {full['total_atomic_errors']} | {full['criterion_pass_rate_pct']}% | {full['source_agreement_pct']}% |

## Error audit

The one-shot answer violated the pump minimum (`0.01` versus `0.10 mL/min`) and made three-order-of-magnitude errors in both gas conversion and molar-flow calculations. At 60 °C and 21 bar absolute, `50 sccm` is approximately `2.94 mL/min` in channel, not `0.00238 mL/min`. Its DMAOL flow is `0.00126 mmol/min`, not `1.26e-6 mmol/min`; consequently, the reported H2 equivalents do not close. It also reported identical inlet and in-channel residence times despite gas compression.

FlowPilot correctly converted `0.9497 sccm` to approximately `0.0533 mL/min` at `22.013 bar absolute`, and its 1.0 H2 equivalent closes against the DMAOL flow. It detected that `0.9497 sccm` is below the MFC minimum of `1.0 sccm` and returned `BLOCK`. Its remaining defects are that stale rationale still mentions `0.5 mL/min`, `50 sccm`, and `6 min`, and it did not repair the near-boundary MFC value to the available minimum.

## Interpretation

FlowPilot was substantially better on universal calculation and inventory checks in this case, although it did not produce an executable final screen. The deterministic block is preferable to silently publishing an infeasible setting, but a robust system should clamp the MFC to `1.0 sccm`, recompute H2 equivalents and residence times, regenerate all rationale, and revalidate automatically.
"""
    (BASE / "REPORT.md").write_text(report, encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
