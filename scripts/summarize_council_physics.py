"""Build a reproducible report/plots from an archived physics-enabled council run."""
import argparse
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--report-name", default="report")
    args = parser.parse_args()
    manifest = json.loads((args.run / "manifest.json").read_text())
    is_mock = manifest.get("offline_selection_test", False)
    folder = args.run / "on"
    result = json.loads((folder / "result.json").read_text())
    audit = result["scientific_assessment"]
    physics = audit["physics_review"]
    report = args.run / args.report_name
    report.mkdir(exist_ok=False)
    simulations = [dict(candidate_id=c["candidate_id"], **s)
                   for c in physics["results"] for s in c.get("simulations", [])]
    keys = ["candidate_id", "variant", "sensitivity", "scenario", "status", "reverse_flow_predicted",
            "reverse_displacement_uL", "peak_reverse_liquid_mL_min", "peak_junction_bar_g",
            "peak_gas_plenum_bar_g", "min_MFC_margin_above_required_bar", "evidence_id"]
    with (report / "simulations.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(simulations)
    from flora_translate.engine.council_v4.scientific import signature
    from flora_translate.schemas import FlowProposal
    verification = {
        "simulations": len(simulations),
        "supported_candidates": sum(c["status"] == "supported_unvalidated" for c in physics["inputs"]),
        "simulation_statuses": {s: sum(r["status"] == s for r in simulations) for s in sorted({r["status"] for r in simulations})},
        "all_conservation_checks_pass": all(r.get("numerical_checks", {}).get("conservation_passed") is True for r in simulations),
        "no_new_agent": {r["role"] for r in audit["calls"]} <= {"DrChemistry", "DrKinetics", "DrFluidics", "DrSafety", "Skeptic", "Chief"},
        "tool_users": [r["role"] for r in physics["tool_calls"]],
        "selected_candidate": audit["selected_candidate_id"],
        "selected_design_preserved": audit.get("selected_design_preserved"),
        "final_contract_status": result["final_design"]["status"],
        "stage_closure": all(s["closure"] for s in result["result_report"]["stages"]),
        "topology_exists": (folder / "topology.png").is_file(),
        "design_modification_applied": physics["design_modification_applied"],
        "laboratory_execution_status": "review_required", "backflow_probability": None,
        "profile_provenance": physics["profile"]["provenance"],
        "model_responses_mocked": is_mock,
    }
    if args.baseline:
        previous = json.loads((args.baseline / "off/result.json").read_text())
        baseline_audit = previous["scientific_assessment"]
        sigs = lambda a: [signature(FlowProposal.model_validate(c["proposal"])) for c in a["candidates"]]
        verification["archived_baseline_comparison"] = {
            "path": str(args.baseline), "same_candidate_physical_signatures": sigs(audit) == sigs(baseline_audit),
            "same_frozen_generation": json.loads((args.run / "frozen_generation.json").read_text()) == json.loads((args.baseline / "frozen_generation.json").read_text()),
            "same_original_intake": json.loads((args.run / "intake_package.json").read_text()) == json.loads((args.baseline / "intake_package.json").read_text()),
            "baseline_model": baseline_audit.get("model"), "new_model": audit.get("model"),
            "baseline_selected_candidate": baseline_audit["selected_candidate_id"],
            "limitation": "Archived one-run comparison: baseline disables both topology review and physics tools. Not an isolated marginal-physics or statistical performance comparison.",
        }
    # Recompute a saved case using a smaller step/tolerance; this is a numerical check, not laboratory validation.
    from flora_translate.engine.council_v4.transient_flow import simulate
    sample = next(r for r in simulations if r["candidate_id"] == 1 and r["variant"] == "as_designed"
                  and r["sensitivity"] == "one_tenth_gas_plenum" and r["scenario"] == "gas_first_start")
    artifact = json.loads((folder / "physics_tools" / Path(sample["artifact_path"]).name).read_text())
    refined = simulate(artifact["network"], artifact["profile"], artifact["scenario"], max_step_s=0.1, rtol=1e-9)
    verification["convergence_check"] = {
        "evidence_id": sample["evidence_id"], "default_reverse_uL": sample["reverse_displacement_uL"],
        "refined_reverse_uL": refined["reverse_displacement_uL"],
        "absolute_difference_uL": abs(sample["reverse_displacement_uL"] - refined["reverse_displacement_uL"]),
        "refined_status": refined["status"], "refined_conservation": refined["numerical_checks"],
    }
    verification["startup_window_check"] = {
        "case": sample["evidence_id"], "BPR_bar_g": artifact["network"]["BPR_bar_g"],
        "peak_junction_bar_g": artifact["peak_junction_bar_g"],
        "opening_pressure_reached": artifact["peak_junction_bar_g"] >= artifact["network"]["BPR_bar_g"],
        "simulated_until_s": artifact["simulated_until_s"],
    }
    write_json(report / "refined_numerical_check.json", refined)
    write_json(report / "verification.json", verification)
    write_json(report / "reviewer_assessments.json", physics.get("reviewer_assessments", {}))
    from flora_translate.engine.council_v4.physics_tools import selected_physics_findings
    selected = next(c for c in physics["results"] if c["candidate_id"] == audit["selected_candidate_id"])
    findings = selected_physics_findings(selected)
    write_json(report / "deterministic_selected_warnings.json", findings)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 12, "axes.titlesize": 13, "axes.labelsize": 12,
                         "legend.fontsize": 10, "svg.fonttype": "none"})
    variants = ["as_designed", "pure_oxygen_equal_molar", "hypothetical_liquid_check"]
    colors = {"as_designed": "#b43d48", "pure_oxygen_equal_molar": "#247eac", "hypothetical_liquid_check": "#548b4b"}
    labels = {"as_designed": "As designed (air)", "pure_oxygen_equal_molar": "Equal-dose oxygen (unapproved)",
              "hypothetical_liquid_check": "Hypothetical liquid check valve"}
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 8.5), constrained_layout=True)
    handles = []
    for col, sensitivity in enumerate(["entered_profile", "one_tenth_gas_plenum"]):
        for variant in variants:
            rows = [r for r in simulations if r["candidate_id"] == 1 and r["variant"] == variant
                    and r["scenario"] == "gas_first_start" and r["sensitivity"] == sensitivity]
            if not rows:
                continue
            data = json.loads((folder / "physics_tools" / Path(rows[0]["artifact_path"]).name).read_text())["trace"]
            line, = axes[0, col].plot([r["time_s"] for r in data], [r["liquid_branch_mL_min"] for r in data],
                label=labels[variant], color=colors[variant], linewidth=2, linestyle="--" if variant == "hypothetical_liquid_check" else "-")
            axes[1, col].plot([r["time_s"] for r in data], [r["junction_bar_g"] for r in data], color=colors[variant], linewidth=2,
                linestyle="--" if variant == "hypothetical_liquid_check" else "-")
            if col == 0:
                handles.append(line)
        axes[0, col].axhline(0, color="#333333", linewidth=0.8)
        axes[0, col].set_title(f"{'a' if col == 0 else 'b'}  Assumed gas plenum: {physics['profile']['gas_plenum_mL'] / (10 if col else 1):g} mL", loc="left")
        axes[1, col].set_title(f"{'c' if col == 0 else 'd'}  Junction pressure", loc="left")
        axes[0, col].set_ylabel("Liquid-branch flow (mL/min)")
        axes[1, col].set_ylabel("Gauge pressure (bar)")
        for ax in axes[:, col]:
            ax.axvline(physics["profile"]["startup_liquid_delay_s"], color="#777777", linewidth=0.8, linestyle=":")
            ax.set_xlabel("Time (s)")
            ax.grid(alpha=0.2)
            ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Candidate 1: gas-first startup under illustrative assumptions\nConditional model predictions, not measured KHU behavior", fontsize=15)
    fig.legend(handles=handles, loc="outside lower center", ncol=3, frameon=False)
    fig.savefig(report / "conditional_startup_comparison.png", dpi=220)
    fig.savefig(report / "conditional_startup_comparison.pdf")
    plt.close(fig)

    table = []
    for variant in variants:
        for sensitivity in ("entered_profile", "one_tenth_gas_plenum"):
            rows = [r for r in simulations if r["candidate_id"] == 1 and r["variant"] == variant and r["sensitivity"] == sensitivity]
            if rows:
                valid = [r for r in rows if r["status"] == "simulated_unvalidated"]
                value = f"{max(r['reverse_displacement_uL'] for r in valid):.6g}" if valid else "not assessable"
                table.append(f"| {variant} | {sensitivity} | {value} |")
    paragraphs = ["# Council transient-tool trial", "", "**Illustrative engineering model; not validated against KHU pressure/flow measurements.**", "",
        f"Council model: `{'MOCK RESPONSES (not a model benchmark)' if is_mock else audit.get('model')}`. Selected candidate: {audit['selected_candidate_id']}. "
        f"Recorded model calls: {len(audit['calls'])}. Tool users: {', '.join(verification['tool_users'])}.", "",
        f"{len(simulations)} simulations across {len(physics['results'])} candidates. "
        f"Status counts: {verification['simulation_statuses']}. All conservation checks pass: {verification['all_conservation_checks_pass']}.", "",
        "## Conditional candidate-1 results", "",
        "Maximum reverse liquid displacement across the four test scenarios. This table is not an experimental backflow probability or a measure of chemical yield.", "",
        "| Alternative | Gas-volume assumption | Maximum reverse displacement (uL) |", "|---|---|---:|", *table, "",
        "The entered and smaller-plenum assumptions can give different conclusions for the same plumbing. "
        "A small computed displacement does not reproduce the severity of the collaborator's incident. "
        "Oxygen arrival at Reactor 1, multiphase startup details and actual failure frequency are not modeled.", "",
        f"In the convergence-check startup case, the peak junction pressure is {artifact['peak_junction_bar_g']:.3g} bar(g) "
        f"against a BPR opening setpoint of {artifact['network']['BPR_bar_g']:g} bar(g). "
        "A finite startup window does not establish completed startup or subsequent stable operation. "
        "The hypothetical liquid valve uses the explicit profile's cracking pressure and leakage, "
        "not the gas-line CV-3301 specifications; naming an inventory valve in a model suggestion does not validate it for this alternative.", "",
        "## Deterministic warnings for the selected design", "",
        "These warnings are recomputed directly from the archived tool output, independently of the Chief's narrative. "
        "They are conditional on the assumptions and do not establish measured failure.", "",
        *[f"- {f['message']}" for f in findings], "",
        "## Actual reviewer recommendations", ""]
    for role, assessment in physics.get("reviewer_assessments", {}).items():
        paragraphs.extend([f"### {role}", "", "Proposals from mocked responses:" if is_mock else "Proposals recorded from the live model (unapproved):", ""])
        paragraphs.extend([f"- {v}" for v in assessment.get("proposed_alternatives", [])])
        paragraphs.extend(["", "Limitations stated by the reviewer:", "", *[f"- {v}" for v in assessment.get("limitations", [])], ""])
    paragraphs.extend(["## Verification", "", "```json", json.dumps(verification, indent=2), "```", "",
        "The tool does not mutate the final design or install proposed accessories. Numerical closure remains separate from laboratory approval.", "",
        "Full raw prompts/responses: `../on/llm_calls.jsonl` and `../on/council_audit.json`. "
        "All simulation inputs/traces: `../on/physics_tools/`. Final process: `../on/result.json`, `../on/topology.png`, `../on/stages.csv` and `../on/streams.csv`.", "",
        "See `docs/council_transient_physics.md` in the code repository for equations, scope, assumptions and how to disable the feature."])
    (report / "REPORT.md").write_text("\n".join(paragraphs) + "\n")
    print(json.dumps(verification, indent=2))
    print("REPORT", report)


if __name__ == "__main__":
    main()
