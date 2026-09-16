from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
OUTPUTS = ROOT / "outputs"


def load_json(name: str) -> dict[str, Any]:
    return json.loads((OUTPUTS / name).read_text())


def fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.4f}".rstrip("0").rstrip(".")
    return str(value)


def build_row(
    *,
    parameter_key: str,
    parameter_label: str,
    unit: str,
    pre_value: Any,
    final_value: Any,
    rationale_summary: str,
    rationale_source: str,
    rationale_quality: str,
) -> dict[str, str]:
    return {
        "parameter_key": parameter_key,
        "parameter_label": parameter_label,
        "unit": unit,
        "pre_council_value": fmt(pre_value),
        "final_value": fmt(final_value),
        "from_to": f"{fmt(pre_value)} -> {fmt(final_value)}",
        "rationale_summary": rationale_summary,
        "rationale_source": rationale_source,
        "rationale_quality": rationale_quality,
    }


def export_csv(path: Path, rows: list[dict[str, str]]) -> None:
    fieldnames = [
        "parameter_key",
        "parameter_label",
        "unit",
        "pre_council_value",
        "final_value",
        "from_to",
        "rationale_summary",
        "rationale_source",
        "rationale_quality",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_12_candidate_rows() -> list[dict[str, str]]:
    run = load_json("snar_council_run.json")
    pre = run["pre_council_proposal"]
    final = run["proposal"]
    calc = run["post_council_calculations"]

    return [
        build_row(
            parameter_key="residence_time_min",
            parameter_label="Residence Time",
            unit="min",
            pre_value=pre["residence_time_min"],
            final_value=final["residence_time_min"],
            rationale_summary=(
                "Dr. Chemistry and Dr. Kinetics both favored the longer-tau region for SNAr at 150 C. "
                "Chief selected candidate 4 because it kept strong chemistry/kinetics while using a shorter coil "
                "than another long-tau option. Kinetics still flagged X≈0.76 vs the 0.85 target, but this was much "
                "safer than the short-tau candidates."
            ),
            rationale_source=(
                "DrChemistryV4, DrKineticsV4, ChiefV4 in outputs/snar_council_run.json deliberation_log"
            ),
            rationale_quality="agent-backed",
        ),
        build_row(
            parameter_key="flow_rate_mL_min",
            parameter_label="Total Flow Rate",
            unit="mL/min",
            pre_value=pre["flow_rate_mL_min"],
            final_value=final["flow_rate_mL_min"],
            rationale_summary=(
                "Chief preferred candidate 4's lower-flow operating window because it had a wider, more forgiving "
                "flowrate envelope (0.0389-0.0722 mL/min) than candidate 12, making it less sensitive to pump precision limits."
            ),
            rationale_source="ChiefV4 selection rationale in outputs/snar_council_run.json",
            rationale_quality="agent-backed",
        ),
        build_row(
            parameter_key="reactor_volume_mL",
            parameter_label="Reactor Volume",
            unit="mL",
            pre_value=pre["reactor_volume_mL"],
            final_value=final["reactor_volume_mL"],
            rationale_summary=(
                "This compact reactor volume is the direct consequence of the winning tau/Q/d combination. "
                "The council favored a smaller reactor in a better geometry regime instead of keeping the original bulky design."
            ),
            rationale_source="Derived from Chief-selected candidate 4 in outputs/snar_council_run.json",
            rationale_quality="derived-from-agent-choice",
        ),
        build_row(
            parameter_key="tubing_ID_mm",
            parameter_label="Tubing Inner Diameter",
            unit="mm",
            pre_value=pre["tubing_ID_mm"],
            final_value=final["tubing_ID_mm"],
            rationale_summary=(
                "Chief chose the 0.75 mm design because candidate 4 had the highest geometry score, an optimal L/d ratio of 21.3, "
                "and better heat-transfer and mixing uniformity. This is the main geometry improvement over the pre-council design."
            ),
            rationale_source="ChiefV4 selection rationale in outputs/snar_council_run.json",
            rationale_quality="agent-backed",
        ),
        build_row(
            parameter_key="tubing_material",
            parameter_label="Tubing Material",
            unit="",
            pre_value=pre["tubing_material"],
            final_value=final["tubing_material"],
            rationale_summary=(
                "Chemistry and Safety both treated FEP as compatible with DMF at 150 C. "
                "The chemistry note explicitly cites a 50 C safety margin, and safety required FEP or PFA for this operating temperature."
            ),
            rationale_source="DrChemistryV4 + DrSafetyV4 in outputs/snar_council_run.json",
            rationale_quality="agent-backed",
        ),
        build_row(
            parameter_key="BPR_bar",
            parameter_label="Back-Pressure Regulator",
            unit="bar",
            pre_value=pre["BPR_bar"],
            final_value=final["BPR_bar"],
            rationale_summary=(
                "Safety judged the original 10 bar setting unnecessarily high. "
                "The council moved to the minimum safe liquid-phase region, consistent with the safety estimate that only about 0.62 bar was required."
            ),
            rationale_source="DrSafetyV4 in outputs/snar_council_run.json",
            rationale_quality="agent-backed",
        ),
        build_row(
            parameter_key="temperature_C",
            parameter_label="Temperature",
            unit="C",
            pre_value=pre["temperature_C"],
            final_value=final["temperature_C"],
            rationale_summary=(
                "No council agent argued for changing temperature. The case was framed as thermal SNAr at 150 C, "
                "so the council focused on geometry, pressure, and materials instead."
            ),
            rationale_source="DesignerV4 framing + no contrary revision in outputs/snar_council_run.json",
            rationale_quality="agent-backed",
        ),
        build_row(
            parameter_key="concentration_M",
            parameter_label="Reactor Concentration",
            unit="M",
            pre_value=pre["concentration_M"],
            final_value=final["concentration_M"],
            rationale_summary=(
                "Dr. Chemistry explicitly treated 0.15 M as well-optimized for this SNAr class, so concentration was left unchanged."
            ),
            rationale_source="DrChemistryV4 in outputs/snar_council_run.json",
            rationale_quality="agent-backed",
        ),
        build_row(
            parameter_key="reactor_type",
            parameter_label="Reactor Type",
            unit="",
            pre_value=pre["reactor_type"],
            final_value=final["reactor_type"],
            rationale_summary=(
                "No agent challenged the coil reactor topology. The council refined the operating point inside the same process architecture."
            ),
            rationale_source="No contrary revision in outputs/snar_council_run.json",
            rationale_quality="agent-backed",
        ),
        build_row(
            parameter_key="mixer_type",
            parameter_label="Mixer Type",
            unit="",
            pre_value=pre["mixer_type"],
            final_value=final["mixer_type"],
            rationale_summary=(
                "Fluidics judged mixing acceptable across the shortlisted designs and especially strong for candidate 4 "
                "(r_mix≈0.018), so the T-mixer stayed unchanged."
            ),
            rationale_source="DrFluidicsV4 in outputs/snar_council_run.json",
            rationale_quality="agent-backed",
        ),
        build_row(
            parameter_key="pressure_drop_bar",
            parameter_label="Pressure Drop",
            unit="bar",
            pre_value=run["pre_council_calculations"]["pressure_drop_bar"],
            final_value=calc["pressure_drop_bar"],
            rationale_summary=(
                "Pressure drop is a derived outcome, not a directly chosen knob. The winning 0.75 mm / low-flow geometry still kept delta-P extremely low, "
                "which supported the fluidics and safety acceptance."
            ),
            rationale_source="DrFluidicsV4 + DrSafetyV4, derived from selected candidate metrics",
            rationale_quality="derived-from-agent-choice",
        ),
        build_row(
            parameter_key="surface_to_volume_m_inv",
            parameter_label="Surface-to-Volume Ratio",
            unit="m^-1",
            pre_value=run["pre_council_calculations"]["surface_to_volume"],
            final_value=calc["surface_to_volume"],
            rationale_summary=(
                "Surface-to-volume more than doubled because the council chose the smaller-ID reactor. "
                "This was one of the main geometric advantages of the 12-candidate search over the single-candidate repair."
            ),
            rationale_source="Derived from Chief-selected candidate 4 geometry",
            rationale_quality="derived-from-agent-choice",
        ),
    ]


def build_1_candidate_rows() -> list[dict[str, str]]:
    run = load_json("snar_1cand_run.json")
    rich_run = load_json("snar_council_run.json")
    pre = run["pre_council_proposal"]
    final = run["one_cand_proposal"]
    calc = run["one_cand_calculations"]

    return [
        build_row(
            parameter_key="residence_time_min",
            parameter_label="Residence Time",
            unit="min",
            pre_value=pre["residence_time_min"],
            final_value=final["residence_time_min"],
            rationale_summary=(
                "The saved 1-candidate ablation file does not retain the detailed agent log. "
                "From the verified before/after state, the council appears to have repaired kinetics locally by increasing tau sharply while leaving Q and d unchanged."
            ),
            rationale_source="outputs/snar_1cand_run.json outcome only",
            rationale_quality="inference-from-outcome",
        ),
        build_row(
            parameter_key="flow_rate_mL_min",
            parameter_label="Total Flow Rate",
            unit="mL/min",
            pre_value=pre["flow_rate_mL_min"],
            final_value=final["flow_rate_mL_min"],
            rationale_summary=(
                "Flow rate stayed unchanged, which indicates the single-candidate ablation did not explore an alternative throughput regime. "
                "It repaired the original design locally instead of moving to a different geometry region."
            ),
            rationale_source="outputs/snar_1cand_run.json outcome only",
            rationale_quality="inference-from-outcome",
        ),
        build_row(
            parameter_key="reactor_volume_mL",
            parameter_label="Reactor Volume",
            unit="mL",
            pre_value=pre["reactor_volume_mL"],
            final_value=final["reactor_volume_mL"],
            rationale_summary=(
                "This larger reactor volume is the direct consequence of increasing tau while leaving flow rate and tubing diameter unchanged."
            ),
            rationale_source="Derived from outputs/snar_1cand_run.json",
            rationale_quality="derived-from-outcome",
        ),
        build_row(
            parameter_key="tubing_ID_mm",
            parameter_label="Tubing Inner Diameter",
            unit="mm",
            pre_value=pre["tubing_ID_mm"],
            final_value=final["tubing_ID_mm"],
            rationale_summary=(
                "Tubing ID stayed at 1.6 mm. The single-candidate ablation file shows no evidence of a geometry search, "
                "so the council remained in the original large-ID regime."
            ),
            rationale_source="outputs/snar_1cand_run.json outcome only",
            rationale_quality="inference-from-outcome",
        ),
        build_row(
            parameter_key="tubing_material",
            parameter_label="Tubing Material",
            unit="",
            pre_value=pre["tubing_material"],
            final_value=final["tubing_material"],
            rationale_summary=(
                "The 1-candidate result corrected material from PFA to FEP. "
                "Its ablation file does not preserve agent text, but the richer 12-candidate log for the same SNAr case explicitly approved FEP compatibility with DMF at 150 C."
            ),
            rationale_source="outputs/snar_1cand_run.json + cross-check with outputs/snar_council_run.json",
            rationale_quality="cross-checked inference",
        ),
        build_row(
            parameter_key="BPR_bar",
            parameter_label="Back-Pressure Regulator",
            unit="bar",
            pre_value=pre["BPR_bar"],
            final_value=final["BPR_bar"],
            rationale_summary=(
                "The 1-candidate result corrected BPR down from 10.0 to 0.6 bar. "
                "The detailed safety explanation was not retained in the ablation file, but the richer 12-candidate log for the same case estimated only about 0.62 bar was required."
            ),
            rationale_source="outputs/snar_1cand_run.json + cross-check with outputs/snar_council_run.json",
            rationale_quality="cross-checked inference",
        ),
        build_row(
            parameter_key="temperature_C",
            parameter_label="Temperature",
            unit="C",
            pre_value=pre["temperature_C"],
            final_value=final["temperature_C"],
            rationale_summary=(
                "Temperature stayed at 150 C. The saved 1-candidate output shows no chemistry-level change, only local repair of the operating point."
            ),
            rationale_source="outputs/snar_1cand_run.json",
            rationale_quality="proposal-backed",
        ),
        build_row(
            parameter_key="concentration_M",
            parameter_label="Reactor Concentration",
            unit="M",
            pre_value=pre["concentration_M"],
            final_value=final["concentration_M"],
            rationale_summary=(
                "Concentration stayed at 0.15 M. The preserved proposal-level reasoning treats 0.15 M as the practical reactor concentration for the mixed feeds, and the ablation kept that value."
            ),
            rationale_source="reasoning_per_field in outputs/snar_1cand_run.json",
            rationale_quality="proposal-backed",
        ),
        build_row(
            parameter_key="reactor_type",
            parameter_label="Reactor Type",
            unit="",
            pre_value=pre["reactor_type"],
            final_value=final["reactor_type"],
            rationale_summary=(
                "Reactor type remained a coil. The 1-candidate ablation did not change the process architecture."
            ),
            rationale_source="outputs/snar_1cand_run.json",
            rationale_quality="proposal-backed",
        ),
        build_row(
            parameter_key="mixer_type",
            parameter_label="Mixer Type",
            unit="",
            pre_value=pre["mixer_type"],
            final_value=final["mixer_type"],
            rationale_summary=(
                "Mixer type remained a T-mixer. The single-candidate result kept the same front-end mixing architecture."
            ),
            rationale_source="outputs/snar_1cand_run.json",
            rationale_quality="proposal-backed",
        ),
        build_row(
            parameter_key="pressure_drop_bar",
            parameter_label="Pressure Drop",
            unit="bar",
            pre_value=run["pre_council_calculations"]["pressure_drop_bar"],
            final_value=calc["pressure_drop_bar"],
            rationale_summary=(
                "Pressure drop increased only because the residence time was lengthened on the same geometry. "
                "This is a derived consequence, not a directly chosen field."
            ),
            rationale_source="Derived from outputs/snar_1cand_run.json",
            rationale_quality="derived-from-outcome",
        ),
        build_row(
            parameter_key="surface_to_volume_m_inv",
            parameter_label="Surface-to-Volume Ratio",
            unit="m^-1",
            pre_value=run["pre_council_calculations"]["surface_to_volume"],
            final_value=calc["surface_to_volume"],
            rationale_summary=(
                "Surface-to-volume ratio did not improve because the tubing diameter never changed. "
                "This is the clearest sign that the 1-candidate run repaired the original design without finding a better geometry regime."
            ),
            rationale_source="Derived from outputs/snar_1cand_run.json",
            rationale_quality="derived-from-outcome",
        ),
    ]


def main() -> None:
    export_csv(OUTPUTS / "snar_12cand_parameter_reasons.csv", build_12_candidate_rows())
    export_csv(OUTPUTS / "snar_1cand_parameter_reasons.csv", build_1_candidate_rows())
    print("Wrote:")
    print(" -", OUTPUTS / "snar_12cand_parameter_reasons.csv")
    print(" -", OUTPUTS / "snar_1cand_parameter_reasons.csv")


if __name__ == "__main__":
    main()
