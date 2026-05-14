from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
VIS_DIR = Path(__file__).resolve().parent

UPSTREAM_ORDER = ["claude", "gpt4o", "gpt4omini"]
COUNCIL_ORDER = ["claude", "gpt4o", "gpt4omini"]


def _format_stream(stream: dict) -> str:
    label = stream.get("stream_label", "?")
    contents = ", ".join(stream.get("contents") or [])
    solvent = stream.get("solvent") or "unspecified solvent"
    flow = stream.get("flow_rate_mL_min")
    conc = stream.get("concentration_M")
    role = stream.get("pump_role") or ""

    details: list[str] = []
    if role:
        details.append(role)
    if contents:
        details.append(contents)
    details.append(f"in {solvent}")
    if conc is not None:
        details.append(f"{conc} M")
    if flow is not None:
        details.append(f"{flow} mL/min")
    return f"Stream {label}: " + "; ".join(details)


def _scheme_text(proposal: dict) -> str:
    steps: list[str] = []

    streams = proposal.get("streams") or []
    if streams:
        for idx, stream in enumerate(streams, start=1):
            steps.append(f"{idx}. Prepare {_format_stream(stream)}.")

    pre_steps = proposal.get("pre_reactor_steps") or []
    next_idx = len(steps) + 1
    for offset, step in enumerate(pre_steps, start=0):
        steps.append(f"{next_idx + offset}. {step}.")

    mixer_type = proposal.get("mixer_type") or "unspecified"
    reactor_type = proposal.get("reactor_type") or "reactor"
    material = proposal.get("tubing_material") or "unspecified material"
    tubing_id = proposal.get("tubing_ID_mm")
    volume = proposal.get("reactor_volume_mL")
    flow_rate = proposal.get("flow_rate_mL_min")
    temperature = proposal.get("temperature_C")
    tau = proposal.get("residence_time_min")
    bpr = proposal.get("BPR_bar")

    reactor_bits = [f"Pass the process through a {reactor_type}"]
    if tubing_id is not None:
        reactor_bits.append(f"with {tubing_id} mm ID")
    if material:
        reactor_bits.append(f"made of {material}")
    if volume is not None:
        reactor_bits.append(f"and {volume} mL reactor volume")
    reactor_sentence = " ".join(reactor_bits)

    operating_bits: list[str] = []
    if mixer_type and mixer_type != "none":
        operating_bits.append(f"using a {mixer_type} mixer")
    elif mixer_type == "none":
        operating_bits.append("with no inline mixer")
    if flow_rate is not None:
        operating_bits.append(f"at {flow_rate} mL/min total flow")
    if temperature is not None:
        operating_bits.append(f"at {temperature} °C")
    if tau is not None:
        operating_bits.append(f"for {tau} min residence time")
    if bpr is not None:
        operating_bits.append(f"under {bpr} bar back-pressure")
    operating_sentence = ", ".join(operating_bits)

    steps.append(f"{len(steps) + 1}. {reactor_sentence}; operate {operating_sentence}.")

    post_steps = proposal.get("post_reactor_steps") or []
    next_idx = len(steps) + 1
    for offset, step in enumerate(post_steps, start=0):
        steps.append(f"{next_idx + offset}. {step}.")

    return " ".join(steps)


def main() -> None:
    rows: list[dict[str, str | float]] = []

    for upstream in UPSTREAM_ORDER:
        for council in COUNCIL_ORDER:
            result_path = (
                ROOT
                / f"U_{upstream}"
                / f"C_{council}"
                / "runs/protocol_isoxazole_des_full/budget_12/repeat_01/result.json"
            )
            if not result_path.exists():
                continue

            data = json.loads(result_path.read_text())
            proposal = (
                data.get("formatted_result", {}).get("proposal")
                or data.get("final_design_candidate", {}).get("proposal")
                or {}
            )
            explanation = data.get("formatted_result", {}).get("explanation", "")

            rows.append(
                {
                    "upstream_bundle": upstream,
                    "council_bundle": council,
                    "temperature_C": proposal.get("temperature_C"),
                    "residence_time_min": proposal.get("residence_time_min"),
                    "flow_rate_mL_min": proposal.get("flow_rate_mL_min"),
                    "tubing_material": proposal.get("tubing_material"),
                    "tubing_ID_mm": proposal.get("tubing_ID_mm"),
                    "reactor_volume_mL": proposal.get("reactor_volume_mL"),
                    "BPR_bar": proposal.get("BPR_bar"),
                    "reactor_type": proposal.get("reactor_type"),
                    "stream_count": len(proposal.get("streams") or []),
                    "process_scheme_text": _scheme_text(proposal),
                    "explanation": explanation,
                    "result_path": str(result_path),
                }
            )

    csv_path = VIS_DIR / "model_process_schemes.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    md_path = VIS_DIR / "model_process_schemes.md"
    lines = ["# Model Benchmark Process Schemes", ""]
    for row in rows:
        lines.append(f"## U_{row['upstream_bundle']} / C_{row['council_bundle']}")
        lines.append("")
        lines.append(row["process_scheme_text"])
        lines.append("")
    md_path.write_text("\n".join(lines), encoding="utf-8")

    print(f"Wrote {csv_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
