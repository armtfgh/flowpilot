"""Thermal Knoevenagel case study — ENGINE Council v4.

Protocol: Knoevenagel condensation of benzaldehyde with ethyl cyanoacetate
in EtOH at 80 °C (above bp 78 °C → BPR required), 8 h batch, 0.5 M.
Exothermic; base-catalysed.  Designed so the naive LLM proposal fails on
BPR (sets 0 bar) while the council correctly adds it, and cuts τ via IF=10×.

Saves: outputs/thermal_council_run.json
  Includes:
    result["pre_council_calculations"]  — 9-step calc on the raw LLM proposal
    result["post_council_calculations"] — 9-step calc on the council winner
    (post_council_calculations == design_calculations, kept for naming symmetry)
"""

import json
import logging
from dataclasses import asdict
from pathlib import Path

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")

PROTOCOL = """
Thermal Knoevenagel condensation of benzaldehyde (0.5 M) with ethyl
cyanoacetate (0.55 M, 1.1 equiv) using piperidine base-catalyst (0.05 M,
10 mol%) in ethanol at 80 °C.  Mildly exothermic aldol-type C-C bond
formation.  Batch reaction time: 8 h at reflux, isolated yield: 92 %.
No photocatalyst.  Product: ethyl (E)-2-cyano-3-phenylacrylate.
"""

OUTPUT_PATH = Path("outputs/thermal_council_run.json")


def _safe(obj):
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, bytes):
        return obj.decode("utf-8", errors="replace")
    return str(obj)


def _run_calc_on_proposal(proposal_dict: dict,
                           batch_record,
                           chemistry_plan,
                           inventory) -> dict:
    """Run the 9-step DesignCalculator on an arbitrary FlowProposal dict."""
    from flora_translate.design_calculator import DesignCalculator
    from flora_translate.schemas import FlowProposal

    fields = FlowProposal.model_fields.keys()
    proposal = FlowProposal(**{k: v for k, v in proposal_dict.items() if k in fields})
    calc = DesignCalculator().run(batch_record, chemistry_plan, proposal, inventory)
    return asdict(calc)


def main():
    from flora_translate.main import translate
    from flora_translate.config import LAB_INVENTORY_PATH
    from flora_translate.schemas import BatchRecord, ChemistryPlan, LabInventory

    print("Running FLORA-Translate — thermal Knoevenagel case study …")
    result = translate(PROTOCOL)

    inventory = LabInventory.from_json(str(LAB_INVENTORY_PATH))

    # Reconstruct BatchRecord from parsed result (reliable fields only)
    cp_raw = result.get("chemistry_plan", {})
    try:
        cp_fields = {k: v for k, v in cp_raw.items() if k in ChemistryPlan.model_fields}
        chem_plan = ChemistryPlan(**cp_fields)
    except Exception:
        chem_plan = None

    batch_rec = BatchRecord(
        reaction_description=PROTOCOL.strip(),
        batch_time_h=8.0,
        temperature_C=80.0,
        solvent="EtOH",
        concentration_M=0.5,
        yield_percent=92.0,
    )

    # ── Pre-council metrics ───────────────────────────────────────────────────
    print("Computing 9-step metrics for PRE-council proposal …")
    try:
        pre_calc = _run_calc_on_proposal(
            result["pre_council_proposal"], batch_rec, chem_plan, inventory
        )
        result["pre_council_calculations"] = pre_calc
        print(f"  τ={pre_calc['residence_time_min']:.1f} min | "
              f"d={pre_calc['tubing_ID_mm']} mm | "
              f"Re={pre_calc['reynolds_number']:.1f} | "
              f"BPR_required={pre_calc['bpr_required']} ({pre_calc['bpr_pressure_bar']} bar) | "
              f"Da_th={pre_calc['thermal_damkohler']} | "
              f"S/V={pre_calc['surface_to_volume']:.0f}")
    except Exception as e:
        print(f"  [warn] pre-council calc failed: {e}")
        result["pre_council_calculations"] = {}

    # ── Post-council metrics (already in result, just rename for symmetry) ────
    result["post_council_calculations"] = result.get("design_calculations", {})
    dc = result["post_council_calculations"]
    p  = result.get("proposal", {})
    print(f"POST-council: τ={p.get('residence_time_min')} min | "
          f"d={p.get('tubing_ID_mm')} mm | "
          f"Re={dc.get('reynolds_number', '?'):.1f} | "
          f"BPR_required={dc.get('bpr_required')} ({dc.get('bpr_pressure_bar')} bar) | "
          f"Da_th={dc.get('thermal_damkohler')} | "
          f"S/V={dc.get('surface_to_volume', 0):.0f}")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(result, f, indent=2, default=_safe)

    print(f"\nSaved → {OUTPUT_PATH}")
    print(f"Confidence: {result.get('confidence')}")

    # ── Quick delta summary ───────────────────────────────────────────────────
    pre = result.get("pre_council_calculations", {})
    post = result.get("post_council_calculations", {})
    if pre and post:
        metrics = [
            ("τ (min)",       "residence_time_min",  None, "lower"),
            ("d (mm)",        "tubing_ID_mm",        None, "lower"),
            ("Re",            "reynolds_number",     2300, "lower"),
            ("BPR (bar)",     "bpr_pressure_bar",    None, "higher"),
            ("bpr_required",  "bpr_required",        None, None),
            ("Da_thermal",    "thermal_damkohler",   1.0,  "lower"),
            ("S/V (m⁻¹)",     "surface_to_volume",   None, "higher"),
            ("Productivity",  "productivity_mmol_h", None, "higher"),
        ]
        print("\nDelta summary:")
        for label, key, thresh, better in metrics:
            pv = pre.get(key, "?")
            cv = post.get(key, "?")
            if pv != "?" and cv != "?":
                if isinstance(pv, float):
                    print(f"  {label:20s}: {pv:8.3f} → {cv:8.3f}")
                else:
                    print(f"  {label:20s}: {pv!s:8s} → {cv!s:8s}")


if __name__ == "__main__":
    main()
