"""SNAr case study — ENGINE Council v4.

Protocol: Nucleophilic aromatic substitution of 4-fluoronitrobenzene with
piperazine in DMF at 120 °C (well below DMF bp 153 °C — BPR not strictly
required but council should still assess).  No photocatalyst.  6 h batch.

Saves: outputs/snar_council_run.json
"""

import json
import logging
from dataclasses import asdict
from pathlib import Path

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")

PROTOCOL = """
Nucleophilic aromatic substitution (SNAr) of 4-fluoronitrobenzene (0.3 M)
with piperazine (1.2 equiv) in N,N-dimethylformamide at 120 °C.
No catalyst required. The reaction was run under air for 6 hours and
gave the N-aryl piperazine product in 81% isolated yield after extraction.
Reaction was performed at 1.0 mmol scale with continuous magnetic stirring.
No special atmosphere control was used.
"""

OUTPUT_PATH = Path("outputs/snar_council_run.json")


def _safe(obj):
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, bytes):
        return obj.decode("utf-8", errors="replace")
    return str(obj)


def _run_calc_on_proposal(proposal_dict, batch_record, chemistry_plan, inventory):
    from flora_translate.design_calculator import DesignCalculator
    from flora_translate.schemas import FlowProposal
    fields   = FlowProposal.model_fields.keys()
    proposal = FlowProposal(**{k: v for k, v in proposal_dict.items() if k in fields})
    calc     = DesignCalculator().run(batch_record, chemistry_plan, proposal, inventory)
    return asdict(calc)


def main():
    from flora_translate.main import translate
    from flora_translate.config import LAB_INVENTORY_PATH
    from flora_translate.schemas import BatchRecord, ChemistryPlan, LabInventory

    print("Running FLORA-Translate — SNAr case study …")
    result = translate(PROTOCOL)

    inventory = LabInventory.from_json(str(LAB_INVENTORY_PATH))

    cp_raw = result.get("chemistry_plan", {})
    try:
        cp_fields = {k: v for k, v in cp_raw.items() if k in ChemistryPlan.model_fields}
        chem_plan = ChemistryPlan(**cp_fields)
    except Exception:
        chem_plan = None

    batch_rec = BatchRecord(
        reaction_description=PROTOCOL.strip(),
        batch_time_h=6.0,
        temperature_C=120.0,
        solvent="DMF",
        concentration_M=0.3,
        yield_pct=81.0,
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

    # ── Post-council metrics ──────────────────────────────────────────────────
    result["post_council_calculations"] = result.get("design_calculations", {})
    dc = result["post_council_calculations"]
    p  = result.get("proposal", {})
    print(f"POST-council: τ={p.get('residence_time_min')} min | "
          f"d={p.get('tubing_ID_mm')} mm | "
          f"Re={dc.get('reynolds_number', '?')} | "
          f"BPR_required={dc.get('bpr_required')} ({dc.get('bpr_pressure_bar')} bar) | "
          f"Da_th={dc.get('thermal_damkohler')} | "
          f"S/V={dc.get('surface_to_volume', 0):.0f}")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(result, f, indent=2, default=_safe)

    print(f"\nSaved → {OUTPUT_PATH}")
    print(f"Confidence: {result.get('confidence')}")

    # ── Delta summary ─────────────────────────────────────────────────────────
    pre  = result.get("pre_council_calculations", {})
    post = result.get("post_council_calculations", {})
    if pre and post:
        metrics = [
            ("τ (min)",      "residence_time_min"),
            ("d (mm)",       "tubing_ID_mm"),
            ("Re",           "reynolds_number"),
            ("BPR (bar)",    "bpr_pressure_bar"),
            ("bpr_required", "bpr_required"),
            ("Da_thermal",   "thermal_damkohler"),
            ("S/V (m⁻¹)",    "surface_to_volume"),
            ("Productivity", "productivity_mmol_h"),
        ]
        print("\nPRE-council → POST-council:")
        for label, key in metrics:
            pv = pre.get(key, "?")
            cv = post.get(key, "?")
            if isinstance(pv, float) and isinstance(cv, float):
                print(f"  {label:20s}: {pv:8.3f} → {cv:8.3f}")
            else:
                print(f"  {label:20s}: {pv!s:8} → {cv!s}")


if __name__ == "__main__":
    main()
