"""SNAr ablation: 1-candidate vs 12-candidate council comparison.

The pre-council proposal (τ=90 min, d=1.6 mm, BPR=10 bar) is passed to the
council as the ONLY candidate — Designer (Stage 1) is bypassed.
Stages 2–4 (scoring → skeptic → chief) run identically to the 12-cand run.

Scientific question: does the council's value come from breadth of exploration
(12 candidates) or from targeted correction of a single design?

Saves: outputs/snar_1cand_run.json
"""

import json
import logging
from dataclasses import asdict
from pathlib import Path

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")

OUTPUT_PATH  = Path("outputs/snar_1cand_run.json")
EXISTING_RUN = Path("outputs/snar_council_run.json")

PROTOCOL = """
Nucleophilic aromatic substitution (SNAr) of 4-fluoronitrobenzene (0.3 M)
with piperazine (1.2 equiv) in N,N-dimethylformamide at 120 °C.
No catalyst required. Batch reaction time 6 h, isolated yield 81%.
No special atmosphere control.
"""


def main():
    from flora_translate.config import LAB_INVENTORY_PATH
    from flora_translate.design_calculator import DesignCalculator
    from flora_translate.engine.sampling import compute_metrics
    from flora_translate.engine.council_v4 import CouncilV4
    from flora_translate.schemas import (
        BatchRecord, ChemistryPlan, FlowProposal, LabInventory,
    )

    # ── Load pre-council proposal from the 12-cand run ────────────────────────
    with open(EXISTING_RUN) as f:
        existing = json.load(f)

    fp_dict  = existing["pre_council_proposal"]
    proposal = FlowProposal(**{k: v for k, v in fp_dict.items()
                               if k in FlowProposal.model_fields})

    cp_raw = existing.get("chemistry_plan", {})
    try:
        cp_fields = {k: v for k, v in cp_raw.items()
                     if k in ChemistryPlan.model_fields}
        chem_plan = ChemistryPlan(**cp_fields)
    except Exception:
        chem_plan = None

    batch_rec = BatchRecord(
        reaction_description=PROTOCOL.strip(),
        batch_time_h=6.0, temperature_C=120.0, solvent="DMF",
        concentration_M=0.3, yield_pct=81.0,
    )
    inventory = LabInventory.from_json(str(LAB_INVENTORY_PATH))

    # ── Run DesignCalculator to get physics metrics of the proposal ───────────
    calc = DesignCalculator().run(batch_rec, chem_plan, proposal, inventory)

    tau_kinetics_min = (getattr(calc, "tau_kinetics_min", None)
                        or calc.residence_time_min or 90.0)
    IF_used   = calc.intensification_factor or 10.0
    pump_max  = calc.pump_max_bar or 20.0
    assumed_MW = 256.3   # N-aryl piperazine

    # ── Construct the single candidate dict ───────────────────────────────────
    single = compute_metrics(
        tau_min          = proposal.residence_time_min,
        d_mm             = proposal.tubing_ID_mm,
        Q_mL_min         = proposal.flow_rate_mL_min,
        solvent          = "DMF",
        temperature_C    = proposal.temperature_C,
        concentration_M  = proposal.concentration_M,
        assumed_MW       = assumed_MW,
        IF_used          = IF_used,
        tau_kinetics_min = tau_kinetics_min,
        pump_max_bar     = pump_max,
        is_photochem     = False,
        tau_source       = "pre_council_proposal",
    )
    single["id"]               = 1
    single["pareto_front"]     = True
    single["feasible"]         = True
    single["hard_gate_flags"]  = []
    single["hard_gate_status"] = "PASS"

    print(f"\nSingle candidate: τ={single['tau_min']} min | "
          f"d={single['d_mm']} mm | Q={single['Q_mL_min']:.4f} mL/min | "
          f"Re={single['Re']:.1f} | r_mix={single['r_mix']:.3f} | "
          f"X={single['expected_conversion']:.2f}")

    # ── Run council with Designer bypassed ────────────────────────────────────
    print("\nRunning council on 1 candidate (Stage 1 bypassed) …")
    design_candidate, final_calc = CouncilV4().run(
        proposal         = proposal,
        batch_record     = batch_rec,
        analogies        = [],
        inventory        = inventory,
        chemistry_plan   = chem_plan,
        calculations     = calc,
        objectives       = "balanced",
        fixed_candidates = [single],     # ← bypass Designer
    )

    cp      = design_candidate.proposal
    cc      = asdict(final_calc)
    fp_calc = asdict(calc)

    # ── Three-way comparison ──────────────────────────────────────────────────
    twelve_cp   = existing.get("proposal", {})
    twelve_cc   = existing.get("post_council_calculations", {})

    def fmt(v):
        try:    return f"{float(v):>10.3f}"
        except: return f"{str(v):>10}"

    print("\n" + "="*68)
    print("SNAr: PRE-COUNCIL  →  1-CAND COUNCIL  vs  12-CAND COUNCIL")
    print("="*68)
    print(f"  {'Parameter':22s} {'Pre-council':>12} {'1-cand':>12} {'12-cand':>12}")
    print("  " + "-"*60)

    rows = [
        ("τ (min)",      fp_calc["residence_time_min"],
                         getattr(cp,"residence_time_min",None),
                         twelve_cc.get("residence_time_min")),
        ("d (mm)",       fp_calc["tubing_ID_mm"],
                         getattr(cp,"tubing_ID_mm",None),
                         twelve_cc.get("tubing_ID_mm")),
        ("Q (mL/min)",   fp_calc["flow_rate_mL_min"],
                         getattr(cp,"flow_rate_mL_min",None),
                         twelve_cc.get("flow_rate_mL_min")),
        ("BPR (bar)",    proposal.BPR_bar,
                         getattr(cp,"BPR_bar",None),
                         twelve_cp.get("BPR_bar")),
        ("Re",           fp_calc["reynolds_number"],
                         cc.get("reynolds_number"),
                         twelve_cc.get("reynolds_number")),
        ("S/V (m⁻¹)",    fp_calc["surface_to_volume"],
                         cc.get("surface_to_volume"),
                         twelve_cc.get("surface_to_volume")),
        ("tubing mat.",  proposal.tubing_material,
                         getattr(cp,"tubing_material",None),
                         twelve_cp.get("tubing_material")),
    ]
    for label, pre, one, tw in rows:
        print(f"  {label:22s} {fmt(pre)} {fmt(one)} {fmt(tw)}")

    # ── Save ──────────────────────────────────────────────────────────────────
    output = {
        "mode":                    "1-candidate (Designer bypassed)",
        "pre_council_proposal":    fp_dict,
        "pre_council_calculations":fp_calc,
        "one_cand_proposal":       cp.model_dump(),
        "one_cand_calculations":   cc,
        "twelve_cand_proposal":    twelve_cp,
        "twelve_cand_calculations":twelve_cc,
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nSaved -> {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
