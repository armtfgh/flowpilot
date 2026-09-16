"""Thermal ablation: 1-candidate vs 12-candidate council comparison.

The pre-council proposal from outputs/thermal_council_run.json is passed to the
council as the ONLY candidate — Designer (Stage 1) is bypassed.
Stages 2–4 run identically to the normal 12-candidate council.

Saves: outputs/thermal_1cand_run.json
"""

import json
import logging
from dataclasses import asdict
from pathlib import Path

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")

OUTPUT_PATH = Path("outputs/thermal_1cand_run.json")
EXISTING_RUN = Path("outputs/thermal_council_run.json")

PROTOCOL = """
Thermal Knoevenagel condensation of benzaldehyde (0.5 M) with ethyl
cyanoacetate (0.55 M, 1.1 equiv) using piperidine base-catalyst (0.05 M,
10 mol%) in ethanol at 80 °C. Mildly exothermic aldol-type C-C bond
formation. Batch reaction time: 8 h at reflux, isolated yield: 92 %.
No photocatalyst. Product: ethyl (E)-2-cyano-3-phenylacrylate.
"""


def main():
    from flora_translate.config import LAB_INVENTORY_PATH
    from flora_translate.design_calculator import DesignCalculator
    from flora_translate.engine.council_v4 import CouncilV4
    from flora_translate.engine.sampling import compute_metrics
    from flora_translate.schemas import (
        BatchRecord, ChemistryPlan, FlowProposal, LabInventory,
    )

    with open(EXISTING_RUN) as f:
        existing = json.load(f)

    fp_dict = existing["pre_council_proposal"]
    proposal = FlowProposal(**{
        k: v for k, v in fp_dict.items() if k in FlowProposal.model_fields
    })

    cp_raw = existing.get("chemistry_plan", {})
    try:
        cp_fields = {
            k: v for k, v in cp_raw.items() if k in ChemistryPlan.model_fields
        }
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
    inventory = LabInventory.from_json(str(LAB_INVENTORY_PATH))

    calc = DesignCalculator().run(batch_rec, chem_plan, proposal, inventory)

    tau_kinetics_min = (
        getattr(calc, "tau_kinetics_min", None)
        or calc.residence_time_min
        or proposal.residence_time_min
        or 90.0
    )
    IF_used = calc.intensification_factor or 10.0
    pump_max = calc.pump_max_bar or 20.0
    assumed_MW = 203.2

    single = compute_metrics(
        tau_min=proposal.residence_time_min,
        d_mm=proposal.tubing_ID_mm,
        Q_mL_min=proposal.flow_rate_mL_min,
        solvent="EtOH",
        temperature_C=proposal.temperature_C,
        concentration_M=proposal.concentration_M,
        assumed_MW=assumed_MW,
        IF_used=IF_used,
        tau_kinetics_min=tau_kinetics_min,
        pump_max_bar=pump_max,
        is_photochem=False,
        tau_source="pre_council_proposal",
    )
    single["id"] = 1
    single["pareto_front"] = True
    single["feasible"] = True
    single["hard_gate_flags"] = []
    single["hard_gate_status"] = "PASS"
    single["BPR_bar"] = proposal.BPR_bar or 0.0
    single["tubing_material"] = proposal.tubing_material
    single["concentration_M"] = proposal.concentration_M
    single["temperature_C"] = proposal.temperature_C

    print(
        f"\nSingle candidate: τ={single['tau_min']} min | "
        f"d={single['d_mm']} mm | Q={single['Q_mL_min']:.4f} mL/min | "
        f"Re={single['Re']:.1f} | r_mix={single['r_mix']:.3f} | "
        f"X={single['expected_conversion']:.2f}"
    )

    print("\nRunning council on 1 candidate (Stage 1 bypassed) …")
    design_candidate, final_calc = CouncilV4().run(
        proposal=proposal,
        batch_record=batch_rec,
        analogies=[],
        inventory=inventory,
        chemistry_plan=chem_plan,
        calculations=calc,
        objectives="balanced",
        fixed_candidates=[single],
    )

    cp = design_candidate.proposal
    cc = asdict(final_calc)
    fp_calc = asdict(calc)
    twelve_cp = existing.get("proposal", {})
    twelve_cc = existing.get("post_council_calculations", {})

    output = {
        "mode": "1-candidate (Designer bypassed)",
        "pre_council_proposal": fp_dict,
        "pre_council_calculations": fp_calc,
        "one_cand_proposal": cp.model_dump(),
        "one_cand_calculations": cc,
        "twelve_cand_proposal": twelve_cp,
        "twelve_cand_calculations": twelve_cc,
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nSaved -> {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
