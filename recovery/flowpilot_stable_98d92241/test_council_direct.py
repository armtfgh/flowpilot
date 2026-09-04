"""Direct ENGINE Council v4 stress-test with a deliberately flawed proposal.

Bypasses the full FLORA pipeline (no RAG, no chemistry agent, no LLM parsing).
We craft a FlowProposal that contains one intentional flaw per domain so that
every specialist has something real to catch.

Intentional flaws
─────────────────
  [Dr. Chemistry  ] Beer-Lambert inner-filter: d = 2.0 mm, ε = 15 000 M⁻¹cm⁻¹,
                    C = 0.1 M  → A ≫ 2  (reactor opaque, photons don't reach core)
  [Dr. Kinetics   ] τ = 4 min  far below τ_kinetics ≈ 60 min for this reaction class
                    (batch 8 h, IF = 8 × → τ_flow ≈ 60 min; 4 min gives X < 10 %)
  [Dr. Fluidics   ] d = 2.0 mm + Q = 2.0 mL/min  → Re ≈ 3 400  (turbulent, not laminar)
  [Dr. Safety     ] BPR = 0 bar while T = 85 °C in MeCN (bp 82 °C, P_vap ≈ 1.1 bar)
                    → solvent will vaporise in the reactor

Reaction: Ir(ppy)3 photoredox C–H functionalisation in MeCN, 85 °C,
          0.1 M, 450 nm, N2, 8 h batch, 74 % yield.
"""

import json
import logging
from dataclasses import asdict
from pathlib import Path

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")

OUTPUT_PATH = Path("outputs/council_stress_test.json")


def main():
    from flora_translate.config import LAB_INVENTORY_PATH
    from flora_translate.design_calculator import DesignCalculator
    from flora_translate.engine.council_v4 import CouncilV4
    from flora_translate.schemas import (
        BatchRecord, ChemistryPlan, FlowProposal,
        LabInventory, StreamAssignment,
    )

    # ── 1. Batch record ───────────────────────────────────────────────────────
    batch = BatchRecord(
        reaction_description=(
            "Ir(ppy)3-catalyzed visible-light photoredox C-H functionalisation "
            "of N-aryl tetrahydroisoquinoline in MeCN at 85 °C, 0.1 M, 450 nm "
            "blue LED, N2 atmosphere, 8 h batch time, 74 % yield."
        ),
        photocatalyst="Ir(ppy)3",
        catalyst_loading_mol_pct=1.0,
        solvent="MeCN",
        temperature_C=85.0,
        reaction_time_h=8.0,
        concentration_M=0.1,
        yield_pct=74.0,
        light_source="450 nm blue LED",
        wavelength_nm=450.0,
        atmosphere="N2",
    )

    # ── 2. Chemistry plan ─────────────────────────────────────────────────────
    chem_plan = ChemistryPlan(
        reaction_name="Ir(ppy)3 photoredox C-H functionalisation",
        reaction_class="photoredox",
        mechanism_type="oxidative quenching SET",
        oxygen_sensitive=True,
        deoxygenation_required=True,
    )

    # ── 3. Deliberately flawed FlowProposal ───────────────────────────────────
    #   ❌ d = 2.0 mm  → Re turbulent + Beer-Lambert failure
    #   ❌ τ = 4 min   → far too short for kinetics
    #   ❌ BPR = 0 bar → MeCN at 85 °C will boil (bp 82 °C)
    #   ❌ Q = 2.0 mL/min  → drives Re into turbulent regime
    flawed = FlowProposal(
        residence_time_min=4.0,           # ❌ kinetics: way too short
        flow_rate_mL_min=2.0,             # ❌ fluidics: drives Re turbulent
        temperature_C=85.0,
        concentration_M=0.1,
        BPR_bar=0.0,                      # ❌ safety: MeCN bp 82 °C, needs BPR
        reactor_type="coil",
        tubing_material="FEP",
        tubing_ID_mm=2.0,                 # ❌ chemistry: Beer-Lambert + turbulence
        reactor_volume_mL=8.0,            # τ × Q = 4 × 2 = 8 mL (self-consistent)
        wavelength_nm=450.0,
        deoxygenation_method=None,        # ❌ safety: O2-sensitive reaction, no deox
        streams=[
            StreamAssignment(
                stream_label="A",
                pump_role="substrate + Ir(ppy)3 photocatalyst",
                contents=["N-aryl THIQ (0.1 M)", "Ir(ppy)3 (1 mol%)"],
                solvent="MeCN",
                concentration_M=0.1,
                flow_rate_mL_min=2.0,
                reasoning="All reagents combined in one stream.",
            ),
        ],
        mixer_type="T-mixer",
        chemistry_notes="Photoredox C-H functionalisation via single-electron oxidation.",
        engine_validated=False,
        confidence="LOW",
    )

    # ── 4. Run design calculator on the flawed proposal ───────────────────────
    inventory = LabInventory.from_json(str(LAB_INVENTORY_PATH))
    print("Running 9-step calculator on flawed proposal …")
    calc = DesignCalculator().run(batch, chem_plan, flawed, inventory)
    print(f"  τ={calc.residence_time_min} min | d={calc.tubing_ID_mm} mm | "
          f"Re={calc.reynolds_number:.0f} | BPR_req={calc.bpr_required} "
          f"({calc.bpr_pressure_bar} bar) | "
          f"Da_mass={calc.damkohler_mass:.2f} | S/V={calc.surface_to_volume:.0f}")

    # ── 5. Pass flawed proposal to council ────────────────────────────────────
    print("\nRunning ENGINE Council v4 on flawed proposal …")
    analogies = [
        {
            "source": "synthetic_analogy",
            "residence_time_min": 60.0,
            "flow_rate_mL_min": 0.1,
            "tubing_ID_mm": 0.75,
            "batch_time_h": 8.0,
            "intensification_factor": 8.0,
            "reaction_class": "photoredox",
            "solvent": "MeCN",
            "temperature_C": 85.0,
            "wavelength_nm": 450.0,
            "description": "Literature Ir(ppy)3 photoredox C-H activation "
                           "in MeCN, 60 min residence time, 0.75 mm PFA tubing.",
        }
    ]

    design_candidate, final_calc = CouncilV4().run(
        proposal=flawed,
        batch_record=batch,
        analogies=analogies,
        inventory=inventory,
        chemistry_plan=chem_plan,
        calculations=calc,
        objectives="de-risk first-run",
    )

    # ── 6. Save results ───────────────────────────────────────────────────────
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    output = {
        "flawed_proposal":    flawed.model_dump(),
        "flawed_calculations": asdict(calc),
        "council_proposal":   design_candidate.proposal.model_dump(),
        "council_calculations": asdict(final_calc),
        "deliberation_log":   design_candidate.deliberation_log.model_dump()
                              if design_candidate.deliberation_log else {},
    }

    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2, default=str)

    # ── 7. Print delta summary ────────────────────────────────────────────────
    cp = design_candidate.proposal
    print("\n" + "="*60)
    print("FLAWED  →  COUNCIL")
    print("="*60)
    metrics = [
        ("τ (min)",      flawed.residence_time_min,   cp.residence_time_min),
        ("d (mm)",       flawed.tubing_ID_mm,         cp.tubing_ID_mm),
        ("Q (mL/min)",   flawed.flow_rate_mL_min,     cp.flow_rate_mL_min),
        ("BPR (bar)",    flawed.BPR_bar,              cp.BPR_bar),
        ("Re",           calc.reynolds_number,        final_calc.reynolds_number),
        ("BPR required", calc.bpr_required,           final_calc.bpr_required),
        ("S/V (m⁻¹)",    calc.surface_to_volume,      final_calc.surface_to_volume),
        ("Deox method",  flawed.deoxygenation_method, cp.deoxygenation_method),
    ]
    for label, before, after in metrics:
        print(f"  {label:20s}: {str(before):>12}  →  {after}")

    dlog = design_candidate.deliberation_log
    if dlog:
        print(f"\nConsensus reached: {dlog.consensus_reached}")
        print(f"Changes applied:   {list(dlog.all_changes_applied.keys())}")

    print(f"\nSaved → {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
