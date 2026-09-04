from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmark.run_model_matrix_benchmark import council_bundle, upstream_bundle
from flora_translate.main import translate


PROMPT_PHOTOREDOX_AEROBIC = """REACTION NAME: One-pot photoredox Giese addition + aerobic oxidation TARGET MOLECULE: Sulfoxide 4a [α-(4-methoxyphenylthio)propionitrile sulfoxide] REAGENTS Substrate / radical precursor: PMPSCH₂TMS (1a) Amount: 0.2 mmol (1.0 equiv.) Oxidation potential: Eox = +0.98 V vs. SCE Michael acceptor: Acrylonitrile (2a) Amount: 2.0 equiv. Photocatalyst: Ir(dF(CF₃)ppy)₂(dtbpy)PF₆ Loading: 0.5 mol% Excited state redox: Ir(III)/Ir(II) = +1.21 V vs. SCE Ground state reductive: Ir(III)/Ir(II) = −1.37 V vs. SCE Solvent: EtOH / pH 9 aqueous buffer (5:1 v/v) Concentration: 0.1 M Oxidant: Molecular oxygen (O₂) from air — introduced only in Step 2 PROCEDURE (TWO SEQUENTIAL STEPS, ONE POT) STEP 1 — C–C Bond Formation (Giese Radical Addition): Atmosphere: Argon (strictly O₂-free) Light source: 10 W blue LED, 452 nm Temperature: 25°C (room temperature) Reaction time: 4 hours Vessel: Sealed pressure tube Outcome: Sulfide 3a formed, 92% yield Mechanism: Ir(III) oxidizes 1a via SET → cationic radical 1a•+ → desilylation → α-thiomethyl radical I → Giese addition onto acrylonitrile 2a → radical II → reduced by Ir(II) → carbanion III → protonated by solvent → sulfide 3a STEP 2 — Aerobic Oxidation (Sulfide → Sulfoxide): Atmosphere: Air (O₂ introduced after Step 1 complete) Light source: 10 W blue LED, 452 nm (same source, kept on) Temperature: 25°C Reaction time: 6 hours Vessel: Same tube, opened to air Outcome: Sulfoxide 4a formed, 95% yield Mechanism: Two parallel pathways — (a) SET: Ir(III)* oxidizes sulfide 3a + O₂ → superoxide → sulfoxide (b) EnT: Ir(III)* converts ³O₂ → ¹O₂ (singlet oxygen) → sulfoxide Sulfide oxidation potential: Eox(3a) = +1.13 V vs. SCE SENSITIVITY FLAGS O₂ during Step 1: CRITICAL — O₂ quenches α-thiomethyl radical I → reaction fails completely Light: ESSENTIAL — no conversion without LED Photocatalyst: ESSENTIAL — no conversion without [Ir] pH control: IMPORTANT — pH 9 buffer suppresses byproduct PMPSCH₃ (1a-I) via protonation of radical intermediate Moisture: TOLERANT — aqueous buffer intentional Scale in batch: PROBLEMATIC — uneven irradiation and poor O₂ mass transfer at >0.2 mmol TOTAL BATCH TIME: 10 hours (4h + 6h) TOTAL BATCH YIELD: 95% isolated (sulfoxide 4a) SCALE: 0.2 mmol"""


PROMPT_BROMINATION_QUENCH = """REACTION NAME: α-Bromination of 3-phenylpropanal. SUBSTRATE: 3-phenylpropanal (1a). TARGET PRODUCT: α-bromophenylpropanal (2a). REACTION CLASS: Electrophilic α-bromination (thermal). SCALE: 10.0 mmol. SCOPE CONSTRAINTS: This is a single-reaction protocol. Only one chemical transformation occurs: 1a + Br2 -> 2a + HBr. There is no second reaction. Do not add thiourea. Do not add a cyclization step. Stop at 2a. The quench Na2S2O3 is not a reaction. It is a safety neutralization step that consumes excess Br2. Assign it as an inline quench stream only, not as a second reactor stage. Workup extraction, brine wash, drying, and column are batch-only offline steps. REAGENTS: 3-phenylpropanal 1.34 g, 10.0 mmol, 1.0 equiv. Br2 1.598 g, 10.0 mmol, 1.0 equiv. MeCN 15 mL reaction solvent. Substrate concentration 0.667 M. QUENCH: Na2S2O3 aqueous solution 10% w/v, 15 mL total. BATCH CONDITIONS: Temperature 25 °C. Reaction time 30 min. Atmosphere N2 throughout because aldehyde is O2-sensitive. Br2 addition dropwise to substrate solution. EXPECTED RESULTS: Yield 83%, conversion 94%, mono/di selectivity 18:1. FLOW TRANSLATION: FLOW STAGE 1 reaction: Stream A 3-phenylpropanal in MeCN 0.667 M. Stream B Br2 in MeCN 0.667 M. Mixing T-mixer equal flow rates 1:1. Reactor coil at 25 °C. Residence time derived from 30 min batch and reaction class. FLOW STAGE 2 inline quench not reaction: Stream C Na2S2O3 aqueous 10% w/v. Role neutralize residual Br2 at reactor outlet. Mixer second T-mixer at reactor outlet. Purpose safety only, no new chemistry. Everything after Stage 2 is offline workup."""


PROMPT_SYNTHETIC_STRESS = """REACTION NAME: Synthetic three-stage gas-liquid exothermic photothermal cascade. TARGET: nitro-ketone intermediate Z. SCALE: 5.0 mmol. This is a deliberately complex validation protocol. Stage 1: exothermic acyl chloride aminolysis. Stream A: amine substrate S in dry MeCN, 0.50 M, 1.0 equiv. Stream B: acid chloride RCOCl in MeCN, 0.50 M, 1.05 equiv. Addition is strongly exothermic; batch uses ice bath and slow addition over 45 min at 0 °C, then 15 min hold. HCl is generated and must be trapped by inline base. Stage 2: gas-liquid hydrogenation of an alkene intermediate. After Stage 1 outlet, introduce H2 gas only at Stage 2 through an MFC. Pd/C packed bed or static-mixer coil acceptable. Batch hydrogenation uses 1 atm H2 balloon for 2 h at 25 °C. H2 mass transfer is critical. Stage 3: aerobic photoredox oxidation. After Stage 2, introduce air only at Stage 3 through a second MFC, blue LED 455 nm, 25 °C, 1 h batch time. Oxygen must not contact Stage 1 acid chloride feed or Stage 2 H2 zone. Heat transfer is important in Stage 1 because aminolysis is exothermic; gas-liquid mass transfer is important in Stages 2 and 3. Required process: Stage 1 liquid-liquid reaction coil with strong heat removal; Stage 2 H2 gas-liquid segment or packed-bed hydrogenation; Stage 3 air/O2 photochemical oxidation. Offline workup only after final outlet. Total isolated yield in batch 72%."""

PROMPT_CO_CARBONYLATION_HEAT = """REACTION NAME: Synthetic aminocarbonylation of aryl bromide with carbon monoxide. TARGET: amide C1. SCALE: 3.0 mmol. This validation protocol tests toxic gas handling, BPR enforcement, and heat-transfer accounting. Reagents: aryl bromide AB-1, 0.30 M in dry 1,4-dioxane, 1.0 equiv; amine nucleophile N-1, 0.60 M in dioxane, 2.0 equiv; Pd catalyst and base premixed with the aryl bromide feed. Gas reagent: carbon monoxide, introduced only as a dedicated CO gas feed through an MFC at the reactor inlet. Batch reference: sealed autoclave, 8 bar CO, 90 °C, 6 h, 82% yield. Reaction is mildly exothermic and gas-liquid mass transfer limits scale-up. Flow translation requirements: use separate liquid pumps for aryl bromide/catalyst/base and amine; introduce CO by MFC, not pump; use heated coil or packed tubular reactor, BPR mandatory; downstream gas-liquid separator and offline workup only. Do not add a second chemistry stage."""

PROMPT_CO2_CARBOXYLATION_QUENCH = """REACTION NAME: Synthetic organometallic carboxylation with carbon dioxide gas. TARGET: carboxylic acid D1. SCALE: 2.5 mmol. Scope constraints: single chemical transformation followed by quench only. Stream A: organozinc reagent in dry THF, 0.25 M, oxygen/moisture-sensitive. Stream B: electrophile solution in THF, 0.25 M. Gas reagent: dry CO2 introduced through an MFC after streams A and B mix; CO2 is a reagent, not an inert blanket. Batch reference: -10 °C, CO2 balloon, 2 h, then acidic aqueous quench, 76% isolated yield. Heat transfer matters because quench is exothermic, but the quench is not a reactor stage. Flow translation: Stage 1 gas-liquid carboxylation coil at controlled temperature with BPR; Stage 2 inline acid quench mixer only, no residence-time reactor; offline extraction after final outlet."""

PROMPT_OZONE_OXIDATION_COLD = """REACTION NAME: Synthetic cold ozonolysis of electron-rich alkene. TARGET: aldehyde O1. SCALE: 1.0 mmol. Stream A: alkene substrate in dry MeOH/DCM 1:1, 0.10 M. Gas reagent: ozone in oxygen, introduced only through an MFC into a cold gas-liquid coil. Batch reference: -40 °C, ozone bubbled until endpoint, 25 min, then dimethyl sulfide reductive workup, 88% yield. Ozone transfer, gas holdup, and heat removal are critical. Safety constraints: no ozone accumulation, small holdup, BPR or controlled gas outlet required, off-gas scrubber after reactor. Flow translation: one cold gas-liquid ozonolysis reactor; reductive quench as inline mixer only if represented, not a second reaction coil; offline workup after quench."""

PROMPT_SYNGAS_HYDROFORMYLATION = """REACTION NAME: Synthetic hydroformylation of terminal alkene under syngas. TARGET: branched aldehyde H1. SCALE: 4.0 mmol. Stream A: terminal alkene in toluene, 0.40 M, with Rh catalyst and phosphine ligand. Gas reagent: syngas CO/H2 = 1:1 introduced by MFC at the reactor inlet. Batch reference: 10 bar syngas, 80 °C, 4 h, 70% yield, branched/linear ratio 8:1. Gas-liquid mass transfer, pressure control, and heat transfer matter. Flow translation: single heated gas-liquid coil or packed static-mixer reactor; BPR mandatory; no oxygen or air anywhere; gas-liquid separator after BPR; offline purification only."""

PROMPT_N2_EXOTHERMIC_CONTROL = """REACTION NAME: Synthetic exothermic sulfonyl chloride substitution under inert nitrogen. TARGET: sulfonamide S1. SCALE: 8.0 mmol. This is a negative-control protocol for gas handling. Nitrogen is only a blanket atmosphere and must not be introduced as a reagent stream. Stream A: amine substrate in MeCN, 0.50 M, 1.0 equiv, with triethylamine 1.2 equiv. Stream B: sulfonyl chloride in MeCN, 0.50 M, 1.05 equiv. Batch reference: N2 atmosphere, 0 °C addition over 45 min, then 30 min at 25 °C, 90% yield. Reaction is strongly exothermic and HCl-forming; heat transfer and mixing are important. Flow translation: two liquid pumps to a T-mixer/static mixer, cooled coil for reaction, optional inline filter for Et3N·HCl, no MFC, no gas-liquid correction, and no second reactor. Quench/workup is offline only."""

PROMPT_CHLORINE_ADDITION_QUENCH = """REACTION NAME: Synthetic chlorination of activated alkene with chlorine gas. TARGET: vicinal dichloride CL1. SCALE: 2.0 mmol. Stream A: activated alkene in dry DCM, 0.20 M. Gas reagent: chlorine gas Cl2 introduced only through an MFC into a cold gas-liquid coil. Batch reference: -20 °C, Cl2 balloon, 20 min, 84% yield. Chlorine is toxic and exothermic; gas-liquid mass transfer and heat removal are critical. Flow translation: one cold gas-liquid chlorination reactor with small holdup, BPR or controlled gas outlet, downstream Na2S2O3 aqueous quench as inline mixer only, off-gas scrubber after outlet. Do not create a second reaction reactor for the quench."""

PROMPT_AMMONIA_AMINOLYSIS_GAS = """REACTION NAME: Synthetic ammonolysis of activated ester with ammonia gas. TARGET: primary amide A1. SCALE: 5.0 mmol. Stream A: activated ester in MeOH, 0.30 M. Gas reagent: anhydrous ammonia NH3 gas introduced by MFC into a gas-liquid segmented flow reactor. Batch reference: ammonia balloon at 25 °C for 3 h, 78% yield; mass transfer and odor/safety are limiting. Reaction is mildly exothermic. Flow translation: single gas-liquid coil at 25-40 °C with BPR, NH3 gas feed must be MFC not pump, no additional chemistry stages, offline solvent removal after outlet."""

PROMPT_SO2_SULFONYLATION_GAS = """REACTION NAME: Synthetic sulfur dioxide insertion sulfonylation. TARGET: sulfinate salt SO1. SCALE: 3.0 mmol. Stream A: aryl lithium equivalent in THF, 0.20 M, oxygen/moisture-sensitive, -20 °C. Gas reagent: sulfur dioxide SO2 gas introduced by MFC after Stream A is cooled. Batch reference: SO2 balloon at -20 °C for 90 min, then methyl iodide trapping offline, 70% yield. SO2 is toxic, mass-transfer limited, and the first contact is exothermic. Flow translation: one cold gas-liquid SO2 insertion coil with BPR and small holdup; methyl iodide trapping is offline only unless represented as a quench mixer, not a second reactor."""

PROMPT_H2_NITRO_REDUCTION_PACKED_BED = """REACTION NAME: Synthetic continuous hydrogenation of nitroarene to aniline. TARGET: aniline N1. SCALE: 10.0 mmol. Stream A: nitroarene in ethanol, 0.20 M. Gas reagent: hydrogen H2 introduced by MFC into a Pd/C packed-bed hydrogenation reactor. Batch reference: H2 balloon, Pd/C, 25 °C, 2 h, 95% yield. H2 mass transfer, catalyst bed pressure drop, and safety are important. Flow translation: one packed-bed reactor or coil packed with Pd/C, H2 via MFC, BPR mandatory, inline filter after packed bed to catch catalyst fines, no photochemistry, no extra reaction stages."""

PROMPT_LIQUID_ONLY_HCL_EVOLUTION = """REACTION NAME: Synthetic Boc deprotection with HCl in dioxane solution. TARGET: amine hydrochloride salt B1. SCALE: 6.0 mmol. This is a negative control for gas detection. HCl is supplied as 4 M HCl in dioxane liquid solution, not HCl gas. Stream A: Boc-protected amine in dioxane, 0.30 M. Stream B: 4 M HCl in dioxane liquid reagent, 5 equiv. Batch reference: 25 °C, 45 min, 98% conversion. CO2 gas evolves as a byproduct but is not fed as a reagent and should not create an MFC. Reaction is exothermic and gas evolution needs a vented backpressure-compatible separator after reaction. Flow translation: two liquid pumps, T-mixer, cooled coil if needed, optional gas-liquid separator after reactor for evolved CO2, no gas-feed MFC."""


CASES = {
    "photoredox_aerobic": PROMPT_PHOTOREDOX_AEROBIC,
    "bromination_quench": PROMPT_BROMINATION_QUENCH,
    "synthetic_gas_heat_multistage": PROMPT_SYNTHETIC_STRESS,
    "co_carbonylation_heat": PROMPT_CO_CARBONYLATION_HEAT,
    "co2_carboxylation_quench": PROMPT_CO2_CARBOXYLATION_QUENCH,
    "ozone_oxidation_cold": PROMPT_OZONE_OXIDATION_COLD,
    "syngas_hydroformylation": PROMPT_SYNGAS_HYDROFORMYLATION,
    "n2_exothermic_control": PROMPT_N2_EXOTHERMIC_CONTROL,
    "chlorine_addition_quench": PROMPT_CHLORINE_ADDITION_QUENCH,
    "ammonia_aminolysis_gas": PROMPT_AMMONIA_AMINOLYSIS_GAS,
    "so2_sulfonylation_gas": PROMPT_SO2_SULFONYLATION_GAS,
    "h2_nitro_reduction_packed_bed": PROMPT_H2_NITRO_REDUCTION_PACKED_BED,
    "liquid_only_hcl_evolution": PROMPT_LIQUID_ONLY_HCL_EVOLUTION,
}


def _topology_summary(result: dict) -> dict:
    topo = result.get("process_topology") or {}
    ops = topo.get("unit_operations") or []
    return {
        "op_count": len(ops),
        "op_types": [op.get("op_type") for op in ops],
        "op_labels": [op.get("label") for op in ops],
        "mfc_count": sum(1 for op in ops if op.get("op_type") == "mfc"),
        "reaction_reactor_count": sum(
            1 for op in ops
            if op.get("op_type") in {"coil_reactor", "photoreactor", "chip_reactor", "packed_bed_reactor"}
            and "quench" not in (op.get("op_id") or "").lower()
            and "quench" not in (op.get("label") or "").lower()
        ),
        "total_flow_rate_mL_min": topo.get("total_flow_rate_mL_min"),
        "residence_time_min": topo.get("residence_time_min"),
        "reactor_volume_mL": topo.get("reactor_volume_mL"),
    }


def _engineering_summary(result: dict) -> dict:
    proposal = result.get("proposal") or {}
    calc = result.get("design_calculations") or {}
    return {
        "tau_min": proposal.get("residence_time_min"),
        "Q_mL_min": proposal.get("flow_rate_mL_min"),
        "ID_mm": proposal.get("tubing_ID_mm"),
        "V_mL": proposal.get("reactor_volume_mL"),
        "BPR_bar": proposal.get("BPR_bar"),
        "is_gas_liquid": calc.get("is_gas_liquid"),
        "gas_species": calc.get("gas_species"),
        "gas_flow_sccm": calc.get("gas_flow_sccm"),
        "gas_holdup": calc.get("gas_holdup"),
        "o2_transfer_sufficiency": calc.get("o2_transfer_sufficiency"),
        "pressure_drop_bar": calc.get("pressure_drop_bar"),
        "UA_W_K": calc.get("UA_W_K"),
        "Da_th": calc.get("thermal_damkohler"),
        "engine_validated": proposal.get("engine_validated"),
        "safety_flags": proposal.get("safety_flags"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--cases", nargs="+", choices=sorted(CASES), default=list(CASES))
    parser.add_argument("--upstream", default="claude")
    parser.add_argument("--council", default="gpt4o")
    parser.add_argument("--output-root", default="benchmark/data")
    args = parser.parse_args()

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.output_root) / f"multiphase_upgrade_validation_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    with upstream_bundle(args.upstream), council_bundle(args.council):
        for case_id in args.cases:
            for repeat in range(1, args.repeats + 1):
                run_dir = out_dir / case_id / f"repeat_{repeat:02d}"
                run_dir.mkdir(parents=True, exist_ok=True)
                (run_dir / "input_protocol.txt").write_text(CASES[case_id], encoding="utf-8")
                started = time.perf_counter()
                status = "completed"
                error = ""
                try:
                    result = translate(CASES[case_id])
                    (run_dir / "result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
                    for key in ("svg_path", "png_path"):
                        src = Path(result.get(key) or "")
                        if src.exists():
                            shutil.copy2(src, run_dir / src.name)
                    topo_summary = _topology_summary(result)
                    eng_summary = _engineering_summary(result)
                except Exception as exc:
                    status = "failed"
                    error = str(exc)
                    result = {}
                    topo_summary = {}
                    eng_summary = {}
                    (run_dir / "error.json").write_text(json.dumps({"error": error}, indent=2), encoding="utf-8")
                row = {
                    "case_id": case_id,
                    "repeat": repeat,
                    "status": status,
                    "runtime_s": round(time.perf_counter() - started, 3),
                    "error": error,
                    **eng_summary,
                    **{f"topology_{k}": v for k, v in topo_summary.items()},
                    "run_dir": str(run_dir),
                }
                rows.append(row)
                print(json.dumps(row, ensure_ascii=False))

    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"output_dir": str(out_dir), "summary_json": str(summary_path)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
