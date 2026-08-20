# FlowPilot systemic failure analysis and correction

Date: 2026-08-19

## Executive finding

FlowPilot was not mainly losing because its language models were weaker. It was
losing because multiple pipeline stages were allowed to publish different
versions of the same design. The one-shot baseline had one state and therefore
often looked more coherent, even when it had no deterministic proof that its
equipment and calculations were valid.

The corrected architecture treats model output as a candidate only. One
deterministic post-council compiler now owns stream identities, component
stoichiometry, pump and MFC assignments, pressure basis, reactor allocation,
gas conversion, residence-time definitions, safety equipment, procedure, final
JSON, process graph, and GUI-facing values.

## Why one-shot appeared to handle KHU inventory better

The one-shot model saw the protocol and inventory together and wrote one
plausible narrative. It did not pass the design through independently mutable
chemistry, proposal, council, calculator, topology, contract, and GUI states.
This avoided internal drift, but it did not prove feasibility.

FlowPilot previously allowed the following failures:

1. Council/grid output could overwrite chemistry-supported residence time.
2. Stage summaries could overwrite quantified global stream records.
3. Inventory allocation could change hardware after calculations were made.
4. Topology could retain stale gas identity, flow, stage time, or geometry.
5. The final contract and GUI could select different residence-time bases.
6. A gas actual-flow conversion passed absolute pressure into a function that
   expected gauge pressure.
7. Multistage gas conversion used the proposal summary temperature instead of
   the temperature of the stage where gas was introduced.
8. Multicomponent feed concentration and equivalent fields lacked a declared
   reference component.
9. Work-up extraction could mistake a request sentence for a source-protocol
   operation.
10. Negated inventory prose could trigger false chemistry or safety modalities.

## Systemic corrections

- Removed the unconditional design-space winner overwrite.
- Made the deterministic realizer the final numerical authority.
- Bound pumps to flow, pressure, material, gas, and quantity capability.
- Bound reactors and lights by exact inventory IDs and stage conditions.
- Coupled liquid and gas rate solving instead of clipping either independently.
- Declared gauge and absolute pressure separately.
- Declared STP as 273.15 K, 1.01325 bar, and 22.414 L/mol.
- Recomputed gas actual flow at the realized gas-introduction-stage temperature.
- Defined V/Q-liquid as a nominal geometric reference; packed-bed fluid contact
  is not claimed without measured holdup or RTD.
- Preserved inlet/STP and pressure-corrected in-channel gas flow as distinct
  quantities.
- Added reference-component semantics for multicomponent streams.
- Generated exact final-volume stock recipes from component concentrations and
  equivalents without inventing molecular weights, assay, or density.
- Assigned declared safety accessories to the canonical manifest and procedure.
- Corrected gas-liquid separator phase semantics.
- Removed topology geometry that was derived but not declared by inventory.
- Prevented request text from being copied as work-up instructions.
- Made non-hazardous first screens end explicitly at crude collection/analysis
  when no chemist-confirmed isolation procedure exists.
- Kept hazardous quench details explicit as unresolved chemistry inputs rather
  than inventing a reagent or endpoint.
- Published final JSON, topology, procedure, safety, and GUI data from the same
  canonical final-design object.

## Fixed benchmark evidence

### Confirmatory latest-compiler result

The final held-input run met the predefined stopping rule and was not rerun:

- GPT-5.4 FlowPilot: 0.9204; GPT-5.4 one-shot: 0.9085; delta +0.0119.
- Qwen3.6-27B FlowPilot: 0.8912; Qwen one-shot: 0.7436; delta +0.1476.
- Overall mean paired delta: +0.0797.
- Generator-family-excluded paired delta: +0.1303.
- FlowPilot critical flags: 0 for both generator families.
- Deterministic closure: 10/10 candidates, zero contract issues.

The confirmatory report is stored at
`deliverables/manuscript_five_case_v4_latest_compiler_20260819/`.

Five cases, two generator families, one repeat, and two blinded LLM judges were
used. The one-shot generations were held constant.

| Campaign | GPT FlowPilot | GPT one-shot | Qwen FlowPilot | Qwen one-shot | Mean paired delta |
|---|---:|---:|---:|---:|---:|
| Before systemic correction | 0.7492 | 0.9022 | 0.7782 | 0.7375 | -0.0562 |
| Canonical-state correction | 0.8795 | 0.9085 | 0.8415 | 0.7436 | +0.0345 |
| Post-compiler rescore | 0.8665 | 0.9085 | 0.8628 | 0.7436 | +0.0386 |

The post-compiler comparison shows:

- Qwen FlowPilot beat Qwen one-shot in all 5 cases.
- Mean Qwen advantage was +0.1192.
- Cross-family OpenAI-judge Qwen advantage was +0.2000.
- GPT FlowPilot matched or exceeded GPT one-shot in 1 of 5 cases and remained
  lower by 0.0420 on average. FlowPilot is therefore not yet proven universally
  superior to a frontier one-shot model.
- Exact inter-judge agreement was 31.7%, within-one-point agreement was 61.9%,
  and mean absolute disagreement was 1.15/4. LLM-judge scores must be reported
  with this uncertainty.

The final post-judge deterministic cleanup was replayed over all ten FlowPilot
candidates. Result: 10/10 executable contracts and zero consistency issues.
This replay is not silently substituted for the judged campaign; it is stored
as separate regression evidence.

## Current KHU baseline result

The final KHU replay is executable with zero consistency issues and exact
inventory IDs.

- Liquid pump: `KHU-PUMP-HAMILTON-PSD4`
- Stage 1 reactor: `KHU-REACTOR-PFA-1MM-2ML`
- Stage 1 light: `KHU-LIGHT-UV150-450`
- Stage 2 reactor: `KHU-REACTOR-PFA-075MM-5ML`
- Stage 2 light: `KHU-LIGHT-MANUAL-BLUE-448`
- Gas device: `KHU-GAS-MFC-O2-BRONKHORST`
- BPR: `KHU-BPR-FIXED-7BAR`
- Liquid flow: 0.117260 mL/min
- O2 inlet/STP: 0.262812 sccm
- O2 in-channel: 0.038098 mL/min at 40 C and 8.01325 bar absolute
- Total reactor volume: 7.0 mL
- Nominal total liquid-only reactor-volume time: 59.6964 min
- Inlet/STP apparent total: 30.2115 min
- In-channel apparent total: 49.2398 min

The mixer is a standard passive accessory assumption because the KHU inventory
does not declare an exact mixer ID. It remains a pre-run verification item. No
wet-lab yield or chemical success is claimed by deterministic closure.

## Evidence locations

- Judged post-compiler report:
  `deliverables/manuscript_five_case_v3_postcompiler_fix_20260819/`
- Judged campaign and raw logs:
  `ablation_results/manuscript_benchmark/manuscript_five_case_v3_postcompiler_fix_20260819/`
- Latest ten-candidate deterministic replay:
  `ablation_results/manuscript_benchmark/manuscript_five_case_v2_systemic_fix_20260819/systemic_revalidation_v4_20260819/`
- Latest KHU replay:
  `outputs/flowpilot_systemic_fix_20260819/khu_revalidated_final_v8/`
- Regression script:
  `scripts/revalidate_manuscript_flowpilot_candidates.py`

## Verification

`python -m pytest flora_translate/tests ablation_test/tests -q`

Result: 285 passed.

Graphviz `dot` and CairoSVG are unavailable in this environment. The canonical
SVG fallback was generated successfully; PNG export was skipped. This does not
affect topology data or final-contract validation.
