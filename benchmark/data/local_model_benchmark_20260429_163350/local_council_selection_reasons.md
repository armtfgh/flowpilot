# Local Benchmark Council Logs and Selection Reasons

This note summarizes the winner selection evidence for the three local benchmark cells. The authoritative sources are:

- `stage_events.jsonl` for actual stage progression and applied winner fields
- `snapshots/stage4_chief_selection.json` for the Chief's rationale
- `result.json -> formatted_result.explanation` for the human-readable final explanation

## U_gemma / C_claude

Run folder:
- `U_gemma/C_claude/runs/protocol_isoxazole_des_full/budget_12/repeat_01`

Key files:
- `stage_events.jsonl`
- `snapshots/stage4_chief_selection.json`
- `result.json`

Final applied design:
- `tau = 120.0 min`
- `Q = 0.0589 mL/min`
- `d = 0.75 mm`
- `V_R = 7.068 mL`
- `BPR = 5.0 bar`
- `material = SS`

Stage log facts:
- Stage 2 blocked candidates: `[3, 6]`
- Stage 3.5 changed candidates: `[1, 2, 3, 5, 6, 8, 9, 11, 12]`
- Stage 4 winner: `candidate_id = 10`
- Stage 4 disqualified ids: `[3, 6, 14]`

Chief rationale:
- Candidate 10 had the top combined score: `0.8725`
- Chemistry score: `0.85`
- Kinetics score: `1.0`
- Fluidics score: `0.85`
- Safety score: `0.85`
- Geometry score: `0.8`
- Estimated conversion cited by Chief: `86%`

Tradeoffs resolved by the Chief:
- Chose candidate 10 over candidate 7 because chemistry score and conversion were slightly better
- Chose candidate 10 over candidate 12 because conversion was higher (`86%` vs `63%`) despite lower productivity
- Balanced objective favored robust conversion over more aggressive throughput

Remaining uncertainty:
- Kinetic rate constant estimated from batch data
- Heat transfer in TBAB/EG DES at `120 C`

Experiment recommendation from Chief:
- Run blank DES first, then switch to reaction mixture and sample at `3τ` and `5τ`

Readable final explanation:
- Long residence time was chosen to support in-situ nitrile oxide generation and subsequent cycloaddition
- `5.0 bar` BPR was used to maintain single-phase operation at `120 C`
- `SS` was selected for high-temperature robustness

## U_claude / C_gemma

Run folder:
- `U_claude/C_gemma/runs/protocol_isoxazole_des_full/budget_12/repeat_01`

Key files:
- `stage_events.jsonl`
- `snapshots/stage4_chief_selection.json`
- `result.json`

Final applied design:
- `tau = 76.5 min`
- `Q = 0.1097 mL/min`
- `d = 0.75 mm`
- `V_R = 8.394 mL`
- `BPR = 5.0 bar`
- `material = PFA`

Stage log facts:
- Stage 2 blocked candidates: `[]`
- Stage 3.5 changed candidates: all `12/12`
- Stage 4 winner: `candidate_id = 1`
- Stage 5 revision applied: `false`

Chief rationale:
- Winner was selected even though it was not the top weighted-score row in the same snapshot
- Chief explicitly chose candidate 1 because it gave the highest conversion among candidates that avoided critical kinetic failures
- Conversion cited by Chief: `X = 0.88`

Tradeoffs resolved by the Chief:
- Candidate 1 over candidate 12: much higher conversion (`0.88` vs `0.72`) despite lower productivity
- Candidate 1 over candidate 8: better conversion and better mixing (`r_mix 0.019` vs `0.178`)
- Balanced objective favored yield and robustness over throughput

Remaining uncertainty:
- DES viscosity at `120 C` may alter pressure drop from the modeled value

Experiment recommendation from Chief:
- Run blank solvent first, then switch to reaction mixture and sample at `3τ` and `5τ`

Readable final explanation:
- Long residence time was selected to accommodate the two-step thermal sequence:
  1. nitrile oxide formation
  2. [3+2] cycloaddition
- Single premixed feed was preferred to avoid mixing complexity in viscous DES
- `5.0 bar` BPR was used to maintain liquid-phase stability

## U_gemma / C_gemma

Run folder:
- `U_gemma/C_gemma/runs/protocol_isoxazole_des_full/budget_12/repeat_01`

Key files:
- `stage_events.jsonl`
- `snapshots/stage4_chief_selection.json`
- `result.json`

Final applied design:
- `tau = 125.8 min`
- `Q = 0.0667 mL/min`
- `d = 0.75 mm`
- `V_R = 8.393 mL`
- `BPR = 5.0 bar`
- `material = SS`

Stage log facts:
- Stage 2 blocked candidates: `[]`
- Stage 3.5 changed candidates: all `12/12`
- Stage 4 winner: `candidate_id = 13`
- Stage 5 revision applied: `false`

Important inconsistency:
- `stage4_chief_selection.json` contains a `final_consensus` with:
  - `tau = 61.4 min`
  - `Q = 0.243 mL/min`
  - `d = 1.0 mm`
  - `V_R = 14.92 mL`
- But Stage 7 actually applied:
  - `tau = 125.8 min`
  - `Q = 0.0667 mL/min`
  - `d = 0.75 mm`
  - `V_R = 8.393 mL`

What to trust:
- Trust `stage_events.jsonl` Stage 7 and `run_summary.json` / `result.json` for the actual final design
- Treat the `final_consensus` block in `stage4_chief_selection.json` as inconsistent metadata for this run

Chief rationale text in the snapshot:
- Candidate 13 was chosen for a balanced compromise between throughput and residence time
- It explicitly tried to avoid the extreme low-productivity of the very long-τ designs
- It cited DES viscosity as the main residual uncertainty

Readable final explanation:
- Final formatter explanation corresponds to the actually applied long-τ compact design
- It reports:
  - `SS 0.75 mm ID`
  - `8.393 mL`
  - `125.8 min`
  - `0.0667 mL/min`
  - `120 C`
  - `5.0 bar`
- Confidence remained `LOW` due to weak precedent similarity
