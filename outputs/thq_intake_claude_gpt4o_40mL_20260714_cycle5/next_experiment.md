# FlowPilot THQ Cycle 5: Proposed Experiment

## Evidence used

- KRICT-only measured results were supplied to the pipeline; KHU data were excluded.
- Best measured anchor: Entry 7 (KRICT 3-1), 51% product at 68.9 min inlet/STP apparent residence time.
- No measured evidence of an overreaction regime was supplied.
- The manual pump minimum is 0.010 mL/min.

## Proposed Entry 8: KRICT 3-2

| Parameter | Proposed value |
|---|---:|
| Reactor | Two 20 mL manual FEP coils connected in series |
| Total reactor volume | 40.0 mL |
| Tubing ID | 1.0 mm |
| Temperature | 40 C |
| Concentration | 0.50 M |
| BPR pressure | 3.0 bar gauge |
| Liquid/substrate flow | 0.01727 mL/min |
| O2 flow at inlet/STP | 0.38717 mL/min |
| O2 flow in-channel | 0.11207 mL/min |
| O2 supplied | 2.00 equiv |
| Inlet/STP apparent residence time | 98.90 min |
| Pressure-corrected in-channel residence time | 309.27 min |
| Irradiation | 448 nm, 62 mW/cm2, 360 degrees around each coil |
| Product result | To be measured |

## Deterministic checks

At 0.50 M, the substrate feed is 0.008635 mmol/min. A 2.00-equivalent O2 feed is therefore 0.01727 mmol/min, corresponding to 0.38717 mL/min at STP.

The primary residence-time calculation is:

```text
tau_inlet = 40.0 / (0.01727 + 0.38717) = 98.90 min
```

Using the pressure-corrected in-channel O2 flow:

```text
tau_in_channel = 40.0 / (0.01727 + 0.11207) = 309.27 min
```

The liquid flow is above the 0.010 mL/min pump minimum, so no flow clamp was applied. Both 20 mL coils must receive equivalent 448 nm irradiation and be connected in series without an unirradiated hold-up section.

This is a next-screen recommendation, not a predicted-yield claim.
