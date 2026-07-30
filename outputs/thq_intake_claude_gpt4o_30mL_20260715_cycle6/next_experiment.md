# FlowPilot THQ Cycle 6: Revised KRICT 4-2

## Inventory change

- The 40 mL serial reactor is unavailable and was removed from the active inventory.
- The available serial option is 30 mL: one 20 mL manual coil plus one 10 mL manual coil.
- Both coils must receive equivalent 448 nm irradiation.

## Proposed design

| Parameter | Proposed value |
|---|---:|
| Reactor | One 20 mL and one 10 mL manual FEP coil in series |
| Total reactor volume | 30.0 mL |
| Tubing ID | 1.0 mm |
| Temperature | 40 C |
| Concentration | 0.50 M |
| BPR pressure | 3.0 bar gauge |
| Liquid/substrate flow | 0.01296 mL/min |
| O2 flow at inlet/STP | 0.29038 mL/min |
| O2 flow in-channel | 0.08405 mL/min |
| O2 supplied | 2.00 equiv |
| Inlet/STP apparent residence time | 98.90 min |
| Pressure-corrected in-channel residence time | 309.26 min |
| Irradiation | 448 nm, 62 mW/cm2, 360 degrees around each coil |
| Product result | To be measured |

## Deterministic checks

```text
tau_inlet = 30.0 / (0.01296 + 0.29038) = 98.90 min
tau_in_channel = 30.0 / (0.01296 + 0.08405) = 309.25 min
```

The liquid flow is 0.00296 mL/min above the 0.010 mL/min manual-pump minimum, so no pump clamp was applied. The 2.00-equivalent oxygen supply is calculated using the inlet/STP MFC flow.

This is a next-screen recommendation, not a predicted-yield claim.
