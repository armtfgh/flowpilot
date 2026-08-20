# FlowPilot Canonical v2 Live Benchmark

## Scope

This is a fresh Qwen-only production-path benchmark of the current FlowPilot
pipeline. It uses three fixed held-out chemistry cases, temperature 0, one
candidate per case, lexical retrieval with the source record excluded, the
multi-agent council, deterministic realization, inventory topology compilation,
and the canonical v2 release contract.

Each run received a frozen standardized `DesignInputPackage` containing the raw
protocol, objective, hard constraints, chemist-confirmed transformation identity,
and exact inventory. Hazard-specific containment was declared for every case and
a hydrogen check valve was declared for the hydrogenolysis case. These additions
are explicit benchmark inputs, not model-generated equipment.

Model for every upstream, translation, council, revision, and safety call:
`/models/Qwen3.6-27B` at `http://10.13.24.169:8000/v1`.

## Result

| Case | Release status | Main result |
|---|---|---|
| CuAAC | Executable screen | 0.30 mL packed bed; 0.11111 mL/min; 2.700 min; 150 C; 20 bar |
| Hydrogenolysis | Executable screen | 3.0 mL packed bed; 0.100 mL/min liquid; 1.0 sccm H2; 3.541 actual H2 equiv; 30.0 min liquid-contact time |
| Two-stage oxidative amidation | Blocked | TBHP quantity is absent from the supplied protocol and remained unresolved |

Two of three cases produced canonical executable screens. The third was blocked
for a chemically meaningful missing input rather than being completed by an
invented value. The hydrogen finalizer initially contained a hidden 10-equivalent
H2 floor; that policy was removed and the hydrogen case was rerun. The final
result preserves the 1.0-equivalent target and reports the 3.541-equivalent actual
delivery caused by the declared MFC minimum of 1.0 sccm.

## Interpretation

This benchmark supports three conclusions:

1. Standardized intake materially improves chemistry-identity preservation.
2. The final contract now distinguishes executable designs from incomplete
   chemistry instead of rewarding a filled JSON object.
3. The pipeline is improved but not proven universally superior. The legacy
   one-shot outputs used different inventory/authority inputs and are therefore
   excluded from a matched architecture comparison in this package.

The internal kinetic estimates for CuAAC were weak because retrieval analogies
had low similarity. The final CuAAC screen is numerically and inventory closed,
but its residence time remains a first-screen hypothesis and requires wet-lab
validation.

## Files

Each case folder contains the frozen public input, complete result JSON, LLM
event log, stage event log, and deterministic metrics. `summary.csv` and
`summary.json` provide the compact outcome table. `checksums.sha256` verifies
the package contents.
