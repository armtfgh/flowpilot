# Council flow-operability review

Date: 2026-09-18

## Status

Implemented and locally tested. After explicit user approval, the live Claude
Sonnet 4.6 feature-off/on comparison completed on 2026-09-18. Both arms produced
closed numerical designs. The enabled council proposed lower-flow pure oxygen
and retained the unresolved reverse-path findings. See the
[live comparison report](../outputs/council_backflow_ab_20260918_111811/REPORT.md)
for full results, usage, and remaining model-narrative errors. The earlier
data-export refusal was resolved by user approval, not bypassed.

The original local comparison described below uses synthetic council responses to test the complete
calculation, inventory, finalization, autosave, diagram, and GUI paths. It is a
software integration test, not evidence that an LLM independently chose oxygen.

## What changed

The scientific council now reviews reverse-flow connectivity before its existing
four specialist reviews, Skeptic, and Chief. The 12-candidate pool is retained.
No manuscript, inventory profile, raw protocol, historical experiment, or prior
benchmark result was rewritten.

1. Build each candidate's inventory-bound process topology.
2. Find gas/liquid junctions and trace upstream physical branches against the
   nominal flow direction. Ignore utility connections.
3. Record which branch has a declared non-return valve, its declared direction,
   available rating information, and the first upstream reactor or delivery
   device exposed by that path. A valve on the gas branch does not protect the
   liquid branch. A valve at the pump does not protect a reactor between it and
   the junction.
4. Give the council's flow-operability reviewer the deterministic findings,
   candidate flows, existing pressure-rating checks, and permitted alternatives.
5. Permit either retention or, for an eligible single air feed, a pure-oxygen
   proposal preserving the delivered oxygen molar feed. The prompt does not
   require the reviewer to choose oxygen. Other equipment or configurations can
   be recommended, but are not represented as installed or verified.
6. Recalculate all 12 candidates for an accepted source proposal. Preserve
   liquid flows, concentrations, reactors, temperatures, pressure setpoints,
   and gas introduction stage. Refresh inventory assignments, engineering,
   objective comparisons, and stage-time calculations. Reject the source
   revision if these invariants cannot be maintained.
7. Let the existing specialists, Skeptic, and Chief review the resulting pool
   with the operability findings attached. Reassess the finalized topology and
   publish the findings alongside the final parameters and diagram.

The added review is normally one extra model call for an applicable gas/liquid
process. Existing JSON-repair handling allows two recorded attempts if needed.
Liquid-only processes do not receive an extra model call or changed specialist
instructions. The legacy council branch is unchanged.

## Authority and limits

- Explicit gas identity answers and prohibitions take precedence. The bounded
  automatic alternative is limited to batch-derived air, one physical gas feed,
  and declared available O2-capable gas hardware.
- Declared MFC capability does not confirm that oxygen supply or the complete
  assembly is suitable for oxygen service.
- Source substitutions remain `council_screening_proposal`, with
  `requires_chemist_confirmation=true`. The saved intake continues to say air.
- A source change cannot silently raise the oxygen dose to satisfy an MFC
  minimum flow. It must preserve the dose within the existing six-decimal flow
  serialization tolerance of 0.000001 mL/min.
- Reverse connectivity indicates a credible pathway, not a prediction that
  backflow will occur. No gas/liquid-ratio threshold or fabricated probability
  is used.
- Pressure-rating checks are not measurements of supply or junction pressures.
  MFC differential-pressure needs, valve leakage, direction, cracking pressure,
  service suitability, and startup/shutdown behavior require confirmation.
- Reducing gas flow does not remove an unprotected reverse path. Pure oxygen
  also changes oxygen partial pressure and oxygen-service/fire considerations.
- This implementation does not provide a transient multiphase-flow solver,
  automatic valve installation, startup control, sensor-based detection, or
  certification of a laboratory setup.

The existing final-contract `status="executable"` continues to mean numerical
and inventory closure. It does not mean laboratory approval. The additional
`final_design.flow_operability.laboratory_execution_status="review_required"`
records that distinction. The webapp displays an amber review-required banner,
a separate closure row, a proposed-topology heading, and the council findings.
Numerical results and diagrams are not hidden simply because confirmation is
still needed. The older Streamlit presentation was not redesigned in this task.

## Figure 5 software A/B check

Source: `outputs/khu_revised_six_20260915/presentation/figure5_set1`.
The original KHU inventory and answers were used without adding the observed
backflow email or a request to prefer oxygen. Archived upstream chemistry and
initial translation were frozen for both arms. Council choices in this local
check were synthetic; candidate 1 was selected in both arms deliberately to
isolate the gas-source recalculation.

| Quantity | Feature off | Feature on, synthetic oxygen proposal |
| --- | ---: | ---: |
| Physical gas | Air | O2 |
| Gas at inlet/STP, mL/min | 0.426909 | 0.089651 |
| Delivered oxygen, equiv | 2.0 | 2.0 |
| Liquid flow, mL/min | 0.020 | 0.020 |
| Gas introduced at | Stage 2 | Stage 2 |
| Stage 1 volume, mL | 2 | 2 |
| Stage 1 liquid-basis time, min | 100 | 100 |
| Stage 2 volume, mL | 20 | 20 |
| Stage 2 inlet/STP apparent time, min | 44.7518 | 182.3969 |
| BPR setpoint, bar(g) | 7 | 7 |
| Numerical stage closure | Passed | Passed |
| Topology generated | Yes | Yes |
| Backflow assessment | Not performed by this feature | Review required |

Equal oxygen delivery gives approximately `Q_O2 = 0.21 * Q_air`, a 79% lower
total inlet gas volume. The reported gas-stage time is the existing convention
`V / (Q_liquid + Q_gas,STP)`. It is an **inlet-referenced apparent space time**,
not a measured in-channel contact time or a validated kinetic requirement.
The apparent time change must not be interpreted as proof of improved yield.

The enabled screen flags the path `st2_mixer -> st1_reactor` as unprotected.
The separate path `st2_mixer -> pump_b_check_valve -> pump_b` contains a declared
gas-branch valve whose service characteristics remain unverified. The oxygen
alternative leaves the first path unresolved and is not ready for wet-lab use.

## Verification and artifacts

Completed Python regression: **159 passed**, including 12 dedicated backflow
tests and existing scientific council, gas, evidence, objective, inventory,
stage-reconciliation, final-contract, reporting, and topology tests.

The dedicated tests cover branch-specific protection, reversed valve direction,
utility-edge exclusion, liquid-only applicability, ordering reproducibility,
cycles, explicit gas restrictions, all-12-candidate recalculation, dose
preservation, source provenance, invalid reviewer output, MFC minimum-flow
failure, retention of air, and final pipeline integration with the flag off/on.

The webapp builds successfully. **Four final Playwright checks passed** across
desktop and mobile. Local checks exercise feature off/on
on desktop and mobile, including warnings, stage results, council findings,
stream flows, embedded-icon topology loading, and page overflow. Screenshots
were inspected; see the saved browser test logs for the final verification.
The review banner, closure checklist, and proposed-topology heading all make
the pending laboratory review explicit.
An initial browser assertion incorrectly required an internal equipment role
label in the diagram; the renderer displays gas identity and setpoints instead.
The corrected assertion checks those actual labels and checks the role in JSON.
The first attempt's artifacts are retained, not removed.

All local comparison material is in:

`outputs/council_backflow_ab_20260918_105512/`

It includes source hashes, frozen inputs, source snapshots, original baseline
source, both result JSONs, candidate audit files, generated review requests,
streams/stages CSVs, topology PNG/SVG files, logs, regression XML, and browser
screenshots. Synthetic requests are explicitly identified as synthetic. There
is no real-model usage or response log for an unexecuted live comparison.

## Reproduction

Run the local integration comparison without provider access:

```bash
.venv-flowpilot/bin/python scripts/run_council_backflow_comparison.py --offline-selection-test
```

After approval for the archived data transfer, run the actual council A/B:

```bash
.venv-flowpilot/bin/python scripts/run_council_backflow_comparison.py \
  --provider anthropic --model claude-sonnet-4-6
```

That script freezes upstream generation and retrieval in both arms, runs the
council live, records model requests/responses and failures, and creates a new
timestamped output folder. It does not rerun until a favorable choice appears.
Either retaining air with specific controls or proposing oxygen with unresolved
controls can be a valid review conclusion. One pair cannot establish statistical
superiority or reproducibility of model choices.

The runtime switch is `council_backflow_review`; its default is `true`. Set it to
`false` for a controlled comparison. Restart a running backend to load the new
Python code. Review archived runs as historical outputs; they are not silently
rewritten with the new feature.

## Engineering references

- [Bronkhorst FLEXI-FLOW Compact manual](https://products.bronkhorst.com/media/prqhjq5u/917158-manual-flexi-flow-compact.pdf): manufacturer guidance on installation, operating pressure, and reverse-flow protection. Verify the actual device configuration.
- [HSE: Oxygen use in the workplace](https://www.hse.gov.uk/pubns/indg459.htm): oxygen-service and fire-risk considerations. A lower volumetric flow is not a substitute for an oxygen-service review.
