# Experimental council pressure-network tools

## Scope and status

This opt-in addition gives **DrFluidics and DrSafety**, not a new agent, access
to `simulate_flow_transients`. Normal runs are unchanged. In this mode the
standalone FlowOperability review is replaced by tool use inside the existing
reviewers. DrChemistry, DrKinetics, Skeptic and Chief retain their existing roles.

The tool is a reduced, isothermal transient pressure network. It is **not a
validated predictor of the KHU incident**, CFD, a gas-front tracker, an online
detector, a probability estimator, or a safety approval system. The collaborator
has not supplied pressure histories, supply settings or measured startup timing.
Every trial therefore labels device dynamics as illustrative assumptions.

## What the tool accepts

The first implementation supports two connected circular-coil stages at the
same temperature, one liquid feed before Stage 1, and one gas feed introduced
at Stage 2 through a declared forward check valve. Geometry, liquid flow, gas
feed, BPR setting, equipment identity and valve cracking pressure come from
the inventory-bound candidate, not from reviewer-generated numbers.

Unsupported topologies, unknown gas-valve cracking pressure, an existing liquid
valve with unknown leakage, unsupported geometry, and insufficient specified
supply pressure produce `not_assessable`. They do not receive invented
hydraulics or a low-risk score. The prototype must be extended before using
it for additional feeds, packed beds, several gas junctions or different
stage temperatures.

The explicit profile supplies fluid viscosity/density, effective liquid-line
compliance, gas plenum volume, actual assumed gas supply pressure, MFC pressure
requirement and response, gas-branch conductance, pump response/cutoff/leakage,
and an effective downstream resistance. These are NOT interchangeable with
maximum equipment ratings. The example profile is for software testing only,
not proposed laboratory regulator or pump settings.

## Model and units

Reported gas feeds remain **mL/min at inlet/STP**, defined as 273.15 K and
1.01325 bar. Internal gas compression requires absolute pressure; the solver
adds 1.01325 bar to the proposal's gauge BPR pressure. Residence-time reporting
in the main pipeline is not changed by this tool.

Let `pL`, `pG`, `pJ` be absolute liquid-pump discharge, gas-plenum and junction
pressures; `VL` be downstream liquid inventory; `V` the downstream reactor
volume; `A = P_STP * T / T_STP`; `CL` effective liquid compliance; and `VG`
gas-line plenum volume. In minutes, the balances are:

```
qL = (pL - pJ) / R1                 # signed upstream liquid-branch flow
qG_STP = KG * max(pG - pJ - p_crack, 0)
qOut = max(pJ - p_BPR_abs, 0) / R2

CL * dpL/dt = qPump - qLeak - qL
(VG/A) * dpG/dt = qMFC_STP - qG_STP
dVL/dt = qL - (VL/V)*qOut
dpJ/dt = [A*qG_STP + pJ*(qL - qOut)] / (V - VL)

reverse displacement = integral(max(-qL, 0) dt)
```

The implementation converts these balances to seconds. Circular-tube laminar
resistance uses `R = 128*mu*L/(pi*d^4)` with SI-to-bar/(mL/min) conversion.
`R2` includes an explicitly assumed two-phase multiplier and outlet resistance.
The latter is not a calibrated two-phase friction correlation. The BPR is an
ideal algebraic opening law, not a model of its mechanical hysteresis.

Pump and MFC flow actuators have first-order response. Delivery falls when the
specified pump pressure limit or actual MFC differential-pressure allowance
is insufficient. A hypothetical liquid non-return valve replaces the signed
liquid branch law with forward cracking and explicitly specified reverse
leakage. Zero leakage is an ideal limit, never proof of real valve performance.

The downstream outlet is treated as a homogeneous gas/liquid mixture. The
reverse branch is assumed to displace liquid. The model does not resolve the
phase distribution at the junction, liquid slugs, capillarity, gas dissolution,
reaction consumption, or an oxygen front entering Stage 1. Reaching invalid
phase inventories or exhausting upstream liquid displacement terminates
validity rather than publishing a successful simulation.

## Fixed tests and alternatives

Each requested candidate is evaluated under four scenarios over 90 seconds:

1. An initialized steady operating point.
2. Initially atmospheric, prefilled system: gas starts first, liquid starts
   after the explicitly entered delay.
3. Liquid delivery interruption from 30 to 60 seconds, while gas continues.
4. A twofold gas-command perturbation from 30 to 40 seconds.

Both the entered gas plenum volume and one tenth that volume are evaluated.
This deterministic sensitivity bracket is **not a measured range or probability
distribution**. Scenario counts must not be interpreted as incident frequency.
No distributions, failure probabilities or random sampling are used.

The fixed comparisons are the as-designed arrangement, pure oxygen at the
same delivered oxygen molar rate when source constraints/MFC limits permit,
and a hypothetical liquid-branch check valve. Oxygen changes partial pressure
and service/fire requirements; the model does not evaluate that chemistry or
those hazards. Hypothetical equipment is never automatically added to inventory.
An explicit chemist gas-identity restriction is respected. Reduced gas volume
is not presumed to prevent backflow, and the tests include a counterexample.

## Council interaction and authority

The tool dialogue uses a bounded JSON dispatcher through the existing model
client. It does not depend on native tool-calling support, so the same contract
can be used with Anthropic, OpenAI or local models. Live local-model performance
is not established by the present Claude trial.

1. DrFluidics requests the named tool for all 12 frozen candidates. Selective
   favorable screening is rejected.
2. Python validates the allowed tool name and IDs. It does not accept physical
   parameters or equipment invented in the tool request.
3. Python runs fixed scenarios and saves complete numerical traces and profile
   provenance. Summaries retain evidence IDs and explicitly distinguish valid,
   invalid and unassessable runs. No failures are silently replaced by zeros.
4. The tool result is returned to DrFluidics for its normal 12-candidate review.
5. DrSafety requests the same tool to cross-check candidates. Identical requests
   use cached deterministic results, not an independent physical measurement.
6. Both reviewers must cite a real result ID and simulation evidence ID(s),
   state limitations and describe any mitigation as a conditional proposal.
7. Skeptic and Chief receive these results and assessments. The final design
   remains an inventory-bound candidate; simulated alternatives do not silently
   alter streams, gas identity, equipment or residence times.

Assumed simulation outputs are not promoted to measured evidence. They do not
replace existing deterministic inventory checks. Missing dynamics remain
visible. The original backflow topology warnings remain unresolved and the
laboratory status remains `review_required`, even if numerical design closure
is complete. A numerical `executable` contract means software consistency,
not laboratory authorization.

The final result also receives deterministic warnings from the selected
candidate's **as-designed** simulations. These preserve conditional reverse
displacement and assumed MFC-rating exceedance even when the Chief overlooks
them in prose. Findings from a hypothetical alternative are not relabeled as
findings about installed hardware. The exact profile, hypothetical valve
parameters and finite-startup-window limitations travel with tool results.

## Reproducibility and verification

The SciPy Radau integrator solves the stiff ordinary differential equations.
Known input discontinuities are explicit integration boundaries. Default
maximum step is 0.5 s, relative tolerance 1e-7 and absolute tolerance 1e-10.
The tests repeat calculations at 0.1 s and 1e-9 relative tolerance.

Every simulation checks liquid and STP-gas conservation, valid inventories,
solver completion and the upstream laminar-flow domain. The model exposes
signed flow, reverse displacement, pressures, MFC differential-pressure margin,
simulation status, assumptions and an input SHA-256. `backflow_probability` is
always null. Full traces, model source snapshots and model responses are archived.

The test suite includes analytical resistance and steady-flow checks, absence
of spurious steady reversal, gas-volume sensitivity, ideal/leaking valve
behavior, insufficient pressure, invalid-state termination, convergence,
repeatability, request validation, and a mocked full `translate()` run through
all existing council reviewers and final rendering.

## Enable a trial

API usage:

```python
import json
from flora_translate.main import translate

with open("docs/examples/transient_assumptions_illustrative.json") as f:
    assumptions = json.load(f)

result = translate(protocol, intake_package=package, runtime_options={
    "design_policy": "scientific_v2",
    "candidate_budget": 12,
    "council_backflow_review": True,
    "council_physics_profile": assumptions,
})
```

Omitting `council_physics_profile` disables the trial. No global GUI defaults
are changed. The existing Council transcript and JSON include the reviews and
selected-candidate physics screen; this implementation adds no separate agent
or new GUI configuration panel.

Frozen KHU replay, real Claude council:

```bash
.venv-flowpilot/bin/python scripts/run_council_backflow_comparison.py \
  --arm on --physics-profile docs/examples/transient_assumptions_illustrative.json
```

Use `--offline-selection-test` for mocked model responses. Numerical simulations
are real in both modes. The script's optional off arm disables both the topology
review and physics tools; that comparison therefore does not isolate the marginal
benefit of physics over topology alone. It is not a performance benchmark or
validation against wet-lab pressure measurements.

## Next validation boundary

To progress beyond an illustrative simulator, obtain independently measured
pressure/flow traces, gas line volumes, pump stop behavior, regulator/MFC
operating pressures and dynamic response, BPR response, and check-valve
leakage/closing characteristics. Fit parameters on a separate calibration set,
then test predictions against held-out, supervised non-reactive hydraulic
measurements. Do not use the single reported incident both to tune and validate
the model. Probability estimates additionally require justified uncertainty and
event-frequency distributions over a defined time horizon.

## September 18 trial and audit

The live Claude Sonnet 4.6 trial is archived at
`outputs/council_physics_claude_20260918/`. Both existing reviewers called the
tool for all 12 candidates. All 288 scenario/variant/sensitivity simulations
completed and passed numerical conservation checks. Chief retained candidate 1;
the final gas source was not changed automatically. The original protocol,
intake, upstream generation and all 12 physical candidate signatures match the
earlier feature-off Claude archive. This is a single-run engineering trial,
not a superiority or reliability benchmark.

Three truncated model replies required the existing bounded JSON retry. All
11 actual model calls, including those incomplete replies, are retained. No
responses were repaired or overwritten. Further audit found that Chief's
summary mentioned a conditional MFC pressure exceedance for the valve
alternative without also identifying it for the as-designed gas-overshoot
case. This motivated the deterministic final warnings described above.

After the live run, audit packaging was tightened to include the full profile
and hypothetical valve parameters in each tool response, preserve intermediate
reviewer assessments, label startup windows that do not reach BPR pressure, and
publish selected-design physics warnings without relying on model prose.
These packaging additions do not change the physical calculations. They are
covered by subsequent full-pipeline mocked tests; the live run retains its
original source snapshot and unmodified responses. The final report separately
recomputes the warnings from that archived numerical output.

## Technical sources

- [SciPy solve_ivp documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html): stiff integrators, events and error tolerances.
- [MathWorks gas-pipe documentation](https://www.mathworks.com/help/simscape/ref/pipeg.html): gas storage and dynamic mass-balance modeling. This implementation does not use or claim equivalence to Simscape.
