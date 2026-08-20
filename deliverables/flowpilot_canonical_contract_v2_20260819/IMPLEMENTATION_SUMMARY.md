# FlowPilot Canonical Contract v2

## Architectural Change

FlowPilot now has one post-realization executable authority. The final contract
contains the parameters, stage design, typed process graph, stream components,
chemistry identity, instrument manifest, safety controls, operating procedure,
validation experiments, and a SHA-256 binding those artifacts together.

Council and upstream outputs remain available as diagnostic reasoning. They are
not executable and cannot overwrite the canonical artifacts after realization.

## New Release Gates

- confirmed chemistry identity preserved;
- every reactive stream component quantified;
- phase-consistent process graph;
- controllers excluded from the material process path;
- mixing-regime claims consistent with topology;
- unambiguous residence-time basis;
- hazard-specific safety controls complete;
- complete preparation-to-waste operating procedure;
- executable operations bound to declared inventory;
- rendered diagram hash equal to canonical topology hash;
- all prior numerical, geometry, gas-bookkeeping, and inventory checks.

Any failed critical gate produces a blocked design. Numerical parameters and an
executable diagram are withheld; diagnostic artifacts remain available for
correction.

## Intake And Inventory

`Q-CHEM-001` now requires chemistry identity confirmation when the protocol does
not explicitly state a recognized transformation. A model-only transformation
inference cannot authorize execution.

Inventory now supports first-class `safety_accessories`, including check valves,
non-return valves, shields, enclosures, secondary containment, emergency stops,
leak detectors, and safe-vent hardware with declared capabilities.

## Frozen Replay Result

All six old outputs had previously passed deterministic validation. Under v2:

- none is executable with the original incomplete identity/safety authority;
- three become executable after chemistry confirmation and safety inventory;
- three Qwen outputs remain blocked for genuine model defects:
  unsupported slug-flow topology, hydrogenolysis identity drift, and incomplete
  TBHP stoichiometry.

See `replay_summary.csv`, `summary.json`, and `contracts/` for the complete
machine-readable evidence.

## Verification

- `189` FlowPilot tests passed.
- Ten canonical-contract tests cover valid compilation and semantic mutations.
- Streamlit AppTest passed with ten result tabs and canonical procedure content.
- Playwright verified the summary, rendered process diagram, and chemistry/
  procedure views with no browser-visible Streamlit exception.
- Browser screenshots are stored under `gui/`.
