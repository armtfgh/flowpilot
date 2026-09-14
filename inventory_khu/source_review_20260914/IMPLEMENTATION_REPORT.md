# KHU six-run update: implementation record

## Scope and provenance

Source files are unchanged. Inventory cells and the original 22-slide PPT are
extracted separately. Only the explicitly revised responses on slides 5, 8,
11, 16, 19 and 22 are active inputs. Slides 1 and 12 supply the batch protocols.
Previous FlowPilot outputs are not experimental measurements.

The new profile is a schema-valid **draft**, not a laboratory safety approval.
Source values and unresolved fields remain in its extraction metadata. The
prior inventory v4 has not been overwritten.

## Changes

- Added shared-resource accounting for pump configurations and photoreactor
  modules. Alternative light sources consume the same physical module resource.
- Added stage-local module capacity and pump-platform compatibility checks.
  Bare tubing can be mounted in a compatible module; named manual and Vapourtec
  assemblies cannot borrow each other's light sources.
- Pump selection prefers one compatible platform and respects shared channel
  and syringe counts. Individually controlled Chemyx streams use separate pump
  units; the optional equal-flow paired mode is not yet enumerated.
- Required outlet check valves are inserted from MFC inventory metadata. The
  1 bar cracking differential contributes to the gas pressure-headroom check.
  Unknown maximum valve pressure remains a pre-experiment verification item.
- Replaced the displayed syringe-specific raster icon with a neutral pump
  symbol. The actual assigned instrument name remains in the diagram label.
- Topology feed labels are now derived from final stream concentrations and
  canonical component quantities, not pre-design alternatives such as "neat
  or in solvent". Display-only updates preserve the selected numerical design,
  rebuild its final contract, and are explicitly distinguished from new model runs.
- Fixed standardized intake parsing of a positive gas destination accompanied
  by an oxygen-free earlier stage. Conflicting positive/exclusion requirements
  still do not resolve silently.
- Preserved an "at least" oxygen requirement as a lower bound rather than an
  exact upper-and-lower constraint.
- During the first live run, detected a software-added molarity for an unnamed
  pH buffer medium. Named solvent-mixture membership is now recognized, so a
  pH description cannot acquire the substrate molarity as an invented buffer
  formulation. Buffer identity/strength still requires laboratory confirmation.
- Extended solvent-mixture normalization to preserve membership when an
  inerting instruction follows the ratio. Added a regression for the exact
  comma-suffixed form found in a live model response.
- Fixed gas-composition validation to distinguish the declared gas identity
  from prose explaining its role. An air feed supplying O2 is not pure O2;
  an explicitly pure-O2 feed must still have active-gas fraction 1.0.
- Added one bounded council JSON retry with a larger response budget after a
  live skeptic response was truncated at the provider token limit. Both raw
  responses are archived. No missing judgment is fabricated and all downstream
  semantic validation remains active; two invalid responses still fail.

## Verification and run policy

After first-pass review, moved the verbatim workbook catalogue from repeated
operating-limit prompt text into profile extraction metadata. The physical
LabInventory objects were verified identical. The intake context decreased
from approximately 87,669 to 28,937 characters; no chemist response was removed.

Focused regression tests cover shared resources, assembly mismatch, module
volume limits, pump-platform support, idempotent check-valve insertion, gas
stage parsing, neutral pump rendering and solvent-medium identification.
Final focused suite: 127 tests passed on 2026-09-14 (output retained with campaign).

The six runs use the production translate entry point, the scientific policy,
12 connected-process candidates, Claude Opus 4.6 upstream and Claude Sonnet 4.6
downstream. OpenAI embeddings may be used by the existing retrieval service;
this does not make OpenAI the upstream/downstream design model.

Each attempt retains input, intake package, inventory snapshot, model call
content and usage, scientific-council artifacts, result JSON, topology and log.
Independent checks recompute component molar flows and cumulative stage flows,
verify stage times and stock geometry, and check resource use and gas placement.
Earlier attempts are not deleted or presented as wet-lab outcomes. Technical
corrections are followed by new attempts where needed, not selection for a
more favorable predicted yield.

## Explicit limitations

- Fabricatable coils are retained as stock, not silently treated as assembled
  fixed-volume reactors. The '<' versus 'up to' source ambiguity is unresolved.
- Red/green strip sources lack numeric wavelengths; their existence is retained
  in the source catalogue but they are not wavelength-selectable candidates.
- H-Cube is retained as an integrated platform in the source catalogue; its
  internal hardware is not offered as independently available components.
- MFC quantity is not explicitly supplied; one listed instrument is assumed.
- Check-valve maximum pressure, pressure reference conventions, full mixture
  compatibility and stock-solubility require laboratory confirmation.
- Feed gas volumes and reported gas-fed stage times use inlet/STP conditions.
  This nominal V/Q quantity is not measured physical gas-liquid residence time.
- No flow yield, conversion, optimized kinetics or wet-lab safety approval is
  claimed by passing the software consistency checks.

## Outputs

Campaign: `outputs/khu_revised_six_20260914/`.
Per-attempt `independent_checks.json` identifies every passed/failed check.
The presentation directory will contain selected checked exports and a new PPTX;
the source KHU PPTX remains untouched.
