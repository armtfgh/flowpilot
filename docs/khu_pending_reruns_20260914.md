# KHU reruns: authorized, installed UV-150 specification pending

## Resolved on 2026-09-15

The notes below describe the earlier uncertainty, not the current policy.
KHU's clarification and the updated workbook explicitly permit each listed
coil in UV-150 and manual modules, subject to module capacity, temperature,
quantity and platform limits. A separate dedicated-only reactor is therefore
not inferred. The current source-derived builder is
`scripts/build_khu_inventory_20260915.py`; the dedicated-only hypothesis builder
is not used for this campaign. The uploaded check-valve icon is present.
All six response sets are being rerun with the newly confirmed pump increments.

## Current status

The user has authorized the three Figure 5 design reruns. The remaining
prerequisite is the installed UV-150 reactor material, volume, ID, pressure
limit and whether it is a separate item or an identified coil from the listed
manual stock. A clarification question was sent before generating a new profile.
The earlier UV-150 light plus manual 2 mL coil pairing is invalid under the
user-confirmed assembly rule. The existing inventory/profile and archived
results have not been silently relabeled or overwritten.

## Rendering changes completed

- Main feed-to-product path uses the actual directed connections and runs
  horizontally through icon-center ports. Side feeds remain branches.
- Fixed a Graphviz port override: a separate headport attribute was replacing
  the named image port and targeting the overall node/caption box.
- Do not invent additional mixers or drop streams for three or more feeds.
- New optional icon path: `flora_design/visualizer/icons/check_valve.png`.
  The renderer reads it on each new render. Until supplied, retain a labeled
  check-valve box. Do not substitute a different equipment icon.
- No calculation, inventory assignment, model response or source topology was
  changed by this layout work.

## Verification

Nine layout/icon tests and 27 related diagram/topology tests passed. Coverage
includes liquid-only, gas-liquid, multistage and empty diagnostic topologies,
long labels, a gas-side check valve, and three/four/five incoming feeds.
An isolated temporary stand-in verifies the icon-loading path; it is not a
delivered check-valve asset.

All six saved KHU topologies were rendered into the separate directory
`outputs/topology_layout_review_20260914/`. Main icon center spread is 1 point
in each rendering; connections and original result-file hashes were checked.
Giese and DPDTC previews were visually inspected. This was a rendering-only
check, not an LLM rerun or browser GUI test. Figure 5 previews explicitly state
that reactor/light correction is pending and they are not for execution.

## Before the authorized reruns

1. Correct the assembly model/profile so only the dedicated Vapourtec reactor
   can use its UV-150 lights, while manual reactors use their permitted manual
   setup. Missing membership must not silently mean universal compatibility.
   Confirm the dedicated installed reactor configuration from source evidence;
   a maximum mounting capacity is not an installed reactor identity.
2. Load and visually verify the user's check-valve image.
3. Rerun Figure 5 sets 1, 2 and 3 with the corrected inventory, rechecking stage
   temperature, illumination, module quantities, flows, times and final exports.
   A light-label substitution cannot correct the previous numerical designs.

## Assembly code correction completed

`ReactorSpec.photoreactor_module_ids` explicitly records permitted modules.
The shared compatibility gate rejects a module-bound light for an unassigned
reactor, and rejects manual/UV-150 cross-pairings even when pump and volume
limits otherwise match. Explicit named legacy assembly matches remain supported.
The final allocator and intake prompt carry the same restriction.

`scripts/build_khu_inventory_v6.py` creates a separate v6 profile only from a
confirmed installed-reactor definition. If the UV-150 coil is part of the
listed manual stock, it marks that stock item as an integrated component to
prevent counting it twice. No installed UV-150 reactor is inferred from the
10 mL capacity. The runner now accepts `--inventory` for the new profile.

75 assembly, allocation, intake, council and layout tests passed. No model
design calls were started with an unconfirmed reactor definition. The user's
new check-valve PNG has been inspected; its direction and file location match
the renderer's expected input.
