# Canonical Contract v2 Frozen Replay

This replay evaluates the six frozen FlowPilot outputs with the new
post-realization semantic release contract. It does not regenerate model
answers and does not alter the original benchmark files.

## Result

- Original inventory/intake packages executable under v2: 0/6
- Executable after explicit chemistry confirmation and declared safety accessories: 3/6
- Remaining model-output defects: 3

The original outputs are correctly blocked because chemistry identity was
not an explicit authority field and required safety accessories were absent
from inventory. After supplying those missing authorities, the candidates
listed below remain blocked for genuine output defects:

- `N2-E9B5A89D` (CuAAC): `FINAL-MIXING-REGIME-TOPOLOGY-CONFLICT`
- `N2-D3A6858C` (Hydrogenolysis): `FINAL-CHEMISTRY-IDENTITY-DRIFT`
- `N2-EE130514` (Two-stage amidation): `FINAL-COMPONENT-STOICHIOMETRY-INCOMPLETE`

## Interpretation

The release contract separates missing user/laboratory authority from
model mistakes. It no longer accepts a model confidence score or a
numerically closed proposal as sufficient evidence of executability.
Topology phase is normalized before rendering, and safety, procedure,
validation experiments, and their canonical SHA-256 are compiled only
after final realization.
