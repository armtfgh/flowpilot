"""Replay archived inputs as software tests; never invoke or rescore a model."""
from pathlib import Path
import pytest
from ablation_test.scripts.verify_review_pipeline_offline import verify_case

ROOT = Path(__file__).resolve().parents[2]
CAMPAIGN = ROOT / "ablation_results/manuscript_benchmark/astra_inventory_repair_pilot_20261012"


@pytest.mark.parametrize("case", ["cuaac", "hydrogenolysis", "photochemical_oxidation"])
@pytest.mark.parametrize("scenario", ["unchanged", "accept_revision", "reject_revision", "missing_revision_approval"])
def test_review_to_delivery_on_archived_inputs(case, scenario):
    path = CAMPAIGN / "generation" / case / "astra_flowpilot/repeat_01/result.json"
    if not path.exists():
        pytest.skip("Optional immutable campaign archive is not present")
    assert verify_case(path, scenario)["passed"]
