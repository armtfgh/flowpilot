import pytest
from ablation_test.scripts import run_astra_verified_pilot as verified


def test_worker_configuration_matches_six_cell_plan():
    verified.configure()
    verified.pilot.evaluation.configure()
    cells = verified.pilot.campaign.campaign_cells(verified.pilot.campaign.selected_cases())
    assert len(cells) == 6
    assert {r["model_id"] for r in cells} == {"gpt-6-astra"}
    assert {r["candidate_budget"] for r in cells} == {2}
    assert verified.pilot.campaign.SEED_NAMESPACE.endswith("v3")


def test_frozen_source_verification_rejects_changed_code(monkeypatch, tmp_path):
    monkeypatch.setattr(verified, "ROOT", tmp_path)
    source = tmp_path / "flora_translate/core.py"
    source.parent.mkdir()
    source.write_text("original\n")
    monkeypatch.setattr(verified, "source_files", lambda: iter([source]))
    output = tmp_path / "campaign"
    verified.pilot.write(output / "frozen/full_source_manifest.json", {
        "flora_translate/core.py": verified.hashlib.sha256(source.read_bytes()).hexdigest()})
    verified.verify_source(output)
    source.write_text("changed\n")
    with pytest.raises(RuntimeError, match="changed after freeze"):
        verified.verify_source(output)
