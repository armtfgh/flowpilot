import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

from flora_translate.input_parser import InputParser
from flora_translate.intake_agent import batch_input_from_package, intake_context_block
from flora_translate.lightweight_upstream import LocalInputParser, EvidenceBackedInputParser
from flora_translate.main import _inventory_from_intake_or_path, _resolve_design_inputs
from flora_translate.protocol_input import extract_protocol_text, protocol_batch_input
from flora_translate.schemas import DesignInputPackage


PROTOCOL = "A solution of N-benzylamine in ethanol was treated with H2 and Pd/C for 2 h."
OBJECTIVE = "Suggest a first practical flow experiment."
INVENTORY = {
    "reactors": [{"name": "hydrogenation skid", "type": "coil", "material": "SS", "volume_mL": 10, "ID_mm": 1}],
    "pumps": [], "tubing": [], "light_sources": [], "BPR_available": [6],
}


def wrapped(inventory=INVENTORY):
    return f"BATCH PROCEDURE\n{PROTOCOL}\n\nREQUEST\n{OBJECTIVE}\n\nAVAILABLE EQUIPMENT\n{json.dumps(inventory)}"


@pytest.mark.parametrize("parser,call_module", [
    (InputParser, "flora_translate.input_parser"),
    (LocalInputParser, "flora_translate.lightweight_upstream"),
    (EvidenceBackedInputParser, "flora_translate.lightweight_upstream"),
])
def test_all_parsers_extract_only_original_protocol(parser, call_module, monkeypatch):
    calls = []

    def fake_call(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(text=json.dumps({"reaction_description": "MODEL SUMMARY", "raw_text": "MODEL INVENTION"}))

    monkeypatch.setattr(f"{call_module}.call_model_text", fake_call)
    record = parser().parse(wrapped())
    assert record.raw_text == PROTOCOL
    assert PROTOCOL in calls[0]["user_content"]
    assert OBJECTIVE not in calls[0]["user_content"]
    assert "hydrogenation skid" not in calls[0]["user_content"]


@pytest.mark.parametrize("name", ["hydrogenation skid", "CuAAC machine", "oxidation module", "amidation rack"])
def test_equipment_names_do_not_change_protocol_and_context_reaches_design(name):
    inventory = deepcopy(INVENTORY)
    inventory["reactors"][0]["name"] = name
    payload, intake = _resolve_design_inputs(wrapped(inventory))
    assert payload == PROTOCOL
    assert intake.objective == OBJECTIVE
    assert intake.inventory_constraints == inventory
    assert intake.chemistry_identity_confirmation == {}
    assert not intake.answers
    assert OBJECTIVE in intake_context_block(intake)
    assert name in intake_context_block(intake)
    assert _inventory_from_intake_or_path(intake, "does-not-exist.json").reactors[0].name == name


def test_existing_benchmark_envelope_retains_explicit_context():
    text = (
        f"RAW BATCH PROTOCOL:\n{PROTOCOL}\n\nOBJECTIVE:\n{OBJECTIVE}\n\n"
        "HARD CONSTRAINTS:\n- Maximum temperature 50 C.\n\n"
        f"AVAILABLE INVENTORY (authoritative JSON):\n{json.dumps(INVENTORY)}\n\n"
        'CHEMIST-CONFIRMED CHEMISTRY IDENTITY:\n{"transformation_family": "hydrogenolysis"}'
    )
    payload, intake = _resolve_design_inputs(text)
    assert payload == PROTOCOL
    assert intake.operating_limits == "- Maximum temperature 50 C."
    assert intake.chemistry_identity_confirmation == {"transformation_family": "hydrogenolysis"}


def test_intake_original_protocol_overrides_model_extracted_raw_text():
    package = DesignInputPackage(
        raw_protocol=PROTOCOL,
        extracted_batch_fields={"reaction_description": "MODEL SUMMARY", "raw_text": "MODEL INVENTION", "solvent": "ethanol"},
        objective=OBJECTIVE,
        inventory_constraints=INVENTORY,
        ready_for_design=True,
    )
    original = package.model_dump()
    for value in (package, original):
        fields = batch_input_from_package(value)
        assert fields["raw_text"] == PROTOCOL
        assert fields["reaction_description"] == "MODEL SUMMARY"
        assert fields["solvent"] == "ethanol"
        assert InputParser().parse(fields).raw_text == PROTOCOL
    assert package.model_dump() == original
    for value in (original, json.dumps(original)):
        payload, intake = _resolve_design_inputs(value)
        assert payload["raw_text"] == PROTOCOL
        assert intake.objective == OBJECTIVE
        assert intake.inventory_constraints == INVENTORY
        assert extract_protocol_text(value) == PROTOCOL


def test_explicit_not_ready_package_still_blocks_before_calls():
    package = DesignInputPackage(raw_protocol=PROTOCOL, ready_for_design=False, missing_question_ids=["Q-OBJ-001"])
    with pytest.raises(ValueError, match="Q-OBJ-001"):
        _resolve_design_inputs(package.model_dump())


@pytest.mark.parametrize("json_encode", [False, True])
@pytest.mark.parametrize("protocol_fields", [
    {"raw_protocol": PROTOCOL},
    {"extracted_batch_fields": {"reaction_description": PROTOCOL, "solvent": "ethanol"}},
])
def test_structured_envelope_aliases_keep_context(protocol_fields, json_encode):
    data = {**deepcopy(protocol_fields), "request": OBJECTIVE, "inventory": deepcopy(INVENTORY),
            "hard_constraints": ["Maximum temperature 50 C."]}
    original = deepcopy(data)
    value = json.dumps(data) if json_encode else data
    payload, intake = _resolve_design_inputs(value)
    assert intake.objective == OBJECTIVE
    assert intake.inventory_constraints == INVENTORY
    assert intake.operating_limits == ["Maximum temperature 50 C."]
    assert "hydrogenation skid" not in extract_protocol_text(payload)
    assert data == original
    data["ready_for_design"] = False
    with pytest.raises(ValueError, match="not ready"):
        _resolve_design_inputs(json.dumps(data) if json_encode else data)


def test_structured_aliases_do_not_override_canonical_fields():
    data = {"raw_protocol": PROTOCOL, "request": "alias objective", "objective": OBJECTIVE,
            "inventory": {"incorrect": True}, "inventory_constraints": INVENTORY,
            "hard_constraints": ["alias constraint"], "operating_limits": []}
    _, intake = _resolve_design_inputs(data)
    assert intake.objective == OBJECTIVE
    assert intake.inventory_constraints == INVENTORY
    assert intake.operating_limits == []


@pytest.mark.parametrize("key,value,canonical", [
    ("hard_constraints", ["Temperature below 35 C."], "operating_limits"),
    ("operating_limits", {"max_temperature_C": 35}, "operating_limits"),
    ("historical_data", "Prior measured yield was 41%.", "historical_data"),
    ("hypotheses", ["Light delivery may limit conversion."], "hypotheses"),
])
@pytest.mark.parametrize("json_encode", [False, True])
def test_context_without_objective_or_inventory_is_not_lost(key, value, canonical, json_encode):
    data = {"reaction_description": PROTOCOL, "raw_text": PROTOCOL, key: value}
    payload, intake = _resolve_design_inputs(json.dumps(data) if json_encode else data)
    assert payload["raw_text"] == PROTOCOL
    assert getattr(intake, canonical) == value
    assert key not in payload


@pytest.mark.parametrize("json_encode", [False, True])
def test_mixed_protocol_and_typed_batch_fields_are_preserved(json_encode):
    data = {"raw_protocol": PROTOCOL, "reaction_time_h": 4, "solvent": "ethanol",
            "objective": OBJECTIVE, "inventory_constraints": INVENTORY}
    original = deepcopy(data)
    value = json.dumps(data) if json_encode else data
    payload, intake = _resolve_design_inputs(value)
    assert payload["reaction_time_h"] == 4
    assert payload["solvent"] == "ethanol"
    assert payload["raw_text"] == PROTOCOL
    assert intake.inventory_constraints == INVENTORY
    # Direct parser callers receive the same structured batch values.
    record = InputParser().parse(value)
    assert record.reaction_time_h == 4
    assert record.solvent == "ethanol"
    assert record.raw_text == PROTOCOL
    assert data == original


def test_schema_metadata_alone_does_not_create_an_intake_envelope():
    data = {"schema_version": "batch_v1", "reaction_description": PROTOCOL, "solvent": "ethanol"}
    payload, intake = _resolve_design_inputs(data)
    assert intake is None
    assert payload["solvent"] == "ethanol"


@pytest.mark.parametrize("json_encode", [False, True])
def test_structured_batch_with_context_preserves_fields_without_inventory_evidence(json_encode):
    fields = {"reaction_description": PROTOCOL, "solvent": "ethanol", "reaction_time_h": 2,
              "objective": OBJECTIVE, "inventory": INVENTORY}
    original = deepcopy(fields)
    value = json.dumps(fields) if json_encode else fields
    payload, intake = _resolve_design_inputs(value)
    record = InputParser().parse(payload)
    assert record.solvent == "ethanol"
    assert record.reaction_time_h == 2
    assert "hydrogenation skid" not in record.raw_text
    assert OBJECTIVE not in record.raw_text
    assert PROTOCOL in record.raw_text
    assert intake.inventory_constraints == INVENTORY
    assert fields == original


def test_ordinary_text_and_structured_batch_record_remain_supported():
    text = "A substrate was combined with B. The request was recorded in the laboratory notebook."
    assert protocol_batch_input(text) == text
    assert _resolve_design_inputs(text) == (text, None)
    fields = {"reaction_description": PROTOCOL, "solvent": "ethanol", "reaction_time_h": 2}
    payload, intake = _resolve_design_inputs(fields)
    assert intake is None
    assert InputParser().parse(payload).reaction_time_h == 2
    assert json.loads(extract_protocol_text(fields)) == fields


def test_saved_practical_inputs_retain_only_the_frozen_protocol():
    root = Path(__file__).resolve().parents[2]
    paths = sorted((root / "ablation_results/practical_input/claude_practical_pilot_20261014/frozen/practical_inputs").glob("*/public_input.json"))
    if not paths:
        pytest.skip("Archived development campaign not installed")
    assert len(paths) == 3
    for path in paths:
        data = json.loads(path.read_text())
        text = (path.parent / "request.txt").read_text()
        payload, intake = _resolve_design_inputs(text)
        # The archive is read-only; these are the supplied facts, not a model summary.
        assert payload == data["protocol"]
        assert intake.inventory_constraints == data["inventory"]
        assert intake.objective == data["objective"]
