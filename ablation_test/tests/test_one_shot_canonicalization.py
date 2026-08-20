from ablation_test.src.runner import _canonicalize_one_shot_proposal
from flora_translate.schemas import FlowProposal


def test_stage_keyed_one_shot_fields_canonicalize_without_losing_total():
    raw = {
        "residence_time_min": {"stage_1": 4.0, "stage_2": 20.0, "total": 24.0},
        "flow_rate_mL_min": 0.654,
        "temperature_C": {"stage_1": 70.0, "stage_2": 80.0},
        "concentration_M": {"A": 0.667, "B": 0.667},
        "BPR_bar": 0,
        "reactor_type": {"stage_1": "coil", "stage_2": "coil"},
        "tubing_material": "PFA",
        "tubing_ID_mm": 0.5,
        "reactor_volume_mL": {"stage_1": 1.96, "stage_2": 13.08, "total": 15.04},
        "stage_parameters": [
            {
                "stage": "1",
                "reactor_volume_mL": 1.96,
                "cumulative_flow_mL_min": 0.49,
                "residence_time_min": 4.0,
            },
            {
                "stage": "2",
                "reactor_volume_mL": 13.08,
                "cumulative_flow_mL_min": 0.654,
                "residence_time_min": 20.0,
            },
        ],
        "streams": [],
        "chemistry_notes": ["First", "Second"],
        "literature_analogies": "No held-out source used.",
    }

    normalized = _canonicalize_one_shot_proposal(raw)
    proposal = FlowProposal(**normalized)

    assert proposal.residence_time_min == 24.0
    assert proposal.reactor_volume_mL == 15.04
    assert proposal.temperature_C == 80.0
    assert proposal.chemistry_notes == "First\nSecond"
    assert proposal.literature_analogies == ["No held-out source used."]


def test_stream_equivalence_mapping_uses_limiting_stream_basis():
    raw = {
        "residence_time_min": 3,
        "flow_rate_mL_min": 0.1,
        "temperature_C": 130,
        "concentration_M": 0.125,
        "BPR_bar": 20,
        "reactor_type": "packed bed",
        "tubing_material": "stainless steel",
        "tubing_ID_mm": 4,
        "reactor_volume_mL": 0.3,
        "streams": [
            {
                "stream_label": "A",
                "contents": ["azide", "alkyne"],
                "concentration_M": {"azide": 0.125, "alkyne": 0.1375},
                "molar_equiv": {"azide": 1.0, "alkyne": 1.1},
            }
        ],
    }

    proposal = FlowProposal(**_canonicalize_one_shot_proposal(raw))

    assert proposal.streams[0].molar_equiv == 1.0
    assert proposal.streams[0].concentration_M == 0.1375


def test_common_direct_baseline_shapes_are_canonicalized():
    raw = {
        "residence_time_min": 3,
        "flow_rate_mL_min": 0.1,
        "reactor_volume_mL": 0.3,
        "pre_reactor_steps": "Prepare feed.",
        "post_reactor_steps": "Collect product.",
        "stage_parameters": {"stage_1": {"temperature_C": 60}},
        "confidence": 0.8,
        "streams": [{"stream_label": "A", "molar_equiv": "substrate 1.0 equiv"}],
    }
    proposal = FlowProposal(**_canonicalize_one_shot_proposal(raw))
    assert proposal.pre_reactor_steps == ["Prepare feed."]
    assert proposal.post_reactor_steps == ["Collect product."]
    assert proposal.stage_parameters[0]["stage_label"] == "stage_1"
    assert proposal.confidence == "HIGH"
    assert proposal.streams[0].molar_equiv == 1.0
