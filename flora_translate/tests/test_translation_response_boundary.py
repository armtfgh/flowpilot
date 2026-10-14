"""Unexpected response envelopes must not silently become default designs."""
from copy import deepcopy
import json
from types import SimpleNamespace

import pytest

from flora_translate import translation_llm


def proposal():
    return {
        'flow_rate_mL_min': 0.5, 'reactor_volume_mL': 1.0,
        'residence_time_min': 2.0, 'temperature_C': 21,
        'concentration_M': 0.1, 'BPR_bar': 3,
        'reactor_type': 'coil', 'tubing_material': 'PFA',
        'wavelength_nm': 420, 'light_setup': '420 nm LED, 1.5 W',
        'streams': [{'stream_label': 'A', 'pump_role': 'substrate',
                     'contents': ['substrate, 0.1 M'], 'solvent': 'acetonitrile',
                     'phase': 'liquid', 'flow_rate_mL_min': 0.5,
                     'concentration_M': 0.1}],
    }


@pytest.mark.parametrize('raw', [None, [], {}, '', {'analysis': 'pending'},
                                  {'proposal': {}},
                                  {'proposal': {'flow_rate_mL_min': 0.1},
                                   'flow_rate_mL_min': 0.2}])
def test_bad_boundary_rejected_before_defaults_mutate_input(raw):
    before = deepcopy(raw)
    with pytest.raises(ValueError):
        translation_llm.TranslationLLM._normalize_proposal_data(raw)
    assert raw == before


def test_existing_flat_scientific_values_remain_unchanged():
    raw = proposal()
    result = translation_llm.TranslationLLM._normalize_proposal_data(deepcopy(raw))
    for key in ('flow_rate_mL_min', 'temperature_C', 'wavelength_nm',
                'light_setup', 'concentration_M', 'BPR_bar'):
        assert result[key] == raw[key]
    assert len(result['streams']) == len(raw['streams'])
    for actual, original in zip(result['streams'], raw['streams']):
        for key, value in original.items():
            assert actual[key] == value


def test_wrapper_is_retried_as_flat_proposal_without_losing_irradiation(monkeypatch):
    raw = proposal()
    answers = iter([{'recommended_disposition': 'SCREEN', 'proposal': raw}, raw])
    calls = []

    def fake(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(text=json.dumps(next(answers)))

    monkeypatch.setattr(translation_llm, 'call_model_text', fake)
    result = translation_llm.TranslationLLM().generate('Return a flat proposal.', 'Batch protocol.')
    assert [call['api_name'] for call in calls] == ['translation_llm', 'translation_llm_retry']
    assert result.wavelength_nm == 420
    assert result.temperature_C == 21
    assert result.flow_rate_mL_min == 0.5
    assert result.streams[0].solvent == 'acetonitrile'


@pytest.mark.parametrize('bad', [
    {'recommended_disposition': 'BLOCK', 'disposition_rationale': 'No feasible apparatus',
     'proposal': {'flow_rate_mL_min': 0.1}},
    {'analysis': 'not a proposal'}, {},
])
def test_repeated_malformed_responses_raise_after_bounded_repair(monkeypatch, bad):
    calls = []

    def fake(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(text=json.dumps(bad))

    monkeypatch.setattr(translation_llm, 'call_model_text', fake)
    with pytest.raises(ValueError):
        translation_llm.TranslationLLM().generate('Return a flat proposal.', 'Batch protocol.')
    assert [call['api_name'] for call in calls] == [
        'translation_llm', 'translation_llm_retry', 'translation_llm_repair']
    assert json.dumps(bad) in calls[-1]['user_content']
