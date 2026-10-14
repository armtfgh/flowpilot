"""No-inference regression tests for final-output instruction scoping."""
from dataclasses import replace
from copy import deepcopy

import pytest

from ablation_test.scripts import run_fair_practical_benchmark as fair


@pytest.fixture(autouse=True)
def preserve_runner(monkeypatch):
    # configure is intentionally process-wide in the command-line harness;
    # tests must restore those references after each isolated check.
    monkeypatch.setattr(fair.runner,'_production_full',fair.runner._production_full)
    monkeypatch.setattr(fair.runner,'_general_one_shot',fair.runner._general_one_shot)


def test_all_scientific_requirements_survive_without_final_envelope_syntax():
    assert fair.SERIALIZATION_PREFIX+fair.SCIENTIFIC_OUTPUT_GUIDANCE==fair.SHARED_OUTPUT
    for exact in (
        'Distinguish inlet/STP gas flow from\npressure-corrected in-channel gas flow.',
        'Ensure reactor_volume_mL is consistent\nwith flow and residence time.',
        'State assumptions rather than inventing missing\nfacts.',
        'recommended_disposition must be exactly EXECUTE, SCREEN, or BLOCK.',
        'Use BLOCK\nwhen the hard constraints make a safe design physically or operationally\nimpossible.',
        'Use SCREEN for a feasible first experiment that still requires\nexperimental validation.',
        'Use EXECUTE only for a directly executable design\nsupported by sufficient evidence.',
    ):
        assert exact in fair.FLOWPILOT_INTERNAL_OBJECTIVE
    for serialization in ('one JSON object','The proposal uses these fields:','Each stream is an object with'):
        assert serialization not in fair.FLOWPILOT_INTERNAL_OBJECTIVE


def test_three_cases_keep_public_contract_and_every_nonobjective_field(monkeypatch):
    records=[];calls=[]
    class Recorder:
        def write_json(self,name,value):records.append((name,value))
    def native(case,recorder,budget,temperature,seed,execution):
        calls.append((case,budget,temperature,seed,execution));return {'native_result':True}
    monkeypatch.setattr(fair,'NORMAL_PRODUCTION',native)
    for _,case in fair.fair_cases():
        before=deepcopy(case.manifest_payload())
        result=fair._production_with_scoped_output(case,Recorder(),2,.2,123,'execution')
        assert result=={'native_result':True}
        assert calls[-1][0]==replace(case,objective=fair.FLOWPILOT_INTERNAL_OBJECTIVE)
        assert calls[-1][1:]==(2,.2,123,'execution')
        assert case.manifest_payload()==before
        assert fair.SHARED_OUTPUT in case.public_payload()['objective']
        assert records[-1][1]['public_objective']==case.objective
        assert records[-1][1]['retained_guidance_exact']==fair.SCIENTIFIC_OUTPUT_GUIDANCE
    assert len(calls)==3


def test_unrecognized_objective_is_rejected_without_losing_user_text(monkeypatch):
    case=fair.fair_cases()[0][1]
    monkeypatch.setattr(fair,'NORMAL_PRODUCTION',lambda *a:pytest.fail('Unrecognized objective was routed'))
    with pytest.raises(ValueError,match='exact fair-benchmark objective'):
        fair._production_with_scoped_output(replace(case,objective=case.objective+'\nAn additional real constraint.'),None,2,.2,123)


def test_configure_twice_calls_native_once_without_nested_wrappers(monkeypatch):
    calls=[]
    class Recorder:
        def write_json(self,*a):pass
    monkeypatch.setattr(fair,'NORMAL_PRODUCTION',lambda *args:(calls.append(args) or {'ok':True}))
    fair.configure();first=fair.runner._production_full
    fair.configure();second=fair.runner._production_full
    assert first is second is fair._production_with_scoped_output
    result=second(fair.fair_cases()[0][1],Recorder(),2,.2,123)
    assert result=={'ok':True} and len(calls)==1
    assert calls[0][0].objective==fair.FLOWPILOT_INTERNAL_OBJECTIVE
    assert fair.runner._general_one_shot is fair.NORMAL_ONE_SHOT


def test_reload_capture_marker_points_to_unwrapped_native():
    # The module import captures this marker rather than the installed wrapper;
    # this prevents a reload from replacing native dispatch with itself.
    native=fair._production_with_scoped_output._fair_native_production
    assert native is fair.NORMAL_PRODUCTION
    assert native is not fair._production_with_scoped_output
    assert getattr(fair._production_with_scoped_output,'_fair_native_production') is native
