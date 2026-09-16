from ablation_test.src.cases import load_cases


def test_literature_suite_has_twelve_unique_non_forbidden_cases():
    cases = load_cases()
    assert len(cases) == 12
    assert len({case.case_id for case in cases}) == 12


def test_public_payload_does_not_contain_hidden_reference():
    for case in load_cases():
        public = case.public_payload()
        assert "reference_flow" not in public
        assert "source_record_id" not in public
        assert "expected_features" not in public


def test_every_literature_case_defines_leave_one_out_source():
    for case in load_cases():
        assert case.source_record_id
        assert case.excluded_record_ids

