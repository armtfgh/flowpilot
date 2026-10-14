import numpy as np

from ablation_test.scripts.coworker_report import within_tolerance


def test_relative_repeatability_tolerance():
    passed, units = within_tolerance([10.0, 10.5, 9.8], "relative", 0.10)
    assert passed is True
    assert units < 1.0


def test_relative_repeatability_detects_unstable_design():
    passed, units = within_tolerance([10.0, 15.0, 20.0], "relative", 0.10)
    assert passed is False
    assert units > 1.0


def test_absolute_repeatability_tolerance():
    passed, units = within_tolerance([39.0, 40.0, 41.0], "absolute", 2.0)
    assert passed is True
    assert np.isclose(units, 1.0)
