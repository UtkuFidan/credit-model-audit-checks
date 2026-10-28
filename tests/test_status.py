import math

from model_audit.status import Status, classify, worst


def test_psi_thresholds_inclusive():
    assert classify(0.0999, 0.10, 0.25) == Status.PASS
    assert classify(0.10, 0.10, 0.25) == Status.WATCH
    assert classify(0.2499, 0.10, 0.25) == Status.WATCH
    assert classify(0.25, 0.10, 0.25) == Status.FAIL


def test_lower_is_worse_for_p_values():
    assert classify(0.20, 0.05, 0.01, higher_is_worse=False) == Status.PASS
    assert classify(0.03, 0.05, 0.01, higher_is_worse=False) == Status.WATCH
    assert classify(0.005, 0.05, 0.01, higher_is_worse=False) == Status.FAIL


def test_strict_mode_for_zero_tolerance():
    assert classify(0.0, 0.0, 0.001, inclusive=False) == Status.PASS
    assert classify(0.0005, 0.0, 0.001, inclusive=False) == Status.WATCH
    assert classify(0.002, 0.0, 0.001, inclusive=False) == Status.FAIL


def test_nan_is_fail_and_worst():
    assert classify(math.nan, 0.1, 0.25) == Status.FAIL
    assert worst([Status.PASS, Status.WATCH]) == Status.WATCH
    assert worst([]) == Status.PASS
