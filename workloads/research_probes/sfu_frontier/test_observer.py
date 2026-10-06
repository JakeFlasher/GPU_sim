import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location("sfu_observer", Path(__file__).with_name("observer.py"))
observer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(observer)


def row(blocks, cycles):
    return {"blocks": blocks, "cycles": cycles}


def data():
    fit = [row(42, [100, 101, 102]), row(105, [200, 201, 202])]
    calibration = [row(b, [101 + 100 * (b > 84) + i])
                   for i, b in enumerate([21, 35, 56, 70, 77, 91, 119, 140, 154])]
    return fit, calibration


def test_split_radius_counts_workloads_and_retains_open_physical_upper():
    result = observer.predict(*data(), 210, 84)
    assert result["point_cycles"] == 301
    assert result["interval_calibration_rank"] == 9
    assert result["radius_cycles"] == 8
    assert result["interval_upper_cycles"] == 309
    assert result["extrapolates_fit_feature_range"]
    assert result["semantic_completion_upper"] is None
    assert not result["coverage_guarantee_for_designed_extrapolation"]


def test_repeating_one_shape_does_not_increase_interval_sample_size():
    fit, calibration = data()
    calibration = [row(calibration[0]["blocks"], [101] * 100)]
    result = observer.predict(fit, calibration, 210, 84)
    assert result["interval_calibration_shape_count"] == 1
    assert result["radius_cycles"] is None
    assert result["interval_upper_cycles"] is None


def test_interval_labels_never_refit_the_point():
    fit, calibration = data()
    first = observer.predict(fit, calibration, 210, 84)
    second = observer.predict(fit, [row(r["blocks"], [900]) for r in calibration], 210, 84)
    assert first["point_cycles"] == second["point_cycles"]
    assert first["radius_cycles"] != second["radius_cycles"]


@pytest.mark.parametrize("target", [42, 105, 21])
def test_refuses_shape_overlap(target):
    with pytest.raises(ValueError, match="disjoint"):
        observer.predict(*data(), target, 84)


def test_refuses_overlapping_fit_and_interval_shapes():
    fit, calibration = data()
    with pytest.raises(ValueError, match="disjoint"):
        observer.predict(fit, calibration + [row(42, [99])], 210, 84)


@pytest.mark.parametrize("bad", [True, -1, 0, float("nan"), float("inf")])
def test_refuses_invalid_cycles(bad):
    fit, calibration = data()
    fit[0]["cycles"] = [bad]
    with pytest.raises(ValueError):
        observer.predict(fit, calibration, 210, 84)


def test_refuses_rank_deficient_fit():
    with pytest.raises(ValueError, match="distinct"):
        observer.predict([row(21, [1]), row(42, [2])], [row(126, [3])], 210, 84)


def test_integer_ceiling_feature_does_not_claim_physical_occupancy():
    fit, calibration = data()
    result = observer.predict(fit, calibration, 168, 84)
    assert result["target_feature"] == 2
    assert "not_hardware_waves" in result["feature"]
    assert not result["extrapolates_fit_feature_range"]


def test_wave_feature_rejects_rank_deficiency_and_keeps_semantic_upper_open():
    fit = [row(42, [100]), row(63, [110]), row(105, [200]), row(126, [210])]
    interval = [row(21, [98]), row(35, [105]), row(77, [120]), row(119, [205])]
    result = observer.predict_wave(fit, interval, 210, 84)
    assert result["target_feature"] == 3
    assert result["semantic_completion_upper"] is None
    assert result["coverage_guarantee_for_designed_extrapolation"] is False
    with pytest.raises(ValueError, match="rank deficient"):
        observer.predict_wave([row(42, [100]), row(63, [110])], interval, 210, 84)
