"""HT calibrated-ratio estimator: census exactness, calibration abort, rates."""
import math

import pytest

from estimator import (
    EstimatorInputError,
    RateTarget,
    Segment,
    Stratum,
    estimate,
)


def _seg(i, stratum, sectors, hits, records):
    return Segment(id=i, stratum=stratum,
                   measured={"sectors": sectors, "hits": hits}, record_count=records)


def test_census_is_exact_and_calibration_holds():
    # Full census of one stratum (n_h == N_h): HT == exact sum, cal factor 1.
    segs = {i: _seg(i, 0, 10.0, 8.0, 100.0) for i in range(8)}
    strata = [Stratum(id=0, N_h=8, sampled_ids=list(range(8)))]
    r = estimate(segs, strata, certainty_ids=[], x_true=800.0,
                 count_targets=["sectors", "hits"],
                 rate_targets=[RateTarget("hit_rate", "hits", "sectors")], seed=1)
    assert r.calibration_ok
    assert abs(r.point["sectors"] - 80.0) < 1e-9
    assert abs(r.point["hit_rate"] - 0.8) < 1e-9
    assert r.status == "converged"
    assert "cycles" in r.not_recoverable


def test_subsample_scales_up_by_inclusion_probability():
    # Sample 2 of 4 identical segments -> HT scales by N_h/n_h = 2.
    segs = {i: _seg(i, 0, 10.0, 8.0, 100.0) for i in range(2)}
    strata = [Stratum(id=0, N_h=4, sampled_ids=[0, 1])]
    r = estimate(segs, strata, certainty_ids=[], x_true=400.0,
                 count_targets=["sectors"], rate_targets=[], seed=2)
    assert abs(r.point["sectors"] - 40.0) < 1e-9  # (4/2)*(10+10)
    assert r.calibration_ok  # X̂ = (4/2)*200 = 400 == x_true


def test_calibration_abort_when_xhat_misses_true():
    # x_true inconsistent with the sampled record counts -> abort (unavailable).
    segs = {i: _seg(i, 0, 10.0, 8.0, 100.0) for i in range(4)}
    strata = [Stratum(id=0, N_h=4, sampled_ids=[0, 1, 2, 3])]
    r = estimate(segs, strata, certainty_ids=[], x_true=1000.0,
                 count_targets=["sectors"], rate_targets=[], seed=3)
    assert not r.calibration_ok
    assert r.status == "unavailable"


def test_certainty_segments_are_not_scaled():
    # A certainty segment contributes its raw total; a sampled stratum scales.
    segs = {0: _seg(0, -1, 5.0, 5.0, 50.0),
            1: _seg(1, 0, 10.0, 8.0, 100.0), 2: _seg(2, 0, 10.0, 8.0, 100.0)}
    strata = [Stratum(id=0, N_h=2, sampled_ids=[1, 2])]
    r = estimate(segs, strata, certainty_ids=[0], x_true=250.0,
                 count_targets=["sectors"], rate_targets=[], seed=4)
    # 5 (certainty) + (2/2)*(10+10) = 25 ; X̂ = 50 + 200 = 250 == x_true
    assert abs(r.point["sectors"] - 25.0) < 1e-9
    assert r.calibration_ok


def test_multiple_strata_use_their_own_inclusion_probabilities():
    segs = {
        0: _seg(0, -1, 5.0, 4.0, 50.0),
        1: _seg(1, 1, 10.0, 8.0, 100.0),
        2: _seg(2, 1, 10.0, 8.0, 100.0),
        3: _seg(3, 2, 20.0, 10.0, 200.0),
        4: _seg(4, 2, 20.0, 10.0, 200.0),
    }
    strata = [Stratum(1, 4, [1, 2]), Stratum(2, 6, [3, 4])]
    # X-hat = 50 + (4/2)*200 + (6/2)*400 = 1650 exactly.
    result = estimate(
        segs, strata, certainty_ids=[0], x_true=1650.0,
        count_targets=["sectors", "hits"], rate_targets=[], seed=5)
    assert result.x_hat == pytest.approx(1650.0)
    assert result.point["sectors"] == pytest.approx(165.0)
    assert result.point["hits"] == pytest.approx(96.0)


def test_empty_or_underfilled_strata_are_typed_refusals():
    segs = {i: _seg(i, 0, 10.0, 8.0, 100.0) for i in range(8)}
    with pytest.raises(EstimatorInputError) as empty:
        estimate(
            segs, [Stratum(0, 8, list(range(8))), Stratum(1, 100, [])],
            [], 800.0, ["sectors"], [], seed=0)
    assert empty.value.reason == "stratum_sample_below_floor"

    with pytest.raises(EstimatorInputError) as singleton:
        estimate(
            {0: segs[0]}, [Stratum(0, 2, [0])], [], 100.0,
            ["sectors"], [], seed=0)
    assert singleton.value.reason == "stratum_sample_below_floor"


def test_stratum_population_and_membership_must_match_exactly():
    segs = {
        0: _seg(0, 0, 10.0, 8.0, 100.0),
        1: _seg(1, 0, 10.0, 8.0, 100.0),
    }
    with pytest.raises(EstimatorInputError) as too_many:
        estimate(
            segs, [Stratum(0, 1, [0, 1])], [], 200.0,
            ["sectors"], [], seed=0)
    assert too_many.value.reason == "stratum_sample_exceeds_population"

    wrong = dict(segs)
    wrong[1] = _seg(1, 7, 10.0, 8.0, 100.0)
    with pytest.raises(EstimatorInputError) as mismatch:
        estimate(
            wrong, [Stratum(0, 2, [0, 1])], [], 200.0,
            ["sectors"], [], seed=0)
    assert mismatch.value.reason == "sampled_segment_stratum_mismatch"

    extra = dict(segs)
    extra[2] = _seg(2, 0, 10.0, 8.0, 100.0)
    with pytest.raises(EstimatorInputError) as unaccounted:
        estimate(
            extra, [Stratum(0, 3, [0, 1])], [], 300.0,
            ["sectors"], [], seed=0)
    assert unaccounted.value.reason == "unaccounted_sample_segments"


def test_certainty_and_ordinary_memberships_are_unique_and_disjoint():
    segs = {
        0: _seg(0, -1, 5.0, 4.0, 50.0),
        1: _seg(1, 0, 10.0, 8.0, 100.0),
        2: _seg(2, 0, 10.0, 8.0, 100.0),
    }
    with pytest.raises(EstimatorInputError) as duplicate_certainty:
        estimate(
            segs, [Stratum(0, 2, [1, 2])], [0, 0], 250.0,
            ["sectors"], [], seed=0)
    assert duplicate_certainty.value.reason == "duplicate_certainty_segment"

    with pytest.raises(EstimatorInputError) as overlap:
        estimate(
            segs, [Stratum(0, 3, [0, 1])], [0], 250.0,
            ["sectors"], [], seed=0)
    assert overlap.value.reason == "certainty_ordinary_overlap"

    ordinary = {
        0: _seg(0, 0, 10.0, 8.0, 100.0),
        1: _seg(1, 0, 10.0, 8.0, 100.0),
        2: _seg(2, 1, 10.0, 8.0, 100.0),
    }
    with pytest.raises(EstimatorInputError) as repeated:
        estimate(
            ordinary,
            [Stratum(0, 2, [0, 1]), Stratum(1, 2, [1, 2])],
            [], 300.0, ["sectors"], [], seed=0)
    assert repeated.value.reason == "segment_in_multiple_strata"


@pytest.mark.parametrize("value", [-1.0, math.inf, math.nan])
def test_nonfinite_or_negative_records_and_targets_refuse(value):
    bad_records = {
        0: _seg(0, 0, 10.0, 8.0, value),
        1: _seg(1, 0, 10.0, 8.0, 100.0),
    }
    with pytest.raises(EstimatorInputError) as record_error:
        estimate(
            bad_records, [Stratum(0, 2, [0, 1])], [], 200.0,
            ["sectors"], [], seed=0)
    assert record_error.value.reason == "invalid_segment_record_count"

    bad_target = {
        0: _seg(0, 0, value, 8.0, 100.0),
        1: _seg(1, 0, 10.0, 8.0, 100.0),
    }
    with pytest.raises(EstimatorInputError) as target_error:
        estimate(
            bad_target, [Stratum(0, 2, [0, 1])], [], 200.0,
            ["sectors"], [], seed=0)
    assert target_error.value.reason == "invalid_segment_target"


def test_required_target_must_exist_on_every_segment():
    segs = {
        0: Segment(0, 0, {}, 100.0),
        1: _seg(1, 0, 10.0, 8.0, 100.0),
    }
    with pytest.raises(EstimatorInputError) as exc_info:
        estimate(
            segs, [Stratum(0, 2, [0, 1])], [], 200.0,
            ["sectors"], [], seed=0)
    assert exc_info.value.reason == "missing_segment_target"


@pytest.mark.parametrize("x_true", [-1.0, math.inf, math.nan])
def test_exact_record_total_must_be_finite_and_nonnegative(x_true):
    segs = {
        0: _seg(0, 0, 10.0, 8.0, 100.0),
        1: _seg(1, 0, 10.0, 8.0, 100.0),
    }
    with pytest.raises(EstimatorInputError) as exc_info:
        estimate(
            segs, [Stratum(0, 2, [0, 1])], [], x_true,
            ["sectors"], [], seed=0)
    assert exc_info.value.reason == "invalid_exact_record_total"


def test_census_stratum_is_fixed_in_bootstrap():
    # Heterogeneous census totals have zero design uncertainty; resampling the
    # observed census would invent a wide interval.
    segs = {
        i: _seg(i, 0, float(i + 1), float(i), 100.0)
        for i in range(8)
    }
    result = estimate(
        segs, [Stratum(0, 8, list(range(8)))], [], 800.0,
        ["sectors"], [], seed=0)
    assert result.point["sectors"] == 36.0
    assert result.ci["sectors"] == (36.0, 36.0)
    assert result.status == "converged"
