import numpy as np
import pytest
from poverty_targeting.targeting import (
    budget_curve,
    calibration_check,
    choose_calibration,
    cross_fit_calibration,
)


def data(n=8000, seed=0, distortion=1.0):
    """Outcomes drawn from true probabilities; scores are a distorted version of them."""
    rng = np.random.default_rng(seed)
    truth = rng.uniform(0.02, 0.9, n)
    y = (rng.random(n) < truth).astype(int)
    scores = truth**distortion  # distortion 1 = perfectly calibrated
    return scores, y, np.ones(n), rng.integers(1, 5, n)


def test_calibration_helps_miscalibrated_scores():
    p, y, w, folds = data(distortion=3.0)  # badly underconfident
    check = calibration_check(p, y, w, folds)
    assert check["platt"]["brier"] < check["none"]["brier"] - 0.005
    assert choose_calibration(check) != "none"


def test_calibrated_scores_are_left_alone():
    p, y, w, folds = data(distortion=1.0)
    assert choose_calibration(calibration_check(p, y, w, folds)) == "none"


def test_cross_fitting_never_uses_a_fold_to_calibrate_itself():
    p, y, w, folds = data(n=400)
    q = cross_fit_calibration(p, y, w, folds, "isotonic")
    y_flipped = y.copy()
    y_flipped[folds == 1] = 1 - y_flipped[folds == 1]  # change fold 1's outcomes only
    q2 = cross_fit_calibration(p, y_flipped, w, folds, "isotonic")
    np.testing.assert_allclose(q[folds == 1], q2[folds == 1])  # fold 1 unaffected


def test_unknown_method_is_rejected():
    p, y, w, folds = data(n=100)
    with pytest.raises(ValueError, match="unknown calibration"):
        cross_fit_calibration(p, y, w, folds, "magic")


def test_budget_curve_against_random_selection():
    y = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
    perfect = np.array([0.9, 0.8, 0.3, 0.2, 0.2, 0.1, 0.1, 0.1, 0.1, 0.1])
    curve = budget_curve(perfect, y, np.ones(10), budgets=(0.2, 0.5))
    first = curve.iloc[0]
    assert first["coverage_poor"] == pytest.approx(1.0)  # both poor reached
    assert first["inclusion_error"] == pytest.approx(0.0)
    assert first["random_coverage_poor"] == pytest.approx(0.2)
    assert first["random_inclusion_error"] == pytest.approx(0.8)
    assert curve["score_cutoff"].is_monotonic_decreasing
