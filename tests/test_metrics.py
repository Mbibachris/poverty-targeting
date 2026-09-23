import numpy as np
import pytest

from poverty_targeting.metrics import (
    calibration,
    discrimination,
    evaluate,
    select_by_budget,
    targeting_errors,
)

# Six households: the first three are poor.
Y = np.array([1, 1, 1, 0, 0, 0])


def test_targeting_errors_by_hand():
    selected = np.array([1, 1, 0, 1, 0, 0], dtype=bool)  # 2 poor reached, 1 poor missed
    e = targeting_errors(Y, selected)
    assert e["exclusion_error"] == pytest.approx(1 / 3)  # 1 of 3 poor missed
    assert e["inclusion_error"] == pytest.approx(1 / 3)  # 1 of 3 reached is not poor
    assert e["coverage_nonpoor"] == pytest.approx(1 / 3)
    assert e["targeting_differential"] == pytest.approx(2 / 3 - 1 / 3)
    assert e["selection_rate"] == pytest.approx(0.5)


def test_errors_are_population_weighted():
    # The missed poor household has 8 members: missing it misses 8 of 10 poor people.
    weights = np.array([1, 1, 8, 1, 1, 1])
    selected = np.array([1, 1, 0, 0, 0, 0], dtype=bool)
    assert targeting_errors(Y, selected, weights)["exclusion_error"] == pytest.approx(0.8)


def test_budget_selects_highest_scores_until_share_is_reached():
    scores = np.array([0.9, 0.2, 0.8, 0.7, 0.1, 0.3])
    assert select_by_budget(scores, share=0.5).tolist() == [True, False, True, True, False, False]
    weights = np.array([3, 1, 1, 1, 1, 1])  # the top household alone is 3/8 of people
    chosen = select_by_budget(scores, weights, share=0.4)
    assert chosen.tolist() == [True, False, True, False, False, False]


def test_perfect_ranking_has_auc_one():
    scores = np.array([0.9, 0.8, 0.7, 0.3, 0.2, 0.1])
    assert discrimination(Y, scores)["roc_auc"] == pytest.approx(1.0)
    assert discrimination(Y, scores)["pr_auc"] == pytest.approx(1.0)


def test_calibration_detects_overconfidence():
    rng = np.random.default_rng(0)
    p = rng.uniform(0, 1, 20_000)
    y = (rng.uniform(0, 1, 20_000) < p).astype(int)  # truly calibrated
    assert calibration(y, p)["ece"] < 0.02
    overconfident = np.clip(p * 1.6 - 0.3, 0, 1)
    assert calibration(y, overconfident)["ece"] > 0.05


def test_evaluate_defaults_budget_to_poverty_rate():
    scores = np.array([0.9, 0.8, 0.4, 0.6, 0.2, 0.1])
    result = evaluate(Y, scores)
    assert result["at_budget"]["budget_share"] == pytest.approx(0.5)
    assert result["at_budget"]["exclusion_error"] == pytest.approx(1 / 3)
    assert result["at_threshold"]["selection_rate"] == pytest.approx(0.5)
