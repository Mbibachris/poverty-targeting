import numpy as np
import pandas as pd

from poverty_targeting.dataset import TARGET
from poverty_targeting.final import budget_cutoffs, fit_final, model_card
from poverty_targeting.metrics import evaluate
from poverty_targeting.modeling import to_model_input
from poverty_targeting.serving import contributions, feature_of
from poverty_targeting.targeting import budget_curve

FEATURES = ["region", "hh_size", "has_bank_account"]


def synthetic(n=1500, seed=0):
    rng = np.random.default_rng(seed)
    region = rng.choice(["north", "south", "coast"], n)
    size = rng.integers(1, 10, n)
    logit = -4 + 3 * (region == "north") + 0.5 * size
    cluster = np.arange(n) // 10
    return pd.DataFrame(
        {
            "region": pd.array(region, dtype="string"),
            "hh_size": pd.array(size, dtype="Int64"),
            "has_bank_account": pd.array(rng.random(n) < 0.3, dtype="boolean"),
            TARGET: (rng.random(n) < 1 / (1 + np.exp(-logit))).astype("int8"),
            "pop_weight": rng.uniform(1, 5, n),
            "cluster": cluster,
            "fold": (cluster % 5).astype("int8"),
        }
    )


def test_transformed_columns_map_back_to_questions():
    features = ["has_tv", "has_tv_x", "head_sex", "hh_size"]
    assert feature_of("categorical__head_sex_female", features) == "head_sex"
    assert feature_of("numeric__hh_size", features) == "hh_size"
    assert feature_of("numeric__has_tv_x", features) == "has_tv_x"  # longest match wins


def test_final_fit_keeps_the_test_fold_out():
    _, train, test = fit_final(synthetic(), FEATURES, "lightgbm")
    assert (test["fold"] == 0).all()
    assert (train["fold"] != 0).all()


def test_explanations_add_up_to_the_prediction():
    pipeline, _, test = fit_final(synthetic(), FEATURES, "lightgbm")
    X = to_model_input(test, FEATURES)
    parts = contributions(pipeline, X, FEATURES)
    assert list(parts.columns) == [*FEATURES, "baseline"]
    rebuilt = 1 / (1 + np.exp(-parts.sum(axis=1)))
    np.testing.assert_allclose(rebuilt, pipeline.predict_proba(X)[:, 1], rtol=1e-6)


def test_bigger_budget_means_lower_cutoff():
    scores = np.random.default_rng(0).random(1000)
    cutoffs = budget_cutoffs(scores, np.ones(1000))
    values = [cutoffs[b] for b in sorted(cutoffs)]
    assert all(a >= b for a, b in zip(values, values[1:], strict=False))
    assert abs(cutoffs[0.5] - np.median(scores)) < 0.01  # half the budget: the median


def test_model_card_reports_flagged_groups_and_limits():
    y = np.array([1, 1, 0, 0, 0, 0])
    p = np.array([0.9, 0.2, 0.3, 0.1, 0.1, 0.1])
    test_eval = evaluate(y, p, budget_share=1 / 3)
    fairness = pd.DataFrame(
        [{"grouping": "residence", "group": "urban", "exclusion_error": 0.6,
          "exclusion_low": 0.5, "exclusion_high": 0.7, "flag_worse": True}]
    )  # fmt: skip
    meta = {
        "name": "XX_tierB_lightgbm", "spec_id": "global_mpi", "features": ["hh_size"],
        "questions": {"hh_size": "How many people live here?"}, "model": "lightgbm",
        "tier": "B", "survey_id": "XX2020DHS", "n_train": 100, "version": "0.1.0",
    }  # fmt: skip
    card = model_card(meta, test_eval, budget_curve(p, y, np.ones(6), (0.5,)), fairness)
    assert "## Intended use" in card and "Not** for" in card
    assert "residence = urban: exclusion 60.0%" in card
    assert "How many people live here?" in card
