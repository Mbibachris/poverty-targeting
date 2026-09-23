import numpy as np
import pandas as pd
import pytest

from poverty_targeting.dataset import TARGET
from poverty_targeting.modeling import MODELS, cross_validate, split_columns, to_model_input


def synthetic(n=1500, seed=0):
    """Poverty driven by region and household size; folds 0-4 by cluster."""
    rng = np.random.default_rng(seed)
    region = rng.choice(["north", "south", "coast"], n)
    size = rng.integers(1, 10, n)
    logit = -4 + 3 * (region == "north") + 0.5 * size
    poor = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype("int8")
    cluster = np.arange(n) // 10
    return pd.DataFrame(
        {
            "region": pd.array(region, dtype="string"),
            "hh_size": pd.array(size, dtype="Int64"),
            "has_bank_account": pd.array(rng.random(n) < 0.3, dtype="boolean"),
            TARGET: poor,
            "pop_weight": rng.uniform(1, 5, n),
            "cluster": cluster,
            "fold": (cluster % 5).astype("int8"),
        }
    )


FEATURES = ["region", "hh_size", "has_bank_account"]


def test_columns_are_split_by_type():
    categorical, numeric = split_columns(synthetic(), FEATURES)
    assert categorical == ["region"]
    assert numeric == ["hh_size", "has_bank_account"]


def test_model_input_turns_missing_values_into_nan_and_none():
    data = synthetic(n=3)
    data.loc[0, "hh_size"] = pd.NA
    data.loc[1, "has_bank_account"] = pd.NA
    data.loc[2, "region"] = pd.NA
    X = to_model_input(data, FEATURES)
    assert np.isnan(X.loc[0, "hh_size"])
    assert np.isnan(X.loc[1, "has_bank_account"])
    assert X.loc[2, "region"] is None
    assert X["has_bank_account"].dtype == float


def test_cross_validation_never_touches_the_test_fold():
    data = synthetic()
    result = cross_validate(data, FEATURES, "logit")
    assert len(result["predictions"]) == int((data["fold"] != 0).sum())
    assert sorted(result["per_fold"]) == [1, 2, 3, 4]
    assert not np.isnan(result["predictions"]).any()


@pytest.mark.parametrize("model_name", sorted(MODELS))
def test_every_model_learns_the_signal(model_name):
    result = cross_validate(synthetic(), FEATURES, model_name)
    assert result["out_of_fold"]["roc_auc"] > 0.75


@pytest.mark.parametrize("model_name", sorted(MODELS))
def test_serving_tolerates_unseen_categories_and_missing_answers(model_name):
    data = synthetic()
    categorical, numeric = split_columns(data, FEATURES)
    model = MODELS[model_name](categorical, numeric)
    model.fit(to_model_input(data, FEATURES), data[TARGET])
    new = pd.DataFrame(
        {
            "region": pd.array(["unknown_region"], dtype="string"),
            "hh_size": pd.array([pd.NA], dtype="Int64"),
            "has_bank_account": pd.array([True], dtype="boolean"),
        }
    )
    p = model.predict_proba(to_model_input(new, FEATURES))[:, 1]
    assert 0 < p[0] < 1
