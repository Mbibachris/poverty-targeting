"""Shared test fixtures: a small model trained on synthetic data, saved like `final` does.

CI has no real model (the .joblib file is git-ignored), so tests that need one train
this tiny stand-in once per test session.
"""

import joblib
import numpy as np
import pandas as pd
import pytest

from poverty_targeting.dataset import TARGET
from poverty_targeting.feature_table import COMPUTED_FEATURES, add_computed_features
from poverty_targeting.features import load_feature_set
from poverty_targeting.final import budget_cutoffs
from poverty_targeting.modeling import MODELS, split_columns, to_model_input

FEATURES = [
    "region", "urban", "hh_size", "head_age", "n_under5", "n_children_5_17",
    "n_elderly_65plus", "sleeping_rooms", "has_bank_account",
    "dependent_share", "persons_per_room",
]  # fmt: skip


def training_data(n=1200, seed=0):
    rng = np.random.default_rng(seed)
    size = rng.integers(1, 9, n)
    under5 = np.minimum(rng.integers(0, 3, n), size - 1)
    region = rng.choice(["north", "south", "coast"], n)
    head_age = rng.integers(20, 80, n).astype(float)
    head_age[:30] = np.nan  # blanks occur in training -> head_age is optional
    logit = -3 + 2.5 * (region == "north") + 0.4 * size - 1.5 * (rng.random(n) < 0.3)
    df = pd.DataFrame(
        {
            "region": pd.array(region, dtype="string"),
            "urban": pd.array(rng.random(n) < 0.4, dtype="boolean"),
            "hh_size": pd.array(size, dtype="Int64"),
            "head_age": pd.array(head_age, dtype="Int64"),
            "n_under5": pd.array(under5, dtype="Int64"),
            "n_children_5_17": pd.array(np.zeros(n, int), dtype="Int64"),
            "n_elderly_65plus": pd.array(np.zeros(n, int), dtype="Int64"),
            "sleeping_rooms": pd.array(rng.integers(1, 4, n), dtype="Int64"),
            "has_bank_account": pd.array(rng.random(n) < 0.3, dtype="boolean"),
        }
    )
    df = add_computed_features(df)
    df[TARGET] = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype("int8")
    return df


@pytest.fixture(scope="session")
def bundle_path(tmp_path_factory):
    data = training_data()
    categorical, numeric = split_columns(data, FEATURES)
    pipeline = MODELS["lightgbm"](categorical, numeric)
    X = to_model_input(data, FEATURES)
    pipeline.fit(X, data[TARGET])
    fs = load_feature_set("pmt")
    meta = {
        "name": "TEST_tierB_lightgbm",
        "version": "0.0.0-test",
        "trained_at": "2026-01-01T00:00:00+00:00",
        "survey_id": "XX2020DHS",
        "tier": "B",
        "feature_set": "pmt",
        "features": FEATURES,
        "questions": {f: fs.features[f].question for f in FEATURES},
        "optional_answers": [
            f for f in FEATURES if f not in COMPUTED_FEATURES and data[f].isna().any()
        ],
        "default_budget": 0.25,
        "budget_cutoffs": budget_cutoffs(pipeline.predict_proba(X)[:, 1], np.ones(len(data))),
    }
    path = tmp_path_factory.mktemp("model") / "bundle.joblib"
    joblib.dump({"pipeline": pipeline, "meta": meta}, path)
    return path


@pytest.fixture
def household():
    """A valid set of answers for the test model (a fresh copy for each test)."""
    return {
        "region": "north", "urban": False, "hh_size": 7, "head_age": 50, "n_under5": 2,
        "n_children_5_17": 0, "n_elderly_65plus": 0, "sleeping_rooms": 1,
        "has_bank_account": False,
    }  # fmt: skip
