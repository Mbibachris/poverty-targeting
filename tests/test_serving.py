import joblib
import numpy as np
import pandas as pd
import pytest

from poverty_targeting.dataset import TARGET
from poverty_targeting.feature_table import COMPUTED_FEATURES, add_computed_features
from poverty_targeting.features import load_feature_set
from poverty_targeting.final import budget_cutoffs
from poverty_targeting.modeling import MODELS, split_columns, to_model_input
from poverty_targeting.serving import ScoringError, load_bundle, score

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


@pytest.fixture(scope="module")
def bundle_path(tmp_path_factory):
    data = training_data()
    categorical, numeric = split_columns(data, FEATURES)
    pipeline = MODELS["lightgbm"](categorical, numeric)
    X = to_model_input(data, FEATURES)
    pipeline.fit(X, data[TARGET])
    fs = load_feature_set("pmt")
    meta = {
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


HOUSEHOLD = {
    "region": "north", "urban": False, "hh_size": 7, "head_age": 50, "n_under5": 2,
    "n_children_5_17": 0, "n_elderly_65plus": 0, "sleeping_rooms": 1, "has_bank_account": False,
}  # fmt: skip


def test_questions_describe_the_form(bundle_path):
    questions = {q.name: q for q in load_bundle(bundle_path).questions}
    assert not set(COMPUTED_FEATURES) & set(questions)  # computed, never asked
    assert questions["region"].kind == "category"
    assert questions["region"].options == ("coast", "north", "south")
    assert questions["urban"].kind == "yes_no"
    assert questions["hh_size"].kind == "number"
    assert not questions["head_age"].required  # was blank in training
    assert questions["hh_size"].required


def test_score_gives_probability_selection_and_reasons(bundle_path):
    [result] = score(load_bundle(bundle_path), [HOUSEHOLD], budget=0.3, top_k=2)
    assert 0 < result["probability"] < 1
    assert result["selected"] == (result["probability"] >= result["score_cutoff"])
    assert len(result["reasons"]) == 2
    for reason in result["reasons"]:
        assert reason["direction"] == ("raises" if reason["effect"] > 0 else "lowers")


def test_serving_matches_training_computation(bundle_path):
    bundle = load_bundle(bundle_path)
    [result] = score(bundle, [HOUSEHOLD])
    frame = add_computed_features(pd.DataFrame([HOUSEHOLD]).convert_dtypes())
    expected = bundle.pipeline.predict_proba(to_model_input(frame, FEATURES))[:, 1][0]
    assert result["probability"] == pytest.approx(expected, abs=1e-4)


def test_blank_optional_answer_is_allowed_and_reported(bundle_path):
    [result] = score(load_bundle(bundle_path), [{**HOUSEHOLD, "head_age": None}])
    assert result["missing_answers"] == ["head_age"]


def test_blank_required_answer_is_refused(bundle_path):
    answers = {k: v for k, v in HOUSEHOLD.items() if k != "hh_size"}
    with pytest.raises(ScoringError, match="missing required answer 'hh_size'"):
        score(load_bundle(bundle_path), [answers])


def test_bad_answers_are_refused_with_reasons(bundle_path):
    bundle = load_bundle(bundle_path)
    with pytest.raises(ScoringError, match="not one of"):
        score(bundle, [{**HOUSEHOLD, "region": "atlantis"}])
    with pytest.raises(ScoringError, match="unknown answer 'favourite_colour'"):
        score(bundle, [{**HOUSEHOLD, "favourite_colour": "blue"}])
    with pytest.raises(ScoringError, match="computed by the model"):
        score(bundle, [{**HOUSEHOLD, "dependent_share": 0.9}])


def test_budget_limits_and_monotone_selection(bundle_path):
    bundle = load_bundle(bundle_path)
    with pytest.raises(ScoringError, match="budget"):
        score(bundle, [HOUSEHOLD], budget=1.5)
    small = score(bundle, [HOUSEHOLD], budget=0.05)[0]
    large = score(bundle, [HOUSEHOLD], budget=0.95)[0]
    assert large["score_cutoff"] <= small["score_cutoff"]
    assert large["selected"] or not small["selected"]  # a bigger budget never drops anyone


def test_wrong_file_is_rejected(tmp_path):
    path = tmp_path / "not_a_model.joblib"
    joblib.dump({"hello": "world"}, path)
    with pytest.raises(ScoringError, match="not a model"):
        load_bundle(path)
