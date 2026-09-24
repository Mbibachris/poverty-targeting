import joblib
import pandas as pd
import pytest

from poverty_targeting.feature_table import COMPUTED_FEATURES, add_computed_features
from poverty_targeting.modeling import to_model_input
from poverty_targeting.serving import ScoringError, load_bundle, score


def test_questions_describe_the_form(bundle_path):
    questions = {q.name: q for q in load_bundle(bundle_path).questions}
    assert not set(COMPUTED_FEATURES) & set(questions)  # computed, never asked
    assert questions["region"].kind == "category"
    assert questions["region"].options == ("coast", "north", "south")
    assert questions["urban"].kind == "yes_no"
    assert questions["hh_size"].kind == "number"
    assert not questions["head_age"].required  # was blank in training
    assert questions["hh_size"].required


def test_score_gives_probability_selection_and_reasons(bundle_path, household):
    [result] = score(load_bundle(bundle_path), [household], budget=0.3, top_k=2)
    assert 0 < result["probability"] < 1
    assert result["selected"] == (result["probability"] >= result["score_cutoff"])
    assert len(result["reasons"]) == 2
    for reason in result["reasons"]:
        assert reason["direction"] == ("raises" if reason["effect"] > 0 else "lowers")


def test_serving_matches_training_computation(bundle_path, household):
    bundle = load_bundle(bundle_path)
    [result] = score(bundle, [household])
    frame = add_computed_features(pd.DataFrame([household]).convert_dtypes())
    expected = bundle.pipeline.predict_proba(to_model_input(frame, bundle.features))[:, 1][0]
    assert result["probability"] == pytest.approx(expected, abs=1e-4)


def test_blank_optional_answer_is_allowed_and_reported(bundle_path, household):
    [result] = score(load_bundle(bundle_path), [{**household, "head_age": None}])
    assert result["missing_answers"] == ["head_age"]


def test_blank_required_answer_is_refused(bundle_path, household):
    del household["hh_size"]
    with pytest.raises(ScoringError, match="missing required answer 'hh_size'"):
        score(load_bundle(bundle_path), [household])


def test_bad_answers_are_refused_with_reasons(bundle_path, household):
    bundle = load_bundle(bundle_path)
    with pytest.raises(ScoringError, match="not one of"):
        score(bundle, [{**household, "region": "atlantis"}])
    with pytest.raises(ScoringError, match="unknown answer 'favourite_colour'"):
        score(bundle, [{**household, "favourite_colour": "blue"}])
    with pytest.raises(ScoringError, match="computed by the model"):
        score(bundle, [{**household, "dependent_share": 0.9}])


def test_budget_limits_and_monotone_selection(bundle_path, household):
    bundle = load_bundle(bundle_path)
    with pytest.raises(ScoringError, match="budget"):
        score(bundle, [household], budget=1.5)
    small = score(bundle, [household], budget=0.05)[0]
    large = score(bundle, [household], budget=0.95)[0]
    assert large["score_cutoff"] <= small["score_cutoff"]
    assert large["selected"] or not small["selected"]  # a bigger budget never drops anyone


def test_wrong_file_is_rejected(tmp_path):
    path = tmp_path / "not_a_model.joblib"
    joblib.dump({"hello": "world"}, path)
    with pytest.raises(ScoringError, match="not a model"):
        load_bundle(path)
