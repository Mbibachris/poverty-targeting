import pandas as pd
import pytest

from poverty_targeting import config
from poverty_targeting.feature_table import (
    add_computed_features,
    build_feature_table,
    roster_answers,
)
from poverty_targeting.features import FeatureSetError, load_feature_set

FS = load_feature_set("pmt")


def persons():
    return pd.DataFrame(
        {
            "hh_id": ["a", "a", "a", "a", "b"],
            "is_head": pd.array([True, False, False, False, True], dtype="boolean"),
            "age_years": pd.array([40, 3, 10, 70, 80], dtype="Int64"),
            "years_schooling": pd.array([9, 0, 4, 2, None], dtype="Int64"),
        }
    )


def households():
    hh = {
        "hh_id": ["a", "b"],
        "region": pd.array(["North", "South"], dtype="string"),
        "urban": pd.array([False, True], dtype="boolean"),
        "hh_size": pd.array([4, 1], dtype="Int64"),
        "head_sex": pd.array(["female", "male"], dtype="string"),
        "head_age": pd.array([40, 80], dtype="Int64"),
        "sleeping_rooms": pd.array([2, 1], dtype="Int64"),
    }
    for f in FS.features.values():  # every other direct feature: a simple placeholder
        column = f.source.partition(".")[2]
        if f.source != "derived" and column not in hh:
            hh[column] = pd.array([True, False], dtype="boolean")
    return pd.DataFrame(hh)


def test_roster_answers_count_by_age_and_read_head_schooling():
    answers = roster_answers(pd.Index(["a", "b"]), persons())
    assert answers["n_under5"].tolist() == [1, 0]
    assert answers["n_children_5_17"].tolist() == [1, 0]
    assert answers["n_elderly_65plus"].tolist() == [1, 1]
    assert answers["head_years_schooling"].tolist()[0] == 9
    assert pd.isna(answers.loc["b", "head_years_schooling"])  # unknown stays unknown


def test_computed_features():
    answers = pd.DataFrame(
        {
            "hh_size": pd.array([4, 1, 3], dtype="Int64"),
            "n_under5": pd.array([1, 0, 0], dtype="Int64"),
            "n_children_5_17": pd.array([1, 0, 0], dtype="Int64"),
            "n_elderly_65plus": pd.array([0, 1, 0], dtype="Int64"),
            "sleeping_rooms": pd.array([2, 1, 0], dtype="Int64"),
        }
    )
    result = add_computed_features(answers)
    assert result["dependent_share"].tolist() == [0.5, 1.0, 0.0]  # elderly alone -> defined
    assert result["persons_per_room"].tolist()[:2] == [2.0, 1.0]
    assert pd.isna(result.loc[2, "persons_per_room"])  # zero rooms -> unknown, not infinite


def test_feature_table_follows_tier_order():
    tables = {"households": households(), "persons": persons()}
    table_a = build_feature_table(tables, FS, "A")
    assert list(table_a.columns) == ["hh_id", *FS.tiers["A"]]
    table_b = build_feature_table(tables, FS, "B")
    assert "head_years_schooling" in table_b and "head_years_schooling" not in table_a
    assert table_a["dependent_share"].tolist() == [0.75, 1.0]


def test_training_and_serving_compute_the_same_features():
    # The API will call add_computed_features on one form's answers: same result as training.
    tables = {"households": households(), "persons": persons()}
    trained = build_feature_table(tables, FS, "A").set_index("hh_id").loc["a"]
    form = pd.DataFrame(
        {
            "hh_size": pd.array([4], dtype="Int64"),
            "n_under5": pd.array([1], dtype="Int64"),
            "n_children_5_17": pd.array([1], dtype="Int64"),
            "n_elderly_65plus": pd.array([1], dtype="Int64"),
            "sleeping_rooms": pd.array([2], dtype="Int64"),
        }
    )
    served = add_computed_features(form).iloc[0]
    assert served["dependent_share"] == trained["dependent_share"]
    assert served["persons_per_room"] == trained["persons_per_room"]


def test_derived_feature_without_implementation_is_caught(tmp_path):
    text = (config.FEATURE_SET_DIR / "pmt.toml").read_text()
    text = text.replace('"owns_livestock",\n]', '"owns_livestock", "mystery_index",\n]', 1)
    text += '\n[features.mystery_index]\nsource = "derived"\ninputs = ["persons.age_years"]\n'
    (tmp_path / "test_set.toml").write_text(text)
    fs = load_feature_set("test_set", set_dir=tmp_path)
    with pytest.raises(FeatureSetError, match="mystery_index"):
        build_feature_table({"households": households(), "persons": persons()}, fs, "A")
