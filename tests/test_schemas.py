import pytest
from pydantic import ValidationError

from poverty_targeting.api.schemas import build_household_model, build_request_models
from poverty_targeting.serving import load_bundle


@pytest.fixture
def Household(bundle_path):
    return build_household_model(load_bundle(bundle_path))


def test_valid_household_passes(Household, household):
    assert Household(**household).hh_size == 7


def test_category_must_be_a_trained_option(Household, household):
    with pytest.raises(ValidationError, match="north"):  # the error lists valid options
        Household(**{**household, "region": "atlantis"})


def test_numbers_are_bounded_whole_numbers(Household, household):
    with pytest.raises(ValidationError):
        Household(**{**household, "hh_size": 0})
    with pytest.raises(ValidationError):
        Household(**{**household, "head_age": 300})
    with pytest.raises(ValidationError):
        Household(**{**household, "n_under5": 1.5})


def test_required_vs_optional_answers(Household, household):
    assert Household(**{**household, "head_age": None}).head_age is None  # optional
    del household["hh_size"]
    with pytest.raises(ValidationError, match="hh_size"):
        Household(**household)


def test_unknown_and_computed_fields_are_rejected(Household, household):
    with pytest.raises(ValidationError, match="favourite_colour"):
        Household(**{**household, "favourite_colour": "blue"})
    with pytest.raises(ValidationError, match="dependent_share"):
        Household(**{**household, "dependent_share": 0.5})


def test_age_counts_cannot_exceed_household_size(Household, household):
    with pytest.raises(ValidationError, match="exceed household size"):
        Household(**{**household, "hh_size": 2, "n_under5": 3})


def test_request_options_are_checked(Household, household):
    Single, Batch = build_request_models(Household)
    assert Single(household=household).top_k == 3
    with pytest.raises(ValidationError):
        Single(household=household, budget=1.5)
    with pytest.raises(ValidationError):
        Batch(households=[])
    assert len(Batch(households=[household, household]).households) == 2


def test_questions_become_documented_fields(Household):
    schema = Household.model_json_schema()
    assert schema["properties"]["hh_size"]["description"] == (
        "How many people live in this household?"
    )
    assert set(schema["properties"]["region"]["enum"]) == {"coast", "north", "south"}
    assert "hh_size" in schema["required"] and "head_age" not in schema["required"]
