from ui_logic import (
    answer_text,
    grouped,
    likelihood_band,
    number_bounds,
    option_label,
    reason_sentence,
    verdict,
)


def test_option_labels_are_readable():
    assert option_label("lpg") == "LPG (cooking gas)"
    assert option_label("unprotected_well") == "Unprotected well"  # tidied automatically
    assert option_label("Greater Accra") == "Greater Accra"  # region names untouched


def test_questions_are_grouped_in_api_order_with_a_catch_all():
    questions = [{"name": n} for n in ["urban", "region", "hh_size", "brand_new_question"]]
    sections = grouped(questions)
    assert [q["name"] for q in sections["Location"]] == ["region", "urban"]
    assert [q["name"] for q in sections["Other"]] == ["brand_new_question"]
    assert "Dwelling" not in sections  # empty sections are dropped


def test_number_bounds_come_from_the_api_schema():
    schema = {
        "components": {
            "schemas": {
                "Household": {
                    "properties": {
                        "hh_size": {"type": "integer", "minimum": 1.0, "maximum": 60.0},
                        "head_age": {
                            "anyOf": [
                                {"type": "integer", "minimum": 10.0, "maximum": 110.0},
                                {"type": "null"},
                            ]
                        },
                    }
                }
            }
        }
    }
    assert number_bounds(schema, "hh_size") == (1, 60)
    assert number_bounds(schema, "head_age") == (10, 110)  # optional answer


def test_answers_read_naturally():
    assert answer_text(True) == "yes"
    assert answer_text(0.0) == "0"
    assert answer_text(None) == "not known"
    assert answer_text("wood") == "Wood"


def test_reason_sentence_states_direction_and_strength():
    reason = {
        "question": "Main fuel used for cooking",
        "answer": "lpg",
        "effect": -1.215,
        "direction": "lowers",
    }
    sentence = reason_sentence(reason)
    assert "LPG (cooking gas)" in sentence
    assert "strongly lowers" in sentence


def test_likelihood_bands_and_verdict():
    assert likelihood_band(0.98) == "Very likely to be poor"
    assert likelihood_band(0.35) == "Uncertain"
    assert likelihood_band(0.003) == "Very unlikely to be poor"
    assert verdict({"selected": True, "budget": 0.25}).startswith("Would be selected")
    assert "covering 25%" in verdict({"selected": False, "budget": 0.25})
