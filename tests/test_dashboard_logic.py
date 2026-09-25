import pytest
from ui_logic import (
    answer_text,
    curve_at,
    grouped,
    likelihood_band,
    number_bounds,
    option_label,
    parse_households,
    reason_sentence,
    results_rows,
    template_rows,
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


# --- budget explorer and many households --------------------------------------------

CURVE = [
    {"budget": 0.10, "coverage_poor": 0.30, "inclusion_error": 0.20, "random_coverage_poor": 0.10},
    {"budget": 0.20, "coverage_poor": 0.50, "inclusion_error": 0.30, "random_coverage_poor": 0.20},
]
QUESTIONS = [
    {"name": "region", "kind": "category", "options": ["North", "South"], "required": True},
    {"name": "electricity", "kind": "yes_no", "options": None, "required": True},
    {"name": "hh_size", "kind": "number", "options": None, "required": True},
    {"name": "head_age", "kind": "number", "options": None, "required": False},
]


def test_curve_is_interpolated_between_measured_budgets():
    point = curve_at(CURVE, 0.15)
    assert point["coverage_poor"] == pytest.approx(0.40)
    assert point["inclusion_error"] == pytest.approx(0.25)
    assert curve_at(CURVE, 0.50)["coverage_poor"] == pytest.approx(0.50)  # clamped to range


def test_template_has_one_column_per_question():
    [row] = template_rows(QUESTIONS)
    assert list(row) == ["region", "electricity", "hh_size", "head_age"]
    assert row["region"] == "North" and row["electricity"] == "no" and row["head_age"] == ""


def test_spreadsheet_rows_become_households():
    rows = [{"region": "North", "electricity": "Yes", "hh_size": "4.0", "head_age": ""}]
    households, problems = parse_households(rows, QUESTIONS)
    assert problems == []
    assert households == [{"region": "North", "electricity": True, "hh_size": 4}]


def test_spreadsheet_problems_name_the_row_and_column():
    rows = [{"region": "North", "electricity": "maybe", "hh_size": "", "head_age": "x"}]
    _, problems = parse_households(rows, QUESTIONS)
    assert "Row 2: 'electricity' should be yes or no, got 'maybe'" in problems
    assert "Row 2: 'hh_size' is empty" in problems
    assert "Row 2: 'head_age' should be a number, got 'x'" in problems


def test_results_rows_are_flat():
    prediction = {
        "probability": 0.8,
        "selected": True,
        "reasons": [{"question": "Main fuel used for cooking", "direction": "raises"}],
    }
    assert results_rows([prediction]) == [
        {
            "likelihood_poor": 0.8,
            "selected": "yes",
            "main_reason": "Main fuel used for cooking",
            "main_reason_direction": "raises",
        }
    ]
