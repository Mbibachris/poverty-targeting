import dataclasses

import numpy as np
import pandas as pd
import pytest

from poverty_targeting import indicators as ind
from poverty_targeting.mpi_spec import load_mpi_spec

SPEC = load_mpi_spec("global_mpi")
NA = np.nan


def households(**changes):
    """Four households a-d; keyword arguments replace whole columns."""
    data = {
        "hh_id": ["a", "b", "c", "d"],
        "anthro_selected": [True, True, False, True],
        "cooking_fuel": ["wood", "lpg", "no_cooking", None],
        "electricity": [False, True, True, None],
        "sanitation": ["pit_with_slab", "pit_with_slab", "no_facility", "flush_septic"],
        "toilet_shared": [True, False, None, False],
        "water_source": ["sachet", "tubewell_borehole", "surface_water", "tubewell_borehole"],
        "water_time_min": [10.0, 40.0, 5.0, None],
        "floor": ["finished", "natural", "finished", "finished"],
        "wall": ["finished", "finished", "other", "finished"],
        "roof": ["finished", "finished", "finished", None],
        "has_radio": [True, True, False, False],
        "has_tv": [True, False, False, None],
        "has_telephone": [False, False, False, False],
        "has_computer": [False, False, False, False],
        "has_animal_cart": [False, False, False, False],
        "has_bicycle": [False, False, False, False],
        "has_motorbike": [False, False, False, False],
        "has_refrigerator": [False, False, False, False],
        "has_car_truck": [False, False, True, False],
    }
    data.update(changes)
    df = pd.DataFrame(data).set_index("hh_id", drop=False)
    for col in ["cooking_fuel", "sanitation", "water_source", "floor", "wall", "roof"]:
        df[col] = df[col].astype("string")
    for col in [c for c in df if c.startswith("has_")] + ["anthro_selected", "electricity"]:
        df[col] = df[col].astype("boolean")
    df["toilet_shared"] = df["toilet_shared"].astype("boolean")
    df["water_time_min"] = df["water_time_min"].astype("Float64")
    return df


def persons(rows):
    """rows: list of dicts; unspecified fields are missing / sensible defaults."""
    defaults = {
        "usual_resident": True, "age_years": 30, "age_months": None,
        "height_for_age_z": None, "weight_for_age_z": None, "bmi": None,
        "pregnant": None, "years_schooling": 8, "attended_school": None,
    }  # fmt: skip
    df = pd.DataFrame([{**defaults, **r} for r in rows])
    types = {
        "usual_resident": "boolean", "age_years": "Int64", "age_months": "Int64",
        "height_for_age_z": "Float64", "weight_for_age_z": "Float64", "bmi": "Float64",
        "pregnant": "boolean", "years_schooling": "Int64", "attended_school": "boolean",
    }  # fmt: skip
    return df.astype(types)


def with_nutrition(**rules):
    new = {**SPEC.indicators, "nutrition": {**SPEC.indicators["nutrition"], **rules}}
    return dataclasses.replace(SPEC, indicators=new)


def values(series):
    return series.to_numpy(dtype=float, na_value=np.nan).tolist()


def assert_same(series, expected):
    np.testing.assert_array_equal(values(series), expected)


# --- health -------------------------------------------------------------------------


def test_nutrition_rules():
    people = persons(
        [
            {"hh_id": "a", "age_years": 2, "age_months": 30, "height_for_age_z": -2.5},  # stunted
            {"hh_id": "b", "age_years": 25, "bmi": 17.0, "pregnant": True},  # pregnant: skip
            {"hh_id": "c", "age_years": 30, "bmi": 16.0},  # never measured household
            {"hh_id": "d", "age_years": 17, "bmi": 17.0},  # thin adolescent
        ]
    )
    result = ind.nutrition(households(), people, SPEC)
    assert_same(result, [1, 0, NA, 0])  # adolescent excluded by default


def test_adolescent_adult_cutoff_bound():
    people = persons([{"hh_id": "d", "age_years": 17, "bmi": 17.0}])
    result = ind.nutrition(households(), people, with_nutrition(adolescents_15_19="adult_cutoff"))
    assert_same(result, [0, 0, NA, 1])


def test_nutrition_without_any_measurements_is_refused():
    with pytest.raises(ind.IndicatorError, match="no nutrition information"):
        ind.nutrition(households(), persons([{"hh_id": "a"}]), SPEC)


def test_child_mortality_window_and_age():
    deaths = pd.DataFrame(
        {
            "hh_id": ["a", "b"],
            "age_at_death_months": pd.array([3, 3], dtype="Int64"),
            "months_since_death": pd.array([20, 70], dtype="Int64"),  # b: too long ago
        }
    )
    assert_same(ind.child_mortality(households(), deaths, SPEC), [1, 0, 0, 0])


# --- education ----------------------------------------------------------------------


def test_years_schooling():
    people = persons(
        [
            {"hh_id": "a", "years_schooling": 6},  # reached 6 -> not deprived
            {"hh_id": "b", "years_schooling": 3},  # nobody reached 6 -> deprived
            {"hh_id": "c", "years_schooling": 3},
            {"hh_id": "c", "years_schooling": None},  # the unknown member might have -> NA
            {"hh_id": "d", "years_schooling": 2},
            {"hh_id": "d", "years_schooling": 12, "usual_resident": False},  # visitor ignored
        ]
    )
    assert_same(ind.years_schooling(households(), people, SPEC), [0, 1, NA, 1])


def test_school_attendance():
    people = persons(
        [
            {"hh_id": "a", "age_years": 8, "attended_school": False},  # school-aged, out
            {"hh_id": "b", "age_years": 15, "attended_school": False},  # beyond class-8 age
            {"hh_id": "c", "age_years": 10, "attended_school": True},
        ]
    )
    assert_same(ind.school_attendance(households(), people, SPEC), [1, 0, 0, 0])


# --- living standards ---------------------------------------------------------------


def test_living_standards():
    hh = households()
    assert_same(ind.cooking_fuel(hh, SPEC), [1, 0, 0, NA])
    assert_same(ind.electricity(hh, SPEC), [1, 0, 0, NA])
    assert_same(ind.sanitation(hh, SPEC), [1, 0, 1, 0])  # a: improved but shared
    assert_same(ind.drinking_water(hh, SPEC), [0, 1, 1, NA])  # b: 40 min; d: time unknown
    assert_same(ind.housing(hh, SPEC), [0, 1, 0, NA])  # c: "other" walls are adequate
    assert_same(ind.assets(hh, SPEC), [0, 1, 0, NA])  # c: car; d: TV unknown


def test_build_indicators_follows_spec_order():
    tables = {
        "households": households().reset_index(drop=True),
        "persons": persons([{"hh_id": "a", "age_years": 30, "bmi": 22.0}]),
        "child_deaths": pd.DataFrame(
            {
                "hh_id": pd.Series([], dtype="string"),
                "age_at_death_months": pd.Series([], dtype="Int64"),
                "months_since_death": pd.Series([], dtype="Int64"),
            }
        ),
    }
    result = ind.build_indicators(tables, SPEC)
    names, _ = SPEC.weights()
    assert list(result.columns) == ["hh_id", *names]
    assert len(result) == 4
