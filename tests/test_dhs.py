import dataclasses

import pandas as pd
import pytest

from poverty_targeting.dhs import (
    UnmappedCodeError,
    build_child_deaths,
    child_deaths_from_raw,
    households_from_raw,
    persons_from_raw,
)
from poverty_targeting.readers import RawTable
from poverty_targeting.registry import SurveySpec

SPEC = SurveySpec(
    survey_id="XX2020DHS",
    country="Testland",
    iso3="TST",
    program="DHS",
    phase=8,
    year=2020,
    files={"household": "hh.zip", "persons": "pr.zip"},
    anthropometry_subsample_var="hv042",
    code_overrides={"water_source": {72: "sachet"}},
)


def make_raw(**changes):
    """Three synthetic DHS households; keyword arguments replace whole columns."""
    data = {
        "hhid": ["   1  1", "   1  2", "   2  1"],
        "hv001": [1, 1, 2],
        "hv005": [1_000_000, 1_500_000, 500_000],
        "hv024": [1, 1, 2],
        "hv025": [1, 2, 2],
        "hv042": [1, 0, 1],
        "hv009": [4, 2, 7],
        "hv219": [1, 2, 1],
        "hv220": [45, 98, 60],
        "hv201": [11, 72, 43],
        "hv204": [996, 20, 998],
        "hv205": [12, 23, 31],
        "hv225": [0.0, 1.0, float("nan")],
        "hv206": [1, 0, 0],
        "hv226": [3, 10, 11],
        "hv213": [34, 11, 21],
        "hv214": [31, 22, 96],
        "hv215": [31, 31, 12],
        "hv207": [1, 0, 1],
        "hv208": [1, 0, 0],
        "hv243a": [1, 0, 1],
        "hv221": [0, 1, 0],
        "hv243e": [1, 0, 0],
        "hv243c": [0, 0, 1],
        "hv210": [0, 1, 1],
        "hv211": [1, 0, 0],
        "hv209": [1, 0, 0],
        "hv212": [1, 0, 0],
        "hv247": [1, 0, 0],
        "hv244": [0, 1, 1],
        "hv246": [0, 1, 1],
        "hv216": [2, 1, 3],
    }
    data.update(changes)
    labels = {
        "hv024": {1: "north", 2: "south east"},
        "hv201": {11: "piped into dwelling", 72: "sachet water", 43: "river", 99: "missing"},
    }
    return RawTable(data=pd.DataFrame(data), value_labels=labels, variable_labels={})


def test_builds_valid_canonical_table():
    hh = households_from_raw(make_raw(), SPEC)
    assert hh.shape == (3, 33)
    assert hh["hh_id"].tolist() == ["1  1", "1  2", "2  1"]
    assert hh["weight"].tolist() == [1.0, 1.5, 0.5]
    assert hh["region"].tolist() == ["North", "North", "South East"]
    assert hh["urban"].tolist() == [True, False, False]


def test_country_override_is_applied():
    hh = households_from_raw(make_raw(), SPEC)
    assert hh["water_source"].tolist() == ["piped_dwelling", "sachet", "surface_water"]


def test_unknown_code_fails_and_names_code_and_label():
    spec_without_override = dataclasses.replace(SPEC, code_overrides={})
    with pytest.raises(UnmappedCodeError, match=r"72 \('sachet water'\)"):
        households_from_raw(make_raw(), spec_without_override)


def test_all_unknown_codes_reported_together():
    raw = make_raw(hv226=[3, 77, 11], hv213=[34, 11, 55])
    with pytest.raises(UnmappedCodeError) as excinfo:
        households_from_raw(raw, SPEC)
    message = str(excinfo.value)
    assert "hv226 -> cooking_fuel" in message
    assert "hv213 -> floor" in message


def test_missing_value_codes_become_na():
    hh = households_from_raw(make_raw(hv201=[11, 99, 43]), SPEC)
    assert pd.isna(hh.loc[1, "water_source"])
    assert pd.isna(hh.loc[1, "head_age"])  # hv220 = 98 (don't know)


def test_water_time_on_premises_is_zero_and_unknown_is_na():
    hh = households_from_raw(make_raw(), SPEC)
    assert hh.loc[0, "water_time_min"] == 0
    assert hh.loc[1, "water_time_min"] == 20
    assert pd.isna(hh.loc[2, "water_time_min"])


def test_materials_use_dhs_tiers():
    hh = households_from_raw(make_raw(), SPEC)
    assert hh["floor"].tolist() == ["finished", "natural", "rudimentary"]
    assert hh["wall"].tolist() == ["finished", "rudimentary", "other"]


def test_telephone_combines_mobile_and_landline():
    hh = households_from_raw(make_raw(), SPEC)
    assert hh["has_telephone"].tolist() == [True, True, True]


def test_no_anthropometry_gives_na():
    spec = dataclasses.replace(SPEC, anthropometry_subsample_var=None)
    hh = households_from_raw(make_raw(), spec)
    assert hh["anthro_selected"].isna().all()


# --- persons ------------------------------------------------------------------------


def make_raw_persons(**changes):
    """Four synthetic household members; keyword arguments replace whole columns."""
    data = {
        "hhid": ["   1  1", "   1  1", "   1  1", "   1  2"],
        "hvidx": [1, 2, 3, 1],
        "hv101": [1, 2, 3, 1],
        "hv102": [1, 1, 1, 1],
        "hv103": [1, 1, 0, 1],
        "hv104": [1, 2, 1, 2],
        "hv105": [40, 35, 3, 98],
        "hv108": [12, 6, 0, 98],
        "hv121": [0, 0, 2, 1],
        "hc1": [float("nan"), float("nan"), 40.0, float("nan")],
        "hc70": [float("nan"), float("nan"), -215.0, float("nan")],
        "hc71": [float("nan"), float("nan"), 9998.0, float("nan")],
        "ha40": [float("nan"), 2150.0, float("nan"), float("nan")],
        "hb40": [1890.0, float("nan"), float("nan"), float("nan")],
        "ha54": [float("nan"), 1.0, float("nan"), float("nan")],
    }
    data.update(changes)
    return RawTable(data=pd.DataFrame(data), value_labels={}, variable_labels={})


def test_persons_build_valid_table():
    persons = persons_from_raw(make_raw_persons(), SPEC)
    assert persons.shape == (4, 15)
    assert persons["is_head"].tolist() == [True, False, False, True]
    assert persons["sex"].tolist() == ["male", "female", "male", "female"]


def test_attendance_counts_both_dhs_yes_codes():
    # hv121: 1 = currently attending, 2 = attended at some time this year -> both "yes".
    persons = persons_from_raw(make_raw_persons(), SPEC)
    assert persons["attended_school"].tolist() == [False, False, True, True]


def test_anthropometry_is_rescaled_and_flags_become_na():
    persons = persons_from_raw(make_raw_persons(), SPEC)
    assert persons.loc[2, "height_for_age_z"] == -2.15
    assert pd.isna(persons.loc[2, "weight_for_age_z"])  # 9998 = flagged
    assert persons["bmi"].tolist()[:2] == [18.9, 21.5]  # men's and women's BMI merged


def test_unknown_values_become_na():
    persons = persons_from_raw(make_raw_persons(), SPEC)
    assert pd.isna(persons.loc[3, "age_years"])  # hv105 = 98
    assert pd.isna(persons.loc[3, "years_schooling"])  # hv108 = 98


def test_unexpected_attendance_code_fails():
    with pytest.raises(UnmappedCodeError, match="hv121 -> attended_school"):
        persons_from_raw(make_raw_persons(hv121=[0, 0, 5, 1]), SPEC)


def test_persons_without_anthropometry():
    spec = dataclasses.replace(SPEC, anthropometry_subsample_var=None)
    raw = make_raw_persons()
    raw.data = raw.data.drop(columns=["hc1", "hc70", "hc71", "ha40", "hb40", "ha54"])
    persons = persons_from_raw(raw, spec)
    assert persons["bmi"].isna().all()
    assert persons["height_for_age_z"].isna().all()


# --- child deaths -------------------------------------------------------------------


def make_raw_births(**changes):
    """Four synthetic births: two alive, two dead (one of them a twin)."""
    data = {
        "caseid": ["       1   1  2", "       1   1  2", "       1   1  2", "       2   5  3"],
        "v003": [2, 2, 2, 3],
        "v008": [1470, 1470, 1470, 1470],  # interview date (century-month code)
        "bidx": [1, 2, 3, 1],
        "b3": [1440, 1420, 1420, 1450],  # birth dates (CMC)
        "b4": [1, 2, 1, 2],
        "b5": [1, 0, 0, 1],
        "b7": [float("nan"), 6.0, 0.0, float("nan")],  # age at death, months
    }
    data.update(changes)
    return RawTable(data=pd.DataFrame(data), value_labels={}, variable_labels={})


def test_only_deaths_are_kept():
    deaths = child_deaths_from_raw(make_raw_births(), SPEC, source="kids")
    assert len(deaths) == 2
    assert deaths["birth_index"].tolist() == [2, 3]  # twins: same mother, both died


def test_household_id_comes_from_caseid():
    deaths = child_deaths_from_raw(make_raw_births(), SPEC, source="kids")
    assert deaths["hh_id"].tolist() == ["1   1", "1   1"]  # caseid minus the mother's line


def test_months_since_death_uses_cmc_dates():
    deaths = child_deaths_from_raw(make_raw_births(), SPEC, source="kids")
    # interview 1470 - (birth 1420 + age at death 6) = 44 ; 1470 - (1420 + 0) = 50
    assert deaths["months_since_death"].tolist() == [44, 50]
    assert deaths["source"].tolist() == ["kids", "kids"]


def test_unknown_survival_code_fails():
    with pytest.raises(UnmappedCodeError, match="b5 -> child_alive"):
        child_deaths_from_raw(make_raw_births(b5=[1, 0, 7, 1]), SPEC, source="kids")


def test_missing_birth_file_is_explained():
    with pytest.raises(KeyError, match="no birth file configured"):
        build_child_deaths(SPEC)  # SPEC lists no births or kids file


def test_births_file_preferred_over_kids(monkeypatch):
    spec = dataclasses.replace(SPEC, files={**SPEC.files, "kids": "kr.zip", "births": "br.zip"})
    chosen = {}

    def fake_read(path, columns):
        chosen["file"] = path.name
        return make_raw_births()

    monkeypatch.setattr("poverty_targeting.dhs.read_stata_zip", fake_read)
    deaths = build_child_deaths(spec)
    assert chosen["file"] == "br.zip"
    assert set(deaths["source"]) == {"births"}
