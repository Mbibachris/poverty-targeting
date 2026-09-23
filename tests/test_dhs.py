import dataclasses

import pandas as pd
import pytest

from poverty_targeting.dhs import UnmappedCodeError, households_from_raw
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
