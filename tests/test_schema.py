import pandas as pd
import pytest

from poverty_targeting.schema import HOUSEHOLD_COLUMNS, SchemaError, validate_households

SAMPLE_VALUES = {"str": "x", "int": 1, "float": 1.0, "bool": True}


def make_valid_households(n=2):
    """A minimal valid table generated from the schema itself."""
    data = {}
    for col in HOUSEHOLD_COLUMNS:
        value = col.allowed[0] if col.allowed else SAMPLE_VALUES[col.kind]
        data[col.name] = [value] * n
    data["hh_id"] = [f"hh{i}" for i in range(n)]
    return pd.DataFrame(data)


def test_valid_table_passes():
    validate_households(make_valid_households())


def test_missing_column_is_reported():
    df = make_valid_households().drop(columns=["electricity"])
    with pytest.raises(SchemaError, match="missing columns.*electricity"):
        validate_households(df)


def test_value_outside_vocabulary_is_reported():
    df = make_valid_households()
    df.loc[0, "water_source"] = "magic_well"
    with pytest.raises(SchemaError, match="magic_well"):
        validate_households(df)


def test_duplicate_household_ids_are_reported():
    df = make_valid_households()
    df["hh_id"] = "same"
    with pytest.raises(SchemaError, match="duplicated"):
        validate_households(df)


def test_missing_value_in_required_column_is_reported():
    df = make_valid_households()
    df["region"] = df["region"].astype(object)
    df.loc[0, "region"] = None
    with pytest.raises(SchemaError, match="region: 1 missing"):
        validate_households(df)


def test_all_problems_reported_together():
    df = make_valid_households().drop(columns=["electricity"])
    df.loc[0, "cooking_fuel"] = "dragon_fire"
    df["weight"] = -1.0
    with pytest.raises(SchemaError) as excinfo:
        validate_households(df)
    message = str(excinfo.value)
    assert "electricity" in message
    assert "dragon_fire" in message
    assert "weight" in message
