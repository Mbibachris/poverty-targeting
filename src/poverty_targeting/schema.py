"""Canonical household table: the contract every survey adapter must deliver.

Adapters translate each survey's country-specific codes into these columns and
vocabularies. Everything downstream (MPI label, features, model, API) reads only
this table, so it works the same way for every country.

Columns hold harmonised *facts* (e.g. water source = "sachet"), not deprivation
flags. Whether a category counts as deprived is decided later by the MPI spec,
so a national MPI can use different rules without touching the adapters.
"""

from dataclasses import dataclass

import pandas as pd

WATER_SOURCES = (
    "piped_dwelling", "piped_yard", "piped_neighbour", "public_tap", "tubewell_borehole",
    "protected_well", "unprotected_well", "protected_spring", "unprotected_spring",
    "surface_water", "rainwater", "tanker_cart", "bottled", "sachet", "other",
)  # fmt: skip

SANITATION_FACILITIES = (
    "flush_sewer", "flush_septic", "flush_pit", "flush_elsewhere", "flush_unknown",
    "flush_biodigester", "vip_latrine", "pit_with_slab", "pit_without_slab",
    "composting", "bucket", "hanging", "no_facility", "other",
)  # fmt: skip

COOKING_FUELS = (
    "electricity", "solar", "lpg", "natural_gas", "biogas", "alcohol_ethanol",
    "gasoline_diesel", "kerosene", "coal_lignite", "charcoal", "wood", "straw_shrubs",
    "crop_residue", "animal_dung", "processed_biomass", "garbage", "sawdust",
    "no_cooking", "other",
)  # fmt: skip

# DHS codes materials in three tiers: 1x natural, 2x rudimentary, 3x finished.
MATERIALS = ("natural", "rudimentary", "finished", "other")

SEXES = ("male", "female")


@dataclass(frozen=True)
class Column:
    name: str
    kind: str  # "str" | "int" | "float" | "bool" | "category"
    description: str
    nullable: bool = True
    allowed: tuple[str, ...] | None = None


HOUSEHOLD_COLUMNS: tuple[Column, ...] = (
    # --- identifiers and survey design ---
    Column("survey_id", "str", "Survey key, e.g. GH2022DHS", nullable=False),
    Column("hh_id", "str", "Household ID, unique within a survey", nullable=False),
    Column("cluster", "int", "Primary sampling unit", nullable=False),
    Column("weight", "float", "Household sampling weight (normalised)", nullable=False),
    Column("region_code", "int", "Region code as in the survey", nullable=False),
    Column("region", "str", "Region name from the survey's value labels", nullable=False),
    Column("urban", "bool", "True if urban residence", nullable=False),
    Column("anthro_selected", "bool", "Selected for anthropometry (NA if not measured)"),
    # --- composition and head ---
    Column("hh_size", "int", "Number of usual members", nullable=False),
    Column("head_sex", "category", "Sex of household head", allowed=SEXES),
    Column("head_age", "int", "Age of household head in years"),
    # --- living standards (harmonised facts, not deprivation flags) ---
    Column("water_source", "category", "Main drinking-water source", allowed=WATER_SOURCES),
    Column("water_time_min", "float", "Round-trip minutes to water; 0 = on premises"),
    Column("sanitation", "category", "Toilet facility type", allowed=SANITATION_FACILITIES),
    Column("toilet_shared", "bool", "Toilet shared with other households"),
    Column("electricity", "bool", "Household has electricity"),
    Column("cooking_fuel", "category", "Main cooking fuel", allowed=COOKING_FUELS),
    Column("floor", "category", "Main floor material tier", allowed=MATERIALS),
    Column("wall", "category", "Main wall material tier", allowed=MATERIALS),
    Column("roof", "category", "Main roof material tier", allowed=MATERIALS),
    # --- assets (global MPI list) ---
    Column("has_radio", "bool", "Owns a radio"),
    Column("has_tv", "bool", "Owns a television"),
    Column("has_telephone", "bool", "Owns a mobile or landline telephone"),
    Column("has_computer", "bool", "Owns a computer"),
    Column("has_animal_cart", "bool", "Owns an animal-drawn cart"),
    Column("has_bicycle", "bool", "Owns a bicycle"),
    Column("has_motorbike", "bool", "Owns a motorcycle or scooter"),
    Column("has_refrigerator", "bool", "Owns a refrigerator"),
    Column("has_car_truck", "bool", "Owns a car or truck"),
    # --- extra predictors, not part of the global MPI (Tier A features) ---
    Column("has_bank_account", "bool", "Any member has a bank account"),
    Column("owns_agri_land", "bool", "Owns land usable for agriculture"),
    Column("owns_livestock", "bool", "Owns livestock, herds or farm animals"),
    Column("sleeping_rooms", "int", "Number of rooms used for sleeping"),
)

HOUSEHOLD_COLUMN_NAMES = tuple(c.name for c in HOUSEHOLD_COLUMNS)

_KIND_CHECKS = {
    "str": pd.api.types.is_string_dtype,
    "int": pd.api.types.is_integer_dtype,
    "float": pd.api.types.is_float_dtype,
    "bool": pd.api.types.is_bool_dtype,
    "category": lambda s: True,  # checked against the allowed vocabulary instead
}


class SchemaError(ValueError):
    """Raised when a table breaks the canonical contract."""


def validate_households(df: pd.DataFrame) -> None:
    """Check a households table against the contract; report every problem at once."""
    problems: list[str] = []

    missing = [n for n in HOUSEHOLD_COLUMN_NAMES if n not in df.columns]
    unexpected = [n for n in df.columns if n not in HOUSEHOLD_COLUMN_NAMES]
    if missing:
        problems.append(f"missing columns: {missing}")
    if unexpected:
        problems.append(f"unexpected columns: {unexpected}")

    for col in HOUSEHOLD_COLUMNS:
        if col.name not in df.columns:
            continue
        series = df[col.name]
        if not _KIND_CHECKS[col.kind](series):
            problems.append(f"{col.name}: expected {col.kind}, got dtype {series.dtype}")
        if not col.nullable and series.isna().any():
            problems.append(f"{col.name}: {int(series.isna().sum())} missing values not allowed")
        if col.allowed is not None:
            bad = sorted(set(series.dropna().astype(str)) - set(col.allowed))
            if bad:
                problems.append(f"{col.name}: values outside vocabulary {bad}")

    if {"survey_id", "hh_id"} <= set(df.columns):
        n_dup = int(df.duplicated(["survey_id", "hh_id"]).sum())
        if n_dup:
            problems.append(f"{n_dup} duplicated (survey_id, hh_id) rows")

    if "weight" in df.columns and (df["weight"] <= 0).any():
        problems.append("weight: must be strictly positive")

    if problems:
        raise SchemaError("Households table breaks the contract:\n- " + "\n- ".join(problems))
