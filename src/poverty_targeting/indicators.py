"""Household deprivation indicators: canonical tables + an MPI spec -> a 0/1/NA matrix.

Each of the ten global-MPI indicators becomes one column with, per household,
1 = deprived, 0 = not deprived, NA = cannot be determined from the data.
Person-level facts (schooling, attendance, nutrition, child deaths) are
aggregated to the household, following the MPI's "any member" rules:
households without the relevant people are not deprived (not applicable).
"""

import devecon
import numpy as np
import pandas as pd

from poverty_targeting.mpi_spec import MPISpec

ADOLESCENT_RULES = ("exclude", "adult_cutoff")


class IndicatorError(ValueError):
    """Raised when an indicator cannot be built as the spec asks."""


def _as_indicator(values: np.ndarray, index: pd.Index) -> pd.Series:
    """Float array with NaN for unknown -> nullable 0/1/NA Series."""
    return pd.Series(values, index=index, dtype="Float64")


def _any_member(flags: pd.Series, hh_ids: pd.Series, households: pd.Index) -> pd.Series:
    """Per household: 1 if any member's flag is True, NA if none True but some unknown, else 0.

    Households with no qualifying members at all get 0 (not applicable = not deprived).
    """
    frame = pd.DataFrame(
        {
            "hh_id": hh_ids.to_numpy(),
            "true": flags.fillna(False).to_numpy(dtype=bool),
            "unknown": flags.isna().to_numpy(dtype=bool),
        }
    )
    grouped = frame.groupby("hh_id")[["true", "unknown"]].any()
    grouped = grouped.reindex(households, fill_value=False)
    values = np.where(grouped["true"], 1.0, np.where(grouped["unknown"], np.nan, 0.0))
    return _as_indicator(values, households)


def _members(persons: pd.DataFrame, spec: MPISpec) -> pd.DataFrame:
    if spec.members == "usual_residents":
        return persons[persons["usual_resident"].fillna(False).to_numpy(dtype=bool)]
    return persons


# --- health ---------------------------------------------------------------------------


def nutrition(households, persons, spec) -> pd.Series:
    rules = spec.indicators["nutrition"]
    adolescent_rule = rules.get("adolescents_15_19", "exclude")
    if adolescent_rule not in ADOLESCENT_RULES:
        raise IndicatorError(f"adolescents_15_19 must be one of {list(ADOLESCENT_RULES)}")

    m = _members(persons, spec)
    age = m["age_years"]
    z_cut, bmi_cut = rules["under5_z_cutoff"], rules["adult_bmi_cutoff"]

    under5 = (m["age_months"] < 60).fillna(False)
    haz, waz = m["height_for_age_z"], m["weight_for_age_z"]
    child_info = under5 & (haz.notna() | waz.notna())
    child_low = (haz < z_cut).fillna(False) | (waz < z_cut).fillna(False)

    min_adult_age = 15 if adolescent_rule == "adult_cutoff" else 20
    adult = ((age >= min_adult_age) & (age < rules["max_age_years"])).fillna(False)
    if rules.get("exclude_pregnant", False):
        adult = adult & ~m["pregnant"].fillna(False)
    adult_info = adult & m["bmi"].notna()
    adult_low = (m["bmi"] < bmi_cut).fillna(False)

    if not (child_info | adult_info).any():
        raise IndicatorError("no nutrition information: use an MPI spec without nutrition")

    # Only people with nutritional information count; people without it are skipped.
    has_info = child_info | adult_info
    deprived = (child_info & child_low) | (adult_info & adult_low)
    indicator = _any_member(deprived[has_info], m.loc[has_info, "hh_id"], households.index)

    # Households outside the anthropometry subsample were never measured: unknown.
    not_measured = households["anthro_selected"].eq(False).fillna(False)
    return indicator.mask(not_measured.to_numpy(dtype=bool))


def child_mortality(households, child_deaths, spec) -> pd.Series:
    rules = spec.indicators["child_mortality"]
    under_age = child_deaths["age_at_death_months"] < rules["max_child_age_years"] * 12
    recent = child_deaths["months_since_death"] < rules["window_months"]
    qualifying = under_age & recent
    return _any_member(qualifying, child_deaths["hh_id"], households.index)


# --- education ------------------------------------------------------------------------


def years_schooling(households, persons, spec) -> pd.Series:
    rules = spec.indicators["years_schooling"]
    m = _members(persons, spec)
    eligible = (m["age_years"] >= rules["eligible_min_age"]).fillna(False)
    e = m[eligible.to_numpy(dtype=bool)]
    achieved = e["years_schooling"] >= rules["min_years"]
    anyone_achieved = _any_member(achieved, e["hh_id"], households.index)

    n_eligible = e.groupby("hh_id").size().reindex(households.index, fill_value=0)
    # Someone reached 6 years -> not deprived; nobody did -> deprived; unclear -> NA.
    values = np.where(
        n_eligible.to_numpy() == 0,
        0.0,
        1.0 - anyone_achieved.to_numpy(dtype=float, na_value=np.nan),
    )
    return _as_indicator(values, households.index)


def school_attendance(households, persons, spec) -> pd.Series:
    rules = spec.indicators["school_attendance"]
    m = _members(persons, spec)
    first = rules["school_entry_age"]
    last = first + rules["span_years"] - 1
    school_aged = m["age_years"].between(first, last).fillna(False)
    c = m[school_aged.to_numpy(dtype=bool)]
    return _any_member(~c["attended_school"], c["hh_id"], households.index)


# --- living standards (household facts) -----------------------------------------------


def cooking_fuel(households, spec) -> pd.Series:
    deprived = spec.indicators["cooking_fuel"]["deprived"]
    values = devecon.deprive_in(households["cooking_fuel"].to_numpy(dtype=object), deprived)
    return _as_indicator(values, households.index)


def electricity(households, spec) -> pd.Series:
    values = devecon.deprive_in(households["electricity"].to_numpy(dtype=object), {False})
    return _as_indicator(values, households.index)


def sanitation(households, spec) -> pd.Series:
    rules = spec.indicators["sanitation"]
    facility, shared = households["sanitation"], households["toilet_shared"]
    unimproved = ~facility.isin(rules["improved"])
    if rules.get("shared_counts_as_deprived", True):
        shared_known = shared.notna().to_numpy(dtype=bool)
        is_shared = shared.fillna(False).to_numpy(dtype=bool)
    else:
        shared_known = np.ones(len(households), dtype=bool)
        is_shared = np.zeros(len(households), dtype=bool)
    values = np.select(
        [
            facility.isna().to_numpy(dtype=bool),
            unimproved.to_numpy(dtype=bool),
            is_shared,
            ~shared_known,
        ],
        [np.nan, 1.0, 1.0, np.nan],
        default=0.0,
    )
    return _as_indicator(values, households.index)


def drinking_water(households, spec) -> pd.Series:
    rules = spec.indicators["drinking_water"]
    source, minutes = households["water_source"], households["water_time_min"]
    unimproved = ~source.isin(rules["improved"])
    too_far = (minutes >= rules["max_round_trip_minutes"]).fillna(False)
    values = np.select(
        [
            source.isna().to_numpy(dtype=bool),
            unimproved.to_numpy(dtype=bool),
            too_far.to_numpy(dtype=bool),
            minutes.isna().to_numpy(dtype=bool),
        ],
        [np.nan, 1.0, 1.0, np.nan],
        default=0.0,
    )
    return _as_indicator(values, households.index)


def housing(households, spec) -> pd.Series:
    rules = spec.indicators["housing"]
    parts = {
        "floor": rules["inadequate_floor"],
        "wall": rules["inadequate_walls"],
        "roof": rules["inadequate_roof"],
    }
    inadequate = np.zeros(len(households), dtype=bool)
    unknown = np.zeros(len(households), dtype=bool)
    for column, bad in parts.items():
        inadequate |= households[column].isin(bad).fillna(False).to_numpy(dtype=bool)
        unknown |= households[column].isna().to_numpy(dtype=bool)
    values = np.where(inadequate, 1.0, np.where(unknown, np.nan, 0.0))
    return _as_indicator(values, households.index)


def assets(households, spec) -> pd.Series:
    rules = spec.indicators["assets"]
    small = households[[f"has_{a}" for a in rules["small"]]]
    large = households[[f"has_{a}" for a in rules["large"]]]
    n_small = small.fillna(False).to_numpy(dtype=bool).sum(axis=1)
    has_large = large.fillna(False).to_numpy(dtype=bool).any(axis=1)
    unknown = small.isna().to_numpy().any(axis=1) | large.isna().to_numpy().any(axis=1)
    well_off = has_large | (n_small >= 2)
    values = np.where(well_off, 0.0, np.where(unknown, np.nan, 1.0))
    return _as_indicator(values, households.index)


# --- assembly ---------------------------------------------------------------------------


def build_indicators(tables: dict[str, pd.DataFrame], spec: MPISpec) -> pd.DataFrame:
    """One row per household, one 0/1/NA column per indicator in the spec (in spec order)."""
    households = tables["households"].set_index("hh_id", drop=False)
    persons, deaths = tables["persons"], tables["child_deaths"]
    builders = {
        "nutrition": lambda: nutrition(households, persons, spec),
        "child_mortality": lambda: child_mortality(households, deaths, spec),
        "years_schooling": lambda: years_schooling(households, persons, spec),
        "school_attendance": lambda: school_attendance(households, persons, spec),
        "cooking_fuel": lambda: cooking_fuel(households, spec),
        "sanitation": lambda: sanitation(households, spec),
        "drinking_water": lambda: drinking_water(households, spec),
        "electricity": lambda: electricity(households, spec),
        "housing": lambda: housing(households, spec),
        "assets": lambda: assets(households, spec),
    }
    names, _ = spec.weights()
    out = pd.DataFrame({"hh_id": households["hh_id"].to_numpy()}, index=households.index)
    for name in names:
        out[name] = builders[name]()
    return out.reset_index(drop=True)
