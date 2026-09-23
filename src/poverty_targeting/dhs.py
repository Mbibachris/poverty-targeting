"""DHS adapter: translate DHS recode files into the canonical households and persons tables.

Standard DHS response codes are mapped here once, for every DHS survey. Codes a
country adds on top (e.g. Ghana's sachet water) go in that survey's config file.
Any code that neither covers makes the build fail and names the code and its
label, so nothing is ever classified by guesswork.
"""

import pandas as pd

from poverty_targeting.readers import RawTable, read_stata_zip
from poverty_targeting.registry import SurveySpec
from poverty_targeting.schema import (
    HOUSEHOLD_COLUMNS,
    PERSON_COLUMNS,
    validate_households,
    validate_persons,
)

# --- standard DHS codes -> canonical vocabularies (see schema.py) -------------------
STANDARD_CODES: dict[str, dict[int, str]] = {
    "water_source": {
        11: "piped_dwelling", 12: "piped_yard", 13: "piped_neighbour", 14: "public_tap",
        21: "tubewell_borehole", 31: "protected_well", 32: "unprotected_well",
        41: "protected_spring", 42: "unprotected_spring", 43: "surface_water",
        51: "rainwater", 61: "tanker_cart", 62: "tanker_cart", 71: "bottled", 96: "other",
    },
    "sanitation": {
        11: "flush_sewer", 12: "flush_septic", 13: "flush_pit", 14: "flush_elsewhere",
        15: "flush_unknown", 21: "vip_latrine", 22: "pit_with_slab", 23: "pit_without_slab",
        31: "no_facility", 41: "composting", 42: "bucket", 43: "hanging", 96: "other",
    },
    "cooking_fuel": {
        1: "electricity", 2: "solar", 3: "lpg", 4: "natural_gas", 5: "biogas",
        6: "alcohol_ethanol", 7: "gasoline_diesel", 8: "kerosene", 9: "coal_lignite",
        10: "charcoal", 11: "wood", 12: "straw_shrubs", 13: "crop_residue",
        14: "animal_dung", 15: "processed_biomass", 16: "garbage", 17: "sawdust",
        95: "no_cooking", 96: "other",
    },
}  # fmt: skip

# DHS missing-value conventions.
CATEGORY_MISSING = (99,)  # "missing"
BINARY_MISSING = (8, 9)  # "don't know", "missing"
ANTHRO_FLAG_FLOOR = 9990  # 9996-9998: implausible or flagged measurements

HR_COLUMNS = [
    "hhid", "hv001", "hv005", "hv024", "hv025", "hv009", "hv219", "hv220",
    "hv201", "hv204", "hv205", "hv225", "hv206", "hv226", "hv213", "hv214", "hv215",
    "hv207", "hv208", "hv243a", "hv221", "hv243e", "hv243c", "hv210", "hv211",
    "hv209", "hv212", "hv247", "hv244", "hv246", "hv216",
]  # fmt: skip

PR_COLUMNS = [
    "hhid", "hvidx", "hv101", "hv102", "hv103", "hv104", "hv105", "hv108", "hv121",
]  # fmt: skip

# Read only when the survey measured height and weight.
PR_ANTHRO_COLUMNS = ["hc1", "hc70", "hc71", "ha40", "hb40", "ha54"]


class UnmappedCodeError(ValueError):
    """Raised when survey data contains codes the adapter cannot classify."""


class _Mapper:
    """Converts raw columns while collecting every problem, so all are reported at once."""

    def __init__(self, raw: RawTable):
        self.raw = raw
        self.problems: list[str] = []

    def _unknown(self, column: str, codes: list[int], target: str) -> None:
        labels = self.raw.labels_for(column)
        described = ", ".join(f"{c} ('{labels.get(c, 'no label')}')" for c in codes)
        self.problems.append(f"{column} -> {target}: no mapping for {described}")

    def _codes(self, column: str, ignore: tuple[int, ...] = ()) -> set[int]:
        return set(self.raw.data[column].dropna().astype(int)) - set(ignore)

    def category(self, column: str, target: str, mapping: dict[int, str]) -> pd.Series:
        unknown = sorted(self._codes(column, CATEGORY_MISSING) - mapping.keys())
        if unknown:
            self._unknown(column, unknown, target)
        return self.raw.data[column].map(mapping).astype("string")

    def binary(
        self,
        column: str,
        target: str,
        true_codes: tuple[int, ...] = (1,),
        false_codes: tuple[int, ...] = (0,),
    ) -> pd.Series:
        known = set(true_codes) | set(false_codes)
        unknown = sorted(self._codes(column, BINARY_MISSING) - known)
        if unknown:
            self._unknown(column, unknown, target)
        mapping = {c: True for c in true_codes} | {c: False for c in false_codes}
        return self.raw.data[column].map(mapping).astype("boolean")

    def material(self, column: str, target: str) -> pd.Series:
        """DHS material codes: 1x natural, 2x rudimentary, 3x finished, 96 other."""
        tiers = {1: "natural", 2: "rudimentary", 3: "finished"}
        mapping = {c: tiers[c // 10] for c in self._codes(column) if c // 10 in tiers}
        mapping[96] = "other"
        return self.category(column, target, mapping)

    def number(self, column: str, missing: tuple[int, ...] = (), dtype: str = "Int64"):
        return self.raw.data[column].where(~self.raw.data[column].isin(missing)).astype(dtype)

    def measure(self, column: str, scale: float = 100.0) -> pd.Series:
        """Anthropometric value stored x100 by DHS; codes >= 9990 are flagged cases."""
        values = self.raw.data[column]
        return (values.where(values < ANTHRO_FLAG_FLOOR) / scale).astype("Float64")


def households_from_raw(raw: RawTable, spec: SurveySpec) -> pd.DataFrame:
    """Map a raw DHS household table (already read) into the canonical households table."""
    d = raw.data
    m = _Mapper(raw)

    def codes_for(target: str) -> dict[int, str]:
        return {**STANDARD_CODES[target], **spec.code_overrides.get(target, {})}

    region_names = {code: name.title() for code, name in raw.labels_for("hv024").items()}
    unknown_regions = sorted(set(d["hv024"].astype(int)) - region_names.keys())
    if unknown_regions:
        m.problems.append(f"hv024 -> region: codes without a label {unknown_regions}")

    anthro_var = spec.anthropometry_subsample_var
    if anthro_var is not None:
        anthro_selected = m.binary(anthro_var, "anthro_selected")
    else:
        anthro_selected = pd.Series(pd.NA, index=d.index, dtype="boolean")

    # hv204: minutes to water, 996 = on premises (0 minutes), 998/999 = unknown.
    water_time = d["hv204"].where(d["hv204"] != 996, 0)
    water_time = water_time.where(~water_time.isin([998, 999])).astype("Float64")

    has_mobile = m.binary("hv243a", "has_telephone")
    has_landline = m.binary("hv221", "has_telephone")

    out = pd.DataFrame(
        {
            "survey_id": pd.Series(spec.survey_id, index=d.index, dtype="string"),
            "hh_id": d["hhid"].astype("string").str.strip(),
            "cluster": m.number("hv001"),
            "weight": (d["hv005"] / 1_000_000).astype("Float64"),
            "region_code": m.number("hv024"),
            "region": d["hv024"].map(region_names).astype("string"),
            "urban": m.binary("hv025", "urban", true_codes=(1,), false_codes=(2,)),
            "anthro_selected": anthro_selected,
            "hh_size": m.number("hv009"),
            "head_sex": m.category("hv219", "head_sex", {1: "male", 2: "female"}),
            "head_age": m.number("hv220", missing=(98, 99)),
            "water_source": m.category("hv201", "water_source", codes_for("water_source")),
            "water_time_min": water_time,
            "sanitation": m.category("hv205", "sanitation", codes_for("sanitation")),
            "toilet_shared": m.binary("hv225", "toilet_shared"),
            "electricity": m.binary("hv206", "electricity"),
            "cooking_fuel": m.category("hv226", "cooking_fuel", codes_for("cooking_fuel")),
            "floor": m.material("hv213", "floor"),
            "wall": m.material("hv214", "wall"),
            "roof": m.material("hv215", "roof"),
            "has_radio": m.binary("hv207", "has_radio"),
            "has_tv": m.binary("hv208", "has_tv"),
            "has_telephone": has_mobile | has_landline,
            "has_computer": m.binary("hv243e", "has_computer"),
            "has_animal_cart": m.binary("hv243c", "has_animal_cart"),
            "has_bicycle": m.binary("hv210", "has_bicycle"),
            "has_motorbike": m.binary("hv211", "has_motorbike"),
            "has_refrigerator": m.binary("hv209", "has_refrigerator"),
            "has_car_truck": m.binary("hv212", "has_car_truck"),
            "has_bank_account": m.binary("hv247", "has_bank_account"),
            "owns_agri_land": m.binary("hv244", "owns_agri_land"),
            "owns_livestock": m.binary("hv246", "owns_livestock"),
            "sleeping_rooms": m.number("hv216", missing=(98, 99)),
        }
    )

    if m.problems:
        raise UnmappedCodeError(
            f"{spec.survey_id}: cannot map household codes:\n- " + "\n- ".join(m.problems)
        )

    out = out[[c.name for c in HOUSEHOLD_COLUMNS]].reset_index(drop=True)
    validate_households(out)
    return out


def build_households(spec: SurveySpec) -> pd.DataFrame:
    """Read a survey's HR file from disk and return the validated households table."""
    columns = list(HR_COLUMNS)
    if spec.anthropometry_subsample_var is not None:
        columns.append(spec.anthropometry_subsample_var)
    raw = read_stata_zip(spec.file_path("household"), columns=columns)
    return households_from_raw(raw, spec)


def persons_from_raw(raw: RawTable, spec: SurveySpec) -> pd.DataFrame:
    """Map a raw DHS person table (PR, already read) into the canonical persons table."""
    d = raw.data
    m = _Mapper(raw)

    if spec.anthropometry_subsample_var is not None:
        age_months = m.number("hc1")
        haz = m.measure("hc70")
        waz = m.measure("hc71")
        # A person is measured in either the women's or the men's module, never both.
        bmi = m.measure("ha40").fillna(m.measure("hb40"))
        pregnant = m.binary("ha54", "pregnant")
    else:
        age_months = pd.Series(pd.NA, index=d.index, dtype="Int64")
        haz = waz = bmi = pd.Series(pd.NA, index=d.index, dtype="Float64")
        pregnant = pd.Series(pd.NA, index=d.index, dtype="boolean")

    out = pd.DataFrame(
        {
            "survey_id": pd.Series(spec.survey_id, index=d.index, dtype="string"),
            "hh_id": d["hhid"].astype("string").str.strip(),
            "line": m.number("hvidx"),
            "is_head": (d["hv101"] == 1).astype("boolean"),
            "usual_resident": m.binary("hv102", "usual_resident"),
            "slept_last_night": m.binary("hv103", "slept_last_night"),
            "sex": m.category("hv104", "sex", {1: "male", 2: "female"}),
            "age_years": m.number("hv105", missing=(98, 99)),
            "years_schooling": m.number("hv108", missing=(97, 98, 99)),
            # hv121: 1 = currently attending, 2 = attended at some time this year.
            "attended_school": m.binary("hv121", "attended_school", true_codes=(1, 2)),
            "age_months": age_months,
            "height_for_age_z": haz,
            "weight_for_age_z": waz,
            "bmi": bmi,
            "pregnant": pregnant,
        }
    )

    if m.problems:
        raise UnmappedCodeError(
            f"{spec.survey_id}: cannot map person codes:\n- " + "\n- ".join(m.problems)
        )

    out = out[[c.name for c in PERSON_COLUMNS]].reset_index(drop=True)
    validate_persons(out)
    return out


def build_persons(spec: SurveySpec) -> pd.DataFrame:
    """Read a survey's PR file from disk and return the validated persons table."""
    columns = list(PR_COLUMNS)
    if spec.anthropometry_subsample_var is not None:
        columns += PR_ANTHRO_COLUMNS
    raw = read_stata_zip(spec.file_path("persons"), columns=columns)
    return persons_from_raw(raw, spec)
