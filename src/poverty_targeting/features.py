"""Feature sets: which household facts a targeting model may use, and how they relate to the label.

A feature set is configuration (feature_sets/*.toml): tiers of features, each with its
source column and the question a targeting form would ask. This module loads it,
checks every source exists in the canonical tables, and audits overlap with the MPI:
Tier A must share no inputs with the label; Tier B's overlap is reported, not hidden.
"""

import tomllib
from dataclasses import dataclass, field
from pathlib import Path

from poverty_targeting import config, schema
from poverty_targeting.mpi_spec import MPISpec

TABLE_COLUMNS = {
    "households": schema.HOUSEHOLD_COLUMN_NAMES,
    "persons": schema.PERSON_COLUMN_NAMES,
    "child_deaths": schema.CHILD_DEATH_COLUMN_NAMES,
}


class FeatureSetError(ValueError):
    """Raised when a feature set is inconsistent or breaks its tier's rules."""


@dataclass(frozen=True)
class Feature:
    name: str
    source: str  # "<table>.<column>" or "derived"
    inputs: tuple[str, ...]  # canonical columns the feature is computed from
    question: str


@dataclass(frozen=True)
class FeatureSet:
    set_id: str
    description: str
    features: dict[str, Feature] = field(default_factory=dict)
    tiers: dict[str, list[str]] = field(default_factory=dict)  # resolved, extends applied

    def tier(self, name: str) -> list[Feature]:
        if name not in self.tiers:
            raise KeyError(f"No tier '{name}'. Available: {sorted(self.tiers)}")
        return [self.features[f] for f in self.tiers[name]]


def indicator_inputs(spec: MPISpec) -> dict[str, set[str]]:
    """The canonical columns each MPI indicator reads to decide deprivation.

    Age is used only to decide who is eligible (e.g. school-aged), not whether anyone
    is deprived, so it is not listed as an input.
    """
    asset_rules = spec.indicators.get("assets", {})
    assets = asset_rules.get("small", []) + asset_rules.get("large", [])
    inputs = {
        "nutrition": {"persons.height_for_age_z", "persons.weight_for_age_z", "persons.bmi"},
        "child_mortality": {"child_deaths.age_at_death_months", "child_deaths.months_since_death"},
        "years_schooling": {"persons.years_schooling"},
        "school_attendance": {"persons.attended_school"},
        "cooking_fuel": {"households.cooking_fuel"},
        "sanitation": {"households.sanitation", "households.toilet_shared"},
        "drinking_water": {"households.water_source", "households.water_time_min"},
        "electricity": {"households.electricity"},
        "housing": {"households.floor", "households.wall", "households.roof"},
        "assets": {f"households.has_{a}" for a in assets},
    }
    names, _ = spec.weights()
    return {name: inputs[name] for name in names}


def overlaps(feature_set: FeatureSet, spec: MPISpec, tier: str) -> dict[str, list[str]]:
    """For each feature in a tier: the MPI indicators whose inputs it shares."""
    by_indicator = indicator_inputs(spec)
    return {
        f.name: sorted(ind for ind, cols in by_indicator.items() if cols & set(f.inputs))
        for f in feature_set.tier(tier)
    }


def check_strict_tier(feature_set: FeatureSet, spec: MPISpec, tier: str = "A") -> None:
    """Fail if any feature in the strict tier is also an input to the MPI label."""
    leaking = {name: inds for name, inds in overlaps(feature_set, spec, tier).items() if inds}
    if leaking:
        detail = ", ".join(f"{name} -> {inds}" for name, inds in leaking.items())
        raise FeatureSetError(f"Tier {tier} must not use label inputs, but: {detail}")


def available_feature_sets(set_dir: Path | None = None) -> list[str]:
    directory = Path(set_dir or config.FEATURE_SET_DIR)
    return sorted(p.stem for p in directory.glob("*.toml"))


def _resolve_tiers(raw_tiers: dict, problems: list[str]) -> dict[str, list[str]]:
    resolved: dict[str, list[str]] = {}

    def resolve(name: str, seen: tuple[str, ...] = ()) -> list[str]:
        if name in resolved:
            return resolved[name]
        if name in seen:
            problems.append(f"tier '{name}' extends itself in a loop")
            return []
        tier = raw_tiers[name]
        base: list[str] = []
        parent = tier.get("extends")
        if parent is not None:
            if parent not in raw_tiers:
                problems.append(f"tier '{name}' extends unknown tier '{parent}'")
            else:
                base = resolve(parent, (*seen, name))
        own = [f for f in tier["features"] if f not in base]
        resolved[name] = base + own
        return resolved[name]

    for name in raw_tiers:
        resolve(name)
    return resolved


def load_feature_set(set_id: str = "pmt", set_dir: Path | None = None) -> FeatureSet:
    """Read <set_id>.toml, resolve tier inheritance and check every source exists."""
    directory = Path(set_dir or config.FEATURE_SET_DIR)
    path = directory / f"{set_id}.toml"
    if not path.exists():
        raise FileNotFoundError(
            f"No feature set '{set_id}' at {path}. Available: {available_feature_sets(directory)}"
        )
    with path.open("rb") as f:
        raw = tomllib.load(f)

    problems: list[str] = []
    features: dict[str, Feature] = {}
    for name, item in raw.get("features", {}).items():
        source = item["source"]
        inputs = tuple(item.get("inputs", [source] if source != "derived" else []))
        if source == "derived" and not inputs:
            problems.append(f"{name}: derived features must list their inputs")
        for column in inputs:
            table, _, col = column.partition(".")
            if col not in TABLE_COLUMNS.get(table, ()):
                problems.append(f"{name}: '{column}' is not a canonical column")
        features[name] = Feature(name, source, inputs, item.get("question", ""))

    tiers = _resolve_tiers(raw.get("tiers", {}), problems)
    for tier_name, names in tiers.items():
        undefined = sorted(set(names) - set(features))
        if undefined:
            problems.append(f"tier '{tier_name}' uses undefined features {undefined}")

    if problems:
        raise FeatureSetError(
            f"Feature set '{set_id}' is inconsistent:\n- " + "\n- ".join(problems)
        )

    meta = raw["feature_set"]
    return FeatureSet(meta["id"], meta["description"], features, tiers)
