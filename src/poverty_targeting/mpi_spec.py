"""MPI specifications: load an MPI definition (dimensions, indicators, cutoffs, weights).

The MPI is configuration, not code. The global MPI is one spec file; a national MPI
would be another. This module reads a spec and checks it is internally consistent
and speaks the canonical vocabularies of schema.py, before any poverty is computed.
"""

import tomllib
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Any

import devecon
import numpy as np

from poverty_targeting import config, schema

# The indicators this package knows how to construct (Step 5).
KNOWN_INDICATORS = (
    "nutrition", "child_mortality", "years_schooling", "school_attendance",
    "cooking_fuel", "sanitation", "drinking_water", "electricity", "housing", "assets",
)  # fmt: skip

# Category lists in a spec must use the canonical vocabularies.
VOCABULARIES = {
    ("cooking_fuel", "deprived"): schema.COOKING_FUELS,
    ("sanitation", "improved"): schema.SANITATION_FACILITIES,
    ("drinking_water", "improved"): schema.WATER_SOURCES,
    ("housing", "inadequate_floor"): schema.MATERIALS,
    ("housing", "inadequate_walls"): schema.MATERIALS,
    ("housing", "inadequate_roof"): schema.MATERIALS,
}

MISSING_DATA_RULES = ("drop", "deprived=0")
MEMBER_RULES = ("usual_residents", "all_members")


class SpecError(ValueError):
    """Raised when an MPI spec is inconsistent."""


@dataclass(frozen=True)
class MPISpec:
    spec_id: str
    name: str
    source: str
    poverty_cutoff: float
    members: str
    missing_data: str
    dimensions: dict[str, dict[str, Any]] = field(default_factory=dict)
    indicators: dict[str, dict[str, Any]] = field(default_factory=dict)

    def weights(self) -> tuple[list[str], np.ndarray]:
        """Indicator names and their nested weights (summing to 1), via devecon."""
        return devecon.build_nested_weights(self.dimensions)


def _fraction(value: Any) -> float:
    """Read '1/3', 0.5 or 1 exactly, then convert to float."""
    return float(Fraction(str(value)))


def available_specs(spec_dir: Path | None = None) -> list[str]:
    directory = Path(spec_dir or config.MPI_SPEC_DIR)
    return sorted(p.stem for p in directory.glob("*.toml"))


def load_mpi_spec(spec_id: str = "global_mpi", spec_dir: Path | None = None) -> MPISpec:
    """Read <spec_id>.toml and check it before anything is computed with it."""
    directory = Path(spec_dir or config.MPI_SPEC_DIR)
    path = directory / f"{spec_id}.toml"
    if not path.exists():
        raise FileNotFoundError(
            f"No MPI spec '{spec_id}' at {path}. Available: {available_specs(directory)}"
        )
    with path.open("rb") as f:
        raw = tomllib.load(f)

    meta = raw["mpi"]
    dimensions = {
        name: {"weight": _fraction(d["weight"]), "indicators": list(d["indicators"])}
        for name, d in raw.get("dimensions", {}).items()
    }
    indicators = raw.get("indicators", {})
    problems: list[str] = []

    k = _fraction(meta["poverty_cutoff"])
    if not 0 < k <= 1:
        problems.append(f"poverty_cutoff must be in (0, 1], got {k}")
    if meta["missing_data"] not in MISSING_DATA_RULES:
        problems.append(f"missing_data must be one of {list(MISSING_DATA_RULES)}")
    if meta["members"] not in MEMBER_RULES:
        problems.append(f"members must be one of {list(MEMBER_RULES)}")

    total = sum(d["weight"] for d in dimensions.values())
    if abs(total - 1) > 1e-9:
        problems.append(f"dimension weights sum to {total:.4f}, not 1")

    listed = [ind for d in dimensions.values() for ind in d["indicators"]]
    repeated = sorted({ind for ind in listed if listed.count(ind) > 1})
    if repeated:
        problems.append(f"indicators in more than one dimension: {repeated}")
    unknown = sorted(set(listed) - set(KNOWN_INDICATORS))
    if unknown:
        problems.append(f"indicators this package cannot construct: {unknown}")
    no_section = sorted(set(listed) - set(indicators))
    if no_section:
        problems.append(f"indicators without an [indicators.*] section: {no_section}")
    orphan_sections = sorted(set(indicators) - set(listed))
    if orphan_sections:
        problems.append(f"[indicators.*] sections not used by any dimension: {orphan_sections}")

    for (indicator, key), vocabulary in VOCABULARIES.items():
        values = indicators.get(indicator, {}).get(key, [])
        bad = sorted(set(values) - set(vocabulary))
        if bad:
            problems.append(f"{indicator}.{key}: not in the canonical vocabulary {bad}")

    for group in ("small", "large"):
        for asset in indicators.get("assets", {}).get(group, []):
            if f"has_{asset}" not in schema.HOUSEHOLD_COLUMN_NAMES:
                problems.append(f"assets.{group}: no household column has_{asset}")

    if problems:
        raise SpecError(f"MPI spec '{spec_id}' is inconsistent:\n- " + "\n- ".join(problems))

    return MPISpec(
        spec_id=meta["id"],
        name=meta["name"],
        source=meta["source"],
        poverty_cutoff=k,
        members=meta["members"],
        missing_data=meta["missing_data"],
        dimensions=dimensions,
        indicators=indicators,
    )
