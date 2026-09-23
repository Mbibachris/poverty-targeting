"""Build pipeline: raw survey files -> validated, linked canonical tables on disk.

Run from the terminal, e.g.:
    python -m poverty_targeting.build GH2022DHS
    python -m poverty_targeting.build --all
"""

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from poverty_targeting import __version__, config, dhs
from poverty_targeting.registry import SurveySpec, available_surveys, load_survey
from poverty_targeting.schema import DEATH_SOURCES

TABLES = ("households", "persons", "child_deaths")

# One adapter per survey programme. A MICS adapter would add a "MICS" entry here.
ADAPTERS = {
    "DHS": {
        "households": dhs.build_households,
        "persons": dhs.build_persons,
        "child_deaths": dhs.build_child_deaths,
    },
}


class LinkError(ValueError):
    """Raised when the canonical tables do not fit together."""


def check_links(tables: dict[str, pd.DataFrame]) -> None:
    """Check the three tables describe the same households; report every problem at once."""
    households = tables["households"]
    persons = tables["persons"]
    deaths = tables["child_deaths"]
    known = set(households["hh_id"])
    problems: list[str] = []

    orphans = int((~persons["hh_id"].isin(known)).sum())
    if orphans:
        problems.append(f"{orphans} persons belong to no household")

    roster = persons.groupby("hh_id").size().reindex(households["hh_id"], fill_value=0)
    wrong_size = int((roster.to_numpy() != households["hh_size"].to_numpy(dtype=int)).sum())
    if wrong_size:
        problems.append(f"{wrong_size} households whose roster size differs from hh_size")

    heads = persons.groupby("hh_id")["is_head"].sum().reindex(households["hh_id"], fill_value=0)
    wrong_heads = int((heads.to_numpy() != 1).sum())
    if wrong_heads:
        problems.append(f"{wrong_heads} households without exactly one head")

    orphan_deaths = int((~deaths["hh_id"].isin(known)).sum())
    if orphan_deaths:
        problems.append(f"{orphan_deaths} child deaths belong to no household")

    roster_people = set(zip(persons["hh_id"], persons["line"], strict=True))
    lost_mothers = sum(
        (hh, line) not in roster_people
        for hh, line in zip(deaths["hh_id"], deaths["mother_line"], strict=True)
    )
    if lost_mothers:
        problems.append(f"{lost_mothers} child deaths whose mother is not in the roster")

    if problems:
        raise LinkError("Canonical tables do not link:\n- " + "\n- ".join(problems))


def build_survey(
    survey_id: str, config_dir: Path | None = None
) -> tuple[SurveySpec, dict[str, pd.DataFrame]]:
    """Build and cross-check all canonical tables for one survey (nothing is written)."""
    spec = load_survey(survey_id, config_dir=config_dir)
    if spec.program not in ADAPTERS:
        raise NotImplementedError(
            f"{survey_id}: no adapter for programme '{spec.program}'. Available: {sorted(ADAPTERS)}"
        )
    builders = ADAPTERS[spec.program]
    tables = {name: builders[name](spec) for name in TABLES}
    check_links(tables)
    return spec, tables


def write_tables(
    spec: SurveySpec, tables: dict[str, pd.DataFrame], out_dir: Path | None = None
) -> Path:
    """Save tables as Parquet plus a manifest describing how they were built."""
    out_dir = Path(out_dir or config.processed_survey_dir(spec.survey_id))
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, df in tables.items():
        df.to_parquet(out_dir / f"{name}.parquet", index=False)

    manifest = {
        "survey_id": spec.survey_id,
        "country": spec.country,
        "year": spec.year,
        "built_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "package_version": __version__,
        "death_source": next((k for k in DEATH_SOURCES if k in spec.files), None),
        "rows": {name: len(df) for name, df in tables.items()},
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return out_dir


def load_tables(survey_id: str, in_dir: Path | None = None) -> dict[str, pd.DataFrame]:
    """Read a survey's processed tables back into memory."""
    in_dir = Path(in_dir or config.processed_survey_dir(survey_id))
    missing = [name for name in TABLES if not (in_dir / f"{name}.parquet").exists()]
    if missing:
        raise FileNotFoundError(
            f"{survey_id}: processed tables {missing} not found in {in_dir}. "
            f"Run: python -m poverty_targeting.build {survey_id}"
        )
    return {name: pd.read_parquet(in_dir / f"{name}.parquet") for name in TABLES}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build canonical tables for surveys.")
    parser.add_argument("surveys", nargs="*", help="Survey IDs, e.g. GH2022DHS")
    parser.add_argument("--all", action="store_true", help="Build every configured survey")
    args = parser.parse_args(argv)

    survey_ids = available_surveys() if args.all else args.surveys
    if not survey_ids:
        parser.error("give one or more survey IDs, or --all")

    for survey_id in survey_ids:
        spec, tables = build_survey(survey_id)
        out_dir = write_tables(spec, tables)
        counts = ", ".join(f"{name}={len(df):,}" for name, df in tables.items())
        print(f"{survey_id}: {counts} -> {out_dir}")


if __name__ == "__main__":
    main()
