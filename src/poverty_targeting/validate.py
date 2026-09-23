"""Validate the MPI label against published figures, with sensitivity scenarios.

Two choices cannot be settled from DHS data alone: the nutrition rule for 15-19
year-olds and which drinking-water sources count as improved. Each scenario
changes one choice; if the published figures fall between our bounds, the
implementation is consistent with the official one.

Run from the terminal, e.g.:
    python -m poverty_targeting.validate GH2022DHS
"""

import argparse
import dataclasses
from pathlib import Path

import pandas as pd

from poverty_targeting import config
from poverty_targeting.build import load_tables
from poverty_targeting.label import build_label
from poverty_targeting.mpi_spec import MPISpec, load_mpi_spec
from poverty_targeting.registry import load_survey

METRICS = ["H", "A", "M0", "vulnerable", "severe", "health", "education", "living_standards"]

# OPHI MPI Methodological Note 58, footnote 7: the default list of safe water sources.
OPHI_WATER_SOURCES = [
    "piped_dwelling", "piped_yard", "piped_neighbour", "public_tap", "tubewell_borehole",
    "protected_well", "protected_spring", "rainwater",
]  # fmt: skip

# Each scenario: (indicator, setting, value) overrides applied to the spec.
SCENARIOS: dict[str, list[tuple[str, str, object]]] = {
    "baseline (adolescents excluded)": [],
    "adolescents: adult BMI cutoff": [("nutrition", "adolescents_15_19", "adult_cutoff")],
    "water: OPHI default source list": [("drinking_water", "improved", OPHI_WATER_SOURCES)],
    "both changes": [
        ("nutrition", "adolescents_15_19", "adult_cutoff"),
        ("drinking_water", "improved", OPHI_WATER_SOURCES),
    ],
}
# The two adolescent rules under- and over-state nutrition deprivation, so together
# they bound what the official WHO BMI-for-age rule would give.
BOUNDS = ("baseline (adolescents excluded)", "adolescents: adult BMI cutoff")


def with_overrides(spec: MPISpec, overrides: list[tuple[str, str, object]]) -> MPISpec:
    """A copy of the spec with some indicator settings replaced (the original is untouched)."""
    indicators = {name: dict(rules) for name, rules in spec.indicators.items()}
    for indicator, setting, value in overrides:
        indicators[indicator][setting] = value
    return dataclasses.replace(spec, indicators=indicators)


def dimension_contributions(contributions: dict[str, float], spec: MPISpec) -> dict[str, float]:
    """Sum indicator contributions to M0 within each dimension."""
    return {
        dim: sum(contributions[ind] for ind in d["indicators"])
        for dim, d in spec.dimensions.items()
    }


def run_scenarios(tables: dict[str, pd.DataFrame], spec: MPISpec) -> pd.DataFrame:
    rows = []
    for name, overrides in SCENARIOS.items():
        variant = with_overrides(spec, overrides)
        summary = build_label(tables, variant).summary
        dims = dimension_contributions(summary["contributions"], variant)
        rows.append({"scenario": name, **{m: summary[m] for m in METRICS[:5]}, **dims})
    return pd.DataFrame(rows).set_index("scenario")[METRICS]


def within_bounds(results: pd.DataFrame, reference: dict) -> dict[str, bool]:
    """Is each published figure between the two adolescent-rule bounds?"""
    low = results.loc[list(BOUNDS)].min()
    high = results.loc[list(BOUNDS)].max()
    return {m: bool(low[m] <= reference[m] <= high[m]) for m in METRICS}


def render_report(survey_id: str, spec: MPISpec, results: pd.DataFrame, reference: dict) -> str:
    def fmt(metric: str, value: float) -> str:
        return f"{value:.3f}" if metric == "M0" else f"{value:.1%}"

    header = "| Scenario | " + " | ".join(METRICS) + " |"
    rule = "|---" * (len(METRICS) + 1) + "|"
    lines = [
        f"# Validation: {survey_id}, {spec.name}",
        "",
        f"Published figures: {reference['source']} ({reference['url']}).",
        "",
        header,
        rule,
        "| **published** | " + " | ".join(fmt(m, reference[m]) for m in METRICS) + " |",
    ]
    for scenario, row in results.iterrows():
        lines.append(f"| {scenario} | " + " | ".join(fmt(m, row[m]) for m in METRICS) + " |")

    checks = within_bounds(results, reference)
    inside = [m for m, ok in checks.items() if ok]
    outside = [m for m, ok in checks.items() if not ok]
    lines += [
        "",
        "H, A and M0 are population-weighted; health, education and living_standards are each",
        "dimension's contribution to M0. Published figure within the adolescent-rule bounds:",
        f"inside for {', '.join(inside) or 'none'}; outside for {', '.join(outside) or 'none'}.",
    ]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Validate the MPI label against published figures."
    )
    parser.add_argument("survey", help="Survey ID, e.g. GH2022DHS")
    parser.add_argument("--spec", default="global_mpi", help="MPI spec ID (default: global_mpi)")
    args = parser.parse_args(argv)

    spec = load_mpi_spec(args.spec)
    reference = load_survey(args.survey).references.get(spec.spec_id)
    if reference is None:
        raise SystemExit(f"{args.survey}: no [reference.{spec.spec_id}] in its survey config")

    results = run_scenarios(load_tables(args.survey), spec)
    report = render_report(args.survey, spec, results, reference)
    out = Path(config.REPORTS_DIR) / "validation" / f"{args.survey}_{spec.spec_id}.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report, newline="\n")
    print(report)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
