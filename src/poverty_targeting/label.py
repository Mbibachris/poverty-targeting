"""The MPI label: who is multidimensionally poor, and how poor, household by household.

Indicators (0/1/NA per household) -> devecon assembles the deprivation matrix
(applying the spec's missing-data rule) -> devecon.af_identify gives each
household its deprivation score and poor/non-poor flag -> population-weighted
H, A and M0 for the survey.

Run from the terminal, e.g.:
    python -m poverty_targeting.label GH2022DHS
"""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import devecon
import numpy as np
import pandas as pd

from poverty_targeting import config
from poverty_targeting.build import load_tables
from poverty_targeting.indicators import build_indicators
from poverty_targeting.mpi_spec import MPISpec, load_mpi_spec

VULNERABLE_FLOOR = 0.2  # OPHI: "vulnerable" if 20% <= score < poverty cutoff
SEVERE_CUTOFF = 0.5  # OPHI: "severe" poverty if score >= 50%


@dataclass
class Label:
    table: pd.DataFrame  # one row per household
    summary: dict


def usual_member_counts(persons: pd.DataFrame, hh_ids: pd.Series) -> np.ndarray:
    usual = persons[persons["usual_resident"].fillna(False).to_numpy(dtype=bool)]
    return usual.groupby("hh_id").size().reindex(hh_ids, fill_value=0).to_numpy()


def label_from_indicators(
    indicators: pd.DataFrame,
    households: pd.DataFrame,
    persons: pd.DataFrame,
    spec: MPISpec,
) -> Label:
    """Identify the poor and aggregate, given the indicator table (same row order as households)."""
    names, weights = spec.weights()
    k = spec.poverty_cutoff

    # devecon assembles the matrix and applies the missing-data rule (e.g. "drop").
    flags = indicators[names].astype("float64")
    rules = [{"name": n, "column": n, "op": ">=", "cutoff": 1} for n in names]
    _, matrix, kept = devecon.build_deprivation_matrix(flags, rules, missing=spec.missing_data)
    ident = devecon.af_identify(matrix, k, weights)

    # MPI poverty is about people: weight each household by its usual members.
    n_usual = usual_member_counts(persons, households["hh_id"])
    pop_weight = households["weight"].to_numpy(dtype=float) * n_usual

    score = np.full(len(indicators), np.nan)
    score[kept] = ident.scores
    poor = np.full(len(indicators), np.nan)
    poor[kept] = ident.poor
    table = indicators.copy()
    table["in_sample"] = kept
    table["n_usual_members"] = pd.array(n_usual, dtype="Int64")
    table["pop_weight"] = pop_weight
    table["deprivation_score"] = pd.array(score, dtype="Float64")
    table["mpi_poor"] = pd.array(poor, dtype="Float64").astype("boolean")

    s = pop_weight[kept]
    result = devecon.alkire_foster(matrix, k, weights, sample_weights=s)
    parts = devecon.af_decompose(matrix, k, weights, sample_weights=s, indicator_names=names)
    scores = ident.scores
    summary = {
        "survey_id": str(households["survey_id"].iloc[0]),
        "spec_id": spec.spec_id,
        "poverty_cutoff": k,
        "H": result.H,
        "A": result.A,
        "M0": result.M0,
        "vulnerable": float(s[(scores >= VULNERABLE_FLOOR) & (scores < k)].sum() / s.sum()),
        "severe": float(s[scores >= SEVERE_CUTOFF].sum() / s.sum()),
        "households_total": int(len(indicators)),
        "households_in_sample": int(kept.sum()),
        "households_dropped": int((~kept).sum()),
        "censored_headcounts": dict(zip(names, map(float, parts.censored_headcounts), strict=True)),
        "contributions": dict(zip(names, map(float, parts.contributions), strict=True)),
    }
    return Label(table=table, summary=summary)


def build_label(tables: dict[str, pd.DataFrame], spec: MPISpec) -> Label:
    indicators = build_indicators(tables, spec)
    return label_from_indicators(indicators, tables["households"], tables["persons"], spec)


def write_label(label: Label, out_dir: Path | None = None) -> Path:
    survey_id, spec_id = label.summary["survey_id"], label.summary["spec_id"]
    out_dir = Path(out_dir or config.processed_survey_dir(survey_id))
    out_dir.mkdir(parents=True, exist_ok=True)
    label.table.to_parquet(out_dir / f"label_{spec_id}.parquet", index=False)
    (out_dir / f"label_{spec_id}.json").write_text(
        json.dumps(label.summary, indent=2), newline="\n"
    )
    return out_dir


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build the MPI label for surveys.")
    parser.add_argument("surveys", nargs="+", help="Survey IDs, e.g. GH2022DHS")
    parser.add_argument("--spec", default="global_mpi", help="MPI spec ID (default: global_mpi)")
    args = parser.parse_args(argv)

    spec = load_mpi_spec(args.spec)
    for survey_id in args.surveys:
        label = build_label(load_tables(survey_id), spec)
        out_dir = write_label(label)
        s = label.summary
        print(
            f"{survey_id} [{spec.spec_id}]: H={s['H']:.1%}  A={s['A']:.1%}  MPI={s['M0']:.3f}  "
            f"vulnerable={s['vulnerable']:.1%}  severe={s['severe']:.1%}  "
            f"households={s['households_in_sample']:,} (dropped {s['households_dropped']:,})"
            f" -> {out_dir}"
        )


if __name__ == "__main__":
    main()
