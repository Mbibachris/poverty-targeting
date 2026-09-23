"""Circularity audit: how strongly each feature is associated with each MPI indicator.

The feature-set check (features.py) catches *mechanical* overlap: a feature that is
itself a label input. This audit measures *statistical* association, which can be
strong even without overlap. Association is Somers' D = 2 x AUC - 1 of the feature
alone predicting an outcome: 0 means none, +/-1 means perfect. Categorical features
are scored by their outcome rate, so for them D measures strength only (always >= 0).

Computed on the training folds only; the test fold stays untouched.

Run from the terminal, e.g.:
    python -m poverty_targeting.audit GH2022DHS
"""

import argparse
from pathlib import Path

import pandas as pd
from sklearn.metrics import roc_auc_score

from poverty_targeting import config
from poverty_targeting.dataset import TARGET, TEST_FOLD
from poverty_targeting.features import FeatureSet, load_feature_set, overlaps
from poverty_targeting.mpi_spec import MPISpec, load_mpi_spec

PREFIX = "dep_"  # indicator columns are prefixed: features and indicators share names


def somers_d(x: pd.Series, y: pd.Series) -> float:
    """Association of one feature with a 0/1 outcome, between -1 and 1."""
    known = (x.notna() & y.notna()).to_numpy(dtype=bool)
    x, y = x[known], y[known].astype(int)
    if y.nunique() < 2:
        return float("nan")
    if not pd.api.types.is_numeric_dtype(x):
        x = x.map(y.groupby(x).mean())  # score each category by its outcome rate
    return float(2 * roc_auc_score(y, x.astype(float)) - 1)


def association_matrix(data: pd.DataFrame, features: list[str], outcomes: list[str]):
    """Features x outcomes table of Somers' D."""
    return pd.DataFrame({f: {o: somers_d(data[f], data[o]) for o in outcomes} for f in features}).T


def audit_table(feature_set: FeatureSet, spec: MPISpec, data: pd.DataFrame) -> pd.DataFrame:
    """One row per feature: its tier, mechanical overlap, and statistical association."""
    indicators, _ = spec.weights()
    widest = max(feature_set.tiers, key=lambda t: len(feature_set.tiers[t]))
    features = feature_set.tiers[widest]
    matrix = association_matrix(data, features, [PREFIX + i for i in indicators] + [TARGET])
    matrix.columns = [c.removeprefix(PREFIX) for c in matrix.columns]

    shared = overlaps(feature_set, spec, widest)
    first_tier = {}
    for tier in sorted(feature_set.tiers, key=lambda t: len(feature_set.tiers[t])):
        for f in feature_set.tiers[tier]:
            first_tier.setdefault(f, tier)

    strongest = matrix[indicators].abs().idxmax(axis=1)
    return pd.DataFrame(
        {
            "tier": [first_tier[f] for f in features],
            "shares_inputs_with": [", ".join(shared[f]) or "-" for f in features],
            "D_mpi_poor": matrix[TARGET].to_numpy(),
            "strongest_indicator": strongest.to_numpy(),
            "D_strongest": [matrix.loc[f, strongest[f]] for f in features],
        },
        index=pd.Index(features, name="feature"),
    ), matrix


def render_audit(survey_id: str, set_id: str, spec: MPISpec, table, matrix) -> str:
    lines = [
        f"# Circularity audit: {survey_id}, feature set '{set_id}' vs {spec.name}",
        "",
        "Somers' D of each feature alone (training folds only; test fold untouched).",
        "0 = no association, +/-1 = perfect. Categorical features show strength only.",
        "",
        "| Feature | Tier | Shares label inputs with | D with MPI-poor | Strongest indicator |",
        "|---|---|---|---|---|",
    ]
    for feature, row in table.iterrows():
        lines.append(
            f"| {feature} | {row['tier']} | {row['shares_inputs_with']} | "
            f"{row['D_mpi_poor']:+.2f} | {row['strongest_indicator']} "
            f"({row['D_strongest']:+.2f}) |"
        )
    lines += ["", "## Full matrix", "", "| Feature | " + " | ".join(matrix.columns) + " |"]
    lines.append("|---" * (len(matrix.columns) + 1) + "|")
    for feature, row in matrix.iterrows():
        lines.append(f"| {feature} | " + " | ".join(f"{v:+.2f}" for v in row) + " |")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Audit feature-label association.")
    parser.add_argument("survey", help="Survey ID, e.g. GH2022DHS")
    parser.add_argument("--spec", default="global_mpi")
    parser.add_argument("--features", default="pmt")
    args = parser.parse_args(argv)

    spec = load_mpi_spec(args.spec)
    feature_set = load_feature_set(args.features)
    indicators, _ = spec.weights()
    folder = config.processed_survey_dir(args.survey)
    data = pd.read_parquet(folder / f"dataset_{args.features}_{args.spec}.parquet")
    label = pd.read_parquet(folder / f"label_{args.spec}.parquet")
    label = label[["hh_id", *indicators]].rename(columns={i: PREFIX + i for i in indicators})
    train = data[data["fold"] != TEST_FOLD].merge(label, on="hh_id", validate="1:1")

    table, matrix = audit_table(feature_set, spec, train)
    report = render_audit(args.survey, args.features, spec, table, matrix)
    out = Path(config.REPORTS_DIR) / "audit" / f"{args.survey}_{args.features}_{args.spec}.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report, newline="\n")
    print(table.round(2).to_string())
    print(f"-> {out}")


if __name__ == "__main__":
    main()
