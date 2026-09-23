"""Training dataset: labelled households + their features, split into folds by cluster.

Only households with a label (in_sample) are kept. Folds are built so that:
- every household of a survey cluster lands in the same fold (grouped), because
  neighbours share schools, water points and prices; splitting a cluster across
  train and test would let a model recognise places instead of households;
- each fold has about the same share of poor households (stratified).
Fold 0 is the held-out test set; folds 1-4 are for cross-validation.

Run from the terminal, e.g.:
    python -m poverty_targeting.dataset GH2022DHS
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from poverty_targeting import config
from poverty_targeting.build import load_tables
from poverty_targeting.feature_table import build_feature_table
from poverty_targeting.features import load_feature_set

N_FOLDS = 5
TEST_FOLD = 0
SEED = 2026
TARGET = "mpi_poor"


def assemble(features: pd.DataFrame, label: pd.DataFrame, households: pd.DataFrame) -> pd.DataFrame:
    """Join features to the label for labelled households; add the columns evaluation needs."""
    labelled = label.loc[label["in_sample"], ["hh_id", TARGET, "deprivation_score", "pop_weight"]]
    data = features.merge(labelled, on="hh_id", how="inner", validate="one_to_one")
    data = data.merge(households[["hh_id", "cluster"]], on="hh_id", how="left", validate="1:1")
    data[TARGET] = data[TARGET].astype(bool).astype("int8")
    return data


def assign_folds(target: pd.Series, groups: pd.Series, n_folds: int = N_FOLDS, seed: int = SEED):
    """Fold number per row: grouped by cluster, stratified by the target, reproducible."""
    splitter = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    folds = np.full(len(target), -1, dtype="int8")
    for fold, (_, test_idx) in enumerate(splitter.split(np.zeros(len(target)), target, groups)):
        folds[test_idx] = fold
    return pd.Series(folds, index=target.index, name="fold")


def fold_summary(data: pd.DataFrame) -> dict:
    by_fold = data.groupby("fold")
    return {
        "households": int(len(data)),
        "clusters": int(data["cluster"].nunique()),
        "poor_rate": float(data[TARGET].mean()),
        "folds": {
            int(fold): {
                "households": int(len(g)),
                "clusters": int(g["cluster"].nunique()),
                "poor_rate": round(float(g[TARGET].mean()), 4),
            }
            for fold, g in by_fold
        },
        "clusters_in_more_than_one_fold": int(
            (data.groupby("cluster")["fold"].nunique() > 1).sum()
        ),
    }


def build_dataset(survey_id: str, spec_id: str = "global_mpi", set_id: str = "pmt") -> pd.DataFrame:
    """Labelled households with every feature of the set's widest tier, plus folds."""
    tables = load_tables(survey_id)
    label_path = config.processed_survey_dir(survey_id) / f"label_{spec_id}.parquet"
    if not label_path.exists():
        raise FileNotFoundError(
            f"{label_path} not found. Run: python -m poverty_targeting.label {survey_id}"
        )
    feature_set = load_feature_set(set_id)
    widest = max(feature_set.tiers, key=lambda t: len(feature_set.tiers[t]))
    features = build_feature_table(tables, feature_set, widest)

    data = assemble(features, pd.read_parquet(label_path), tables["households"])
    data["fold"] = assign_folds(data[TARGET], data["cluster"])
    return data


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build the model training dataset.")
    parser.add_argument("survey", help="Survey ID, e.g. GH2022DHS")
    parser.add_argument("--spec", default="global_mpi")
    parser.add_argument("--features", default="pmt")
    args = parser.parse_args(argv)

    data = build_dataset(args.survey, args.spec, args.features)
    out_dir = Path(config.processed_survey_dir(args.survey))
    name = f"dataset_{args.features}_{args.spec}"
    data.to_parquet(out_dir / f"{name}.parquet", index=False)
    summary = fold_summary(data)
    (out_dir / f"{name}.json").write_text(json.dumps(summary, indent=2), newline="\n")
    print(json.dumps(summary, indent=2))
    print(f"-> {out_dir / name}.parquet")


if __name__ == "__main__":
    main()
