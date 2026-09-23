"""Model training and cross-validation on the training folds.

Every model is a scikit-learn Pipeline: preprocessing + estimator in one object, so
exactly the same transformations are applied when the model is trained, evaluated
and later served. Cross-validation uses folds 1-4 only; fold 0 (the test set) is
never touched here.

Run from the terminal, e.g.:
    python -m poverty_targeting.modeling GH2022DHS --model logit --tier A
"""

import argparse
import json
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from poverty_targeting import config
from poverty_targeting.dataset import SEED, TARGET, TEST_FOLD
from poverty_targeting.features import load_feature_set
from poverty_targeting.metrics import discrimination, evaluate


def split_columns(data: pd.DataFrame, features: list[str]) -> tuple[list[str], list[str]]:
    """Categorical (text) vs numeric (numbers and yes/no) feature columns."""
    categorical = [f for f in features if not pd.api.types.is_numeric_dtype(data[f])]
    numeric = [f for f in features if f not in categorical]
    return categorical, numeric


def to_model_input(data: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """Plain numpy-friendly columns: numbers as float (NaN = missing), text as object."""
    categorical, numeric = split_columns(data, features)
    out = pd.DataFrame(index=data.index)
    for f in numeric:
        out[f] = data[f].astype("Float64").to_numpy(dtype=float, na_value=np.nan)
    for f in categorical:
        out[f] = data[f].astype(object).where(data[f].notna(), None)
    return out[features]


def make_logit(categorical: list[str], numeric: list[str]) -> Pipeline:
    """Logistic regression: the classic proxy means test."""
    preprocess = ColumnTransformer(
        [
            (
                "numeric",
                make_pipeline(
                    SimpleImputer(strategy="median", add_indicator=True), StandardScaler()
                ),
                numeric,
            ),
            (
                "categorical",
                make_pipeline(
                    SimpleImputer(strategy="constant", fill_value="missing"),
                    OneHotEncoder(handle_unknown="ignore"),
                ),
                categorical,
            ),
        ]
    )
    return Pipeline([("preprocess", preprocess), ("model", LogisticRegression(max_iter=5000))])


def make_lightgbm(categorical: list[str], numeric: list[str]) -> Pipeline:
    """Gradient-boosted trees, deliberately regularised: ~7,000 training households.

    Shallow trees (7 leaves), many small steps, large minimum leaf size and an L2
    penalty. Missing numbers are handled natively by LightGBM, so no imputation.
    """
    preprocess = ColumnTransformer(
        [
            ("numeric", "passthrough", numeric),
            (
                "categorical",
                make_pipeline(
                    SimpleImputer(strategy="constant", fill_value="missing"),
                    OneHotEncoder(handle_unknown="ignore"),
                ),
                categorical,
            ),
        ]
    )
    model = LGBMClassifier(
        n_estimators=800,
        learning_rate=0.02,
        num_leaves=7,
        min_child_samples=80,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        reg_lambda=5.0,
        random_state=SEED,
        verbose=-1,
    )
    return Pipeline([("preprocess", preprocess), ("model", model)])


MODELS: dict[str, Callable[[list[str], list[str]], Pipeline]] = {
    "logit": make_logit,
    "lightgbm": make_lightgbm,
}


def _normalised(weights: pd.Series) -> np.ndarray:
    w = weights.to_numpy(dtype=float)
    return w / w.mean()


def cross_validate(data: pd.DataFrame, features: list[str], model_name: str) -> dict:
    """Out-of-fold predictions on folds 1-4 (fold 0 untouched) and their evaluation."""
    train = data[data["fold"] != TEST_FOLD].reset_index(drop=True)
    categorical, numeric = split_columns(train, features)
    X = to_model_input(train, features)
    y = train[TARGET].to_numpy()
    w = train["pop_weight"]

    oof = np.full(len(train), np.nan)
    per_fold = {}
    for fold in sorted(train["fold"].unique()):
        fit_rows = (train["fold"] != fold).to_numpy()
        model = MODELS[model_name](categorical, numeric)
        model.fit(X[fit_rows], y[fit_rows], model__sample_weight=_normalised(w[fit_rows]))
        oof[~fit_rows] = model.predict_proba(X[~fit_rows])[:, 1]
        per_fold[int(fold)] = discrimination(y[~fit_rows], oof[~fit_rows], w[~fit_rows])

    return {
        "model": model_name,
        "features": features,
        "out_of_fold": evaluate(y, oof, w.to_numpy()),
        "per_fold": per_fold,
        "predictions": oof,
    }


def summary_line(result: dict, label: str) -> str:
    r = result["out_of_fold"]
    b = r["at_budget"]
    aucs = [f["roc_auc"] for f in result["per_fold"].values()]
    return (
        f"{label}: ROC-AUC {r['roc_auc']:.3f} (folds {min(aucs):.3f}-{max(aucs):.3f})  "
        f"PR-AUC {r['pr_auc']:.3f}  ECE {r['calibration']['ece']:.3f}  |  "
        f"at budget {b['budget_share']:.1%}: exclusion {b['exclusion_error']:.1%}, "
        f"inclusion {b['inclusion_error']:.1%}"
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Cross-validate a targeting model.")
    parser.add_argument("survey", help="Survey ID, e.g. GH2022DHS")
    parser.add_argument("--model", default="logit", choices=sorted(MODELS))
    parser.add_argument("--tier", default="A")
    parser.add_argument("--spec", default="global_mpi")
    parser.add_argument("--features", default="pmt")
    args = parser.parse_args(argv)

    folder = config.processed_survey_dir(args.survey)
    data = pd.read_parquet(folder / f"dataset_{args.features}_{args.spec}.parquet")
    features = load_feature_set(args.features).tiers[args.tier]
    result = cross_validate(data, features, args.model)

    label = f"{args.model} / tier {args.tier}"
    report = {k: v for k, v in result.items() if k != "predictions"}
    out = Path(config.REPORTS_DIR) / "cv" / f"{args.survey}_{args.model}_tier{args.tier}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), newline="\n")
    print(summary_line(result, label))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
