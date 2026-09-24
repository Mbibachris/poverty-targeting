"""Calibration check and targeting curve for the chosen model of each tier.

1. Calibration. A recalibration layer (Platt or isotonic) is only worth adding if it
   improves the Brier score, a proper scoring rule, on predictions it was not fitted
   to (cross-fitted over folds 1-4). ECE alone can be gamed by binning, so it is
   reported but does not decide. Ties (within BRIER_TOLERANCE) keep "none".
2. Targeting curve. For a range of budgets (share of the population a programme can
   cover): the score cut-off, coverage of the poor, exclusion and inclusion errors,
   next to what random selection would achieve with the same budget.

Run from the terminal, e.g.:
    python -m poverty_targeting.targeting GH2022DHS
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from poverty_targeting import config
from poverty_targeting.dataset import TARGET, TEST_FOLD
from poverty_targeting.features import load_feature_set
from poverty_targeting.metrics import (
    calibration,
    discrimination,
    select_by_budget,
    targeting_errors,
)
from poverty_targeting.modeling import cross_validate

CALIBRATORS = ("none", "platt", "isotonic")
BRIER_TOLERANCE = 0.0005
BUDGETS = (0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60)


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p)).reshape(-1, 1)


def cross_fit_calibration(p, y, w, folds, method: str) -> np.ndarray:
    """Recalibrate each fold's predictions with a calibrator fitted on the other folds."""
    if method == "none":
        return np.asarray(p, dtype=float)
    out = np.empty(len(p))
    for fold in np.unique(folds):
        fit = folds != fold
        if method == "platt":
            calibrator = LogisticRegression().fit(_logit(p[fit]), y[fit], sample_weight=w[fit])
            out[~fit] = calibrator.predict_proba(_logit(p[~fit]))[:, 1]
        elif method == "isotonic":
            calibrator = IsotonicRegression(out_of_bounds="clip")
            calibrator.fit(p[fit], y[fit], sample_weight=w[fit])
            out[~fit] = calibrator.predict(p[~fit])
        else:
            raise ValueError(f"unknown calibration method '{method}'; use one of {CALIBRATORS}")
    return out


def calibration_check(p, y, w, folds) -> dict[str, dict]:
    """Brier, ECE and ROC-AUC with each calibration method (cross-fitted)."""
    check = {}
    for method in CALIBRATORS:
        q = cross_fit_calibration(p, y, w, folds, method)
        cal = calibration(y, q, w)
        check[method] = {
            "brier": cal["brier"],
            "ece": cal["ece"],
            "roc_auc": discrimination(y, q, w)["roc_auc"],
        }
    return check


def choose_calibration(check: dict[str, dict]) -> str:
    """Lowest Brier score; 'none' unless another method beats it by more than the tolerance."""
    best = min(check, key=lambda m: check[m]["brier"])
    if check["none"]["brier"] - check[best]["brier"] <= BRIER_TOLERANCE:
        return "none"
    return best


def budget_curve(p, y, w, budgets=BUDGETS) -> pd.DataFrame:
    """Targeting outcomes at each budget, next to random selection with the same budget."""
    y = np.asarray(y).astype(bool)
    poverty_rate = w[y].sum() / w.sum()
    rows = []
    for share in budgets:
        selected = select_by_budget(p, w, share)
        e = targeting_errors(y, selected, w)
        rows.append(
            {
                "budget": share,
                "score_cutoff": float(p[selected].min()),
                "coverage_poor": e["coverage_poor"],
                "exclusion_error": e["exclusion_error"],
                "inclusion_error": e["inclusion_error"],
                "random_coverage_poor": share,  # random selection reaches `share` of the poor
                "random_inclusion_error": float(1 - poverty_rate),
            }
        )
    return pd.DataFrame(rows)


def render(survey_id: str, tier: str, model: str, check: dict, chosen: str, curve) -> str:
    lines = [
        f"# Targeting: {survey_id}, tier {tier} ({model})",
        "",
        "Out-of-fold predictions on folds 1-4; population-weighted; test fold unused.",
        "",
        "## Calibration check (cross-fitted)",
        "",
        "| Method | Brier | ECE | ROC-AUC |",
        "|---|---|---|---|",
    ]
    for method, m in check.items():
        lines.append(f"| {method} | {m['brier']:.4f} | {m['ece']:.4f} | {m['roc_auc']:.4f} |")
    lines += [
        "",
        f"Chosen: **{chosen}** (Brier decides; ties within {BRIER_TOLERANCE} keep 'none').",
        "",
        "## Targeting curve",
        "",
        "| Budget | Score cut-off | Poor reached | Exclusion | Inclusion | "
        "Random: poor reached | Random: inclusion |",
        "|---|---|---|---|---|---|---|",
    ]
    for _, r in curve.iterrows():
        lines.append(
            f"| {r['budget']:.0%} | {r['score_cutoff']:.3f} | {r['coverage_poor']:.1%} | "
            f"{r['exclusion_error']:.1%} | {r['inclusion_error']:.1%} | "
            f"{r['random_coverage_poor']:.1%} | {r['random_inclusion_error']:.1%} |"
        )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Calibration check and targeting curves.")
    parser.add_argument("survey", help="Survey ID, e.g. GH2022DHS")
    parser.add_argument("--spec", default="global_mpi")
    parser.add_argument("--features", default="pmt")
    args = parser.parse_args(argv)

    folder = config.processed_survey_dir(args.survey)
    data = pd.read_parquet(folder / f"dataset_{args.features}_{args.spec}.parquet")
    feature_set = load_feature_set(args.features)
    reports = Path(config.REPORTS_DIR)
    choices = json.loads((reports / "cv" / f"{args.survey}_choices.json").read_text())

    train = data[data["fold"] != TEST_FOLD].reset_index(drop=True)
    y = train[TARGET].to_numpy()
    w = train["pop_weight"].to_numpy(dtype=float)
    folds = train["fold"].to_numpy()

    out_dir = reports / "targeting"
    out_dir.mkdir(parents=True, exist_ok=True)
    for tier, model in choices.items():
        p = cross_validate(data, feature_set.tiers[tier], model)["predictions"]
        check = calibration_check(p, y, w, folds)
        chosen = choose_calibration(check)
        curve = budget_curve(cross_fit_calibration(p, y, w, folds, chosen), y, w)
        report = render(args.survey, tier, model, check, chosen, curve)
        (out_dir / f"{args.survey}_tier{tier}.md").write_text(report, newline="\n")
        summary = {"model": model, "calibration": chosen, "check": check}
        summary["curve"] = curve.to_dict(orient="records")
        (out_dir / f"{args.survey}_tier{tier}.json").write_text(
            json.dumps(summary, indent=2), newline="\n"
        )
        print(report)


if __name__ == "__main__":
    main()
