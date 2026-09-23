"""Compare every model on every feature tier, on identical folds, and pick one per tier.

Selection rule (fixed before the test fold is ever used): keep the simpler baseline
unless a challenger beats it in at least MIN_FOLD_WINS of the four folds AND by at
least MIN_MEAN_GAIN in mean ROC-AUC. Small gains inside fold-to-fold noise do not
justify a more complex model that is harder to explain, calibrate and serve.

Run from the terminal, e.g.:
    python -m poverty_targeting.compare GH2022DHS
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from poverty_targeting import config
from poverty_targeting.features import load_feature_set
from poverty_targeting.modeling import MODELS, cross_validate

BASELINE = "logit"
MIN_FOLD_WINS = 3
MIN_MEAN_GAIN = 0.005


def paired_gain(challenger: dict, baseline: dict) -> dict:
    """Per-fold ROC-AUC difference, challenger minus baseline (same folds, so paired)."""
    diffs = {
        fold: challenger["per_fold"][fold]["roc_auc"] - baseline["per_fold"][fold]["roc_auc"]
        for fold in baseline["per_fold"]
    }
    values = np.array(list(diffs.values()))
    return {
        "per_fold": {int(f): round(float(d), 4) for f, d in diffs.items()},
        "mean": float(values.mean()),
        "wins": int((values > 0).sum()),
    }


def choose(results: dict[str, dict]) -> tuple[str, dict[str, dict]]:
    """The model to use for one tier, and every challenger's paired gain over the baseline."""
    gains = {
        name: paired_gain(result, results[BASELINE])
        for name, result in results.items()
        if name != BASELINE
    }
    qualifying = {
        name: g
        for name, g in gains.items()
        if g["wins"] >= MIN_FOLD_WINS and g["mean"] >= MIN_MEAN_GAIN
    }
    if not qualifying:
        return BASELINE, gains
    return max(qualifying, key=lambda name: qualifying[name]["mean"]), gains


def leaderboard(rows: list[dict]) -> str:
    header = (
        "| Tier | Model | ROC-AUC | PR-AUC | ECE | Exclusion @ budget | "
        "AUC gain vs logit (folds won) | Chosen |"
    )
    lines = [header, "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        gain = "-" if r["gain"] is None else f"{r['gain']['mean']:+.4f} ({r['gain']['wins']}/4)"
        lines.append(
            f"| {r['tier']} | {r['model']} | {r['roc_auc']:.3f} | {r['pr_auc']:.3f} | "
            f"{r['ece']:.3f} | {r['exclusion']:.1%} | {gain} | {'yes' if r['chosen'] else ''} |"
        )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Compare models across feature tiers.")
    parser.add_argument("survey", help="Survey ID, e.g. GH2022DHS")
    parser.add_argument("--spec", default="global_mpi")
    parser.add_argument("--features", default="pmt")
    args = parser.parse_args(argv)

    folder = config.processed_survey_dir(args.survey)
    data = pd.read_parquet(folder / f"dataset_{args.features}_{args.spec}.parquet")
    feature_set = load_feature_set(args.features)

    rows, choices = [], {}
    for tier, features in feature_set.tiers.items():
        results = {name: cross_validate(data, features, name) for name in MODELS}
        chosen, gains = choose(results)
        choices[tier] = chosen
        for name, result in results.items():
            r = result["out_of_fold"]
            rows.append(
                {
                    "tier": tier,
                    "model": name,
                    "roc_auc": r["roc_auc"],
                    "pr_auc": r["pr_auc"],
                    "ece": r["calibration"]["ece"],
                    "exclusion": r["at_budget"]["exclusion_error"],
                    "gain": gains.get(name),
                    "chosen": name == chosen,
                }
            )

    table = leaderboard(rows)
    out_dir = Path(config.REPORTS_DIR) / "cv"
    out_dir.mkdir(parents=True, exist_ok=True)
    rule = (
        f"Selection rule: keep '{BASELINE}' unless a challenger wins in >= {MIN_FOLD_WINS} "
        f"of 4 folds and gains >= {MIN_MEAN_GAIN} mean ROC-AUC. Folds 1-4 only; test fold unused."
    )
    report = f"# Model comparison: {args.survey}\n\n{rule}\n\n{table}"
    (out_dir / f"{args.survey}_leaderboard.md").write_text(report, newline="\n")
    (out_dir / f"{args.survey}_choices.json").write_text(
        json.dumps(choices, indent=2), newline="\n"
    )
    print(table)
    print(f"Chosen per tier: {choices}")


if __name__ == "__main__":
    main()
