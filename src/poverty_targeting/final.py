"""Final model: train on folds 1-4, evaluate ONCE on the sealed test fold, package it.

Outputs:
- models/<survey>_tier<T>_<model>.joblib: the fitted pipeline plus everything needed to
  serve it (feature list, questions, budget-to-cut-off table). Contains no survey records.
- models/<survey>_tier<T>_<model>.metrics.json: test-fold results.
- models/<survey>_tier<T>_<model>.model_card.md: what the model is, how well it works,
  for whom it works less well, and what it must not be used for.

Explanations use LightGBM's built-in TreeSHAP (pred_contrib): each prediction's
log-odds split into per-feature contributions that add up exactly. One-hot columns
are summed back to the question they came from, so explanations speak form language.

Run from the terminal, e.g.:
    python -m poverty_targeting.final GH2022DHS --tier B
"""

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from poverty_targeting import __version__, config
from poverty_targeting.dataset import TARGET, TEST_FOLD
from poverty_targeting.fairness import groupings, subgroup_table
from poverty_targeting.feature_table import COMPUTED_FEATURES
from poverty_targeting.features import load_feature_set
from poverty_targeting.metrics import evaluate, select_by_budget
from poverty_targeting.modeling import MODELS, cross_validate, split_columns, to_model_input
from poverty_targeting.serving import contributions
from poverty_targeting.targeting import budget_curve

BUDGET_GRID = np.round(np.arange(0.01, 1.0, 0.01), 2)


def budget_cutoffs(scores, weights) -> dict[float, float]:
    """For each budget share, the lowest score that still gets selected."""
    scores = np.asarray(scores, dtype=float)
    return {
        float(b): float(scores[select_by_budget(scores, weights, float(b))].min())
        for b in BUDGET_GRID
    }


def fit_final(data: pd.DataFrame, features: list[str], model_name: str):
    """Fit on every non-test fold; return the pipeline and the train/test split."""
    train = data[data["fold"] != TEST_FOLD].reset_index(drop=True)
    test = data[data["fold"] == TEST_FOLD].reset_index(drop=True)
    categorical, numeric = split_columns(train, features)
    pipeline = MODELS[model_name](categorical, numeric)
    w = train["pop_weight"].to_numpy(dtype=float)
    pipeline.fit(to_model_input(train, features), train[TARGET], model__sample_weight=w / w.mean())
    return pipeline, train, test


def model_card(meta: dict, test_eval: dict, curve: pd.DataFrame, fairness: pd.DataFrame) -> str:
    b = test_eval["at_budget"]
    flagged = fairness[fairness["flag_worse"]]
    lines = [
        f"# Model card: {meta['name']}",
        "",
        "## What it does",
        f"Estimates the probability that a household is multidimensionally poor under the "
        f"{meta['spec_id']} definition, from {len(meta['features'])} questionnaire answers. "
        f"Model: {meta['model']} (tier {meta['tier']}); trained on {meta['survey_id']} "
        f"(folds 1-4, {meta['n_train']:,} households); package version {meta['version']}.",
        "",
        "## Intended use",
        "Ranking households for a social-protection programme with a fixed budget, as one "
        "input to a process that includes community validation and appeals. **Not** for "
        "automated eligibility decisions without human review.",
        "",
        "## Test-fold performance (fold 0, used once)",
        f"- Households: {test_eval['n']:,}; "
        f"population poverty rate {test_eval['poverty_rate']:.1%}",
        f"- ROC-AUC {test_eval['roc_auc']:.3f}; PR-AUC {test_eval['pr_auc']:.3f}; "
        f"Brier {test_eval['calibration']['brier']:.4f}; ECE {test_eval['calibration']['ece']:.3f}",
        f"- At a {b['budget_share']:.1%} budget: {b['coverage_poor']:.1%} of poor people reached; "
        f"exclusion error {b['exclusion_error']:.1%}; inclusion error {b['inclusion_error']:.1%}",
        "",
        "## Targeting curve (test fold)",
        "",
        "| Budget | Poor reached | Inclusion error |",
        "|---|---|---|",
        *[
            f"| {r.budget:.0%} | {r.coverage_poor:.1%} | {r.inclusion_error:.1%} |"
            for r in curve.itertuples()
        ],
        "",
        "## Who it serves less well (test fold, 95% cluster-bootstrap intervals)",
    ]
    if flagged.empty:
        lines.append("No group's exclusion error is significantly above the overall rate.")
    for r in flagged.itertuples():
        lines.append(
            f"- {r.grouping} = {r.group}: exclusion {r.exclusion_error:.1%} "
            f"({r.exclusion_low:.0%}-{r.exclusion_high:.0%})"
        )
    lines += [
        "",
        "## Known limitations",
        "- The label is the global MPI computed from DHS data; child mortality uses births in "
        "the last five years (KR), which undercounts; adolescent nutrition is excluded.",
        "- Tier B includes items that are also MPI inputs, so part of its accuracy is the "
        "label's own ingredients (see reports/audit and the Tier A benchmark in reports/cv).",
        "- Poor urban and female-headed households are reached less often (see above).",
        "- One country and one survey year; accuracy elsewhere or later is unknown.",
        "",
        "## Questions the model uses",
        *[f"- `{f}`: {meta['questions'][f]}" for f in meta["features"]],
    ]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train, test once, and package the model.")
    parser.add_argument("survey", help="Survey ID, e.g. GH2022DHS")
    parser.add_argument("--tier", default="B")
    parser.add_argument("--spec", default="global_mpi")
    parser.add_argument("--features", default="pmt")
    args = parser.parse_args(argv)

    folder = config.processed_survey_dir(args.survey)
    data = pd.read_parquet(folder / f"dataset_{args.features}_{args.spec}.parquet")
    feature_set = load_feature_set(args.features)
    features = feature_set.tiers[args.tier]
    choices = json.loads(
        (Path(config.REPORTS_DIR) / "cv" / f"{args.survey}_choices.json").read_text()
    )
    model_name = choices[args.tier]

    pipeline, train, test = fit_final(data, features, model_name)
    w_train = train["pop_weight"].to_numpy(dtype=float)
    budget = float(w_train[train[TARGET] == 1].sum() / w_train.sum())  # fixed before testing

    y, w = test[TARGET].to_numpy(), test["pop_weight"].to_numpy(dtype=float)
    p = pipeline.predict_proba(to_model_input(test, features))[:, 1]
    test_eval = evaluate(y, p, w, budget_share=budget)
    curve = budget_curve(p, y, w)
    fairness, overall = subgroup_table(
        y, select_by_budget(p, w, budget), w, groupings(test), test["cluster"].to_numpy()
    )
    importance = contributions(pipeline, to_model_input(test, features), features)
    importance = importance[features].abs().mean().sort_values(ascending=False)

    oof = cross_validate(data, features, model_name)["predictions"]
    name = f"{args.survey}_tier{args.tier}_{model_name}"
    meta = {
        "name": name,
        "survey_id": args.survey,
        "spec_id": args.spec,
        "feature_set": args.features,
        "tier": args.tier,
        "model": model_name,
        "version": __version__,
        "trained_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "features": features,
        "questions": {f: feature_set.features[f].question for f in features},
        "n_train": int(len(train)),
        # Answers the model has seen missing in training; every other answer is required.
        "optional_answers": [
            f for f in features if f not in COMPUTED_FEATURES and train[f].isna().any()
        ],
        "default_budget": budget,
        "budget_cutoffs": budget_cutoffs(oof, w_train),  # from cross-validated scores
    }

    out_dir = Path(config.MODELS_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump({"pipeline": pipeline, "meta": meta}, out_dir / f"{name}.joblib")
    results = {
        "meta": {k: v for k, v in meta.items() if k != "budget_cutoffs"},
        "test": test_eval,
        "test_overall_exclusion": overall,
        "global_importance": importance.round(4).to_dict(),
    }
    (out_dir / f"{name}.metrics.json").write_text(json.dumps(results, indent=2), newline="\n")
    card = model_card(meta, test_eval, curve, fairness)
    (out_dir / f"{name}.model_card.md").write_text(card, newline="\n")
    print(card)
    print("Global importance (mean |log-odds contribution|):")
    print(importance.round(3).to_string())


if __name__ == "__main__":
    main()
