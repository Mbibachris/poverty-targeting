"""Fairness across subgroups: does the model miss the poor more often in some groups?

A single selection rule is applied to everyone (the budget is spent on the
highest scores overall); errors are then measured within each group. The key
quantity is the exclusion error among each group's poor: equal treatment means a
poor household is equally likely to be reached whoever it is.

Groups include household composition, because the circularity audit showed the
MPI's "any member" rules tie deprivation to having children.

Uncertainty: 95% intervals from a cluster bootstrap (whole survey clusters are
resampled, matching how the survey was drawn). A group is flagged when even the
lower end of its interval is above the overall exclusion error.

Run from the terminal, e.g.:
    python -m poverty_targeting.fairness GH2022DHS
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from poverty_targeting import config
from poverty_targeting.dataset import SEED, TARGET, TEST_FOLD
from poverty_targeting.features import load_feature_set
from poverty_targeting.metrics import select_by_budget, targeting_errors
from poverty_targeting.modeling import cross_validate

N_BOOT = 500


def groupings(data: pd.DataFrame) -> dict[str, pd.Series]:
    """Subgroup labels per household, built only from features a form would collect."""
    children = data["n_under5"] + data["n_children_5_17"]
    size = data["hh_size"].astype(float)
    return {
        "residence": data["urban"].map({True: "urban", False: "rural"}),
        "head_sex": data["head_sex"],
        "children": pd.Series(np.where(children > 0, "has children", "no children")),
        "elderly_only": pd.Series(
            np.where(data["n_elderly_65plus"] == data["hh_size"], "elderly only", "other")
        ),
        "household_size": pd.cut(size, [0, 2, 4, 6, np.inf], labels=["1-2", "3-4", "5-6", "7+"]),
        "region": data["region"],
    }


def _exclusion(y: np.ndarray, sel: np.ndarray, w: np.ndarray) -> float:
    poor = w[y].sum()
    return float(w[y & ~sel].sum() / poor) if poor else float("nan")


def bootstrap_exclusion(y, sel, w, groups, clusters, n_boot=N_BOOT, seed=SEED) -> dict:
    """95% interval of each group's exclusion error, resampling whole clusters."""
    rng = np.random.default_rng(seed)
    unique, cluster_idx = np.unique(clusters, return_inverse=True)
    draws = {g: [] for g in pd.unique(groups)}
    for _ in range(n_boot):
        counts = np.bincount(rng.integers(0, len(unique), len(unique)), minlength=len(unique))
        w_rep = w * counts[cluster_idx]  # a cluster drawn twice counts twice
        for g in draws:
            k = groups == g
            draws[g].append(_exclusion(y[k], sel[k], w_rep[k]))
    return {g: tuple(np.nanpercentile(v, [2.5, 97.5])) for g, v in draws.items()}


def subgroup_table(y, sel, w, labels: dict[str, pd.Series], clusters, n_boot=N_BOOT):
    """One row per group: size, poverty rate, selection rate, errors and a flag."""
    y = np.asarray(y).astype(bool)
    sel = np.asarray(sel).astype(bool)
    w = np.asarray(w, dtype=float)
    overall = _exclusion(y, sel, w)
    rows = []
    for grouping, series in labels.items():
        groups = pd.Series(series).astype(str).to_numpy()
        intervals = bootstrap_exclusion(y, sel, w, groups, clusters, n_boot)
        for g in sorted(pd.unique(groups)):
            k = groups == g
            e = targeting_errors(y[k], sel[k], w[k])
            low, high = intervals[g]
            rows.append(
                {
                    "grouping": grouping,
                    "group": g,
                    "population_share": float(w[k].sum() / w.sum()),
                    "households": int(k.sum()),
                    "poor_households": int(y[k].sum()),
                    "poverty_rate": float(w[k][y[k]].sum() / w[k].sum()),
                    "selection_rate": e["selection_rate"],
                    "exclusion_error": e["exclusion_error"],
                    "exclusion_low": float(low),
                    "exclusion_high": float(high),
                    "inclusion_error": e["inclusion_error"],
                    "flag_worse": bool(low > overall),
                }
            )
    return pd.DataFrame(rows), overall


def render(survey_id: str, tier: str, model: str, table: pd.DataFrame, overall: float) -> str:
    lines = [
        f"# Fairness: {survey_id}, tier {tier} ({model})",
        "",
        "Budget = poverty rate, one rule for everyone; out-of-fold predictions, folds 1-4.",
        f"Overall exclusion error: {overall:.1%}. Intervals: 95% cluster bootstrap.",
        "**Flag** = even the interval's lower end is above the overall exclusion error.",
        "",
        "| Grouping | Group | Pop. share | Poor hh | Poverty rate | Selected | "
        "Exclusion (95% CI) | Inclusion | Flag |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for _, r in table.iterrows():
        lines.append(
            f"| {r['grouping']} | {r['group']} | {r['population_share']:.1%} | "
            f"{r['poor_households']} | {r['poverty_rate']:.1%} | {r['selection_rate']:.1%} | "
            f"{r['exclusion_error']:.1%} ({r['exclusion_low']:.0%}-{r['exclusion_high']:.0%}) | "
            f"{r['inclusion_error']:.1%} | {'**worse**' if r['flag_worse'] else ''} |"
        )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Subgroup targeting errors.")
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
    y = train[TARGET].to_numpy().astype(bool)
    w = train["pop_weight"].to_numpy(dtype=float)
    budget = float(w[y].sum() / w.sum())
    labels = groupings(train)

    out_dir = reports / "fairness"
    out_dir.mkdir(parents=True, exist_ok=True)
    for tier, model in choices.items():
        p = cross_validate(data, feature_set.tiers[tier], model)["predictions"]
        sel = select_by_budget(p, w, budget)
        table, overall = subgroup_table(y, sel, w, labels, train["cluster"].to_numpy())
        report = render(args.survey, tier, model, table, overall)
        (out_dir / f"{args.survey}_tier{tier}.md").write_text(report, newline="\n")
        (out_dir / f"{args.survey}_tier{tier}.json").write_text(
            json.dumps({"overall_exclusion": overall, "rows": table.to_dict("records")}, indent=2),
            newline="\n",
        )
        print(report)


if __name__ == "__main__":
    main()
