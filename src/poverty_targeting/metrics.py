"""Targeting metrics: how well a score picks out the poor, in the terms programmes use.

A targeting model is used to *select* households for a programme. What matters is:
- exclusion error: share of poor people the programme misses (undercoverage);
- inclusion error: share of the people it reaches who are not poor (leakage);
- coverage of the poor vs of the non-poor, and their difference (targeting differential).
All are population-weighted: a missed household of eight is eight missed people.

Selection can be made by a probability threshold, or (as programmes usually do) by a
budget: cover the highest-scoring households until a set share of the population is
reached.
"""

import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


def _weights(weights, n: int) -> np.ndarray:
    return np.ones(n) if weights is None else np.asarray(weights, dtype=float)


def select_by_budget(scores, weights=None, share: float = 0.25) -> np.ndarray:
    """Select the highest scores until `share` of the (weighted) population is covered."""
    if not 0 < share <= 1:
        raise ValueError("share must be in (0, 1]")
    scores = np.asarray(scores, dtype=float)
    w = _weights(weights, len(scores))
    order = np.argsort(-scores, kind="stable")  # highest first; ties keep input order
    covered_before = np.cumsum(w[order]) - w[order]
    selected = np.zeros(len(scores), dtype=bool)
    selected[order[covered_before < share * w.sum()]] = True
    return selected


def targeting_errors(y, selected, weights=None) -> dict:
    """Exclusion and inclusion errors, coverage and targeting differential (weighted)."""
    y = np.asarray(y).astype(bool)
    sel = np.asarray(selected).astype(bool)
    w = _weights(weights, len(y))
    poor, nonpoor = w[y].sum(), w[~y].sum()
    reached = w[sel].sum()

    coverage_poor = w[y & sel].sum() / poor if poor else np.nan
    coverage_nonpoor = w[~y & sel].sum() / nonpoor if nonpoor else np.nan
    return {
        "selection_rate": float(reached / w.sum()),
        "exclusion_error": float(1 - coverage_poor),
        "inclusion_error": float(w[~y & sel].sum() / reached) if reached else float("nan"),
        "coverage_poor": float(coverage_poor),
        "coverage_nonpoor": float(coverage_nonpoor),
        "targeting_differential": float(coverage_poor - coverage_nonpoor),
    }


def discrimination(y, scores, weights=None) -> dict:
    """Ranking quality: ROC-AUC and PR-AUC (average precision), weighted."""
    w = _weights(weights, len(y))
    return {
        "roc_auc": float(roc_auc_score(y, scores, sample_weight=w)),
        "pr_auc": float(average_precision_score(y, scores, sample_weight=w)),
    }


def calibration(y, probabilities, weights=None, n_bins: int = 10) -> dict:
    """Do predicted probabilities match observed rates? Brier score, ECE and a bin table."""
    y = np.asarray(y, dtype=float)
    p = np.asarray(probabilities, dtype=float)
    w = _weights(weights, len(y))
    edges = np.unique(np.quantile(p, np.linspace(0, 1, n_bins + 1)))
    bins = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, len(edges) - 2)

    table = []
    for b in range(len(edges) - 1):
        in_bin = bins == b
        if not in_bin.any():
            continue
        wb = w[in_bin]
        table.append(
            {
                "predicted": float(np.average(p[in_bin], weights=wb)),
                "observed": float(np.average(y[in_bin], weights=wb)),
                "weight_share": float(wb.sum() / w.sum()),
            }
        )
    ece = sum(row["weight_share"] * abs(row["predicted"] - row["observed"]) for row in table)
    return {
        "brier": float(brier_score_loss(y, p, sample_weight=w)),
        "ece": float(ece),
        "bins": table,
    }


def evaluate(y, probabilities, weights=None, threshold: float = 0.5, budget_share=None) -> dict:
    """Everything at once. Budget defaults to the weighted poverty rate itself."""
    y = np.asarray(y).astype(int)
    p = np.asarray(probabilities, dtype=float)
    w = _weights(weights, len(y))
    if budget_share is None:
        budget_share = float(w[y == 1].sum() / w.sum())
    return {
        "n": int(len(y)),
        "poverty_rate": float(w[y == 1].sum() / w.sum()),
        **discrimination(y, p, w),
        "calibration": calibration(y, p, w),
        "at_threshold": {"threshold": threshold, **targeting_errors(y, p >= threshold, w)},
        "at_budget": {
            "budget_share": budget_share,
            **targeting_errors(y, select_by_budget(p, w, budget_share), w),
        },
    }
