"""Serving core: load a trained model and score households from their answers.

No web code lives here, so the logic can be tested (and reused by the dashboard) on
its own. It reuses the exact training functions: add_computed_features builds the
computed features and to_model_input prepares the columns, so a household is scored
exactly as training households were.
"""

from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from poverty_targeting import schema
from poverty_targeting.feature_table import COMPUTED_FEATURES, add_computed_features
from poverty_targeting.features import load_feature_set
from poverty_targeting.modeling import to_model_input

HOUSEHOLD_KINDS = {c.name: c.kind for c in schema.HOUSEHOLD_COLUMNS}


class ScoringError(ValueError):
    """Raised when answers or options cannot be scored; the message says why."""


@dataclass(frozen=True)
class Question:
    name: str
    kind: str  # "category" | "yes_no" | "number"
    question: str
    options: tuple[str, ...] | None = None
    required: bool = True


@dataclass
class ModelBundle:
    pipeline: object
    meta: dict
    questions: list[Question] = field(default_factory=list)

    @property
    def features(self) -> list[str]:
        return self.meta["features"]


# --- explanations (TreeSHAP, built into LightGBM) --------------------------------------


def feature_of(column: str, features: list[str]) -> str:
    """Map a transformed column (e.g. 'categorical__water_source_sachet') to its feature."""
    name = column.split("__", 1)[-1]
    matches = [f for f in features if name == f or name.startswith(f + "_")]
    return max(matches, key=len)  # longest match: 'has_tv' must not claim 'has_tv_x'


def contributions(pipeline, X: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """Per-feature log-odds contributions (TreeSHAP), plus the 'baseline' column."""
    transformed = pipeline.named_steps["preprocess"].transform(X)
    if hasattr(transformed, "toarray"):  # one-hot output can be sparse
        transformed = transformed.toarray()
    columns = pipeline.named_steps["preprocess"].get_feature_names_out()
    raw = pipeline.named_steps["model"].booster_.predict(transformed, pred_contrib=True)
    per_column = pd.DataFrame(raw[:, :-1], columns=columns, index=X.index)
    grouped = per_column.T.groupby([feature_of(c, features) for c in columns]).sum().T
    grouped["baseline"] = raw[:, -1]
    return grouped[[*features, "baseline"]]


# --- loading --------------------------------------------------------------------------


def _category_options(pipeline) -> dict[str, tuple[str, ...]]:
    """The categories each categorical feature took in training, read from the encoder."""
    preprocess = pipeline.named_steps["preprocess"]
    columns = next(cols for name, _, cols in preprocess.transformers_ if name == "categorical")
    encoder = preprocess.named_transformers_["categorical"][-1]
    return {
        column: tuple(str(c) for c in categories if c != "missing")
        for column, categories in zip(columns, encoder.categories_, strict=True)
    }


def _questions(pipeline, meta: dict) -> list[Question]:
    """What a form must ask: every model feature except the computed ones.

    An answer may be left blank only if blanks occurred in training (meta
    'optional_answers'); elsewhere the model has never seen a blank and its output
    for one would be meaningless.
    """
    feature_set = load_feature_set(meta["feature_set"])
    options = _category_options(pipeline)
    optional = set(meta.get("optional_answers", []))
    questions = []
    for name in meta["features"]:
        if name in COMPUTED_FEATURES:
            continue
        feature = feature_set.features[name]
        column = feature.source.partition(".")[2]
        if name in options:
            kind = "category"
        elif HOUSEHOLD_KINDS.get(column) == "bool":
            kind = "yes_no"
        else:
            kind = "number"
        questions.append(
            Question(name, kind, feature.question, options.get(name), name not in optional)
        )
    return questions


def load_bundle(path: Path) -> ModelBundle:
    """Load a model saved by poverty_targeting.final and describe the questions it needs."""
    saved = joblib.load(path)
    if not isinstance(saved, dict) or not {"pipeline", "meta"} <= set(saved):
        raise ScoringError(f"{path} is not a model saved by poverty_targeting.final")
    return ModelBundle(
        saved["pipeline"], saved["meta"], _questions(saved["pipeline"], saved["meta"])
    )


# --- scoring --------------------------------------------------------------------------


def answers_frame(bundle: ModelBundle, answers: list[dict]) -> pd.DataFrame:
    """Typed table of answers; unknown keys and invalid options are refused."""
    known = {q.name: q for q in bundle.questions}
    problems = []
    for i, answer in enumerate(answers):
        for key in answer:
            if key in COMPUTED_FEATURES:
                problems.append(f"household {i}: '{key}' is computed by the model; do not send it")
            elif key not in known:
                problems.append(f"household {i}: unknown answer '{key}'")
    for q in bundle.questions:
        for i, answer in enumerate(answers):
            value = answer.get(q.name)
            if value is None and q.required:
                problems.append(f"household {i}: missing required answer '{q.name}'")
            elif value is not None and q.kind == "category" and str(value) not in q.options:
                problems.append(f"household {i}: {q.name}={value!r} not one of {list(q.options)}")
    if problems:
        raise ScoringError("; ".join(problems))

    dtypes = {"category": "string", "yes_no": "boolean", "number": "Float64"}
    return pd.DataFrame(
        {
            q.name: pd.array([a.get(q.name) for a in answers], dtype=dtypes[q.kind])
            for q in bundle.questions
        }
    )


def cutoff_for(bundle: ModelBundle, budget: float | None) -> tuple[float, float]:
    """(budget, score cut-off). Default budget is the training poverty rate."""
    budget = bundle.meta["default_budget"] if budget is None else budget
    key = round(float(budget), 2)
    cutoffs = bundle.meta["budget_cutoffs"]
    if key not in cutoffs:
        raise ScoringError(f"budget must be between 0.01 and 0.99, got {budget}")
    return key, float(cutoffs[key])


def _plain(value):
    """numpy / pandas scalars -> JSON-friendly Python values."""
    if value is None or value is pd.NA or (isinstance(value, float) and np.isnan(value)):
        return None
    if isinstance(value, np.generic):
        return value.item()
    return value


def score(bundle: ModelBundle, answers: list[dict], budget=None, top_k: int = 3) -> list[dict]:
    """Probability, selection at the budget, and the top reasons, for each household."""
    budget, cutoff = cutoff_for(bundle, budget)
    frame = add_computed_features(answers_frame(bundle, answers))
    X = to_model_input(frame, bundle.features)
    probabilities = bundle.pipeline.predict_proba(X)[:, 1]
    parts = contributions(bundle.pipeline, X, bundle.features)[bundle.features]

    results = []
    for i, p in enumerate(probabilities):
        top = parts.iloc[i].abs().sort_values(ascending=False).index[:top_k]
        reasons = [
            {
                "feature": f,
                "question": bundle.meta["questions"][f],
                "answer": _plain(frame[f].iloc[i]),
                "effect": round(float(parts.iloc[i][f]), 3),
                "direction": "raises" if parts.iloc[i][f] > 0 else "lowers",
            }
            for f in top
        ]
        results.append(
            {
                "probability": round(float(p), 4),
                "budget": budget,
                "score_cutoff": round(cutoff, 4),
                "selected": bool(p >= cutoff),
                "reasons": reasons,
                "missing_answers": [
                    q.name for q in bundle.questions if answers[i].get(q.name) is None
                ],
            }
        )
    return results
