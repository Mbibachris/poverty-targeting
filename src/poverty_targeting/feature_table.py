"""Feature table: one row per household holding the features of a tier.

Derived features come in two kinds:
- roster answers: member counts by age and the head's schooling. A targeting form
  asks these directly; for training we count them from the survey roster.
- computed features: arithmetic on other answers (dependent_share, persons_per_room).
  add_computed_features() is the single definition, used here for training and later
  by the API on a form's answers, so the model sees identical features in both.
"""

import pandas as pd

from poverty_targeting.features import FeatureSet, FeatureSetError

# Member counts by age band: (lowest age, highest age or None for open-ended).
AGE_BANDS = {
    "n_under5": (0, 4),
    "n_children_5_17": (5, 17),
    "n_elderly_65plus": (65, None),
}


def roster_answers(hh_ids: pd.Index, persons: pd.DataFrame) -> pd.DataFrame:
    """Answers a form would collect about members, counted from the survey roster."""
    age = persons["age_years"]
    out = {}
    for name, (low, high) in AGE_BANDS.items():
        in_band = (age >= low) if high is None else age.between(low, high)
        counts = in_band.fillna(False).astype(int).groupby(persons["hh_id"]).sum()
        out[name] = counts.reindex(hh_ids, fill_value=0).astype("Int64")

    heads = persons[persons["is_head"].fillna(False).to_numpy(dtype=bool)]
    schooling = heads.set_index("hh_id")["years_schooling"]
    out["head_years_schooling"] = schooling.reindex(hh_ids).astype("Int64")
    return pd.DataFrame(out, index=hh_ids)


def add_computed_features(answers: pd.DataFrame) -> pd.DataFrame:
    """Add features computed from other answers. Used for training AND by the API."""
    df = answers.copy()
    size = df["hh_size"].astype("Float64").replace(0, pd.NA) if "hh_size" in df else None

    if size is not None and set(AGE_BANDS) <= set(df.columns):
        dependents = df["n_under5"] + df["n_children_5_17"] + df["n_elderly_65plus"]
        df["dependent_share"] = (dependents / size).astype("Float64")

    if size is not None and "sleeping_rooms" in df:
        rooms = df["sleeping_rooms"].astype("Float64").replace(0, pd.NA)
        df["persons_per_room"] = (df["hh_size"] / rooms).astype("Float64")
    return df


def build_feature_table(
    tables: dict[str, pd.DataFrame], feature_set: FeatureSet, tier: str
) -> pd.DataFrame:
    """hh_id plus the tier's features, in tier order, one row per household."""
    households = tables["households"].set_index("hh_id", drop=False)
    wanted = feature_set.tiers[tier]

    # Every direct answer in the feature set (computed features may need them).
    direct = {}
    for feature in feature_set.features.values():
        if feature.source == "derived":
            continue
        table, _, column = feature.source.partition(".")
        if table != "households":
            raise FeatureSetError(f"{feature.name}: direct features must come from households")
        direct[feature.name] = households[column]

    answers = pd.DataFrame(direct, index=households.index)
    answers = answers.join(roster_answers(households.index, tables["persons"]))
    table = add_computed_features(answers)

    missing = [name for name in wanted if name not in table.columns]
    if missing:
        raise FeatureSetError(f"No implementation for derived features: {missing}")

    out = table[wanted].copy()
    out.insert(0, "hh_id", households["hh_id"].to_numpy())
    return out.reset_index(drop=True)
