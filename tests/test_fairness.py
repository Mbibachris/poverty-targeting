import numpy as np
import pandas as pd

from poverty_targeting.fairness import bootstrap_exclusion, groupings, subgroup_table


def households():
    return pd.DataFrame(
        {
            "urban": pd.array([True, False, False], dtype="boolean"),
            "head_sex": pd.array(["female", "male", "male"], dtype="string"),
            "hh_size": pd.array([1, 6, 2], dtype="Int64"),
            "n_under5": pd.array([0, 2, 0], dtype="Int64"),
            "n_children_5_17": pd.array([0, 2, 0], dtype="Int64"),
            "n_elderly_65plus": pd.array([1, 0, 0], dtype="Int64"),
            "region": pd.array(["North", "North", "Coast"], dtype="string"),
        }
    )


def test_groupings_from_form_answers():
    g = groupings(households())
    assert g["residence"].tolist() == ["urban", "rural", "rural"]
    assert g["children"].tolist() == ["no children", "has children", "no children"]
    assert g["elderly_only"].tolist() == ["elderly only", "other", "other"]
    assert g["household_size"].astype(str).tolist() == ["1-2", "5-6", "1-2"]


def synthetic(n=2000, seed=0):
    """Group 'b' poor households are never selected; group 'a' poor usually are."""
    rng = np.random.default_rng(seed)
    group = np.where(rng.random(n) < 0.5, "a", "b")
    y = rng.random(n) < 0.3
    sel = y & (group == "a") & (rng.random(n) < 0.8)
    clusters = np.arange(n) // 10
    return y, sel, np.ones(n), pd.Series(group), clusters


def test_group_left_behind_is_flagged():
    y, sel, w, group, clusters = synthetic()
    table, overall = subgroup_table(y, sel, w, {"g": group}, clusters, n_boot=100)
    rows = table.set_index("group")
    assert rows.loc["b", "exclusion_error"] == 1.0
    assert rows.loc["b", "flag_worse"]
    assert not rows.loc["a", "flag_worse"]
    assert rows.loc["a", "exclusion_error"] < overall < rows.loc["b", "exclusion_error"]


def test_bootstrap_interval_brackets_the_estimate():
    y, sel, w, group, clusters = synthetic()
    table, _ = subgroup_table(y, sel, w, {"g": group}, clusters, n_boot=200)
    for _, r in table.iterrows():
        assert r["exclusion_low"] <= r["exclusion_error"] <= r["exclusion_high"]


def test_bootstrap_is_reproducible():
    y, sel, w, group, clusters = synthetic(n=500)
    groups = group.to_numpy()
    first = bootstrap_exclusion(y, sel, w, groups, clusters, n_boot=50)
    second = bootstrap_exclusion(y, sel, w, groups, clusters, n_boot=50)
    assert first == second
