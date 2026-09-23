import numpy as np
import pandas as pd

from poverty_targeting.dataset import TARGET, assemble, assign_folds, fold_summary


def synthetic(n_clusters=40, per_cluster=10, seed=0):
    rng = np.random.default_rng(seed)
    cluster = np.repeat(np.arange(n_clusters), per_cluster)
    poor = (rng.random(len(cluster)) < 0.25).astype("int8")
    return pd.DataFrame({TARGET: poor, "cluster": cluster})


def test_no_cluster_is_split_across_folds():
    data = synthetic()
    data["fold"] = assign_folds(data[TARGET], data["cluster"])
    assert (data.groupby("cluster")["fold"].nunique() == 1).all()
    assert set(data["fold"]) == {0, 1, 2, 3, 4}


def test_folds_have_similar_poverty_rates():
    data = synthetic(n_clusters=200)
    data["fold"] = assign_folds(data[TARGET], data["cluster"])
    rates = data.groupby("fold")[TARGET].mean()
    assert rates.max() - rates.min() < 0.05


def test_folds_are_reproducible():
    data = synthetic()
    first = assign_folds(data[TARGET], data["cluster"])
    second = assign_folds(data[TARGET], data["cluster"])
    pd.testing.assert_series_equal(first, second)


def test_assemble_keeps_only_labelled_households():
    features = pd.DataFrame({"hh_id": ["a", "b", "c"], "hh_size": [3, 5, 2]})
    label = pd.DataFrame(
        {
            "hh_id": ["a", "b", "c"],
            "in_sample": [True, False, True],
            "mpi_poor": pd.array([True, pd.NA, False], dtype="boolean"),
            "deprivation_score": pd.array([0.5, pd.NA, 0.1], dtype="Float64"),
            "pop_weight": [3.0, 5.0, 2.0],
        }
    )
    households = pd.DataFrame({"hh_id": ["a", "b", "c"], "cluster": [1, 1, 2]})
    data = assemble(features, label, households)
    assert data["hh_id"].tolist() == ["a", "c"]
    assert data[TARGET].tolist() == [1, 0]
    assert data["cluster"].tolist() == [1, 2]


def test_summary_reports_leakage_check():
    data = synthetic()
    data["fold"] = assign_folds(data[TARGET], data["cluster"])
    summary = fold_summary(data)
    assert summary["clusters_in_more_than_one_fold"] == 0
    assert sum(f["households"] for f in summary["folds"].values()) == len(data)
