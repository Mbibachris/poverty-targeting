import json

import numpy as np
import pandas as pd
import pytest

from poverty_targeting.label import label_from_indicators, write_label
from poverty_targeting.mpi_spec import load_mpi_spec

SPEC = load_mpi_spec("global_mpi")
NAMES, _ = SPEC.weights()
LIVING = ["cooking_fuel", "sanitation", "drinking_water", "electricity", "housing", "assets"]


def indicator_rows(*rows):
    """Each row: set of deprived indicators, or None for a household with unknown nutrition."""
    records = []
    for i, deprived in enumerate(rows):
        record = {"hh_id": f"h{i}"}
        for name in NAMES:
            record[name] = 1.0 if deprived and name in deprived else 0.0
        if deprived is None:
            record["nutrition"] = np.nan
        records.append(record)
    df = pd.DataFrame(records)
    df[NAMES] = df[NAMES].astype("Float64")
    return df


def households(n, weights=None):
    return pd.DataFrame(
        {
            "survey_id": ["XX2020DHS"] * n,
            "hh_id": [f"h{i}" for i in range(n)],
            "weight": weights or [1.0] * n,
        }
    )


def persons(sizes):
    """sizes: number of usual members in each household h0, h1, ..."""
    hh_ids = [f"h{i}" for i, size in enumerate(sizes) for _ in range(size)]
    return pd.DataFrame(
        {"hh_id": hh_ids, "usual_resident": pd.array([True] * len(hh_ids), dtype="boolean")}
    )


def test_identification_follows_weights_and_cutoff():
    ind = indicator_rows(
        {"nutrition", "child_mortality"},  # 1/6 + 1/6 = 1/3 -> poor (exactly on k)
        set(LIVING[:5]),  # 5/18 < 1/3 -> not poor
        set(LIVING),  # 6/18 = 1/3 -> poor
        set(),  # nothing -> not poor
    )
    label = label_from_indicators(ind, households(4), persons([1, 1, 1, 1]), SPEC)
    assert label.table["mpi_poor"].tolist() == [True, False, True, False]
    assert label.table["deprivation_score"].tolist() == pytest.approx([1 / 3, 5 / 18, 1 / 3, 0])


def test_unknown_indicator_drops_household():
    ind = indicator_rows({"nutrition", "child_mortality"}, None)
    label = label_from_indicators(ind, households(2), persons([1, 1]), SPEC)
    assert label.table["in_sample"].tolist() == [True, False]
    assert pd.isna(label.table.loc[1, "deprivation_score"])
    assert pd.isna(label.table.loc[1, "mpi_poor"])
    assert label.summary["households_dropped"] == 1


def test_headcount_is_population_weighted():
    # Poor household has 4 usual members, non-poor has 1: 4 of 5 people are poor.
    ind = indicator_rows({"nutrition", "child_mortality"}, set())
    label = label_from_indicators(ind, households(2), persons([4, 1]), SPEC)
    assert label.summary["H"] == pytest.approx(0.8)
    assert label.summary["A"] == pytest.approx(1 / 3)
    assert label.summary["M0"] == pytest.approx(0.8 / 3)


def test_vulnerable_and_severe_shares():
    ind = indicator_rows(
        set(LIVING[:4]),  # 4/18 = 0.22 -> vulnerable, not poor
        {"nutrition", "child_mortality", "years_schooling"},  # 1/2 -> severe
    )
    label = label_from_indicators(ind, households(2), persons([1, 1]), SPEC)
    assert label.summary["vulnerable"] == pytest.approx(0.5)
    assert label.summary["severe"] == pytest.approx(0.5)


def test_contributions_sum_to_one_and_label_is_written(tmp_path):
    ind = indicator_rows({"nutrition", "child_mortality", "cooking_fuel"}, set())
    label = label_from_indicators(ind, households(2), persons([2, 3]), SPEC)
    assert sum(label.summary["contributions"].values()) == pytest.approx(1)

    write_label(label, out_dir=tmp_path)
    saved = json.loads((tmp_path / "label_global_mpi.json").read_text())
    assert saved["H"] == pytest.approx(label.summary["H"])
    assert (tmp_path / "label_global_mpi.parquet").exists()
