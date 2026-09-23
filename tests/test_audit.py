import numpy as np
import pandas as pd
import pytest

from poverty_targeting.audit import PREFIX, audit_table, render_audit, somers_d
from poverty_targeting.dataset import TARGET
from poverty_targeting.features import load_feature_set
from poverty_targeting.mpi_spec import load_mpi_spec

Y = pd.Series([0, 0, 1, 1, 0, 1])


def test_somers_d_direction_and_strength():
    assert somers_d(pd.Series([1, 2, 3, 4, 1, 5]), Y) == pytest.approx(1.0)
    assert somers_d(pd.Series([9, 8, 1, 2, 7, 0]), Y) == pytest.approx(-1.0)
    assert somers_d(pd.Series([1, 1, 1, 1, 1, 1]), Y) == pytest.approx(0.0)


def test_categories_are_scored_by_outcome_rate():
    x = pd.Series(["rural", "rural", "town", "town", "rural", "town"], dtype="string")
    assert somers_d(x, Y) == pytest.approx(1.0)


def test_missing_values_are_skipped_and_constant_outcome_is_nan():
    x = pd.array([1, None, 3, 4, 1, 5], dtype="Int64")
    assert somers_d(pd.Series(x), Y) == pytest.approx(1.0)
    assert np.isnan(somers_d(pd.Series([1, 2, 3]), pd.Series([1, 1, 1])))


def test_audit_table_and_report():
    fs, spec = load_feature_set("pmt"), load_mpi_spec("global_mpi")
    indicators, _ = spec.weights()
    rng = np.random.default_rng(0)
    n = 200
    data = pd.DataFrame({f: rng.integers(0, 5, n) for f in fs.tiers["B"]})
    for name in indicators:
        data[PREFIX + name] = rng.integers(0, 2, n)
    data[TARGET] = rng.integers(0, 2, n)

    table, matrix = audit_table(fs, spec, data)
    assert list(table.index) == fs.tiers["B"]
    assert table.loc["hh_size", "tier"] == "A"
    assert table.loc["electricity", "tier"] == "B"
    assert table.loc["electricity", "shares_inputs_with"] == "electricity"
    assert table.loc["hh_size", "shares_inputs_with"] == "-"

    report = render_audit("XX2020DHS", "pmt", spec, table, matrix)
    assert "| electricity | B | electricity |" in report
    assert "## Full matrix" in report
