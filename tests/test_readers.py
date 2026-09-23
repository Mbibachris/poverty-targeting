import zipfile

import pandas as pd
import pytest

from poverty_targeting.readers import read_stata_zip


def make_survey_zip(tmp_path, n_dta=1):
    """Build a tiny fake survey: a .dta with value labels, zipped with a non-data file."""
    df = pd.DataFrame({"hhid": ["a", "b", "c"], "region": [1, 2, 1], "size": [3, 5, 2]})
    dta_path = tmp_path / "TEST.DTA"
    df.to_stata(
        dta_path,
        write_index=False,
        value_labels={"region": {1: "north", 2: "south"}},
        variable_labels={"region": "region of residence"},
    )
    zip_path = tmp_path / "TEST.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        for i in range(n_dta):
            zf.write(dta_path, arcname=f"TEST{i}.DTA")
        zf.writestr("README.txt", "not data")
    return zip_path


def test_reads_codes_and_labels(tmp_path):
    table = read_stata_zip(make_survey_zip(tmp_path))
    assert table.data.shape == (3, 3)
    assert table.data["region"].tolist() == [1, 2, 1]  # codes stay numeric
    assert table.labels_for("region") == {1: "north", 2: "south"}
    assert table.variable_labels["region"] == "region of residence"
    assert table.labels_for("size") == {}  # unlabelled column


def test_reads_selected_columns_only(tmp_path):
    table = read_stata_zip(make_survey_zip(tmp_path), columns=["hhid", "region"])
    assert list(table.data.columns) == ["hhid", "region"]


def test_missing_column_is_named(tmp_path):
    with pytest.raises(KeyError, match="hv999"):
        read_stata_zip(make_survey_zip(tmp_path), columns=["hhid", "hv999"])


def test_zip_with_two_dta_files_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="exactly one"):
        read_stata_zip(make_survey_zip(tmp_path, n_dta=2))
