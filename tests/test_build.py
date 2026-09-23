import json

import pandas as pd
import pytest

from poverty_targeting.build import LinkError, build_survey, check_links, load_tables, write_tables
from poverty_targeting.registry import SurveySpec


def make_tables():
    """Two households, three people, one child death; everything links up."""
    households = pd.DataFrame(
        {"hh_id": pd.array(["a", "b"], dtype="string"), "hh_size": pd.array([2, 1], dtype="Int64")}
    )
    persons = pd.DataFrame(
        {
            "hh_id": pd.array(["a", "a", "b"], dtype="string"),
            "line": pd.array([1, 2, 1], dtype="Int64"),
            "is_head": pd.array([True, False, True], dtype="boolean"),
        }
    )
    deaths = pd.DataFrame(
        {"hh_id": pd.array(["a"], dtype="string"), "mother_line": pd.array([2], dtype="Int64")}
    )
    return {"households": households, "persons": persons, "child_deaths": deaths}


def test_consistent_tables_pass():
    check_links(make_tables())


def test_link_problems_are_reported_together():
    tables = make_tables()
    tables["persons"].loc[1, "is_head"] = True  # household a now has two heads
    tables["persons"].loc[2, "hh_id"] = "zzz"  # person in a household that doesn't exist
    tables["child_deaths"].loc[0, "mother_line"] = 9  # mother not in the roster
    with pytest.raises(LinkError) as excinfo:
        check_links(tables)
    message = str(excinfo.value)
    assert "belong to no household" in message
    assert "without exactly one head" in message
    assert "roster size differs" in message
    assert "mother is not in the roster" in message


def test_write_and_load_round_trip(tmp_path):
    spec = SurveySpec("XX2020DHS", "Testland", "TST", "DHS", 8, 2020, files={"kids": "kr.zip"})
    tables = make_tables()
    write_tables(spec, tables, out_dir=tmp_path)

    loaded = load_tables("XX2020DHS", in_dir=tmp_path)
    for name, df in tables.items():
        pd.testing.assert_frame_equal(loaded[name], df)  # values AND nullable dtypes survive

    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["rows"] == {"households": 2, "persons": 3, "child_deaths": 1}
    assert manifest["death_source"] == "kids"


def test_load_missing_tables_says_how_to_build(tmp_path):
    with pytest.raises(FileNotFoundError, match="python -m poverty_targeting.build"):
        load_tables("XX2020DHS", in_dir=tmp_path)


def test_unknown_programme_is_rejected(tmp_path):
    (tmp_path / "XX2020MICS.toml").write_text(
        '[survey]\nid = "XX2020MICS"\ncountry = "Testland"\niso3 = "TST"\n'
        'program = "MICS"\nphase = 6\nyear = 2020\n\n'
        '[files]\nhousehold = "hh.zip"\npersons = "hl.zip"\n'
    )
    with pytest.raises(NotImplementedError, match="no adapter for programme 'MICS'"):
        build_survey("XX2020MICS", config_dir=tmp_path)
