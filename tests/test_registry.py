import pytest

from poverty_targeting.registry import available_surveys, load_survey

VALID_TOML = """
[survey]
id = "XX2020DHS"
country = "Testland"
iso3 = "TST"
program = "DHS"
phase = 8
year = 2020

[files]
household = "hh.zip"
persons = "pr.zip"
"""


def write_config(tmp_path, survey_id, text):
    (tmp_path / f"{survey_id}.toml").write_text(text)
    return tmp_path


def test_ghana_config_ships_with_package():
    spec = load_survey("GH2022DHS")
    assert spec.country == "Ghana"
    assert spec.anthropometry_subsample_var == "hv042"
    assert spec.file_path("household").name == "GHHR8CDT.zip"


def test_load_valid_config(tmp_path):
    config_dir = write_config(tmp_path, "XX2020DHS", VALID_TOML)
    spec = load_survey("XX2020DHS", config_dir=config_dir)
    assert spec.year == 2020
    assert spec.anthropometry_subsample_var is None  # no [anthropometry] section
    assert available_surveys(config_dir) == ["XX2020DHS"]


def test_unknown_survey_lists_available(tmp_path):
    config_dir = write_config(tmp_path, "XX2020DHS", VALID_TOML)
    with pytest.raises(FileNotFoundError, match="XX2020DHS"):
        load_survey("ZZ1999DHS", config_dir=config_dir)


def test_id_mismatch_is_rejected(tmp_path):
    config_dir = write_config(tmp_path, "YY2021DHS", VALID_TOML)
    with pytest.raises(ValueError, match="declares id"):
        load_survey("YY2021DHS", config_dir=config_dir)


def test_missing_required_file_is_rejected(tmp_path):
    text = VALID_TOML.replace('persons = "pr.zip"', "")
    config_dir = write_config(tmp_path, "XX2020DHS", text)
    with pytest.raises(ValueError, match="persons"):
        load_survey("XX2020DHS", config_dir=config_dir)


def test_unknown_file_key_lists_available():
    spec = load_survey("GH2022DHS")
    with pytest.raises(KeyError, match="household"):
        spec.file_path("births")
