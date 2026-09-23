from poverty_targeting import config


def test_project_root_is_repo_root():
    assert (config.PROJECT_ROOT / "pyproject.toml").exists()


def test_raw_survey_dir_is_built_from_program_and_id():
    path = config.raw_survey_dir("DHS", "GH2022DHS")
    assert path.parts[-3:] == ("raw", "dhs", "GH2022DHS")


def test_processed_survey_dir():
    assert config.processed_survey_dir("GH2022DHS").name == "GH2022DHS"
