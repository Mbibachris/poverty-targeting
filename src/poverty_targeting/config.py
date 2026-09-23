"""Project paths.

Every file location used by the project is defined here and nowhere else.
Set the POVERTY_TARGETING_DATA_DIR environment variable to keep data
somewhere other than the repo's data/ folder (e.g. inside Docker).
"""

import os
from pathlib import Path

# src/poverty_targeting/config.py -> parents[0]=poverty_targeting, [1]=src, [2]=repo root
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = Path(os.environ.get("POVERTY_TARGETING_DATA_DIR", PROJECT_ROOT / "data"))
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"

# Survey config files ship inside the package, so they are available however it is installed.
SURVEY_CONFIG_DIR = Path(__file__).resolve().parent / "surveys"


def raw_survey_dir(program: str, survey_id: str) -> Path:
    """Folder holding one survey's raw files, e.g. data/raw/dhs/GH2022DHS."""
    return RAW_DIR / program.lower() / survey_id


def processed_survey_dir(survey_id: str) -> Path:
    """Folder for one survey's cleaned, canonical tables."""
    return PROCESSED_DIR / survey_id


# MPI definitions (dimensions, indicators, cutoffs, weights) also ship inside the package.
MPI_SPEC_DIR = Path(__file__).resolve().parent / "mpi_specs"
