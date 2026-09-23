"""Survey registry: load per-survey config files into typed SurveySpec objects."""

import tomllib
from dataclasses import dataclass, field
from pathlib import Path

from poverty_targeting import config

REQUIRED_FILES = ("household", "persons")


@dataclass(frozen=True)
class SurveySpec:
    """Everything the pipeline needs to know about one survey."""

    survey_id: str
    country: str
    iso3: str
    program: str
    phase: int
    year: int
    files: dict[str, str] = field(default_factory=dict)
    anthropometry_subsample_var: str | None = None

    @property
    def raw_dir(self) -> Path:
        return config.raw_survey_dir(self.program, self.survey_id)

    def file_path(self, key: str) -> Path:
        """Full path to one of this survey's raw files, e.g. file_path('household')."""
        if key not in self.files:
            raise KeyError(
                f"Survey {self.survey_id} has no '{key}' file configured. "
                f"Available: {sorted(self.files)}"
            )
        return self.raw_dir / self.files[key]


def available_surveys(config_dir: Path | None = None) -> list[str]:
    """IDs of every survey that has a config file."""
    directory = Path(config_dir or config.SURVEY_CONFIG_DIR)
    return sorted(p.stem for p in directory.glob("*.toml"))


def load_survey(survey_id: str, config_dir: Path | None = None) -> SurveySpec:
    """Read <survey_id>.toml and validate it into a SurveySpec."""
    directory = Path(config_dir or config.SURVEY_CONFIG_DIR)
    path = directory / f"{survey_id}.toml"
    if not path.exists():
        raise FileNotFoundError(
            f"No config for survey '{survey_id}' at {path}. "
            f"Available: {available_surveys(directory)}"
        )

    with path.open("rb") as f:
        raw = tomllib.load(f)

    meta = raw["survey"]
    if meta["id"] != survey_id:
        raise ValueError(f"{path.name} declares id '{meta['id']}', expected '{survey_id}'")

    files = raw.get("files", {})
    missing = [key for key in REQUIRED_FILES if key not in files]
    if missing:
        raise ValueError(f"{path.name} is missing required file entries: {missing}")

    anthro = raw.get("anthropometry", {})
    return SurveySpec(
        survey_id=meta["id"],
        country=meta["country"],
        iso3=meta["iso3"],
        program=meta["program"],
        phase=meta["phase"],
        year=meta["year"],
        files=files,
        anthropometry_subsample_var=anthro.get("subsample_var"),
    )
