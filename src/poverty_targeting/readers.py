"""Read raw survey files (a Stata .dta inside a zip) together with their labels."""

import io
import zipfile
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass
class RawTable:
    """Raw survey data plus the labels that give its numeric codes meaning."""

    data: pd.DataFrame
    value_labels: dict[str, dict[int, str]]
    variable_labels: dict[str, str]

    def labels_for(self, column: str) -> dict[int, str]:
        """Code -> label mapping for one column (empty dict if it has none)."""
        return self.value_labels.get(column, {})


def _find_dta_member(zf: zipfile.ZipFile, zip_path: Path) -> str:
    members = [name for name in zf.namelist() if name.lower().endswith(".dta")]
    if len(members) != 1:
        raise ValueError(
            f"Expected exactly one .dta file inside {zip_path.name}, found: {members or 'none'}"
        )
    return members[0]


def read_stata_zip(zip_path: Path, columns: list[str] | None = None) -> RawTable:
    """Read selected columns of the .dta file inside a zip, keeping codes and labels.

    Codes are kept numeric (convert_categoricals=False) because deprivation rules are
    defined on codes; labels are returned separately so we can always check what a
    code means.
    """
    zip_path = Path(zip_path)
    with zipfile.ZipFile(zip_path) as zf:
        member = _find_dta_member(zf, zip_path)
        buffer = io.BytesIO(zf.read(member))

    with pd.read_stata(buffer, iterator=True, convert_categoricals=False) as reader:
        all_variable_labels = reader.variable_labels()
        if columns is not None:
            missing = [c for c in columns if c not in all_variable_labels]
            if missing:
                raise KeyError(f"Columns not found in {zip_path.name}: {missing}")
        data = reader.read(columns=columns)
        label_sets = reader.value_labels()

    value_labels = {}
    for column in data.columns:
        # DHS names each label set after its variable in upper case (hv024 -> "HV024").
        label_set = label_sets.get(column) or label_sets.get(column.upper())
        if label_set:
            value_labels[column] = {int(code): text for code, text in label_set.items()}

    variable_labels = {column: all_variable_labels.get(column, "") for column in data.columns}
    return RawTable(data=data, value_labels=value_labels, variable_labels=variable_labels)
