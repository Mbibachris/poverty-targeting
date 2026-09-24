"""The API must run without the training-only dependency (devecon).

Each check starts a fresh Python process in which importing devecon is made to fail,
then imports a module the API relies on. If any of them pulls in devecon, this fails.
"""

import subprocess
import sys

import pytest

SERVING_MODULES = [
    "poverty_targeting.features",
    "poverty_targeting.feature_table",
    "poverty_targeting.modeling",
]


@pytest.mark.parametrize("module", SERVING_MODULES)
def test_serving_modules_do_not_need_devecon(module):
    code = f"import sys; sys.modules['devecon'] = None; import {module}"
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_label_building_still_uses_devecon():
    code = (
        "import sys; sys.modules['devecon'] = None\n"
        "from poverty_targeting.mpi_spec import load_mpi_spec\n"
        "load_mpi_spec().weights()"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode != 0 and "devecon" in result.stderr
