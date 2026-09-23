"""Guard the devecon dependency: the label is only as correct as the version behind it."""

import devecon
import numpy as np
import pandas as pd


def test_devecon_version_is_the_pinned_release():
    assert devecon.__version__ == "0.0.2"


def test_devecon_treats_pandas_na_as_missing():
    # The v0.0.2 fix: a missing category must be unknown, not "not deprived".
    out = devecon.deprive_in(pd.array(["surface_water", pd.NA], dtype="string"), {"surface_water"})
    assert out[0] == 1.0
    assert np.isnan(out[1])


def test_devecon_identifies_who_is_poor():
    ident = devecon.af_identify([[1, 1], [0, 0]], cutoff_k=0.5)
    assert list(ident.poor) == [True, False]
