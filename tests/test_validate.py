import pandas as pd
import pytest

from poverty_targeting.mpi_spec import load_mpi_spec
from poverty_targeting.validate import (
    BOUNDS,
    METRICS,
    dimension_contributions,
    render_report,
    with_overrides,
    within_bounds,
)

SPEC = load_mpi_spec("global_mpi")


def test_overrides_change_a_copy_only():
    variant = with_overrides(SPEC, [("nutrition", "adolescents_15_19", "adult_cutoff")])
    assert variant.indicators["nutrition"]["adolescents_15_19"] == "adult_cutoff"
    assert SPEC.indicators["nutrition"]["adolescents_15_19"] == "exclude"


def test_dimension_contributions_add_up():
    names, _ = SPEC.weights()
    contributions = dict.fromkeys(names, 0.1)
    dims = dimension_contributions(contributions, SPEC)
    assert dims == pytest.approx({"health": 0.2, "education": 0.2, "living_standards": 0.6})


def fake_results(low, high):
    return pd.DataFrame(
        {m: [low, high] for m in METRICS}, index=pd.Index(list(BOUNDS), name="scenario")
    )


def test_reference_between_bounds_is_inside():
    reference = dict.fromkeys(METRICS, 0.25)
    checks = within_bounds(fake_results(0.20, 0.30), reference)
    assert all(checks.values())


def test_reference_outside_bounds_is_flagged():
    reference = dict.fromkeys(METRICS, 0.35)
    checks = within_bounds(fake_results(0.20, 0.30), reference)
    assert not any(checks.values())


def test_report_lists_published_and_scenarios():
    reference = {**dict.fromkeys(METRICS, 0.25), "source": "Test source", "url": "https://x"}
    report = render_report("XX2020DHS", SPEC, fake_results(0.2, 0.3), reference)
    assert "**published**" in report
    assert BOUNDS[0] in report and BOUNDS[1] in report
    assert "Test source" in report
