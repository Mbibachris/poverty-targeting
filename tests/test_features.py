import pytest

from poverty_targeting import config
from poverty_targeting.features import (
    FeatureSetError,
    check_strict_tier,
    load_feature_set,
    overlaps,
)
from poverty_targeting.mpi_spec import load_mpi_spec

SPEC = load_mpi_spec("global_mpi")
PMT_TEXT = (config.FEATURE_SET_DIR / "pmt.toml").read_text()


def write_set(tmp_path, text):
    (tmp_path / "test_set.toml").write_text(text)
    return tmp_path


def test_pmt_loads_and_tier_b_extends_tier_a():
    fs = load_feature_set("pmt")
    tier_a = fs.tiers["A"]
    tier_b = fs.tiers["B"]
    assert tier_b[: len(tier_a)] == tier_a  # B starts with all of A, in order
    assert "electricity" in tier_b and "electricity" not in tier_a


def test_tier_a_shares_no_inputs_with_the_label():
    check_strict_tier(load_feature_set("pmt"), SPEC, tier="A")


def test_tier_b_overlap_is_reported():
    found = overlaps(load_feature_set("pmt"), SPEC, tier="B")
    assert found["electricity"] == ["electricity"]
    assert found["head_years_schooling"] == ["years_schooling"]
    assert found["has_tv"] == ["assets"]
    assert found["hh_size"] == []


def test_label_input_smuggled_into_tier_a_is_caught(tmp_path):
    text = PMT_TEXT.replace('"owns_livestock",\n]', '"owns_livestock", "floor",\n]', 1)
    fs = load_feature_set("test_set", set_dir=write_set(tmp_path, text))
    with pytest.raises(FeatureSetError, match=r"floor -> \['housing'\]"):
        check_strict_tier(fs, SPEC, tier="A")


def test_unknown_source_column_is_rejected(tmp_path):
    text = PMT_TEXT.replace('source = "households.urban"', 'source = "households.urbn"')
    with pytest.raises(FeatureSetError, match="households.urbn"):
        load_feature_set("test_set", set_dir=write_set(tmp_path, text))


def test_undefined_feature_in_tier_is_rejected(tmp_path):
    text = PMT_TEXT.replace('"has_motorbike",\n]', '"has_motorbike", "has_yacht",\n]')
    with pytest.raises(FeatureSetError, match="has_yacht"):
        load_feature_set("test_set", set_dir=write_set(tmp_path, text))
