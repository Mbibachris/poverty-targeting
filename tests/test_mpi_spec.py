import pytest

from poverty_targeting import config
from poverty_targeting.mpi_spec import SpecError, load_mpi_spec

GLOBAL_TEXT = (config.MPI_SPEC_DIR / "global_mpi.toml").read_text()


def write_spec(tmp_path, text, spec_id="test_mpi"):
    (tmp_path / f"{spec_id}.toml").write_text(text)
    return tmp_path


def test_global_mpi_loads_with_official_structure():
    spec = load_mpi_spec("global_mpi")
    assert spec.poverty_cutoff == pytest.approx(1 / 3)
    names, weights = spec.weights()
    assert len(names) == 10
    assert sum(weights) == pytest.approx(1)
    by_name = dict(zip(names, weights, strict=True))
    assert by_name["nutrition"] == pytest.approx(1 / 6)  # health: 1/3 split over 2
    assert by_name["assets"] == pytest.approx(1 / 18)  # living standards: 1/3 over 6


def test_weights_that_do_not_sum_to_one_are_rejected(tmp_path):
    text = GLOBAL_TEXT.replace(
        '[dimensions.health]\nweight = "1/3"', '[dimensions.health]\nweight = "1/2"'
    )
    with pytest.raises(SpecError, match="weights sum to"):
        load_mpi_spec("test_mpi", spec_dir=write_spec(tmp_path, text))


def test_category_outside_vocabulary_is_rejected(tmp_path):
    text = GLOBAL_TEXT.replace('"sachet",\n]', '"sachet", "magic_well",\n]')
    with pytest.raises(SpecError, match="magic_well"):
        load_mpi_spec("test_mpi", spec_dir=write_spec(tmp_path, text))


def test_unknown_asset_is_rejected(tmp_path):
    text = GLOBAL_TEXT.replace('large = ["car_truck"]', 'large = ["car_truck", "yacht"]')
    with pytest.raises(SpecError, match="has_yacht"):
        load_mpi_spec("test_mpi", spec_dir=write_spec(tmp_path, text))


def test_indicator_in_two_dimensions_is_rejected(tmp_path):
    text = GLOBAL_TEXT.replace(
        'indicators = ["nutrition", "child_mortality"]',
        'indicators = ["nutrition", "child_mortality", "electricity"]',
    )
    with pytest.raises(SpecError, match="more than one dimension"):
        load_mpi_spec("test_mpi", spec_dir=write_spec(tmp_path, text))


def test_unknown_spec_lists_available():
    with pytest.raises(FileNotFoundError, match="global_mpi"):
        load_mpi_spec("no_such_mpi")
