"""Tests for saving and loading results (io/hdf5_io.py)."""
import numpy as np
import pytest

from TCAT2Phase.io.hdf5_io import load_solutions_from_hdf5, save_solutions_to_hdf5

from conftest import build_mesh


@pytest.fixture
def results():
    _, mesh = build_mesh(5)
    solutions = {
        0.0: {"p_w": np.linspace(-1, -5, 5), "e_w": np.full(5, 0.3)},
        10.0: {"p_w": np.linspace(-2, -6, 5), "e_w": np.full(5, 0.25)},
    }
    return mesh, solutions


def test_round_trip(tmp_path, results):
    mesh, solutions = results
    save_solutions_to_hdf5(tmp_path, "run.h5", mesh, solutions)
    loaded, points, metadata = load_solutions_from_hdf5(tmp_path, "run.h5")

    np.testing.assert_array_equal(points, mesh.points)
    assert sorted(loaded) == [0.0, 10.0]
    for t in solutions:
        for name, values in solutions[t].items():
            np.testing.assert_array_equal(loaded[t][name], values)
    assert metadata == {}


def test_metadata_scalars_and_arrays(tmp_path, results):
    mesh, solutions = results
    meta = {"nodes": 5, "tau": 1.0e7, "model": "RE", "pw_values": [0.0, -1500.0, 0.0]}
    save_solutions_to_hdf5(tmp_path, "run.h5", mesh, solutions, metadata=meta)
    _, _, loaded = load_solutions_from_hdf5(tmp_path, "run.h5")
    assert loaded["nodes"] == 5
    assert loaded["tau"] == 1.0e7
    assert loaded["model"] == "RE"
    np.testing.assert_array_equal(loaded["pw_values"], [0.0, -1500.0, 0.0])


def test_save_creates_missing_output_directory(tmp_path, results):
    """Fresh clones have no outputs/<case>/outputs folder; saving must create it."""
    mesh, solutions = results
    target = tmp_path / "outputs" / "TestcaseX" / "outputs"
    save_solutions_to_hdf5(str(target), "run.h5", mesh, solutions)
    assert (target / "run.h5").is_file()


def test_save_overwrites_existing_file(tmp_path, results):
    mesh, solutions = results
    save_solutions_to_hdf5(tmp_path, "run.h5", mesh, solutions)
    save_solutions_to_hdf5(tmp_path, "run.h5", mesh, {1.0: {"p_w": np.zeros(5)}})
    loaded, _, _ = load_solutions_from_hdf5(tmp_path, "run.h5")
    assert list(loaded) == [1.0]
