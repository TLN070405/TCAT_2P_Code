"""Tests for the 1D uniform mesh (mesh/grid.py)."""
import numpy as np
import pytest

from conftest import build_mesh


def test_points_are_uniform_and_span_domain(mesh11):
    _, mesh = mesh11
    assert mesh.num_points == 11
    assert mesh.points.shape == (11, 1)
    np.testing.assert_allclose(mesh.points[:, 0], np.linspace(0.0, 1.0, 11))


def test_counts(mesh11):
    _, mesh = mesh11
    assert mesh.num_cells == 11
    assert mesh.num_faces == 12  # one face between each pair of cells plus two boundary faces


def test_only_end_cells_are_boundary(mesh11):
    _, mesh = mesh11
    flags = [c.boundary for c in mesh.cells]
    assert flags[0] and flags[-1]
    assert not any(flags[1:-1])


def test_only_end_faces_are_boundary(mesh11):
    _, mesh = mesh11
    flags = [f.boundary for f in mesh.faces]
    assert flags[0] and flags[-1]
    assert not any(flags[1:-1])


def test_cell_connectivity(mesh11):
    _, mesh = mesh11
    for cell in mesh.cells:
        n = cell.ID
        assert cell.faces == [n, n + 1]
    assert mesh.cells[0].neighbors == [-1, 1]
    assert mesh.cells[5].neighbors == [4, 6]
    assert mesh.cells[-1].neighbors == [9, -1]


def test_cell_lengths_and_distances_equal_spacing(mesh11):
    _, mesh = mesh11
    dx = 0.1
    for cell in mesh.cells:
        assert cell.length == pytest.approx(dx)
    for cell in mesh.cells[1:-1]:
        assert np.allclose(cell.distance, [dx, dx])


def test_interior_faces_sit_midway_between_cells(mesh11):
    _, mesh = mesh11
    for face in mesh.faces[1:-1]:
        left, right = mesh.points[face.ID - 1, 0], mesh.points[face.ID, 0]
        assert face.x == pytest.approx(0.5 * (left + right))
        assert face.distance == pytest.approx(0.05)
        assert face.owner == face.ID - 1


def test_boundary_faces_lie_half_a_cell_outside_domain(mesh11):
    _, mesh = mesh11
    assert mesh.faces[0].x == pytest.approx(-0.05)
    assert mesh.faces[-1].x == pytest.approx(1.05)


def test_boundary_mesh_keeps_only_boundary_entities(mesh11):
    _, mesh = mesh11
    assert mesh.boundary.num_cells == 2
    assert mesh.boundary.num_faces == 2
    assert [c.ID for c in mesh.boundary.cells] == [0, 10]


def test_offset_domain():
    _, mesh = build_mesh(5, length=2.0, start=-1.0)
    np.testing.assert_allclose(mesh.points[:, 0], [-1.0, -0.5, 0.0, 0.5, 1.0])
    assert mesh.cells[2].length == pytest.approx(0.5)
