"""Tests for the finite-volume helpers in numerics/utils.py."""
import numpy as np
import pytest

from TCAT2Phase.numerics import utils


@pytest.fixture
def quadratic(mesh11):
    """y = x^2 sampled at the 11 mesh points, with exact face gradients and averages."""
    _, mesh = mesh11
    x = mesh.points[:, 0]
    y = x**2
    dx = x[1] - x[0]
    face_gradient = (y[1:] - y[:-1]) / dx
    face_average = 0.5 * (y[1:] + y[:-1])
    return mesh, y, face_gradient, face_average


def test_gradient_at_interior_faces(quadratic):
    mesh, y, face_gradient, _ = quadratic
    grad = utils.gradient(y, mesh)
    np.testing.assert_allclose(grad[1:-1], face_gradient, rtol=0, atol=1e-12)


def test_gradient_is_zero_on_boundary_faces(quadratic):
    mesh, y, _, _ = quadratic
    grad = utils.gradient(y, mesh)
    assert grad[0] == 0.0 and grad[-1] == 0.0


def test_gradient_of_linear_field_is_constant(mesh11):
    _, mesh = mesh11
    y = 3.0 * mesh.points[:, 0] + 2.0
    np.testing.assert_allclose(utils.gradient(y, mesh)[1:-1], 3.0, atol=1e-12)


def test_interpolate_to_interior_faces(quadratic):
    mesh, y, _, face_average = quadratic
    values = utils.interpolate(y, mesh)
    np.testing.assert_allclose(values[1:-1], face_average, rtol=0, atol=1e-12)


def test_cell_center_applies_function_per_cell(mesh11):
    _, mesh = mesh11
    x = np.arange(mesh.num_cells, dtype=float)
    np.testing.assert_allclose(utils.cell_center(x, lambda v: 2 * v + 1, mesh), 2 * x + 1)


def test_averages():
    assert utils.calc_average(2.0, 4.0) == 3.0
    assert utils.calc_function_average(lambda v: v**2, 1.0, 3.0) == 5.0
