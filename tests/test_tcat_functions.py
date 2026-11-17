"""Tests for the TCAT geometric closure relations (state_equations_interface.TCAT)."""
import pytest

from TCAT2Phase.state_equations import state_equations_interface


@pytest.fixture
def tcat():
    return state_equations_interface.TCAT(
        b1=0.1, b2=0.01, b3=0.001, porosity=0.4, e_ss=8000.0, js_ns=100.0, kn_ns=50.0
    )


def test_derived_length_scale(tcat):
    assert tcat.D == pytest.approx(6 * 0.4 / 8000.0)
    assert tcat.b1D == pytest.approx(0.1 * tcat.D)
    assert tcat.b2D2 == pytest.approx(0.01 * tcat.D**2)
    assert tcat.b3D3 == pytest.approx(0.001 * tcat.D**3)


@pytest.mark.parametrize("e_w", [0.0, 0.1, 0.25, 0.4])
def test_solid_interfacial_areas_partition_solid_surface(tcat, e_w):
    assert tcat.interfacial_area_ws(e_w) + tcat.interfacial_area_ns(e_w) == pytest.approx(8000.0)


def test_solid_interfacial_area_endpoints(tcat):
    assert tcat.interfacial_area_ns(0.4) == pytest.approx(0.0)
    assert tcat.interfacial_area_ws(0.0) == pytest.approx(0.0)


@pytest.mark.parametrize("e_w,e_wn,j_wn", [(0.3, 500.0, 2000.0), (0.2, 1200.0, 800.0)])
def test_gaussian_curvature_satisfies_geometric_relation(tcat, e_w, e_wn, j_wn):
    """
    gaussian_curvature solves the linear geometric relation for the n phase:
        porosity - e_w = b1 D (e_wn + e_ns) + b2 D^2 (-e_wn J_wn + e_ns J_ns) + b3 D^3 (e_wn K_wn + e_ns K_ns)
    """
    k_wn = tcat.gaussian_curvature(e_w, e_wn, j_wn)
    e_ns = tcat.interfacial_area_ns(e_w)
    rhs = (
        tcat.b1D * (e_wn + e_ns)
        + tcat.b2D2 * (-e_wn * j_wn + e_ns * tcat.js_ns)
        + tcat.b3D3 * (e_wn * k_wn + e_ns * tcat.kn_ns)
    )
    assert rhs == pytest.approx(0.4 - e_w, rel=1e-10)


def test_relative_permeability_is_square_of_volume_fraction(tcat):
    assert tcat.w_relative_permeability(0.3) == pytest.approx(0.09)
    assert tcat.n_relative_permeability(0.1) == pytest.approx(0.01)


@pytest.mark.xfail(
    strict=True,
    reason=(
        "TCAT.volume_fraction_w and TCAT.gaussian_curvature use different forms of the "
        "geometric relation, so one is not the inverse of the other. The solvers only call "
        "gaussian_curvature; confirm which form volume_fraction_w should use."
    ),
)
def test_volume_fraction_inverts_gaussian_curvature(tcat):
    e_w, e_wn, j_wn = 0.3, 500.0, 2000.0
    k_wn = tcat.gaussian_curvature(e_w, e_wn, j_wn)
    assert tcat.volume_fraction_w(e_wn, j_wn, k_wn) == pytest.approx(e_w, rel=1e-8)
