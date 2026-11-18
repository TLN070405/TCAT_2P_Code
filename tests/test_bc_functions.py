"""Tests for the Brooks-Corey closure relations (state_equations_interface.BrooksCorey)."""
import numpy as np
import pytest

TOL = 1.0e-12
PC_ABOVE_ENTRY = [1100.0, 1500.0, 4000.0, 2.0e4]


# Regression values: computed by the code before May 2025 and unchanged since.
def test_volume_fraction_regression(bc_legacy):
    assert bc_legacy.volume_fraction_w(p_c=100.0) == pytest.approx(0.448, abs=TOL)


def test_d_volume_fraction_regression(bc_legacy):
    assert bc_legacy.d_volume_fraction_w(p_c=100.0) == pytest.approx(0.0, abs=TOL)


def test_capillary_pressure_regression(bc_legacy):
    assert bc_legacy.capillary_pressure(e_w=0.3) == pytest.approx(3760.0341571957747, abs=TOL)


def test_d_capillary_pressure_regression(bc_legacy):
    assert bc_legacy.d_capillary_pressure(e_w=0.3) == pytest.approx(-2689.5809422001253, abs=TOL)


def test_relative_permeability_regression(bc_legacy):
    assert bc_legacy.w_relative_permeability(100.0) == pytest.approx(1.0, abs=TOL)
    assert bc_legacy.n_relative_permeability(100.0) == pytest.approx(0.0, abs=TOL)


# Analytic formula and consistency checks.
@pytest.mark.parametrize("p_c", PC_ABOVE_ENTRY)
def test_volume_fraction_matches_analytic_formula(bc_residual, p_c):
    se = (p_c / 1000.0) ** (-2.0)
    assert bc_residual.volume_fraction_w(p_c) == pytest.approx(0.03 + se * 0.35, rel=1e-14)


@pytest.mark.parametrize("p_c", PC_ABOVE_ENTRY)
def test_capillary_pressure_inverts_volume_fraction(bc_residual, p_c):
    assert bc_residual.capillary_pressure(bc_residual.volume_fraction_w(p_c)) == pytest.approx(p_c, rel=1e-10)


@pytest.mark.parametrize("p_c", PC_ABOVE_ENTRY)
def test_d_volume_fraction_matches_finite_difference(bc_residual, fd, p_c):
    assert bc_residual.d_volume_fraction_w(p_c) == pytest.approx(
        fd(bc_residual.volume_fraction_w, p_c, 1e-4 * p_c), rel=1e-6
    )


@pytest.mark.parametrize("e_w", [0.05, 0.15, 0.25, 0.35])
def test_d_capillary_pressure_matches_finite_difference(bc_residual, fd, e_w):
    assert bc_residual.d_capillary_pressure(e_w) == pytest.approx(
        fd(bc_residual.capillary_pressure, e_w, 1e-7), rel=1e-6
    )


@pytest.mark.parametrize("p_c", [0.0, 500.0, 1000.0])
def test_saturated_below_entry_pressure(bc_residual, p_c):
    assert bc_residual.volume_fraction_w(p_c) == pytest.approx(0.4 - 0.02)
    assert bc_residual.d_volume_fraction_w(p_c) == 0.0
    assert bc_residual.w_relative_permeability(p_c) == 1.0
    assert bc_residual.n_relative_permeability(p_c) == 0.0


def test_capillary_pressure_never_below_entry_pressure(bc_residual):
    for e_w in np.linspace(0.031, 0.38, 50):
        assert bc_residual.capillary_pressure(e_w) >= 1000.0


def test_relative_permeabilities_in_unit_interval_and_monotone(bc_residual):
    pcs = np.linspace(1001.0, 2.0e4, 200)
    k_w = np.array([bc_residual.w_relative_permeability(pc) for pc in pcs])
    k_n = np.array([bc_residual.n_relative_permeability(pc) for pc in pcs])
    assert np.all((0 <= k_w) & (k_w <= 1)) and np.all((0 <= k_n) & (k_n <= 1))
    assert np.all(np.diff(k_w) < 0) and np.all(np.diff(k_n) > 0)
