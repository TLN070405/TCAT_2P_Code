"""Tests for the Van Genuchten closure relations (state_equations_interface.VanGenuchten)."""
import numpy as np
import pytest

TOL = 1.0e-12
PC_VALUES = [10.0, 100.0, 500.0, 2000.0, 1.0e4]


# Regression values: computed by the code before May 2025 and unchanged since.
def test_volume_fraction_regression(vg_legacy):
    assert vg_legacy.volume_fraction_w(p_c=100.0) == pytest.approx(0.44714173673915236, abs=TOL)


def test_d_volume_fraction_regression(vg_legacy):
    assert vg_legacy.d_volume_fraction_w(p_c=100.0) == pytest.approx(-2.1350378040960543e-05, abs=TOL)


def test_capillary_pressure_regression(vg_legacy):
    assert vg_legacy.capillary_pressure(e_w=0.3) == pytest.approx(980.9295867317348, abs=TOL)


def test_d_capillary_pressure_regression(vg_legacy):
    assert vg_legacy.d_capillary_pressure(e_w=0.3) == pytest.approx(-4484.85878135136, abs=TOL)


def test_relative_permeability_regression(vg_legacy):
    assert vg_legacy.w_relative_permeability(100.0) == pytest.approx(0.9361235565573552, abs=TOL)
    assert vg_legacy.n_relative_permeability(100.0) == pytest.approx(4.482348759703877e-05, abs=TOL)


# Analytic formula, written out independently of the implementation.
@pytest.mark.parametrize("p_c", PC_VALUES)
def test_volume_fraction_matches_analytic_formula(vg_residual, p_c):
    d = vg_residual.data
    se = (1.0 + (d.alpha * p_c) ** d.n) ** (-d.m)
    expected = 0.03 + se * (0.4 - 0.03 - 0.02)
    assert vg_residual.volume_fraction_w(p_c) == pytest.approx(expected, rel=1e-14)


# Consistency checks.
@pytest.mark.parametrize("p_c", PC_VALUES)
def test_capillary_pressure_inverts_volume_fraction(vg_residual, p_c):
    e_w = vg_residual.volume_fraction_w(p_c)
    assert vg_residual.capillary_pressure(e_w) == pytest.approx(p_c, rel=1e-10)


@pytest.mark.parametrize("p_c", PC_VALUES)
def test_d_volume_fraction_matches_finite_difference(vg_residual, fd, p_c):
    assert vg_residual.d_volume_fraction_w(p_c) == pytest.approx(
        fd(vg_residual.volume_fraction_w, p_c, 1e-4 * p_c), rel=1e-6
    )


@pytest.mark.parametrize("e_w", [0.05, 0.15, 0.25, 0.35])
def test_d_capillary_pressure_matches_finite_difference(vg_residual, fd, e_w):
    assert vg_residual.d_capillary_pressure(e_w) == pytest.approx(
        fd(vg_residual.capillary_pressure, e_w, 1e-7), rel=1e-6
    )


def test_volume_fraction_decreases_with_capillary_pressure(vg_residual):
    e_w = [vg_residual.volume_fraction_w(pc) for pc in np.linspace(1.0, 1e4, 200)]
    assert np.all(np.diff(e_w) < 0)


def test_volume_fraction_bounded_by_residuals(vg_residual):
    for pc in np.logspace(-1, 6, 50):
        assert 0.03 <= vg_residual.volume_fraction_w(pc) <= 0.4 - 0.02


def test_relative_permeabilities_in_unit_interval_and_monotone(vg_residual):
    pcs = np.linspace(1.0, 1e4, 200)
    k_w = np.array([vg_residual.w_relative_permeability(pc) for pc in pcs])
    k_n = np.array([vg_residual.n_relative_permeability(pc) for pc in pcs])
    assert np.all((0 <= k_w) & (k_w <= 1)) and np.all((0 <= k_n) & (k_n <= 1))
    assert np.all(np.diff(k_w) < 0) and np.all(np.diff(k_n) > 0)


@pytest.mark.parametrize("p_c", [0.0, -50.0])
def test_saturated_branch_for_non_positive_capillary_pressure(vg_residual, p_c):
    assert vg_residual.volume_fraction_w(p_c) == pytest.approx(0.4 - 0.02)
    assert vg_residual.d_volume_fraction_w(p_c) == 0.0
    assert vg_residual.w_relative_permeability(p_c) == 1.0
    assert vg_residual.n_relative_permeability(p_c) == 0.0


@pytest.mark.parametrize("e_w", [0.38, 0.39])
def test_zero_capillary_pressure_at_or_above_full_effective_saturation(vg_residual, e_w):
    assert vg_residual.capillary_pressure(e_w) == 0.0
    assert vg_residual.d_capillary_pressure(e_w) == 0.0
