"""Tests for phase objects (properties/phase.py) and equations of state (state_equations.py)."""
import pytest

import TCAT2Phase
from TCAT2Phase.state_equations import state_equations

from conftest import air, water

TOL = 1.0e-12


def test_linear_density():
    rho = state_equations.density_linear_pressure(p=-1000.0, rho0=1.2, alpha=1.189e-5)
    assert rho == pytest.approx(1.18811, abs=TOL)


def test_exponential_density():
    rho = state_equations.density_exp_pressure(p=-1000.0, rho0=1.0e3, beta=4.9e-10)
    assert rho == pytest.approx(999.99951000012, abs=TOL)


@pytest.mark.parametrize("p", [-5000.0, 0.0, 2.0e4])
def test_density_derivatives_match_finite_difference(fd, p):
    d_lin = state_equations.d_density_linear_pressure(p, rho0=1.2, alpha=1.189e-5)
    assert d_lin == pytest.approx(
        fd(lambda q: state_equations.density_linear_pressure(q, 1.2, 1.189e-5), p, 1.0), rel=1e-9
    )
    d_exp = state_equations.d_density_exp_pressure(p, rho0=1.0e3, beta=1.0e-8)
    assert d_exp == pytest.approx(
        fd(lambda q: state_equations.density_exp_pressure(q, 1.0e3, 1.0e-8), p, 1.0), rel=1e-6
    )


def test_volume_fraction_is_complement_within_pore_space():
    assert state_equations.volume_fraction(e_in=0.3, porosity=0.4) == pytest.approx(0.1, abs=TOL)


def test_effective_saturation_endpoints():
    se = state_equations.effective_saturation
    assert se(0.03, e_wr=0.03, e_nr=0.02, porosity=0.4) == pytest.approx(0.0)
    assert se(0.38, e_wr=0.03, e_nr=0.02, porosity=0.4) == pytest.approx(1.0)


def test_fluid_binds_density_parameters():
    w = water(residual_volume_fraction=0.025)
    assert w.phase == "w"
    assert w.viscosity == 1.0e-3
    assert w.residual_volume_fraction == 0.025
    assert w.density_function(0.0) == pytest.approx(1.0e3)
    assert w.d_density_function(0.0) == pytest.approx(1.0e-5)
    assert w.relative_permeability_function is None


def test_fluid_relative_permeability_is_settable(vg_legacy):
    n = air()
    n.set_relative_permeability_function(vg_legacy.n_relative_permeability)
    assert n.relative_permeability_function(100.0) == pytest.approx(vg_legacy.n_relative_permeability(100.0))


def test_solid_stores_properties():
    s = TCAT2Phase.properties.phase.Solid(
        "s", porosity=0.4, permeability=1e-10, gravity=9.81, dynamic_capillary_pressure_parameter=None
    )
    assert (s.phase, s.porosity, s.permeability, s.gravity) == ("s", 0.4, 1e-10, 9.81)
    assert s.dynamic_capillary_pressure_parameter is None


def test_darcy_velocity_hydrostatic_is_zero(mesh11):
    """A hydrostatic gradient (dp/dx = -rho g) must give zero flux at interior faces."""
    import numpy as np

    _, mesh = mesh11
    rho, g = 1000.0, 9.81
    del_p = np.full(mesh.num_faces, -rho * g)
    q = state_equations.darcy_velocity(
        mesh, k=1e-10, k_r=np.ones(mesh.num_faces), mu=1e-3, del_p=del_p,
        rho=np.full(mesh.num_faces, rho), g=g,
    )
    np.testing.assert_allclose(q, 0.0, atol=1e-15)
