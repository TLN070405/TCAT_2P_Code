"""Shared fixtures and helpers for the TCAT2Phase test suite."""
from types import SimpleNamespace

import numpy as np
import pytest

import TCAT2Phase
from TCAT2Phase.state_equations import state_equations
from TCAT2Phase.state_equations import state_equations_interface


def central_difference(func, x, h):
    """Second-order central finite-difference derivative of func at x."""
    return (func(x + h) - func(x - h)) / (2.0 * h)


@pytest.fixture
def fd():
    """Finite-difference derivative helper, used to check analytic derivatives."""
    return central_difference


def build_mesh(nodes, length=1.0, start=0.0):
    """Return (domain, mesh) for a uniform 1D grid."""
    domain = TCAT2Phase.properties.domain.initialize_domain(
        dims=1, domain_size=np.array([[start, start + length]])
    )
    mesh = TCAT2Phase.mesh.grid.Mesh()
    mesh.uniform_mesh_1d(domain, nodes)
    return domain, mesh


@pytest.fixture
def mesh11():
    """Uniform 11-node mesh on [0, 1]."""
    return build_mesh(11)


@pytest.fixture
def vg_legacy():
    """Van Genuchten model used by the original (pre-2025) tests; no residual saturations."""
    return state_equations_interface.VanGenuchten(
        m=0.5990376904570971, n=2.494, alpha=1.0e-3, porosity=0.448, e_wr=0.0, e_nr=0.0
    )


@pytest.fixture
def vg_residual():
    """Van Genuchten model with non-zero residual volume fractions."""
    return state_equations_interface.VanGenuchten(
        m=0.6, n=2.5, alpha=2.0e-3, porosity=0.4, e_wr=0.03, e_nr=0.02
    )


@pytest.fixture
def bc_legacy():
    """Brooks-Corey model used by the original (pre-2025) tests."""
    return state_equations_interface.BrooksCorey(
        pd=3450.0, lamda=4.66, porosity=0.448, e_wr=0.0, e_nr=0.0
    )


@pytest.fixture
def bc_residual():
    """Brooks-Corey model with non-zero residual volume fractions."""
    return state_equations_interface.BrooksCorey(
        pd=1000.0, lamda=2.0, porosity=0.4, e_wr=0.03, e_nr=0.02
    )


def water(residual_volume_fraction=0.0):
    """Wetting phase with the exponential density used by the test-case drivers."""
    return TCAT2Phase.properties.phase.Fluid(
        "w",
        viscosity=1.0e-3,
        momentum_resistance=0.0,
        residual_volume_fraction=residual_volume_fraction,
        density_function=state_equations.density_exp_pressure,
        d_density_function=state_equations.d_density_exp_pressure,
        rho0=1.0e3,
        beta=1.0e-8,
    )


def air(residual_volume_fraction=0.0):
    """Non-wetting phase with the linear density used by the test-case drivers."""
    return TCAT2Phase.properties.phase.Fluid(
        "n",
        viscosity=1.78e-5,
        momentum_resistance=0.0,
        residual_volume_fraction=residual_volume_fraction,
        density_function=state_equations.density_linear_pressure,
        d_density_function=state_equations.d_density_linear_pressure,
        rho0=1.2,
        alpha=1.189e-5,
    )


def fake_model(porosity, e_wr, e_nr, nodes=1):
    """
    Minimal stand-in for numerics.model.Model with just the attributes the
    hysteresis model reads, so hysteresis can be tested without PETSc.
    """
    return SimpleNamespace(
        nodes=nodes,
        current_cell=0,
        solid=SimpleNamespace(porosity=porosity),
        w_phase=SimpleNamespace(residual_volume_fraction=e_wr),
        n_phase=SimpleNamespace(residual_volume_fraction=e_nr),
    )
