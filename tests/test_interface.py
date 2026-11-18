"""Tests that the interface classes (properties/interface.py) wire up their closure functions."""
import pytest

import TCAT2Phase
from TCAT2Phase.state_equations import state_equations_interface


def test_traditional_interface_exposes_psk_functions(vg_residual):
    iface = TCAT2Phase.properties.interface.TraditionalInterface(vg_residual)
    assert iface.pc_function(0.2) == vg_residual.capillary_pressure(0.2)
    assert iface.d_pc_function(0.2) == vg_residual.d_capillary_pressure(0.2)
    assert iface.volume_fraction_function(800.0) == vg_residual.volume_fraction_w(800.0)
    assert iface.d_volume_fraction_function(800.0) == vg_residual.d_volume_fraction_w(800.0)
    assert iface.w_relative_permeability(800.0) == vg_residual.w_relative_permeability(800.0)
    assert iface.n_relative_permeability(800.0) == vg_residual.n_relative_permeability(800.0)


def test_traditional_interface_accepts_brooks_corey(bc_residual):
    iface = TCAT2Phase.properties.interface.TraditionalInterface(bc_residual)
    assert iface.pc_function(0.2) == bc_residual.capillary_pressure(0.2)


def test_tcat_interface_stores_parameters_and_closures():
    tcat = state_equations_interface.TCAT(
        b1=0.1, b2=0.01, b3=0.001, porosity=0.4, e_ss=8000.0, js_ns=100.0, kn_ns=50.0
    )
    iface = TCAT2Phase.properties.interface.TCATInterface(
        tcat,
        interfacial_dynamics=1.0,
        interfacial_tension=0.072,
        interfacial_area_relaxation=2.0,
        interfacial_area_equilibrium=3.0,
    )
    assert (iface.interfacial_dynamics, iface.interfacial_tension) == (1.0, 0.072)
    assert (iface.interfacial_area_relaxation, iface.interfacial_area_equilibrium) == (2.0, 3.0)
    assert iface.gaussian_curvature(0.3, 500.0, 2000.0) == pytest.approx(tcat.gaussian_curvature(0.3, 500.0, 2000.0))
    assert iface.w_relative_permeability(0.3) == pytest.approx(0.09)
    assert iface.n_relative_permeability(0.1) == pytest.approx(0.01)


def _tcat_interface(**kwargs):
    tcat = state_equations_interface.TCAT(
        b1=0.1, b2=0.01, b3=0.001, porosity=0.4, e_ss=8000.0, js_ns=100.0, kn_ns=50.0
    )
    return TCAT2Phase.properties.interface.TCATInterface(
        tcat, interfacial_dynamics=1.0, interfacial_tension=0.072,
        interfacial_area_relaxation=2.0, interfacial_area_equilibrium=3.0, **kwargs)


def test_gaussian_curvature_sign_defaults_to_theory():
    assert _tcat_interface().gaussian_curvature_sign == 1.0
    assert _tcat_interface(gaussian_curvature_sign=-1).gaussian_curvature_sign == -1.0
    with pytest.raises(ValueError):
        _tcat_interface(gaussian_curvature_sign=2)


@pytest.mark.parametrize("sign", [1.0, -1.0])
def test_curvature_equation_uses_gaussian_curvature_sign(sign):
    """Mean-curvature row: e_wn dJ/dt + J de_wn/dt + s*2K de_w/dt (s = +1 theory, -1 legacy)."""
    import numpy as np
    from types import SimpleNamespace
    from TCAT2Phase.numerics.residual_tcat_cgp import ResidualTcatCGP
    from TCAT2Phase.numerics.residual_full_tcat import ResidualTcatFULL

    iface = SimpleNamespace(gaussian_curvature_sign=sign)
    phase = SimpleNamespace(d_density_function=lambda p: 4.5e-7)
    model = SimpleNamespace(current_cell=0, interface=iface, w_phase=phase, n_phase=phase)
    values = dict(e_w=0.3, e_n=0.069, p_w=-500.0, p_n=0.0, e_wn=700.0, j_wn=6944.0,
                  g_wn=0.3, w_wn=0.0, k_wn=2.5e7)
    pv = {name: np.array([value]) for name, value in values.items()}
    fluxes = {"rho_w": np.array([1000.0]), "rho_n": np.array([1.2])}

    for cls, num_u, row in [(ResidualTcatCGP, 6, 3), (ResidualTcatFULL, 8, 4)]:
        res = object.__new__(cls)
        res.model, res.time_mat = model, np.zeros([num_u, num_u])
        res.compute_time_matrix(pv, fluxes)
        assert res.time_mat[row, 0] == pytest.approx(sign * 2.0 * 2.5e7)
        assert res.time_mat[row, 3 if num_u == 6 else 4] == pytest.approx(700.0)  # e_wn dJ/dt
