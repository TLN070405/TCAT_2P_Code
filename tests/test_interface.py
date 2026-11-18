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
