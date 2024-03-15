import numpy as np
import TCAT2Phase

# Set solid phase properties
s_phase = TCAT2Phase.properties.phase.Solid('s')
s_phase.set_porosity(0.4)

# Set wetting-phase properties
w_phase = TCAT2Phase.properties.phase.Fluid('w')
w_phase.set_e_wr(0.0)

# Set non-wetting phase properties
n_phase = TCAT2Phase.properties.phase.Fluid('n')
n_phase.set_e_nr(0.0)

# Set wn-interface properties
wn_interface = TCAT2Phase.properties.interface.Interface()

m = 0.5990376904570971
n = 2.494
alpha = 1.e-3

vg = TCAT2Phase.state_equations_interface.VanGenuchten(
    m,
    n,
    alpha,
    w_phase,
    n_phase,
    s_phase
    )

def test_w_volume_fraction():
    """
    e_w = f(p_c,Van_Genuchten)
    """
    w_phase.set_volume_fraction_function(vg.volume_fraction_w)
    e_w = w_phase.volume_fraction_function(p_c = 100)
    assert e_w == 0.39923369351710036

def test_capillary_pressure():
    """
    p_c = f(e_w,Van_Genuchten)
    """
    wn_interface.set_pc_function(vg.capillary_pressure)
    p_c = wn_interface.pc_function(e_w = 0.3)
    assert p_c == 823.6839383186395

def test_d_capillary_pressure():
    """
    d_p_c = f(e_w,Van_Genuchten)
    """
    wn_interface.set_d_pc_function(vg.d_capillary_pressure)
    d_p_c = wn_interface.d_pc_function(e_w = 0.3)
    assert d_p_c == -4818.89597078695



test_w_volume_fraction()
test_capillary_pressure()
test_d_capillary_pressure()
