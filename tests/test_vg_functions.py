import TCAT2Phase

# Set solid phase properties
s_phase = TCAT2Phase.properties.phase.Solid('s')
s_phase.set_porosity(0.4)

# Set wetting-phase properties
w_phase = TCAT2Phase.properties.phase.Fluid('w')
w_phase.set_residual_volume_fraction(0.0)

# Set non-wetting phase properties
n_phase = TCAT2Phase.properties.phase.Fluid('n')
n_phase.set_residual_volume_fraction(0.0)

# Set wn-interface properties
wn_interface = TCAT2Phase.properties.interface.Interface()

m = 0.5990376904570971
n = 2.494
alpha = 1.e-3

vg = TCAT2Phase.state_equations_interface.VanGenuchten(
    m = m,
    n = n,
    alpha = alpha,
    porosity = s_phase.porosity,
    e_wr = w_phase.residual_volume_fraction,
    e_nr = n_phase.residual_volume_fraction,
    )

allowable_error = 1.e-12

def test_w_volume_fraction():
    """
    e_w = f(p_c,Van_Genuchten)
    """
    w_phase.set_volume_fraction_function(vg.volume_fraction_w)
    e_w = w_phase.volume_fraction_function(p_c = 100)

    assert abs(e_w - 0.39923369351710036) <= allowable_error

def test_d_w_volume_fraction():
    """
    d_e_w = f(p_c,Van_Genuchten)
    """
    w_phase.set_d_volume_fraction_function(vg.d_volume_fraction_w)
    d_e_w = w_phase.d_volume_fraction_function(p_c = 100)

    assert abs(d_e_w - 0.0) <= allowable_error

def test_capillary_pressure():
    """
    p_c = f(e_w,Van_Genuchten)
    """
    wn_interface.set_pc_function(vg.capillary_pressure)
    p_c = wn_interface.pc_function(e_w = 0.3)
    
    assert abs(p_c - 823.6839383186395) <= allowable_error

def test_d_capillary_pressure():
    """
    d_p_c = f(e_w,Van_Genuchten)
    """
    wn_interface.set_d_pc_function(vg.d_capillary_pressure)
    d_p_c = wn_interface.d_pc_function(e_w = 0.3)
    
    assert abs(d_p_c - -4818.89597078695) <= allowable_error

def test_w_relative_permability():
    """
    k_rw = f(p_c,Van_Genuchten)
    """
    w_phase.set_rel_perm_function(vg.w_relative_permability)
    k_wr = w_phase.rel_perm_function(100)

    assert abs(k_wr - 0.9361235565573552) <= allowable_error

def test_n_relative_permability():
    """
    k_nw = f(p_c,Van_Genuchten)
    """
    n_phase.set_rel_perm_function(vg.n_relative_permability)
    k_nr = n_phase.rel_perm_function(100)

    assert abs(k_nr - 0.04376946660914664) <= allowable_error



test_w_volume_fraction()
test_d_w_volume_fraction()
test_capillary_pressure()
test_d_capillary_pressure()
test_w_relative_permability()
test_n_relative_permability()
