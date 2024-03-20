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

pd    = 3450
lamda = 4.66

bc = TCAT2Phase.state_equations_interface.BrooksCorey(
    pd = pd,
    lamda = lamda,
    porosity = s_phase.porosity,
    e_wr = w_phase.residual_volume_fraction,
    e_nr = n_phase.residual_volume_fraction,
    )

allowable_error = 1.e-12

def test_w_volume_fraction():
    """
    e_w = f(p_c,Brooks_Corey)
    """
    w_phase.set_volume_fraction_function(bc.volume_fraction_w)
    e_w = w_phase.volume_fraction_function(p_c = 4000)

    assert abs(e_w -  0.2007693508484843) <= allowable_error

def test_d_w_volume_fraction():
    """
    d_e_w = f(p_c,Brooks_Corey)
    """
    w_phase.set_d_volume_fraction_function(bc.d_volume_fraction_w)
    d_e_w = w_phase.d_volume_fraction_function(p_c = 4000)

    assert abs(d_e_w - -0.00023389629373848418) <= allowable_error

def test_capillary_pressure():
    """
    p_c = f(e_w,Brooks_Corey)
    """
    wn_interface.set_pc_function(bc.capillary_pressure)
    p_c = wn_interface.pc_function(e_w = 0.3)
    
    assert abs(p_c - 3669.6951064537348) <= allowable_error

def test_d_capillary_pressure():
    """
    d_p_c = f(e_w,Brooks_Corey)
    """
    wn_interface.set_d_pc_function(bc.d_capillary_pressure)
    d_p_c = wn_interface.d_pc_function(e_w = 0.3)
    
    assert abs(d_p_c - -2624.960734230139) <= allowable_error

def test_w_relative_permability():
    """
    k_rw = f(p_c,Brooks_Corey)
    """
    w_phase.set_rel_perm_function(bc.w_relative_permability)
    k_wr = w_phase.rel_perm_function(4000)

    assert abs(k_wr - 0.09406552372665236) <= allowable_error

def test_n_relative_permability():
    """
    k_nw = f(p_c,Brooks_Corey)
    """
    n_phase.set_rel_perm_function(bc.n_relative_permability)
    k_nr = n_phase.rel_perm_function(4000)

    assert abs(k_nr - 0.15545111477640028) <= allowable_error



test_w_volume_fraction()
test_d_w_volume_fraction()
test_capillary_pressure()
test_d_capillary_pressure()
test_w_relative_permability()
test_n_relative_permability()


 
