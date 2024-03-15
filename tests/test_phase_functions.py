import TCAT2Phase

# Set solid phase properties
s_phase = TCAT2Phase.properties.phase.Solid('s')
s_phase.set_porosity(0.4)

# Set generic-phase properties
i_phase = TCAT2Phase.properties.phase.Fluid('i')

allowable_error = 1.e-12


def test_linear_density():
    """
    rho_i = f(p_i) where i is phase ID
    """
    linear_density_params = {'rho0':1.2,'alpha':1.189e-5}
    i_phase.set_density_function(TCAT2Phase.state_equations.density_linear_pressure,**linear_density_params)
    density = i_phase.density_function(p = -1000)
    assert abs(density - 1.18811) <= allowable_error

def test_exponential_density():
    """
    rho_i = f(p_i) where i is phase ID
    """
    exp_density_params = {'rho0':1.e3,'beta':4.9e-10}
    i_phase.set_density_function(TCAT2Phase.state_equations.density_exp_pressure,**exp_density_params)
    density = i_phase.density_function(p = -1000)
    assert abs(density - 999.99951000012) <= allowable_error

def test_volume_fraction():
    """
    e_n = f()
    """
    i_phase.set_volume_fraction_function(TCAT2Phase.state_equations.volume_fraction,porosity = s_phase.porosity)
    e_i = i_phase.volume_fraction_function(e_in = 0.3)
    assert abs(e_i - 0.1) <= allowable_error

test_linear_density()
test_exponential_density()
test_volume_fraction()
