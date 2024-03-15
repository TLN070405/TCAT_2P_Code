import numpy as np
import unittest
import TCAT2Phase

# Set wetting-phase properties
w_phase = TCAT2Phase.properties.phase.Fluid('w')
# Set wetting-phase properties
n_phase = TCAT2Phase.properties.phase.Fluid('n')
# Set parameters for density function
w_den_params = {'rho0':1.e3,
                'beta':4.9e-10}
n_den_params = {'rho0':1.2,
                'alpha':1.189e-5}
w_phase.set_density_function(TCAT2Phase.state_equations.density_exp_pressure,**w_den_params)
n_phase.set_density_function(TCAT2Phase.state_equations.density_linear_pressure,**n_den_params)


rho_new = w_phase.density_function(-1000)
np.testing.assert_almost_equal(rho_new,999.99951000012)

rho_new = n_phase.density_function(-1000)
np.testing.assert_almost_equal(rho_new,1.18811)
