import numpy as np
import unittest
import TCAT2Phase


# Set wetting-phase properties
w_phase = TCAT2Phase.properties.phase.Fluid('w')
n_phase = TCAT2Phase.properties.phase.Fluid('n')
# Set parameters for derivative density functions
w_den_params = {'rho0':1.e3,
                'beta':4.9e-10}
n_den_params = {'rho0':1.2,
                'alpha':1.189e-5}
# Set derivative density functions
w_phase.set_d_density_function(TCAT2Phase.state_equations.d_density_exp_pressure,**w_den_params)
n_phase.set_d_density_function(TCAT2Phase.state_equations.d_density_linear_pressure,**n_den_params)


rho_new = w_phase.d_density_function(-1000)
np.testing.assert_almost_equal(rho_new, 4.899997599000587e-07)

rho_new = n_phase.d_density_function(-1000)
np.testing.assert_almost_equal(rho_new,1.189e-05)
