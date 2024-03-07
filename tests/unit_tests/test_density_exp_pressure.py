import numpy as np
import unittest
import TCAT2Phase

# Set wetting-phase properties
w_phase = TCAT2Phase.properties.phase.Fluid('w')
#w_phase.set_viscosity(1.e-3)
w_den_params = {'rho0':1.e3,
                'beta':4.9e-10}
w_phase.set_density_function(TCAT2Phase.state_equations.density_exp_pressure,**w_den_params)
#w_phase.set_d_density_function(TCAT2Phase.state_equations.d_density_exp_pressure,**w_den_params)

rho_new = w_phase.density_function(-1000)
# unittest.assertAlmostEqual(e_n,0.1)
np.testing.assert_almost_equal(rho_new,999.99951000012)
print(rho_new)