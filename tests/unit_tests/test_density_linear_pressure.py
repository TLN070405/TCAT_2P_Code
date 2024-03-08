import numpy as np
import unittest
import TCAT2Phase

# Set wetting-phase properties
n_phase = TCAT2Phase.properties.phase.Fluid('n')
n_den_params = {'rho0':1.2,
                'alpha':1.189e-5}
n_phase.set_density_function(TCAT2Phase.state_equations.density_linear_pressure,**n_den_params)

rho_new = n_phase.density_function(-1000)
np.testing.assert_almost_equal(rho_new,1.18811)
print(rho_new)