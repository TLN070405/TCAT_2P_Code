import numpy as np
import unittest
import TCAT2Phase

# Set solid phase properties
s_phase = TCAT2Phase.properties.phase.Solid('s')
s_phase.set_porosity(0.4)

# Set nonwetting-phase properties
n_phase = TCAT2Phase.properties.phase.Fluid('n')
n_en_params = {'porosity':s_phase.porosity
                }
n_phase.set_volume_fraction_funtion(TCAT2Phase.state_equations.volume_fraction,**n_en_params)
e_n = n_phase.volume_fraction_function(.3)
# unittest.assertAlmostEqual(e_n,0.1)
np.testing.assert_almost_equal(e_n,0.1)
print(e_n)