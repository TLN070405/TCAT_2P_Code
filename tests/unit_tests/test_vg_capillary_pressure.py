import numpy as np
import unittest
import TCAT2Phase

# Set solid phase properties
s_phase = TCAT2Phase.properties.phase.Solid('s')
s_phase.set_porosity(0.4)

# Set wetting-phase properties and pressure-saturation-permeability parameters
wn_interface = TCAT2Phase.properties.interface.Interface()
wn_psk_params = {'e_wr': 0.0,
                'e_nr': 0.0 ,
                'porosity': s_phase.porosity,
                'm_inv': 1.6693440428380186 ,
                'n_inv': 0.4009623095429029,
                'alpha_inv': 1000.0 ,
                }

wn_interface.set_pc_function(TCAT2Phase.state_equations.vg_capillary_pressure,**wn_psk_params)
pc = wn_interface.pc_function(0.3)

np.testing.assert_almost_equal(pc,823.6839383186395 )