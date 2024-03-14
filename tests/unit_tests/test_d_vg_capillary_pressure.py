import numpy as np
import unittest
import TCAT2Phase

# Set solid phase properties
s_phase = TCAT2Phase.properties.phase.Solid('s')
s_phase.set_porosity(0.4)

# Set wetting-phase properties and pressure-saturation-permeability parameters
wn_interface = TCAT2Phase.properties.interface.Interface()
wn_psk_params = {'e_wr': 0.0,
                'e_nr': 0.0,
                'porosity': s_phase.porosity,
                'alpha_m_n': 0.001494,
                'm_inv': 1.6693440428380186,
                'n_inv1': -0.5990376904570971,
                }

wn_interface.set_d_pc_function(TCAT2Phase.state_equations.d_vg_capillary_pressure,**wn_psk_params)
d_pc = wn_interface.d_pc_function(0.3)

np.testing.assert_almost_equal(d_pc, -4818.89597078695)
 

