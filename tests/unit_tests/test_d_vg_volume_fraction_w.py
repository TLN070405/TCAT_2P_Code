import numpy as np
import unittest
import TCAT2Phase

# Set solid phase properties
s_phase = TCAT2Phase.properties.phase.Solid('s')
s_phase.set_porosity(0.4)

# Set wetting-phase properties and pressure-saturation-permeability parameters
w_phase = TCAT2Phase.properties.phase.Fluid('w')
w_psk_params = {'e_wr:wetting phase residual volume fraction': 0.0,
                'e_nr:nonwetting phase residual volume fraction': 0.0 ,
                'porosity': s_phase.porosity,
                'psk:alpha': 1e-3,
                'psk:n': 2.494,
                'alpha_m_n': 0.001494,
                'n1': 1.4940000000000002,
                'm1': -1.599037690457097,
                }
w_phase.set_d_volume_fraction_function(TCAT2Phase.state_equations.d_vg_volume_fraction_w,**w_psk_params)
d_e_w = w_phase.d_volume_fraction_function(100)


np.testing.assert_almost_equal(d_e_w,0.39923369351710036 )