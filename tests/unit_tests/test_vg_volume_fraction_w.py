import numpy as np
import unittest
import TCAT2Phase

# Set solid phase properties
s_phase = TCAT2Phase.properties.phase.Solid('s')
s_phase.set_porosity(0.4)

# Set wetting-phase properties and pressure-saturation-permeability parameters
w_phase = TCAT2Phase.properties.phase.Fluid('w')
w_psk_params = {'e_wr:wetting phase residual volume fraction':  0.0,
                'e_nr:nonwetting phase residual volume fraction': 0.0 ,
                'porosity':s_phase.porosity,
                'psk:alpha': 1e-3 ,
                'psk:n': 2.494,
                'psk:m': 0.5990376904570971,
                }

w_phase.set_volume_fraction_funtion(TCAT2Phase.state_equations.vg_volume_fraction_w,**w_psk_params)
e_w = w_phase.volume_fraction_function(100)


np.testing.assert_almost_equal(e_w,0.39923369351710036 )