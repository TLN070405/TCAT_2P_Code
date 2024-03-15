import numpy as np
import unittest
import TCAT2Phase

# Set solid phase properties
s_phase = TCAT2Phase.properties.phase.Solid('s')
s_phase.set_porosity(0.4)

# Set wetting-phase properties and pressure-saturation-permeability parameters
w_phase = TCAT2Phase.properties.phase.Fluid('w')
w_phase.set_e_wr(0.0)
w_phase.set_e_nr(0.0)
w_psk_params = {'e_wr':w_phase.e_wr,
                'e_nr':w_phase.e_nr,
                'porosity':s_phase.porosity,
                'alpha': 1e-3 ,
                'n': 2.494,
                'm': 0.5990376904570971,
                }

w_phase.set_volume_fraction_function(TCAT2Phase.state_equations.vg_volume_fraction_w,**w_psk_params)
e_w = w_phase.volume_fraction_function(100)


np.testing.assert_almost_equal(e_w,0.39923369351710036 )