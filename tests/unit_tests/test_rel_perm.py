import numpy as np
import unittest
import TCAT2Phase

# Set wetting-phase properties 
w_phase = TCAT2Phase.properties.phase.Fluid('w')
n_phase = TCAT2Phase.properties.phase.Fluid('n')
# Set pressure-saturation-permeability parameters
w_psk_params = {'psk:alpha': 1e-3,
                'psk:n': 2.494,
                'psk:m': 0.5990376904570971 ,
                'm_inv': 1.6693440428380186 ,
                }
n_psk_params = {'psk:alpha': 1e-3,
                'psk:n': 2.494,
                'psk:m': 0.5990376904570971 ,
                'm_inv': 1.6693440428380186 ,
                'm2'   : 1.1980753809141942,
                }

w_phase.set_rel_perm_function(TCAT2Phase.state_equations.vg_rel_perm_w,**w_psk_params)
n_phase.set_rel_perm_function(TCAT2Phase.state_equations.vg_rel_perm_n,**n_psk_params)


krw= w_phase.rel_perm_function(100)
np.testing.assert_almost_equal(krw, 0.9361235565573552 )

krn= n_phase.rel_perm_function(100)
np.testing.assert_almost_equal(krn, 4.482348759703877e-05 )