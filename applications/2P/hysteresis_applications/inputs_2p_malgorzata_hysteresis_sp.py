import numpy as np

# Domain
domain_size = np.array([[0.0,1.0]])
dims = 1
nodes = 101
model_type = 'two_phase'
form = 'sp'

# boundary types: 0 - Dirchlet
#                 1 - Neumann
boundary_type = {'e_w': [[0, 0]],
                 'p_w': [[0, 0]]}

# boundary names
boundary_name = {'e_w': [['e_w', 'e_w']],
                 'p_w': [['p_w', 'p_w']]}

# solid phase properties
porosity = 0.4
permeability = 1e-10
gravity = 0.0

# wetting-phase properties
ewr = 0.0
w_viscosity = 1.e-3
w_den_params = {'rho0':1.e3,
                'beta':4.9e-10}

# Set nonwetting-phase properties
enr = 0.0
n_viscosity = 1.78e-5
n_den_params = {'rho0':1.2,
                'alpha':1.189e-5
                }
n_en_params = {'porosity':0.4
                }

# w-n interphase imbibition
n_imb = 2.494
m_imb= 1.0 - (1.0 / n_imb)
alpha_imb = 1e-3

# wn-interface properties drainage
n_dr = 2.494
m_dr = 1.0 - (1.0 / n_dr)
alpha_dr = 5e-4

# Parker Lenhard hysteresis data
drainage_new = np.full((nodes), True)
p_c_reversal = np.full((nodes), 0.0)
p_c_endpoint = np.full((nodes), 52880.51596256569)
e_w_reversal = np.full((nodes), 0.4)
e_w_endpoint = np.full((nodes), 0.003)
e_w_old = np.full((nodes), 0.397)
p_c_old = np.full((nodes), 346.73)
hysteresis_tol = 1e-6

# initial conditions
ew_ic = np.full((nodes), 0.397)
pw_ic = np.full((nodes), -346.73)

initial_conditions = {'e_w': ew_ic,
                      'p_w': pw_ic}

# Time
#time_interval = np.array([1.0,10.0,20.0,30.0,100.0,200.0,500.0,600,700,800,900.0,1000,1100,1200,1300,1400,1500,1600,1700,1800,1900,2000])
time_interval = np.array([1.0,10.0,20.0,30.0,100.0,200.0,500.0,600.0,700.0])

# Numerical Parameters
bdf_order = 5
ts_max_steps = 100000
ts_max_reject = 500
max_snes_failures = 10000
rtol = 1.e-8
atol = 1.e-8
pc_type = 'sor'

"""
we need to add also the bc functions ew0(t), ewL(t), pw0(t), pwL(t),...
"""