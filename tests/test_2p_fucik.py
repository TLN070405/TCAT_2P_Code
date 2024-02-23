#Fucik paper inputs
import numpy as np
import TCAT2Phase

# Domain Info
L = [1.0, 1.0]

# Discretization 
nodes = 101

# Time
time = np.linspace(0,1000,10)

############ physical parameters of the problem and PSK model parameters
phi = 0.448
epsilon_nr = 0.0
epsilon_wr = phi*0.265
rho_w0 = 997.8  
rho_n0 = 1.205
tau = 1.1e6
lamda = 4.66
Pd = 3450
K = 1.63e-11
mu_w = 9.77e-4
mu_n = 1.82e-5
a = 4.9e-10
b = 1.189e-5
g = 0.0
d = 1 

# Boundary and initial conditions
ew_bc = [phi*(1.0-0.73),phi]
ew_ic = phi

pw_bc = [-10067.198963341432, -3450.0]
pw_ic = -3450.0

pn_bc = [0.0, 0.0]
pn_ic = 0.0





## Solver parameters inputs
r_tol= 1e-8
a_tol= 1e-8
max_step_rejections=500
max_SNES_Failures=10000
max_time_tteps=200000
