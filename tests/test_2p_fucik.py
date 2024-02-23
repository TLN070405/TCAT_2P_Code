#Fucik paper inputs
import numpy as np
############space domain inputs
Nz= 101
Omega_0= 0.0
Omega_L= 1.0
############ time inputs 
t0= 0.0 # first time to start with
time_interval=np.array([1000.0]) # time_interval contains all the time outputs to be shown after solving the problem
############ physical parameters of the problem and PSK model parameters
phi= 0.448
Epsilon_nr= 0.0
Epsilon_wr= phi*0.265
rho_w0 =997.8  
rho_n0 =1.205
tau=1.1e6
lamda=4.66
Pd=3450
K=1.63e-11
mu_w=9.77e-4
mu_n=1.82e-5
a=4.9e-10
b=1.189e-5
g=0.0
d=1
Lx=1.0   # x dimension 
Ly=1.0   # y dimension
############ boundary and initial conditions of the problem
Ew_bc_0=phi*(1.0-0.73)  
Ew_bc_L=  phi
Pw_bc_0= -10067.198963341432 
Pw_bc_L=  -3450.0
Pn_bc_0= 0.0
Pn_bc_L= 0.0
Ew_IC= phi
Pw_IC=  -3450.0
Pn_IC=  0.0
############ ts solver parameters inputs
rtol= 1e-8
atol= 1e-8
MaxStepRejections=500
MaxSNESFailures=10000
MaxTimeSteps=200000
############
dz = (Omega_L - Omega_0) / (Nz - 1.0)
Cz = 1.0 / dz
k_muw_half=0.5*(K/mu_w)
k_mun_half=0.5*(K/mu_n)
Const_Epsilon = phi-Epsilon_nr - Epsilon_wr
############

