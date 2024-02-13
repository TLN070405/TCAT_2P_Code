#Helmig paper inputs
import numpy as np
############space domain inputs
Nz= 101
Omega_0= 0.0
Omega_L= 0.6
############ time inputs 
t0= 0.0 # first time to start with
time_interval=np.array([0.1,10.0,50.0,100.0,150.0,200.0,300.0,400.0,500.0,600.0,700.0,800.0,900.0,1000.0,1200.0,1400.0,1600.0,1800.0,2000.0,2100.0,2200.0,2300.0,2400.0,2500.0])
# time_interval contains all the time outputs to be shown after solving the problem
############ physical parameters of the problem and van genuchten model parameters
phi= 0.37
Epsilon_nr= 0.0
Epsilon_wr= 0.1036
n= 2.0 
rho_w0= 997.8  
rho_n0= 1.205 
mu_w= 1.0e-3 
mu_n= 1.7e-5
g= 0.0
d=1 
alpha= 3.4e-4
a= 4.9e-10
b= 1.189e-5
K= 9.4e-10
tau= 1e5
Lx=0.1   # x dimension 
Ly=1.0   # y dimension
############ boundary and initial conditions of the problem
Ew_bc_0= 0.1998
Ew_bc_L= 0.111
Pw_bc_0= 92642.5
Pw_bc_L= 1900
Pn_bc_0= 1e5
Pn_bc_L= 1e5
Ew_IC= 0.111
Pw_IC= 1900
Pn_IC= 1e5
############ ts solver parameters inputs
rtol= 1e-8
atol= 1e-8
MaxStepRejections=500
MaxSNESFailures=10000  
MaxTimeSteps=20000 
############
dz = (Omega_L - Omega_0) / (Nz - 1.0)
Cz = 1.0 / dz
m = 1.0 - (1.0 / n)
k_muw_half=0.5*(K/mu_w)
k_mun_half=0.5*(K/mu_n)
Const_Epsilon = phi-Epsilon_nr - Epsilon_wr
n_inv=1.0/n
n_inv1=n_inv-1.0
n1=n-1.0
m_inv=1.0/m
m1=-m-1.0
m2=2.0*m
alpha_inv=1.0/alpha
alpha_m_n=alpha*m*n
############

