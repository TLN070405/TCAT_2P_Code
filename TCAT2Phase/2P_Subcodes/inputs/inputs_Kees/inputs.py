#Malgorzata paper inputs (coarse sand)
import numpy as np
############space domain inputs
Nz= 101
Omega_0= 5.0 
Omega_L=  10.0
############ time inputs 
t0= 0.0 # first time to start with
time_interval=np.array([864.0,1728.0,2592.0,3456.0,6048.0 ]) 
# time_interval contains all the time outputs to be shown after solving the problem
############ physical parameters of the problem and van genuchten model parameters
phi= 0.4 
Epsilon_nr= 0.0 
Epsilon_wr= 0.0
n= 2.494
rho_w0= 1e3  
rho_n0= 1.2 
mu_w=  1e-3 
mu_n=  1.78e-5
g= 9.8066
d=-1  
alpha= 1e-3 
a= 4.9e-10 
b= 1.189e-5 
K= 1e-10
tau= 0.0  
Lx=1.0   # x dimension 
Ly=1.0   # y dimension
############ boundary and initial conditions of the problem
#S0= 0.02
#SL= 0.3
Ew_bc_0= 0.12                         # phi*SL
Ew_bc_L= 0.008                        # phi*S0
Pw_bc_0= -2113.152788450121           #-Pc(phi*SL)
Pw_bc_L= -13706.965861446095          #-Pc(phi*S0)
Pn_bc_0= 0.0 
Pn_bc_L= 0.0
Ew_IC= 0.008                          #phi*S0
Pw_IC= -13706.965861446095            #-Pc(phi*S0)
Pn_IC= 0.0

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

