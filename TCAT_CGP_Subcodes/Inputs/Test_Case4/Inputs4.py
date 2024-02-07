import numpy as np
 
Two_by3=2/3
pi=3.141592653589793
############
time_interval=np.array([1e-5,1e-4,1e-3,1e-2])
 
t0=0.0
######Space domain parameters######
Nx= 101
Omega_0= 0.0
Omega_L= 1.0
dx =(Omega_L - Omega_0) / (Nx - 1.0)
Cx =  1.0 / dx
Two_Cx=2*Cx
Half_Cx=0.5*Cx
Cx2= Cx*Cx
Two_by3_Cx2=Two_by3*Cx2

#####Soil properties#######
epsilon=0.36
g=9.81
d=0  # "d" is an integer to switch between horizontal and vertical domains (accounting or not for gravity)
#####TCAT parameters#######
Rww=1.5e6
Gamma_wn=35e-3
C_wn=-3.42e-7
k_wn_hat=-1.41e-2
rho_w0 = 1000.3
Beta_w= 1e-8 
##### Linear geometric state equation parameters
b0=1.47e-1
b1=-2.08e-3
b2=(5.33e-2)/(4*pi)

   
#####
epsilon6=epsilon*6
Half_Inv_Rww=1/(2*Rww)
Prod_Gamma_K_wn=Gamma_wn*k_wn_hat
Tol=1e-10
#########PETSc Solvers parameters###
rtol=1e-8  
atol=1e-8
MaxStepRejections=20000
MaxSNESFailures=20000
MaxTimeSteps=20000
 
