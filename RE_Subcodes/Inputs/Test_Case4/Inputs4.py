import numpy as np
 
############ TCAT paramtrs
Rww=1.5e6                                     #[Rww]=N*s/m^4
Gamma_wn=35e-3

############ time variables
First_Time_Step  = 1e-10
time_interval=np.array([1e-5,1e-4,1e-3,1e-2])
t0               = 0.0
############ spatial discretization variables
Nx               = 101
Omega_0          = 0.0
Omega_L          = 1.0
dx               = (Omega_L - Omega_0) / (Nx - 1.0)
Cx               = 1.0 / dx
############ General constants
g                = 9.81                        #[g]=m/s2
d                = 0
############ wetting fluid parameters
Beta_w           = 1e-8                        #[Beta_w]=(m*s^2)/kg=1/Pa
rho_w0           = 1000                        #[rho_w0]=kg/m3
############ soil proprerties
epsilon          =  0.369                     #[epsilon]=null
Epsilon_wr       =  0.0                       #[Epsilon_wr]=null
Epsilon_nr       =  0.0                       #[Epsilon_nr]=null

############ VG parameters from fitting Amanda data ( Main imbibition)

n                = 3.4361               #[n]=null        
m                = 1.0 - (1.0 / n)      #[m]=null       
alpha            = 0.2001*1e-3          #[alpha]=1/Pa   

############
k_muw            = 1/Rww 
############ Constants calculated based on other Constants
k_muw_half       = 0.5*k_muw         
Const_Epsilon    = epsilon-Epsilon_nr - Epsilon_wr

n_inv            = 1.0/n
n_inv1           = n_inv-1.0
n1               = n-1.0
m_inv            = 1.0/m
m1               = -m-1.0
m2               = 2.0*m
alpha_inv        = 1.0/alpha
alpha_m_n        = alpha*m*n

############Solvers parameters
rtol             = 1e-8
atol             = 1e-8
MaxStepRejections= 500
MaxSNESFailures  = 10000     
MaxTimeSteps     = 10000
MaxTimeSteps     = 100000
############


