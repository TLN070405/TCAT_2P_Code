import petsc4py
import time
import sys
import math
import numpy as np
import cython
import matplotlib.pyplot as plt
petsc4py.init(sys.argv)
from petsc4py import PETSc
from libc.stdlib cimport malloc





###############  Variables declaration
cdef unsigned int MaxStepRejections,MaxSNESFailures,MaxTimeSteps,Nx
cdef double t0,First_Time_Step
cdef double Omega_0,Omega_L,dx ,Cx
cdef double Epsilon_nr,Epsilon_wr,Ks,alpha,n,m,rho_w0,Beta_w,g,epsilon
cdef double Const_Epsilon,n_inv,n_inv1,n1,m_inv,m1,m2,alpha_inv,alpha_m_n,k_muw_half
cdef double rtol,atol

############ TCAT paramtrs
Rww=1.5e6                                     #[Rww]=N*s/m^4
Gamma_wn=35e-3

############ time variables
First_Time_Step  = 1e-10
time_interval=np.array([1,10,100])
t0               = 0.0
############ spatial discretization variables
Nx               = 101
Omega_0          = 0.0
Omega_L          = 10.0
dx               = (Omega_L - Omega_0) / (Nx - 1.0)
Cx               = 1.0 / dx
############ General constants
g                = 9.81                        #[g]=m/s2
d                = 1
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






#######################################
########### Arrays for calcualtions to enhance the speed of code
cdef double *Vp_c              = <double*>malloc(Nx * sizeof(double))
cdef double *Vp_w              = <double*>malloc(Nx * sizeof(double))
cdef double *Vrho_w_of_P_w     = <double*>malloc(Nx * sizeof(double))
cdef double *VKrw_of_P_c       = <double*>malloc(Nx * sizeof(double))
cdef double *VKrw_rho_w        = <double*>malloc(Nx * sizeof(double))
cdef double *VPw_bc            = <double*>malloc(Nx * sizeof(double))
######## Arrays for visualization and storing results
Ve_w_Time_Results              = np.zeros([len(time_interval),Nx], dtype=np.double) 
Vp_w_Time_Results              = np.zeros([len(time_interval),Nx], dtype=np.double) 
Vj_wn_Time_Results             = np.zeros([len(time_interval),Nx], dtype=np.double) 
######## Arrays for time of simulations and CPU time
Clock_Array                    = np.zeros([len(time_interval)], dtype=float) 
ftime                          = np.zeros([len(time_interval)], dtype=float) 
#######################################
cdef double Pc(double Epsilon_w):
    cdef double Se, P_c
    Se = (Epsilon_w - Epsilon_wr) / Const_Epsilon
    if (Se > 0.0 and Se < 1.0):
       P_c = (  ( Se**(-m_inv)  - 1.0)**n_inv )*alpha_inv
    else:
       P_c = 0.0
    return P_c
####################################### 
cdef double Drvt_Epsilon_w_of_Pc(double P_c):
    cdef double Drvt_Se,Drvt_Epsilon_w
    if (P_c > 0.0):
       Drvt_Se = -alpha_m_n*((alpha*P_c)**n1)*((1.0 + ((alpha*P_c)**n))**m1)
       Drvt_Epsilon_w = Const_Epsilon*Drvt_Se
    else:
       Drvt_Epsilon_w = 0.0
    return Drvt_Epsilon_w

####################################### 

cdef double  Krw_of_Pc( double P_c):
    cdef double Se,Krw
    if (P_c > 0.0):
       Se =  (1.0 + ((alpha*P_c)**n))**(-m)
       Krw =    (Se**0.5)  *(  (1.0 - (  (1.0 - Se**m_inv)**m  )   )**2.0)
    else:
       Krw = 1.0
    return Krw


####################################### 
cdef double  Epsilon_w_of_Pc(double P_c):
    cdef double Se,Epsilon_w
    if (P_c > 0.0):
       Se =  (1.0 + (alpha*P_c)**n)**(-m)
       Epsilon_w =Const_Epsilon*  Se + Epsilon_wr
    else:
       Epsilon_w = epsilon-Epsilon_nr
    return Epsilon_w
 
#######################################
cdef double  rhow_of_Pw( double P_w):
    cdef double rho_w
    rho_w =rho_w0*math.exp(Beta_w*P_w)
    return rho_w
#######################################
cdef double  Drvt_rhow_of_Pw(double  P_w):
    cdef double Drvt_rho_w
    Drvt_rho_w =Beta_w*rho_w0*math.exp(Beta_w*P_w)
    return Drvt_rho_w


########################################################
#########################BC and IC######################
########################################################
######################################################## 
for i in range(0,Nx):
    x = Omega_0+i*dx
    if(i==0):
      VPw_bc[i]        = -1000-rho_w0*g*x 
      print(  Epsilon_w_of_Pc( -VPw_bc[i] ))
    elif(i==Nx-1):
      VPw_bc[i]        = -1000-rho_w0*g*x  
      print(  Epsilon_w_of_Pc( -VPw_bc[i] ))
    else:
      VPw_bc[i]        = -1000-rho_w0*g*x 
      

#######################################
cdef int Residual(ts,t,U,Udot,F):
# residual function is like sundials residual, the prototype is fixed but the name of the function can be changed
###    u[i]=p_w[i]
    cdef unsigned int i
    cdef double A_PR, Kw_plus, Kw_minus, rho_w_plus, rho_w_minus,Od_w
    da=ts.getDM()
    #######
    localU=da.getLocalVec()
    da.globalToLocal(U,localU)
    #######
    u=da.getVecArray(localU)
    udot=da.getVecArray(Udot)
    f=da.getVecArray(F)

  
    for i in range(0,Nx):
        Vp_w[i] = u[i]
        Vp_c[i] = -Vp_w[i]
        Vrho_w_of_P_w[i]=rhow_of_Pw(Vp_w[i])
        VKrw_of_P_c[i]=Krw_of_Pc(Vp_c[i])
        VKrw_rho_w[i]=Vrho_w_of_P_w[i]* VKrw_of_P_c[i]
    for i in range(1,Nx-1):
        A_PR=  -Vrho_w_of_P_w[i]*Drvt_Epsilon_w_of_Pc(Vp_c[i])+Epsilon_w_of_Pc(Vp_c[i])*Drvt_rhow_of_Pw(Vp_w[i])
              
        #########
        Kw_plus= k_muw_half*(VKrw_rho_w[i+1] + VKrw_rho_w[i])
        Kw_minus= k_muw_half*(VKrw_rho_w[i-1] + VKrw_rho_w[i])
        rho_w_plus = 0.5*(Vrho_w_of_P_w[i+1] + Vrho_w_of_P_w[i])
        rho_w_minus =0.5*(Vrho_w_of_P_w[i-1] + Vrho_w_of_P_w[i])
        Od_w = Cx * (Kw_plus*(Cx*(Vp_w[i+1] - Vp_w[i]) + rho_w_plus*g*d) - Kw_minus * (Cx*(Vp_w[i] - Vp_w[i-1]) + rho_w_minus*g*d))
        f[i] = A_PR * udot[i]  - Od_w
    f[0]    =  u[0] -VPw_bc[0]
    f[Nx-1] =  u[Nx-1]  -VPw_bc[Nx-1]
        

 

#######################################
cdef int Initialization(ts,U):
# a private function that we created to initialize the solution vector at t=t0
    cdef unsigned int i
    #######
    da=ts.getDM()
    u=da.getVecArray(U)
    for i in range(0,Nx):
        if(i==0):
          u[i]        = VPw_bc[i]
        elif(i==Nx-1):
          u[i]        = VPw_bc[i]
        else:
          u[i]        = VPw_bc[i]                # here we perturb the pressure at interior domain IC

#######################################
cdef void main():
#######################################    
# here we created a 1D grid, to let the solver store the jacobian as a sparse matrix (this is one of the ways to do that) 
    comm = PETSc.COMM_WORLD 
    bz= PETSc.DMDA.BoundaryType.NONE
    da=PETSc.DMDA().create(dof= 1,sizes=(Nx, ),boundary_type=bz,stencil_type=None,stencil_width=1, comm=comm ) 
    da.setUp()
    u= da.createGlobalVec()
    
#######################################    
    da.setMatType(PETSc.Mat.Type.AIJ)           # create a sparse matrix related to the grid created before (actually this is the type)
    da.createMatrix()                           # create the matrix
#######################################
    ts = PETSc.TS().create(PETSc.COMM_WORLD)    # create a time-stepping solver object 
    ts.setProblemType(ts.ProblemType.NONLINEAR) # set type of problem 
    ts.setType(ts.Type.BDF)                     # set type to BDF method	
    ts.setIFunction(Residual)                   # pass the residual function to "ts" so it can use it 
#######################################    
    ts.setDM(da)                                # attach the ts solver to the grid created
#######################################      
    snes=ts.getSNES()                           # get the snes solver
    snes.setFromOptions()                       # give the user the possibility to personalize the "snes" at running time
#######################################        
    ksp=snes.getKSP()                           # get the ksp solver
    ksp.setFromOptions()                        # give the user the possibility to personalize the "ksp" at running time
#######################################
    pc=ksp.getPC()                              # get the pc object from ksp
    pc.setFromOptions()                         # give the user the possibility to personalize the "pc" at running time
####################################### 
    ts.setMaxSteps(MaxTimeSteps)                    # maximum steps allowed to take to reach final time
    ts.setMaxStepRejections(MaxStepRejections)  # maximum steps rejections allowed , this is related to snes functions         
    ts.setMaxSNESFailures(MaxSNESFailures)
    ts.setTolerances(rtol, atol)                # set tolerances of the the chosen ts (here bdf )
    ts.setTime(t0)                              # set initial time t0 (here t0=0)
    ts.setExactFinalTime(ts.ExactFinalTime.MATCHSTEP)  # this option is to match the output time of ts with the output time defined by user.
    # ts.setTimeStep(First_Time_Step)
    ts.setFromOptions()                         # give the user the possibility to personalize the ts solver at running time
#######################################
    Initialization(ts,u)  
        

#############Calling the solver and start solving the DAE system ###################### 
    mesh = np.linspace(Omega_0, Omega_L, Nx)
    for i in range(len(time_interval)):
       ts.setMaxTime(time_interval[i])
       CPU_start = time.process_time()
       ts.solve(u)
       CPU_end = time.process_time()
       ftime[i]=ts.getTime()
       Clock_Array[i]=CPU_end-CPU_start
       #####visualization
       U=da.getVecArray(u)
       for j in range(0,Nx):
           Ve_w_Time_Results[i][j]   = Epsilon_w_of_Pc(-U[j])
           Vp_w_Time_Results[i][j]   = U[j]
           Vj_wn_Time_Results[i][j]  = -Vp_w_Time_Results[i][j]/Gamma_wn
       NTimeSteps=ts.getStepNumber()
       if(NTimeSteps==MaxTimeSteps): 
         break
    ####################  
    Output_Path='Outputs_RE_Parameters'
    file = open(Output_Path, "w")
    file.write('Nx= %d ' %Nx)
    file.write('\nOmega_0= %g ' %Omega_0)
    file.write('\nOmega_L= %g ' %Omega_L)
    file.write('\nTime(s)= ')   
    for i in range(len(time_interval)):
        file.write('%g ' %time_interval[i])
    file.close() 
    ####################
    Output_Path='Outputs_RE_Ew'
    file = open(Output_Path, "w")
    for i in range(len(time_interval)):
        file.write('Ew= ')
        for j in range(0,Nx):
            file.write('%f ' %Ve_w_Time_Results[i][j] )
        file.write('\n')    
    file.close()  
    ####################
    Output_Path='Outputs_RE_Pw'
    file = open(Output_Path, "w")
    for i in range(len(time_interval)):
        file.write('Pw= ')
        for j in range(0,Nx):
            file.write('%f ' %Vp_w_Time_Results[i][j] )
        file.write('\n')    
    file.close()  
    ####################
    Output_Path='Outputs_RE_Jwn'
    file = open(Output_Path, "w")
    for i in range(len(time_interval)):
        file.write('Jwn= ')
        for j in range(0,Nx):
            file.write('%f ' %Vj_wn_Time_Results[i][j] )
        file.write('\n')    
    file.close()  

#######################################"""

main() 

