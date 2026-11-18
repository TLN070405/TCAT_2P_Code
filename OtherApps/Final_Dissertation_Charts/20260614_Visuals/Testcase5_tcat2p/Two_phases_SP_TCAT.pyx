import petsc4py
import time
import sys
import math
import numpy as np
import cython
import matplotlib.pyplot as plt
petsc4py.init(sys.argv)
from petsc4py import PETSc
from libc.stdlib cimport malloc,free

OptDB = PETSc.Options()




###############  Variables declaration
cdef unsigned int MaxStepRejections,MaxSNESFailures,MaxTimeSteps,Nx
cdef double t0,First_Time_Step
cdef double Omega_0,Omega_L,dx ,Cx
cdef double Epsilon_nr,Epsilon_wr,Ks,alpha,n,m,rho_w0,Beta_w,g,epsilon
cdef double Const_Epsilon,n_inv,n_inv1,n1,m_inv,m1,m2,alpha_inv,alpha_m_n,k_muw_half
cdef double rtol,atol

############ TCAT paramtrs
Rww       = 3.75e5                                     #[Rww]=N*s/m^4
Rnn       = 9.07e2
Gamma_wn  = 7.20e-2

############ time variables
First_Time_Step  = 1e-10
time_interval=np.array([1e-8,1e-1,1,10,50,100,200,300]) 


t0               = 0.0
############ spatial discretization variables
Nx               = OptDB.getInt('Nx', 101)
Omega_0          = 0.0
Omega_L          = 1.0
dx               = (Omega_L - Omega_0) / (Nx - 1.0)
Cx               = 1.0 / dx
############ General constants
g                = 9.81                        #[g]=m/s2
d                = 0
############ wetting fluid parameters
Beta_w           = 4.5e-10                        #[Beta_w]=(m*s^2)/kg=1/Pa
rho_w0           = 1000                        #[rho_w0]=kg/m3
b                = 1.189e-5
rho_n0           = 1.2
############ soil proprerties
epsilon          =  0.369                     #[epsilon]=null
Epsilon_wr       =  0.0687*epsilon                       #[Epsilon_wr]=null
Epsilon_nr       =  0.0                      #[Epsilon_nr]=null

############ VG parameters from fitting Amanda data ( Main imbibition)

n                = 37.8              #[n]=null        
m                = 1.0 - (1.0 / n)      #[m]=null       
alpha            = 2.28e-3         #[alpha]=1/Pa   

############
k_muw            = 1/Rww 
k_mun            = 1/Rnn
############ Constants calculated based on other Constants
k_muw_half       = 0.5*k_muw         
k_mun_half       = 0.5*k_mun    
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






node_i=int((Nx-1)/2)
#######################################
########### Arrays for calcualtions to enhance the speed of code
cdef double *Vp_c              = <double*>malloc(Nx * sizeof(double))
cdef double *Vp_w              = <double*>malloc(Nx * sizeof(double))
cdef double *Vp_n              = <double*>malloc(Nx * sizeof(double))
cdef double *Vrho_w_of_P_w     = <double*>malloc(Nx * sizeof(double))
cdef double *VKrw_of_P_c       = <double*>malloc(Nx * sizeof(double))
cdef double *VKrw_rho_w        = <double*>malloc(Nx * sizeof(double))

cdef double *Vrho_n_of_P_n     = <double*>malloc(Nx * sizeof(double))
cdef double *VKrn_of_P_c       = <double*>malloc(Nx * sizeof(double))
cdef double *VKrn_rho_n        = <double*>malloc(Nx * sizeof(double))

cdef double *VDrvt_Pc_by_Epsilon_w        = <double*>malloc(Nx * sizeof(double))



cdef double *VPw_bc            = <double*>malloc(Nx * sizeof(double))
cdef double *VEw_bc            = <double*>malloc(Nx * sizeof(double))


######## Arrays for visualization and storing results
Ve_w_Time_Results              = np.zeros([len(time_interval),Nx], dtype=np.double) 
Vp_w_Time_Results              = np.zeros([len(time_interval),Nx], dtype=np.double) 
Vp_n_Time_Results              = np.zeros([len(time_interval),Nx], dtype=np.double)

Vew_time  = np.zeros([MaxTimeSteps], dtype=np.double) 
Vpw_time  = np.zeros([MaxTimeSteps], dtype=np.double) 
Vpn_time  = np.zeros([MaxTimeSteps], dtype=np.double) 



Vew_normalized  = np.zeros([MaxTimeSteps], dtype=np.double) 
Vpw_normalized  = np.zeros([MaxTimeSteps], dtype=np.double) 
Vpn_normalized  = np.zeros([MaxTimeSteps], dtype=np.double) 

Vtime     = np.zeros([MaxTimeSteps], dtype=np.double) 


######## Arrays for time of simulations and CPU time
Clock_Array                    = np.zeros([len(time_interval)], dtype=float) 
ftime                          = np.zeros([len(time_interval)], dtype=float) 
#######################################
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
def  Drvt_Pc_by_Epsilon_w( Epsilon_w):
    Se = (Epsilon_w - Epsilon_wr) / Const_Epsilon
    if (Se > 0.0 and Se < 1.0):
       Drvt_P_c = -( ( Se**(-m_inv) - 1.0)**n_inv1 )*(1.0/(alpha_m_n*(Epsilon_w-Epsilon_wr)*(Se**m_inv)))
    else:
       Drvt_P_c = 0.0
    return Drvt_P_c
####################################### 
cdef double  Krw_of_Pc(double P_c):
    if (P_c > 0.0):
       Se =  (1.0 + ((alpha*P_c)**n))**(-m)
       Krw = (Se**0.5)  *(  (1.0 - (  (1.0 - Se**m_inv)**m  )   )**2.0)
    else:
       Krw = 1.0
    return Krw
####################################### 
cdef double  Krn_of_Pc(double  P_c):
    if (P_c > 0.0):
       Se =  (1.0 + (alpha*P_c)**n)**(-m)
       Krn = (  (1.0 - Se)**(0.5)  )*(   (1.0 - (Se**(m_inv)))**(m2)   )
    else:
       Krn = 0.0
    return Krn
####################################### 
cdef double  Epsilon_w_of_Pc(double  P_c):
    if (P_c > 0.0):
       Se =  (1.0 + (alpha*P_c)**n)**(-m)
       Epsilon_w =Const_Epsilon*  Se + Epsilon_wr
    else:
       Epsilon_w = epsilon-Epsilon_nr
    return Epsilon_w
####################################### 
cdef double  Epsilon_n_of_Pc(double Epsilon_w):
    Epsilon_n = epsilon - Epsilon_w
    return Epsilon_n
####################################### 
cdef double  rhow_of_Pw(double P_w):
    rho_w =rho_w0*math.exp(Beta_w*P_w)
    return rho_w
#######################################  
cdef double  Drvt_rhow_of_Pw(double P_w):
    Drvt_rho_w =Beta_w*rho_w0*math.exp(Beta_w*P_w)
    return Drvt_rho_w
#######################################
cdef double  rhon_of_Pn(double P_n):
    rho_n =b* P_n+rho_n0
    return rho_n
#######################################
cdef double  Drvt_rhon_of_Pn(double P_n):
    Drvt_rho_n =b
    return Drvt_rho_n   


########################################################
#########################BC and IC######################
########################################################
######################################################## 
for i in range(0,Nx):
    x = Omega_0+i*dx
    if(i==0):

      VEw_bc[i]        = 0.2746
      VPw_bc[i]        = -Pc(0.2746)
    elif(i==Nx-1):
      VEw_bc[i]        = 0.2746
      VPw_bc[i]        = -Pc(0.2746)
    else:
      VEw_bc[i]        = 0.3 
      VPw_bc[i]        = -Pc(0.3)
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
        Vp_c[i] = Pc(u[i][0])
        VDrvt_Pc_by_Epsilon_w[i]=Drvt_Pc_by_Epsilon_w(u[i][0] )
        Vp_w[i] = u[i][1]
        Vp_n[i] = Vp_c[i] + Vp_w[i]
        Vrho_w_of_P_w[i]=rhow_of_Pw(Vp_w[i])
        VKrw_of_P_c[i]=Krw_of_Pc(Vp_c[i])
        Vrho_n_of_P_n[i]=rhon_of_Pn(Vp_n[i])
        VKrn_of_P_c[i]=Krn_of_Pc(Vp_c[i])
        VKrw_rho_w[i]=Vrho_w_of_P_w[i]* VKrw_of_P_c[i]
        VKrn_rho_n[i]=Vrho_n_of_P_n[i]* VKrn_of_P_c[i]


    for i in range(1,Nx-1):
        CEpsilon_n_of_P_c=Epsilon_n_of_Pc(u[i][0])
        CDrvt_rho_n_of_P_n=Drvt_rhon_of_Pn(Vp_n[i])
        #########
        A_SP_11 = Vrho_w_of_P_w[i]
        A_SP_12 = u[i][0] *Drvt_rhow_of_Pw(Vp_w[i])
        A_SP_21 = CEpsilon_n_of_P_c*CDrvt_rho_n_of_P_n*VDrvt_Pc_by_Epsilon_w[i]-Vrho_n_of_P_n[i]
        A_SP_22 = CEpsilon_n_of_P_c*CDrvt_rho_n_of_P_n
        #########
        Kw_plus= k_muw_half*(VKrw_rho_w[i+1] + VKrw_rho_w[i])
        Kw_minus= k_muw_half*(VKrw_rho_w[i-1] + VKrw_rho_w[i])
        rho_w_plus = 0.5*(Vrho_w_of_P_w[i+1] + Vrho_w_of_P_w[i])
        rho_w_minus =0.5*(Vrho_w_of_P_w[i-1] + Vrho_w_of_P_w[i])
        Kn_plus = k_mun_half*(VKrn_rho_n[i+1] + VKrn_rho_n[i])
        Kn_minus = k_mun_half*(VKrn_rho_n[i-1] + VKrn_rho_n[i])
        rho_n_plus = 0.5*(Vrho_n_of_P_n[i+1] + Vrho_n_of_P_n[i])
        rho_n_minus =0.5*(Vrho_n_of_P_n[i-1] + Vrho_n_of_P_n[i])
        Od_w = Cx * (Kw_plus*(Cx*(Vp_w[i+1] - Vp_w[i]) + d*g*rho_w_plus) - Kw_minus * (Cx*(Vp_w[i] - Vp_w[i-1]) + d*g*rho_w_minus))
        Od_n = Cx * (Kn_plus*(Cx*(Vp_n[i+1] - Vp_n[i]) + d*g*rho_n_plus) - Kn_minus * (Cx*(Vp_n[i] - Vp_n[i-1]) + d*g*rho_n_minus))
        f[i][0] =  A_SP_11 * udot[i][0] + A_SP_12 * udot[i][1] - Od_w
        f[i][1] =  A_SP_21 * udot[i][0] + A_SP_22 * udot[i][1] - Od_n

    f[0][0]    =  u[0][0]     - VEw_bc[0]
    f[0][1]    =  u[0][1]     - VPw_bc[0]
    f[Nx-1][0]    =  u[Nx-1][0]     - VEw_bc[Nx-1]
    f[Nx-1][1]    =  u[Nx-1][1]     - VPw_bc[Nx-1]

 

#######################################
cdef int Initialization(ts,U):
# a private function that we created to initialize the solution vector at t=t0
    cdef unsigned int i
    #######
    da=ts.getDM()
    u=da.getVecArray(U)
    for i in range(0,Nx):
        u[i][0]        = VEw_bc[i]
        u[i][1]        = VPw_bc[i]

cdef store_results(ts):
    ndt=ts.getStepNumber()
    dt = ts.getTimeStep()
    
    Vtime[ndt]=ts.getTime()   #Vtime[ndt-1]+dt
    Solution=ts.getSolution()

    
    Vew_time[ndt] = Solution[node_i]
    Vpw_time[ndt] = Solution[node_i+1]
    Vpn_time[ndt] = Pc(Vew_time[ndt])+Vpw_time[ndt]

#######################################
cdef void main():
#######################################    
# here we created a 1D grid, to let the solver store the jacobian as a sparse matrix (this is one of the ways to do that) 
    comm = PETSc.COMM_WORLD 
    bz= PETSc.DMDA.BoundaryType.NONE
    da=PETSc.DMDA().create(dof= 2,sizes=(Nx, ),boundary_type=bz,stencil_type=None,stencil_width=1, comm=comm ) 
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
    ts.setPostStep(store_results)

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
           Ve_w_Time_Results[i][j]   = U[j][0]
           Vp_w_Time_Results[i][j]   = U[j][1]
           Vp_n_Time_Results[i][j]   = Pc(Ve_w_Time_Results[i][j]) + Vp_w_Time_Results[i][j] 
       NTimeSteps=ts.getStepNumber()
       if(NTimeSteps==MaxTimeSteps): 
         break
    ####################     
    #####  convergence of variables
    e_w_factor  =  1/( Vew_time[NTimeSteps] - Vew_time[1])
    p_w_factor  =  1/( Vpw_time[NTimeSteps] - Vpw_time[1])
    p_n_factor  =  1/( Vpn_time[NTimeSteps] - Vpn_time[1])

    Vew_normalized[1:NTimeSteps+1]   = e_w_factor*(Vew_time[1:NTimeSteps+1]-Vew_time[1])
    Vpw_normalized[1:NTimeSteps+1]   = p_w_factor*(Vpw_time[1:NTimeSteps+1]-Vpw_time[1])
    Vpn_normalized[1:NTimeSteps+1]   = p_n_factor*(Vpn_time[1:NTimeSteps+1]-Vpn_time[1])

    arrays = {
    'e_w_2p': Ve_w_Time_Results,
    'p_w_2p': Vp_w_Time_Results,
    'p_n_2p': Vp_n_Time_Results,
    'e_w_normalized_2p': Vew_normalized[1:NTimeSteps+1],
    'p_w_normalized_2p': Vpw_normalized[1:NTimeSteps+1],
    'p_n_normalized_2p': Vpn_normalized[1:NTimeSteps+1],
    'time_steps_array_2p': Vtime[1:NTimeSteps+1],
    'time_2p': time_interval }

    # Loop through and save each array
    for name, array in arrays.items():
        np.savetxt(f'results_2p/{name}_{Nx}.csv', array, delimiter=',', fmt='%s')



    #free memory
    free(Vp_c)
    free(Vp_w)
    free(Vrho_w_of_P_w )
    free(VKrw_of_P_c)
    free(VKrw_rho_w )
    free(VPw_bc  )
    free(Vp_n)
    free(Vrho_n_of_P_n)
    free(VKrn_of_P_c)
    free(VKrn_rho_n)
    free(VDrvt_Pc_by_Epsilon_w)
    free(VEw_bc)
main() 

