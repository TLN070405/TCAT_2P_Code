# the parameters are for coarse sand

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


OptDB = PETSc.Options()
###############
cdef unsigned int MaxStepRejections,MaxSNESFailures,MaxTimeSteps
cdef unsigned int Nz
cdef int d
cdef double t_out,t0,Omega_0,Omega_L,dz,Cz,phi,mu_w,mu_n,g,alpha,n,K,S0,SL,Epsilon_nr,Epsilon_wr,rho_w0,rho_n0,a,b
cdef double Const_Epsilon,k_mun_half,k_muw_half,m,n_inv,n_inv1,n1,m_inv,m1,m2,alpha_inv,alpha_m_n
cdef double atol,rtol
cdef double Ew_bc_0,Ew_bc_L,Pw_bc_0,Pw_bc_L,Pn_bc_0,Pn_bc_L,Ew_IC,Pw_IC,Pn_IC
############
Counter=1
Node_hyst=12
############
time_interval=np.array([1000])
t0=0.0
############
Nz=101
Omega_0= 0.0
Omega_L= 1.0

############
phi  = 0.4
mu_w = 1e-3
mu_n = 1.78e-5
############
g    = 9.8066
d=0
####
alpha_dr=5e-4 
n_dr=  2.494
####
K= 1e-10
############
Epsilon_nr= 0.0
Epsilon_wr= 0.0
rho_w0 =1e3
rho_n0 = 1.2
a= 4.9e-10
b= 1.189e-5
####
Const_Epsilon = phi-Epsilon_nr - Epsilon_wr
k_mun_half=0.5*(K/mu_n)
k_muw_half=0.5*(K/mu_w)
####

############
rtol=1e-8
atol=1e-8
htol=1e-8
MaxStepRejections=500
MaxSNESFailures=5000
MaxTimeSteps=10000

Ew_bc_L = 0.35 
Pw_bc_L =  -3000
Ew_IC   = 0.35                     
Pw_IC   =  -3000 #-346.7309109719586  

############
dz =(Omega_L - Omega_0) / (Nz - 1.0)
Cz =  1.0 / dz

###
m_dr = 1.0 - (1.0 / n_dr)
n_inv_dr=1.0/n_dr
n_inv1_dr=n_inv_dr-1.0
n1_dr=n_dr-1.0
m_inv_dr=1.0/m_dr
m1_dr=-m_dr-1.0
m2_dr=2.0*m_dr
alpha_inv_dr=1.0/alpha_dr
alpha_m_n_dr=alpha_dr*m_dr*n_dr




######################dynamic arrays #################
cdef double *Vp_c = <double*>malloc(Nz * sizeof(double))
cdef double *Vp_w = <double*>malloc(Nz * sizeof(double))
cdef double *Vp_n = <double*>malloc(Nz * sizeof(double))
cdef double *Vrho_w_of_P_w = <double*>malloc(Nz * sizeof(double))
cdef double *Vrho_n_of_P_n = <double*>malloc(Nz * sizeof(double))
cdef double *VKrw_of_P_c = <double*>malloc(Nz * sizeof(double))
cdef double *VKrn_of_P_c = <double*>malloc(Nz * sizeof(double))
cdef double *VKrw_rho_w = <double*>malloc(Nz * sizeof(double))
cdef double *VKrn_rho_n = <double*>malloc(Nz * sizeof(double))
cdef double *VDrvt_Pc_by_Epsilon_w = <double*>malloc(Nz * sizeof(double))
Clock_Array=np.zeros([len(time_interval)], dtype=float) 
ftime=np.zeros([len(time_interval)], dtype=float) 
######################dynamic arrays related to hysteresis model #################



  
#######################################
#######################################
cdef double  Pc_dr(double Epsilon_w):
    Se = (Epsilon_w - Epsilon_wr) / Const_Epsilon
    if (Se > 0.0 and Se < 1.0):
       P_c = (  ( Se**(-m_inv_dr)  - 1.0)**n_inv_dr )*alpha_inv_dr
    else:
       P_c = 0.0
    return P_c

#######################################
cdef double  Drvt_Pc_by_Epsilon_w_dr(double Epsilon_w):
    Se = (Epsilon_w - Epsilon_wr) / Const_Epsilon
    if (Se > 0.0 and Se < 1.0):
       Drvt_P_c = -( ( Se**(-m_inv_dr) - 1.0)**n_inv1_dr )*(1.0/(alpha_m_n_dr*(Epsilon_w-Epsilon_wr)*(Se**m_inv_dr)))
    else:
       Drvt_P_c = 0.0
    return Drvt_P_c
####################################### 
cdef double  Epsilon_w_of_Pc_dr(double  P_c):
    if (P_c > 0.0):
       Se =  (1.0 + (alpha_dr*P_c)**n_dr)**(-m_dr)
       Epsilon_w =Const_Epsilon*  Se + Epsilon_wr
    else:
       Epsilon_w = phi-Epsilon_nr
    return Epsilon_w
    
    
####################################### 
cdef double  Krw_of_Pc(double P_c):
    if (P_c > 0.0):
       Se =  (1.0 + ((alpha_dr*P_c)**n_dr))**(-m_dr)
       Krw = (Se**0.5)  *(  (1.0 - (  (1.0 - Se**m_inv_dr)**m_dr  )   )**2.0)
    else:
       Krw = 1.0
    return Krw
####################################### 
cdef double  Krn_of_Pc(double  P_c):
    if (P_c > 0.0):
       Se =  (1.0 + (alpha_dr*P_c)**n_dr)**(-m_dr)
       Krn = (  (1.0 - Se)**(0.5)  )*(   (1.0 - (Se**(m_inv_dr)))**(m2_dr)   )
    else:
       Krn = 0.0
    return Krn
####################################### 
cdef double  Epsilon_n_of_Pc(double Epsilon_w):
    Epsilon_n = phi - Epsilon_w
    return Epsilon_n
####################################### 
cdef double  rhow_of_Pw(double P_w):
    rho_w =rho_w0*math.exp(a*P_w)
    return rho_w
#######################################  
cdef double  Drvt_rhow_of_Pw(double P_w):
    Drvt_rho_w =a*rho_w0*math.exp(a*P_w)
    return Drvt_rho_w
#######################################
cdef double  rhon_of_Pn(double P_n):
    rho_n =b* P_n+rho_n0
    return rho_n
#######################################
cdef double  Drvt_rhon_of_Pn(double P_n):
    Drvt_rho_n =b
    return Drvt_rho_n   


#######################################
cdef double func_P_w0_bc(double t): 
    VPw =[-346.7309109719586,-5000,-2000,-3000,-100]
    Vtime=[0, 500,800,1000,2000]

    i=1
    if Vtime[i-1]<=t<=Vtime[i]:
      aa=(VPw[i]-VPw[i-1])/(Vtime[i]-Vtime[i-1])
      bb=VPw[i-1]-aa*Vtime[i-1]
      P_w0=aa*t+bb
    i=2
    if Vtime[i-1]<t<=Vtime[i]:
      aa=(VPw[i]-VPw[i-1])/(Vtime[i]-Vtime[i-1])
      bb=VPw[i-1]-aa*Vtime[i-1]
      P_w0=aa*t+bb
    i=3
    if Vtime[i-1]<t<=Vtime[i]:  
      aa=(VPw[i]-VPw[i-1])/(Vtime[i]-Vtime[i-1])
      bb=VPw[i-1]-aa*Vtime[i-1]
      P_w0=aa*t+bb
    i=4 
    if Vtime[i-1]<t<=Vtime[i]: 
      aa=(VPw[i]-VPw[i-1])/(Vtime[i]-Vtime[i-1])
      bb=VPw[i-1]-aa*Vtime[i-1]
      P_w0=aa*t+bb  
    return P_w0

#######################################
cdef int Residual(ts,t,U,Udot,F):
# residual function is like sundials residual, the prototype is fixed but the name of the function can be changed
#u[i][0]=Epsilon_w[i]
#u[i][1]=Psi_w[i]
 
    cdef unsigned int i
    cdef double CEpsilon_n_of_Pc,CDrvt_rho_n_of_Pn
    cdef double A_SP_11,A_SP_12,A_SP_21,A_SP_22
    cdef double Kw_plus,Kw_minus,rho_w_plus,rho_w_minus,Kn_plus,Kn_minus,rho_n_plus,rho_n_minus,Od_w,Od_n
    global Counter,process_old, process_new
    da=ts.getDM()
    #######
    localU=da.getLocalVec()
    da.globalToLocal(U,localU)
    #######
    u=da.getVecArray(localU)
    udot=da.getVecArray(Udot)
    f=da.getVecArray(F)
    Ndt=ts.getStepNumber()

    for i in range(0,Nz):
        Vp_c[i] =  Pc_dr(u[i][0])   
        VDrvt_Pc_by_Epsilon_w[i]= Drvt_Pc_by_Epsilon_w_dr(u[i][0])
        Vp_w[i] = u[i][1]
        Vp_n[i] = Vp_c[i] + Vp_w[i]
        Vrho_w_of_P_w[i]=rhow_of_Pw(Vp_w[i])
        VKrw_of_P_c[i]=Krw_of_Pc(Vp_c[i])
        Vrho_n_of_P_n[i]=rhon_of_Pn(Vp_n[i])
        VKrn_of_P_c[i]=Krn_of_Pc(Vp_c[i])
        VKrw_rho_w[i]=Vrho_w_of_P_w[i]* VKrw_of_P_c[i]
        VKrn_rho_n[i]=Vrho_n_of_P_n[i]* VKrn_of_P_c[i]
          
    for i in range(1,Nz-1):
        #########
        CEpsilon_n_of_P_c  = Epsilon_n_of_Pc(u[i][0])
        CDrvt_rho_n_of_P_n = Drvt_rhon_of_Pn(Vp_n[i])
        #########
        A_SP_11 = Vrho_w_of_P_w[i]
        A_SP_12 = u[i][0] *Drvt_rhow_of_Pw(Vp_w[i])
        A_SP_21 = CEpsilon_n_of_P_c*CDrvt_rho_n_of_P_n*VDrvt_Pc_by_Epsilon_w[i]-Vrho_n_of_P_n[i]
        A_SP_22 = CEpsilon_n_of_P_c*CDrvt_rho_n_of_P_n
        #########
        Kw_plus     = k_muw_half*(VKrw_rho_w[i+1] + VKrw_rho_w[i])
        Kw_minus    = k_muw_half*(VKrw_rho_w[i-1] + VKrw_rho_w[i])
        rho_w_plus  = 0.5 * (Vrho_w_of_P_w[i+1]   + Vrho_w_of_P_w[i])
        rho_w_minus = 0.5 * (Vrho_w_of_P_w[i-1]   + Vrho_w_of_P_w[i])
        Kn_plus     = k_mun_half*(VKrn_rho_n[i+1] + VKrn_rho_n[i])
        Kn_minus    = k_mun_half*(VKrn_rho_n[i-1] + VKrn_rho_n[i])
        rho_n_plus  = 0.5 * (Vrho_n_of_P_n[i+1]   + Vrho_n_of_P_n[i])
        rho_n_minus = 0.5 * (Vrho_n_of_P_n[i-1]   + Vrho_n_of_P_n[i])
        Od_w = Cz * (Kw_plus*(Cz*(Vp_w[i+1] - Vp_w[i]) + d*g*rho_w_plus) - Kw_minus * (Cz*(Vp_w[i] - Vp_w[i-1]) + d*g*rho_w_minus))
        Od_n = Cz * (Kn_plus*(Cz*(Vp_n[i+1] - Vp_n[i]) + d*g*rho_n_plus) - Kn_minus * (Cz*(Vp_n[i] - Vp_n[i-1]) + d*g*rho_n_minus))
        f[i][0] =  A_SP_11 * udot[i][0] + A_SP_12 * udot[i][1] - Od_w
        f[i][1] =  A_SP_21 * udot[i][0] + A_SP_22 * udot[i][1] - Od_n
    #########
    f[0][0] = u[0][0] - Epsilon_w_of_Pc_dr(-func_P_w0_bc(t))
    f[0][1] = u[0][1] - func_P_w0_bc(t)
    f[Nz-1][0] = u[Nz-1][0] - Ew_bc_L
    f[Nz-1][1] = u[Nz-1][1] - Pw_bc_L

 
        
#######################################
def Initialization(ts,U):
# a private function that we created to initialize the solution vector at t=t0
    cdef unsigned int i
    da=ts.getDM()
    u=da.getVecArray(U)
    #####
    u[0][0] = Epsilon_w_of_Pc_dr(-func_P_w0_bc(0))
    u[0][1] = func_P_w0_bc(0) 
    u[Nz-1][0] = Ew_bc_L
    u[Nz-1][1] = Pw_bc_L 
    for i in range(1,Nz-1):
        u[i][0] = Ew_IC
        u[i][1] = Pw_IC

#######################################
cdef void main():
#######################################    
# here we created a 1D grid, to let the solver store the jacobian as a sparse matrix (this is one of the ways to do that) 
    cdef unsigned int k
    comm = PETSc.COMM_WORLD 
    bz= PETSc.DMDA.BoundaryType.NONE
    da=PETSc.DMDA().create(dof= 2,sizes=(Nz, ),boundary_type=bz,stencil_type=None,stencil_width=1, comm=comm ) 
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
    ts.setMaxStepRejections(MaxStepRejections)  # maximum time steps rejections allowed ,       
    ts.setMaxSNESFailures(MaxSNESFailures)
    ts.setTolerances(rtol, atol)                # set tolerances of the the chosen ts (here bdf )
    ts.setTime(t0)                              # set initial time t0 
    ts.setExactFinalTime(ts.ExactFinalTime.MATCHSTEP)  # this option is to match the output time of ts with the output time defined by user.
    #ts.setTimeStep(First_Time_Step)
    #ts.setPostStep(Hysteresis_module)
    ts.setFromOptions()                         # give the user the possibility to personalize the ts solver at running time
#######################################
    Initialization(ts,u)
#############Calling the solver and start solving the DAE system ###################### 
    ThetaW=PETSc.Vec().createSeq(Nz) 
    Ew_Node_hyst=np.zeros([len(time_interval)], dtype=np.double) 
    mesh = np.linspace(Omega_0, Omega_L, Nz)
    for i in range(len(time_interval)):
       ts.setMaxTime(time_interval[i])
       CPU_start = time.process_time()
       ts.solve(u)
       CPU_end = time.process_time()
       ftime[i]=ts.getTime()
       Clock_Array[i]=CPU_end-CPU_start
       #####visualization
       U=da.getVecArray(u)
       for k in range(0,Nz):
          ThetaW[k]=U[k][0]
       plt.plot(mesh,ThetaW,label='t(sec)=%5.4f'%ftime[i])
       plt.legend()
       NTimeSteps=ts.getStepNumber()
       if(NTimeSteps==MaxTimeSteps): 
         break
    ######  
    plt.xlabel("Position (m)")
    plt.ylabel("e_w") 
    plt.title("Solution 2Phase-SP 1D Maalgorzata Paper ")
    plt.show()
    ######

    # Hysteresis visualization
    e_w_range    = np.linspace(0.1, phi, 1000)
    p_c_mpc_dr   = np.zeros([len(e_w_range)], dtype=np.double)
    p_c_mpc_imb  = np.zeros([len(e_w_range)], dtype=np.double)

    for n, value in enumerate(e_w_range):
        p_c_mpc_dr[n]   = Pc_dr(value)

    plt.plot(e_w_range,p_c_mpc_dr,label="MDC VG",linestyle="dashed")
    plt.xlabel('e_w')
    plt.ylabel('p_c')
    plt.title('Hysteresis pc-ew curves')
    plt.legend()
    plt.show()


    
main() 

