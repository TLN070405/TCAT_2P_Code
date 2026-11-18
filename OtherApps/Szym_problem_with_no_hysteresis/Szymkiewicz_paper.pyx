


#  THis is example 1 from Szym paper with modified boundary conditions

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
cdef unsigned int Nz, taunumber,Nznumber
cdef int d
cdef double t_out,t0,Omega_0,Omega_L,dz,Cz,phi,mu_w,mu_n,g,alpha,n,K,S0,SL,Epsilon_nr,Epsilon_wr,rho_w0,rho_n0,a,b
cdef double Const_Epsilon,k_mun_half,k_muw_half,m,n_inv,n_inv1,n1,m_inv,m1,m2,alpha_inv,alpha_m_n
cdef double atol,rtol
cdef double Ew_bc_0,Ew_bc_L,Pw_bc_0,Pw_bc_L,Pn_bc_0,Pn_bc_L,Ew_IC,Pw_IC,Pn_IC


############
Nz= 401
Omega_0= 0.0
Omega_L= 0.8
############
#time_interval=np.array([50,100,150,200,250,300,350,400,450,500,550,600,650,700,750,800,850,900,950,1000,3000])
time_interval=np.array([10,100,1000,3000,5000])
t0=0.0
############
rho_w0 =1000
rho_n0 =1.204
mu_w=9.77e-4
mu_n=1.82e-5
g=9.81
a=4.9e-10
b=1.189e-5
############
phi= 0.3
Epsilon_nr= 0.0
Epsilon_wr= 0.0
############

Pd=1000
lamda=2.0
K=(9.81e-4*mu_w)/(rho_w0*g)

############
d=0
############ boundary and initial conditions of the problem
Ew_bc_0=0.27
Pw_bc_0=-0.107*(rho_w0*g)
#Ew_bc_L= we have Neumman BC  qn=0
#Pw_bc_L= we have Neumman BC  qw=0
Ew_IC= 0.003
Pw_IC= -1.019*(rho_w0*g)
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
rtol=1e-8
atol=1e-8
MaxStepRejections=500
MaxSNESFailures=100000
MaxTimeSteps=100000

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
cdef double *VPc_tn1 = <double*>malloc(Nz * sizeof(double))
cdef double *VSw_tn1 = <double*>malloc(Nz * sizeof(double))
cdef double *VDrvt_Pc_by_Epsilon_w = <double*>malloc(Nz * sizeof(double))


Clock_Array=np.empty([len(time_interval)], dtype=float)
ftime=np.empty([len(time_interval)], dtype=float)
#######################################
cdef  double  Pc(double Epsilon_w):
    cdef double Se,P_c
    Se = (Epsilon_w - Epsilon_wr) / Const_Epsilon
    if (Se > 0.0 and Se <= 1.0):
       P_c = Pd*((Se)**(-1.0/lamda))
    else:
       P_c = Pd
    return P_c
#######################################
cdef  double  Drvt_Pc_by_Epsilon_w(double Epsilon_w):
    cdef double Se,Drvt_P_c
    Se = (Epsilon_w - Epsilon_wr) / Const_Epsilon
    if (Se > 0.0 and Se < 1.0):
       Drvt_P_c =  (-Pd/(lamda*(phi-Epsilon_wr)))*(Se**((-1/lamda)-1))
    else:
       Drvt_P_c = 0.0
    return Drvt_P_c
#######################################
cdef  double  Epsilon_w_of_Pc(double Pc):
    cdef double Se,Epsilon_w
    if (Pc > Pd):
       Se = (Pc/Pd)**(-lamda)
    else:
       Se = 1.0
    Epsilon_w=Const_Epsilon*Se+Epsilon_wr
    return Epsilon_w
#######################################
cdef  double  Krw_of_Pc(double P_c):
    cdef double Se,Krw
    if (P_c > Pd):
       Se =   (P_c/Pd)**(-lamda)
       Krw = Se**(3+(2/lamda))
    else:
       Krw = 1.0
    return Krw
#######################################
cdef  double  Krn_of_Pc(double  P_c):
    cdef double Se,Krn
    if (P_c > Pd):
       Se =   (P_c/Pd)**(-lamda)
       Krn = ((1.0-Se)**2)*(1-(Se**(1.0+(2/lamda))))
    else:
       Krn = 0.0
    return Krn
#######################################
cdef  double  Epsilon_n_of_Pc(double Epsilon_w):
    cdef double Epsilon_n
    Epsilon_n = phi - Epsilon_w
    return Epsilon_n
#######################################
cdef  double  rhow_of_Pw(double P_w):
    cdef double rho_w
    rho_w =rho_w0*math.exp(a*P_w)
    return rho_w
#######################################
cdef  double  Drvt_rhow_of_Pw(double P_w):
    cdef double Drvt_rho_w
    Drvt_rho_w =a*rho_w0*math.exp(a*P_w)
    return Drvt_rho_w
#######################################
cdef  double  rhon_of_Pn(double P_n):
    cdef double rho_n
    rho_n =b* P_n+rho_n0
    return rho_n
#######################################
cdef  double Drvt_rhon_of_Pn(double P_n):
    cdef double Drvt_rho_n
    Drvt_rho_n =b
    return Drvt_rho_n

#######################################
cdef double func_P_w0_bc(double t):
    cdef double aa,bb
    if(t>=0 and t<=200):
      aa=(-0.104*(rho_w0*g)+0.107*(rho_w0*g))/(200-0)
      bb=-0.107*(rho_w0*g)-(aa*0)
      P_w0 = aa*t+bb
    elif(t>200 and t<=400):
      P_w0 = -0.104*(rho_w0*g)
    elif(t>400 and t<=600):
      aa=(-0.106*(rho_w0*g)+0.104*(rho_w0*g))/(600-400)
      bb=-0.104*(rho_w0*g)-(aa*400)
      P_w0 = aa*t+bb
    elif(t>600 and t<=800):
      P_w0=-0.106*(rho_w0*g)
    elif(t>800 and t<=1000):
      aa=(-0.105*(rho_w0*g)+0.106*(rho_w0*g))/(1000-800)
      bb=-0.106*(rho_w0*g)-(aa*800)
      P_w0 = aa*t+bb
    return P_w0
#######################################
cdef int Residual(ts,t,U,Udot,F):
# residual function is like sundials residual, the prototype is fixed but the name of the function can be changed
#u[i][0]=Epsilon_w[i]
#u[i][1]=Psi_w[i]

    cdef unsigned int i
    cdef double CEpsilon_n_of_P_c,CDrvt_rho_n_of_P_n
    cdef double A_SP_11,A_SP_12,A_SP_21,A_SP_22
    cdef double Kw_plus,Kw_minus,rho_w_plus,rho_w_minus,Kn_plus,Kn_minus,rho_n_plus,rho_n_minus,Od_w,Od_n

    da=ts.getDM()
    #######
    localU=da.getLocalVec()
    da.globalToLocal(U,localU)
    #######
    u=da.getVecArray(localU)
    udot=da.getVecArray(Udot)
    f=da.getVecArray(F)
    Ndt=ts.getStepNumber()
    #######
    for i in range(0,Nz):
          Vp_c[i]=Pc(u[i][0])
          VDrvt_Pc_by_Epsilon_w[i]=Drvt_Pc_by_Epsilon_w(u[i][0])
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
          Od_w = Cz * (Kw_plus*(Cz*(Vp_w[i+1] - Vp_w[i]) + d*g*rho_w_plus) - Kw_minus * (Cz*(Vp_w[i] - Vp_w[i-1]) + d*g*rho_w_minus))
          Od_n = Cz * (Kn_plus*(Cz*(Vp_n[i+1] - Vp_n[i]) + d*g*rho_n_plus) - Kn_minus * (Cz*(Vp_n[i] - Vp_n[i-1]) + d*g*rho_n_minus))
          f[i][0] =  A_SP_11 * udot[i][0] + A_SP_12 * udot[i][1] - Od_w
          f[i][1] =  A_SP_21 * udot[i][0] + A_SP_22 * udot[i][1] - Od_n
    f[0][0] = u[0][0] - Epsilon_w_of_Pc(-func_P_w0_bc(t))
    f[0][1] = u[0][1] - func_P_w0_bc(t)
    f[Nz-1][0] = (K/mu_w)*VKrw_of_P_c[Nz-1]*(Cz*(Vp_w[Nz-1]-Vp_w[Nz-2])+d*g*Vrho_w_of_P_w[Nz-1])
    f[Nz-1][1] = (K/mu_n)*VKrn_of_P_c[Nz-1]*(Cz*(Vp_n[Nz-1]-Vp_n[Nz-2])+d*g*Vrho_n_of_P_n[Nz-1])

#######################################

def FormFunction(snes,U,F):

    cdef unsigned int i
    da=snes.getDM()
    localU=da.getLocalVec()
    da.globalToLocal(U,localU)
    #######
    u=da.getVecArray(localU)
    f=da.getVecArray(F)
    for i in range(0,Nz):
        Vp_c[i] = Pc(u[i][0])
        Vp_w[i] = u[i][1]
        Vp_n[i] = Vp_c[i] + Vp_w[i]
        Vrho_w_of_P_w[i]=rhow_of_Pw(Vp_w[i])
        VKrw_of_P_c[i]=Krw_of_Pc(Vp_c[i])
        Vrho_n_of_P_n[i]=rhon_of_Pn(Vp_n[i])
        VKrn_of_P_c[i]=Krn_of_Pc(Vp_c[i])
        VKrw_rho_w[i]=Vrho_w_of_P_w[i]* VKrw_of_P_c[i]
        VKrn_rho_n[i]=Vrho_n_of_P_n[i]* VKrn_of_P_c[i]

    f[0][0] = u[0][0] - Ew_bc_0
    f[0][1] = u[0][1] - Pw_bc_0
    f[Nz-1][0] = (K/mu_w)*VKrw_of_P_c[Nz-1]*(Cz*(Vp_w[Nz-1]-Vp_w[Nz-2])+d*g*Vrho_w_of_P_w[Nz-1])
    f[Nz-1][1] = (K/mu_n)*VKrn_of_P_c[Nz-1]*(Cz*(Vp_n[Nz-1]-Vp_n[Nz-2])+d*g*Vrho_n_of_P_n[Nz-1])
    for i in range(1,Nz-1):
        f[i][1]= u[i][1] + 1.019
        f[i][0]= u[i][0] - 0.003

#######################################
def Initialization():
# a private function that we created to initialize the solution vector at t=t0
    cdef unsigned int i
    comm = PETSc.COMM_WORLD
    bz= PETSc.DMDA.BoundaryType.NONE
    da1=PETSc.DMDA().create(dof= 2,sizes=(Nz, ),boundary_type=bz,stencil_type=None,stencil_width=1, comm=comm )
    da1.setUp()
    x= da1.createGlobalVec()                     # create the solution vector related to the grid created
    r=x.duplicate()
    snes=PETSc.SNES().create(PETSc.COMM_WORLD)
    snes.setFunction(FormFunction,r)
    snes.setDM(da1)
    snes.setFromOptions()
    u=da1.getVecArray(x)

    for i in range(1,Nz-1):
        u[i][1] = -1.019
        u[i][0] = 0.003
    u[0][0] = Ew_bc_0
    u[0][1] = Pw_bc_0
    u[Nz-1][0] = u[Nz-2][0]
    u[Nz-1][1] = u[Nz-2][1]

    snes.solve(None,x)
    return x,da1


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
    ts.setFromOptions()                         # give the user the possibility to personalize the ts solver at running time
#######################################
    x,da1=Initialization()                          # initialization of solution vector (initial conditions)
    U=da.getVecArray(u)
    X=da1.getVecArray(x)
    for i in range(0,Nz):
        U[i][0] = X[i][0]
        U[i][1] = X[i][1]
#############Calling the solver and start solving the DAE system ######################
    ThetaW=PETSc.Vec().createSeq(Nz)
    ThetaW_L1=np.empty([len(time_interval)], dtype=np.double)
    ThetaW_L2=np.empty([len(time_interval)], dtype=np.double)
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
       ThetaW_L1[i]=U[Nz-5][0]
       ThetaW_L2[i]=U[Nz-10][0]
       for k in range(0,Nz):
          ThetaW[k]=U[k][0]
       plt.plot(mesh, ThetaW, label='2P-TM, t=%d sec' % round(ftime[i]))
       plt.legend()
       NTimeSteps=ts.getStepNumber()
       if(NTimeSteps==MaxTimeSteps):
         break
    ######

    plt.xlabel(r'$x\,[m]$', fontsize=30)
    plt.ylabel(r'$\epsilon^{\overline{\overline{w}}}\,[-]$', fontsize=30)
    plt.legend(fontsize=24)
    plt.xticks(fontsize=16)  # Change 16 to your desired font size
    plt.yticks(fontsize=16)

    #plt.title("Solution 2Phase-SP 1D Szymkiewicz Paper ")
    plt.show()

    plt.plot(time_interval,ThetaW_L1)
    plt.xlabel("time(s)")
    plt.ylabel("Epsilon_w at Node Nz-5 ")
    #plt.title("Saturation in function of time at Node Nz-5")
    plt.show()

    plt.plot(time_interval,ThetaW_L2)
    plt.xlabel("time(s)")
    plt.ylabel("Epsilon_w at Node Nz-10")
    #plt.title("Saturation in function of time at Node Nz-10")
    plt.show()
    ## print results about solver on screen
    ts.view()


main()
