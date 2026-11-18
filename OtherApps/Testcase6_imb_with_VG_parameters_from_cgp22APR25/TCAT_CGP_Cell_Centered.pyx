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

############### Variables declaration
cdef unsigned int MaxStepRejections,MaxSNESFailures,MaxTimeSteps,Nx
cdef int d
cdef double  Two_by3, pi 
cdef double  t0, Omega_0, Omega_L, dx, Cx, Two_Cx, Cx2, Two_by3_Cx2
cdef double  epsilon, g, epsilon6
cdef double  Rww, Gamma_wn, k_wn1_hat,Half_Inv_Rww, Prod_Gamma_K_wn1
cdef double  rho_w0, Beta_w
cdef double  b0, b1, b2, e_ns_0, e_ws_0, kn_ns_0, js_ns_0, b2_4pi
cdef double  rtol, atol

##################################################################
##################################################################
##################################################################
#########################Constants################################
##################################################################
##################################################################
##################################################################
Two_by3=2/3
pi=3.141592653589793
############0
time_interval=np.array([1e-5,1e-2,10,20,30,40,50,60,80,90,100,118])
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
epsilon=0.369
g=9.81
d=0 # "d" is an integer to switch between horizontal and vertical domains (accounting or not for gravity)


#####TCAT parameters#######
Rww       = 3.75e5 
Gamma_wn  = 7.20e-2
C_wn      = 6.398675e-05
k_wn1_hat = 1.003597e-10 

#####fluid parameters#######
rho_w0 = 1000
Beta_w= 4.5e-10

##### Linear geometric state equation parameters
b0=1.47e-1
b1=-2.08e-3
b2=(5.33e-2)/(4*pi)

#####
epsilon6=epsilon*6
Half_Inv_Rww=1/(2*Rww)
Prod_Gamma_K_wn1=Gamma_wn*k_wn1_hat
#########PETSc Solvers parameters###
rtol=1e-8  
atol=1e-8
MaxStepRejections=20000
MaxSNESFailures=20000
MaxTimeSteps=20000

 
node_i=20
node_i2=100
##################################################################
##################################################################
##################################################################
######################dynamic arrays #############################
##################################################################
##################################################################
################# Solution variables arrays#######################
cdef double *Ve_w     = <double*>malloc(Nx * sizeof(double))
cdef double *Vp_w     = <double*>malloc(Nx * sizeof(double))
cdef double *Ve_wn    = <double*>malloc(Nx * sizeof(double))
cdef double *Vj_wn    = <double*>malloc(Nx * sizeof(double))
cdef double *Vw_xwn   = <double*>malloc(Nx * sizeof(double))
cdef double *Vkn_wn   = <double*>malloc(Nx * sizeof(double))
################# Constants arrays################################
cdef double *Ve_wn_eq  = <double*>malloc(Nx * sizeof(double))
################ Arrays for intermediate calculations #############
cdef double *Vrho_w               = <double*>malloc(Nx * sizeof(double))
cdef double *VKrw_rho_w           = <double*>malloc(Nx * sizeof(double))
cdef double *VDif_ewn_eq_ewn      = <double*>malloc(Nx * sizeof(double))
cdef double *VOd_pc               = <double*>malloc(Nx * sizeof(double))
################################################################
 
cdef double *VEw_bc       = <double*>malloc(Nx * sizeof(double))
cdef double *VPw_bc       = <double*>malloc(Nx * sizeof(double))
cdef double *VEwn_bc       = <double*>malloc(Nx * sizeof(double))
cdef double *VJwn_bc       = <double*>malloc(Nx * sizeof(double))
cdef double *VWxwn_bc       = <double*>malloc(Nx * sizeof(double))
cdef double *VKwn_bc       = <double*>malloc(Nx * sizeof(double))
####################(Arrays to store results)#######################
Ve_w_Time_Results=np.zeros([len(time_interval),Nx], dtype=np.double) 
Vp_w_Time_Results=np.zeros([len(time_interval),Nx], dtype=np.double) 
Ve_wn_Time_Results=np.zeros([len(time_interval),Nx], dtype=np.double) 
Vj_wn_Time_Results=np.zeros([len(time_interval),Nx], dtype=np.double) 
Vw_xwn_Time_Results=np.zeros([len(time_interval),Nx], dtype=np.double) 
Vkn_wn_Time_Results=np.zeros([len(time_interval),Nx], dtype=np.double) 



Vew_time  = np.zeros([MaxTimeSteps], dtype=np.double) 
Vjwn_time = np.zeros([MaxTimeSteps], dtype=np.double) 
Vew_time2  = np.zeros([MaxTimeSteps], dtype=np.double) 
Vjwn_time2 = np.zeros([MaxTimeSteps], dtype=np.double) 

Vpw_time  = np.zeros([MaxTimeSteps], dtype=np.double) 
Vewn_time  = np.zeros([MaxTimeSteps], dtype=np.double) 
Vkwn_time  = np.zeros([MaxTimeSteps], dtype=np.double) 
Vtime     = np.zeros([MaxTimeSteps], dtype=np.double) 



##############(Arrays to store time of simulation)#################
Clock_Array=np.empty([len(time_interval)], dtype=float) 
ftime=np.empty([len(time_interval)], dtype=float) 

########################################################
########################################################
####################(state equations)###################
########################################################
########################################################

#######################################
cdef  double  rhow_of_Pw(double P_w):
    cdef  double rho_w
    rho_w =rho_w0*math.exp(Beta_w*P_w)
    return rho_w
#######################################  
cdef  double  Drvt_rhow_of_Pw(double P_w):
    cdef  double Drvt_rho_w
    Drvt_rho_w =Beta_w*rho_w0*math.exp(Beta_w*P_w)
    return Drvt_rho_w
#######################################
cdef  Linear_Geometric_Constants( double ew):
    cdef  double Swi,Sni,e_ns,e_ws,js_ns,kn_ns,D_sauter,b0D1,b1D2,b2D3
    e_ss             = 1.89e3 
    js_ns            = 2.00e3
    kn_ns            = 1.00e6

    Swi              = ew/epsilon
    e_ns             = (1-Swi)*e_ss
    e_ws             = Swi*e_ss
    D_sauter         = epsilon6/(e_ns+e_ws)
    b0D1             = b0*D_sauter
    b1D2             = b1*D_sauter*D_sauter
    b2D3             = b2*D_sauter*D_sauter*D_sauter
    return e_ns,e_ws,js_ns,kn_ns,b0D1,b1D2,b2D3
#######################################
cdef  double  Kwn_Linear_Geometric(double ew, double e_wn,double j_wn, double b0D1,double b1D2,double b2D3, double e_ns, double js_ns,double kn_ns):
    cdef  double  c0, c1,c2,c3,c4, kwn
    c0=epsilon-ew
    c1=b0D1*(e_wn+e_ns)
    c2=b1D2*(-e_wn*j_wn+e_ns*js_ns)
    c3=b2D3*e_ns*kn_ns
    c4=b2D3*e_wn
    kwn=(c0-c1-c2-c3)/c4
    return kwn 
#######################################
cdef  double  Wx_wn_IC_func(double dew_dx,double Ewn_eq, double Pw, double Ewn, double Jwn):
    cdef  double Wx_wn  
    Wx_wn=-(dew_dx/Ewn**2)*(   C_wn*(Pw+Gamma_wn*Jwn)  -(Prod_Gamma_K_wn1/(Pw))*(Ewn_eq-Ewn)   )
    return Wx_wn 



########################################################
#########################BC and IC######################
########################################################
######################################################## 
for i in range(0,Nx):
    x = Omega_0+i*dx
    if(i==0):
      VEw_bc[i]        = 0.369
      VPw_bc[i]        = 1e-7
      VEwn_bc[i]       = 1e-6
      VJwn_bc[i]       = -VPw_bc[i]/Gamma_wn
      Ve_wn_eq[i]      = VEwn_bc[i]
    elif(i==Nx-1):
      VEw_bc[i]        = VEw_bc[0]
      VPw_bc[i]        = VPw_bc[0]
      VEwn_bc[i]       = VEwn_bc[0]
      VJwn_bc[i]       = VJwn_bc[0]
      Ve_wn_eq[i]      = Ve_wn_eq[0]
    else:
      VEw_bc[i]        = 0.25
      VPw_bc[i]        = -518.5       # best so far :-518.5
      VEwn_bc[i]       =  857.7         # best so far : 857.7
      VJwn_bc[i]       = -VPw_bc[i]/Gamma_wn
      Ve_wn_eq[i]      = VEwn_bc[0]
 
#########

for i in range(0,Nx):
    if(i==0):
      dew_dx           =Cx*(VEw_bc[i+1]-VEw_bc[i])
      e_ns,e_ws,js_ns,kn_ns,b0D1,b1D2,b2D3= Linear_Geometric_Constants(VEw_bc[i])
      ##### 
      VWxwn_bc[i]      = 0.0 #Wx_wn_IC_func(dew_dx,Ve_wn_eq[i], VPw_bc[i] , VEwn_bc[i] ,  VJwn_bc[i] ) 
      VKwn_bc[i]       =Kwn_Linear_Geometric( VEw_bc[i], VEwn_bc[i]  ,VJwn_bc[i],  b0D1, b1D2, b2D3,  e_ns,  js_ns, kn_ns)
    elif(i==Nx-1):
      dew_dx           = Cx*(VEw_bc[i]-VEw_bc[i-1])
      e_ns,e_ws,js_ns,kn_ns,b0D1,b1D2,b2D3= Linear_Geometric_Constants(VEw_bc[i])
      #####
      VWxwn_bc[i]      =0.0 #Wx_wn_IC_func(dew_dx,Ve_wn_eq[i], VPw_bc[i] , VEwn_bc[i] ,  VJwn_bc[i] ) 
      VKwn_bc[i]       =Kwn_Linear_Geometric( VEw_bc[i], VEwn_bc[i]  ,VJwn_bc[i],  b0D1, b1D2, b2D3,  e_ns,  js_ns, kn_ns)
    else: 
      dew_dx           = Cx*(VEw_bc[i]-VEw_bc[i-1])
      e_ns,e_ws,js_ns,kn_ns,b0D1,b1D2,b2D3= Linear_Geometric_Constants(VEw_bc[i])
      #####
      VWxwn_bc[i]      =0.0 #Wx_wn_IC_func(dew_dx,Ve_wn_eq[i], VPw_bc[i] , VEwn_bc[i] ,  VJwn_bc[i] ) 
      VKwn_bc[i]       =Kwn_Linear_Geometric( VEw_bc[i], VEwn_bc[i]  ,VJwn_bc[i],  b0D1, b1D2, b2D3,  e_ns,  js_ns, kn_ns)
      

##############################################################################
##############################################################################
##############################################################################
##############################################################################
##############################################################################
##############################################################################
cdef int Residual(ts,t,U,Udot,F):

##### When a variable starts with capital "C" it generally means a constant (opposite of vector), when it starts with "V" it means its a vector.
##### "Dif" in variables names stands for difference, Prod stands for product.
    #######
    cdef unsigned int i 
    cdef double  A_SPIJWK_11,A_SPIJWK_12,A_SPIJWK_21,A_SPIJWK_22,A_SPIJWK_31,A_SPIJWK_33,A_SPIJWK_41,A_SPIJWK_43,A_SPIJWK_44
    cdef double  Kw_plus,Kw_minus,rho_w_plus,rho_w_minus,Od_w,Od_wn,Od_J,Od_pc
    cdef double  Cdew_dx, e_wn2, Cewn_ens, Cejwn_ejns, Cekwn_ekns,e_ns,e_ws,js_ns,kn_ns,b0D1,b1D2,b2D3
    global counter
    #######
    da=ts.getDM()
    #######
    localU=da.getLocalVec()
    da.globalToLocal(U,localU)
    #######
    u=da.getVecArray(localU)
    udot=da.getVecArray(Udot)
    f=da.getVecArray(F)
    Ndt=ts.getStepNumber()
    for i in range(0,Nx):
        #store u[i][0,1,2,3,4,5] at Ve_w[i],...Vkn_wn[i], because we will use it multiple times and using u[i][...] makes code slower
        #the 1e-5 term is added to avoid the division by zero, because at equilibrium CPw_J_wn will go to 0.
        Ve_w[i]               = u[i][0]
        Vp_w[i]               = u[i][1]         
        Ve_wn[i]              = u[i][2]
        Vj_wn[i]              = u[i][3]
        Vw_xwn[i]             = u[i][4]
        Vkn_wn[i]             = u[i][5]
        Vrho_w[i]             = rhow_of_Pw(Vp_w[i])      
        VKrw_rho_w[i]         = Ve_w[i] * Ve_w[i] * Vrho_w[i]
        VDif_ewn_eq_ewn[i]    = Ve_wn_eq[i] - Ve_wn[i]
        CPw_J_wn              = Vp_w[i] + Gamma_wn * Vj_wn[i]

        VOd_pc[i]             = C_wn * CPw_J_wn - (Prod_Gamma_K_wn1/(Vp_w[i])) * VDif_ewn_eq_ewn[i]  
    for i in range(1,Nx-1):
        ######### calculate the A_SPIJWK matrix components (we type only the non zeros terms of the Matrix)
        A_SPIJWK_11 = 1                 
        A_SPIJWK_21 = Vrho_w[i]
        A_SPIJWK_22 = Ve_w[i] *Drvt_rhow_of_Pw(Vp_w[i])
        A_SPIJWK_31 = Vj_wn[i]
        A_SPIJWK_33 = 1
        A_SPIJWK_41 = -2*Vkn_wn[i]
        A_SPIJWK_43 = Vj_wn[i]
        A_SPIJWK_44 = Ve_wn[i]
        ######### calculate the spacial descretization operators 
        Kw_plus     = Half_Inv_Rww * (VKrw_rho_w[i+1] + VKrw_rho_w[i])
        Kw_minus    = Half_Inv_Rww * (VKrw_rho_w[i-1] + VKrw_rho_w[i])
        rho_w_plus  = 0.5 * (Vrho_w[i+1] + Vrho_w[i])
        rho_w_minus = 0.5 * (Vrho_w[i-1] + Vrho_w[i])
        Od_w        = Cx*(Kw_plus*(Cx*(Vp_w[i+1] - Vp_w[i]) + rho_w_plus*g*d) - Kw_minus*(Cx*(Vp_w[i] - Vp_w[i-1]) + rho_w_minus*g*d)) 




        ##### Upwind 
        Wx_wn_up   = 0.5*(Vw_xwn[i] + Vw_xwn[i+1])
        Wx_wn_down = 0.5*(Vw_xwn[i] + Vw_xwn[i-1])

        if Wx_wn_up>=0:
            Fe_up = Wx_wn_up*Ve_wn[i]
            Fj_up = Wx_wn_up*Ve_wn[i]*Vj_wn[i]
        else:
            Fe_up = Wx_wn_up*Ve_wn[i+1]
            Fj_up = Wx_wn_up*Ve_wn[i+1]*Vj_wn[i+1]

        if Wx_wn_down>=0:
            Fe_down = Wx_wn_down*Ve_wn[i-1]
            Fj_down = Wx_wn_down*Ve_wn[i-1]*Vj_wn[i-1]
            Few_down = Ve_w[i-1]
        else:
            Fe_down  = Wx_wn_down*Ve_wn[i]
            Fj_down = Wx_wn_down*Ve_wn[i]*Vj_wn[i]

        ##### Lax-Friedrichs 
        alpha = max(abs(Vw_xwn[i]), abs(Vw_xwn[i+1]))
        Fe_up = 0.5 * (Vw_xwn[i] * Ve_wn[i] + Vw_xwn[i+1] * Ve_wn[i+1]) - 0.5 * alpha * (Ve_wn[i+1] - Ve_wn[i])
        Fj_up = 0.5 * (Vw_xwn[i] * Ve_wn[i]* Vj_wn[i] + Vw_xwn[i+1] * Ve_wn[i+1] * Vj_wn[i+1]) - 0.5 * alpha * (Ve_wn[i+1]*Vj_wn[i+1] - Ve_wn[i]*Vj_wn[i])
        alpha = max(abs(Vw_xwn[i-1]), abs(Vw_xwn[i]))
        Fe_down = 0.5 * (Vw_xwn[i] * Ve_wn[i] + Vw_xwn[i-1] * Ve_wn[i-1]) - 0.5 * alpha * (Ve_wn[i] - Ve_wn[i-1])
        Fj_down = 0.5 * (Vw_xwn[i] * Ve_wn[i] * Vj_wn[i] + Vw_xwn[i-1] * Ve_wn[i-1] * Vj_wn[i-1]) - 0.5 * alpha * (Ve_wn[i]*Vj_wn[i] - Ve_wn[i-1]*Vj_wn[i-1])

        #####
        D_ewn_wx_dx = Cx*(Fe_up-Fe_down)
        Od_wn = -(k_wn1_hat*(Vp_w[i] + Gamma_wn * Vj_wn[i])/Vp_w[i]) * VDif_ewn_eq_ewn[i]- D_ewn_wx_dx
        #####
        D_ewn_wx_jwn_dx = Cx*(Fj_up-Fj_down)
        Od_J  = Two_by3_Cx2 * (VOd_pc[i+1] - 2 * VOd_pc[i] + VOd_pc[i-1]) - 2*D_ewn_wx_jwn_dx
        #####
        Cdew_dx  = 0.5*Cx  * ( Ve_w[i+1] - Ve_w[i-1] )
        e_wn2    = Ve_wn[i] * Ve_wn[i]
        #####Calculate necessary vectors for equation 6
        e_ns,e_ws,js_ns,kn_ns,b0D1,b1D2,b2D3= Linear_Geometric_Constants(Ve_w[i])
        #####
        Cewn_ens         = Ve_wn[i] + e_ns
        Cejwn_ejns       =-Ve_wn[i] * Vj_wn[i]  + e_ns * js_ns
        Cekwn_ekns       = Ve_wn[i] * Vkn_wn[i] + e_ns * kn_ns
        ######### define the  residuals developed in the SPIJWK formulation inside the domain
        f[i][0] = A_SPIJWK_11 * udot[i][0]                                                        - VOd_pc[i]  #eq 2
        f[i][1] = A_SPIJWK_21 * udot[i][0] + A_SPIJWK_22 * udot[i][1]                             - Od_w       #eq 1 
        f[i][2] = A_SPIJWK_31 * udot[i][0] + A_SPIJWK_33 * udot[i][2]                             - Od_wn      #eq 3
        f[i][3] = A_SPIJWK_41 * udot[i][0] + A_SPIJWK_43 * udot[i][2] + A_SPIJWK_44 * udot[i][3]  - Od_J       #eq 4
        f[i][4] = Cdew_dx * VOd_pc[i] + e_wn2 * Vw_xwn[i]                                                      #eq 5
        f[i][5] = Vkn_wn[i] - Kwn_Linear_Geometric(Ve_w[i], Ve_wn[i]  ,Vj_wn[i],  b0D1, b1D2, b2D3,  e_ns,  js_ns, kn_ns)

 
    ######### Boundary points residuals #########    
    f[0][0] = u[0][0]  - VEw_bc[0]      
    f[0][1] = u[0][1]  - VPw_bc[0]    
    f[0][2] = u[0][2]  - VEwn_bc[0]      
    f[0][3] = u[0][3]  - VJwn_bc[0]            
    f[0][4] = u[0][4]   - VWxwn_bc[0]               #u[1][4]  - u[0][4]           
    f[0][5] = u[0][5]  - VKwn_bc[0]                    
    
    f[Nx-1][0] = u[Nx-1][0] - VEw_bc[Nx-1]    
    f[Nx-1][1] = u[Nx-1][1] - VPw_bc[Nx-1]   
    f[Nx-1][2] = u[Nx-1][2] - VEwn_bc[Nx-1]                 
    f[Nx-1][3] = u[Nx-1][3] - VJwn_bc[Nx-1]                     
    f[Nx-1][4] = u[Nx-1][4]   - VWxwn_bc[Nx-1]               #  u[Nx-1][4] - u[Nx-2][4]             
    f[Nx-1][5] = u[Nx-1][5] - VKwn_bc[Nx-1]               
  
#######################################

cdef int Initialization(ts,U):
# a private function that we created to initialize the solution vector at t=t0
    cdef unsigned int i
    #######
    da=ts.getDM()
    u=da.getVecArray(U)
    ####### Initial conditions and boundary conditions
    for i in range(0,Nx):
        if(i==0):
          u[i][0]   = VEw_bc[i] 
          u[i][1]   = VPw_bc[i] 
          u[i][2]   = VEwn_bc[i]
          u[i][3]   = VJwn_bc[i]
          u[i][4]   = VWxwn_bc[i]
          u[i][5]   = VKwn_bc[i] 
        elif(i==Nx-1):
          u[i][0]   = VEw_bc[i]
          u[i][1]   = VPw_bc[i]
          u[i][2]   = VEwn_bc[i]
          u[i][3]   = VJwn_bc[i]
          u[i][4]   = VWxwn_bc[i]
          u[i][5]   = VKwn_bc[i] 
        else:
          u[i][0]   = VEw_bc[i]
          u[i][1]   = VPw_bc[i]  
          u[i][2]   = VEwn_bc[i]
          u[i][3]   = VJwn_bc[i]
          u[i][4]   = VWxwn_bc[i]
          u[i][5]   = VKwn_bc[i] 


#######################################
cdef void main():
#######################################    
    # here we created a 1D grid, to let the solver store the jacobian as a sparse matrix
    cdef unsigned int i,j
    comm = PETSc.COMM_WORLD 
    bx= PETSc.DMDA.BoundaryType.NONE
    da=PETSc.DMDA().create(dof= 6,sizes=(Nx, ),boundary_type=bx,stencil_type=None,stencil_width=1, comm=comm ) 
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
    ts.setMaxSteps(MaxTimeSteps)                # maximum steps allowed to reach final time
    ts.setMaxStepRejections(MaxStepRejections)  # maximum time steps rejections allowed ,       
    ts.setMaxSNESFailures(MaxSNESFailures)
    ts.setTolerances(rtol, atol)                # set tolerances of the the chosen ts (here bdf )
    ts.setTime(t0)                              # set initial time t0 
    ts.setExactFinalTime(ts.ExactFinalTime.MATCHSTEP)  # this option is to match the output time of ts with the output time defined by user.
    #ts.setTimeStep(First_Time_Step)
    ts.setFromOptions()                         # give the user the possibility to personalize the ts solver at running time
#######################################
    Initialization(ts,u)                        # initialization of solution vector (initial conditions)
    U=da.getVecArray(u)                         # get the solution vector right after initialization 
    
    
#############Calling the solver and start solving the DAE system ###################### 
    mesh = np.linspace(Omega_0, Omega_L, Nx) # mesh just for plots
    for i in range(len(time_interval)):
       ts.setMaxTime(time_interval[i])
       CPU_start = time.process_time()
       ts.solve(u)                            # call ts solver to solve the system
       CPU_end = time.process_time()
       ftime[i]=ts.getTime()
       Clock_Array[i]=CPU_end-CPU_start
       ##### 
       Solution=ts.getSolution() 
       #####visualization
       U=da.getVecArray(u)
       ##### store the solution vector at each time_interval value
       for j in range(0,Nx):
           Ve_w_Time_Results[i][j]    = U[j][0]
           Vp_w_Time_Results[i][j]    = U[j][1]
           Ve_wn_Time_Results[i][j]   = U[j][2]
           Vj_wn_Time_Results[i][j]   = U[j][3] 
           Vw_xwn_Time_Results[i][j]  = U[j][4]
           Vkn_wn_Time_Results[i][j]  = U[j][5]
       ##### This condition is to stop looping (the time loop) if the MaxTimeSteps is reached
       NTimeSteps=ts.getStepNumber()
       if(NTimeSteps==MaxTimeSteps): 
         break
    ####################     
    Output_Path='Outputs_TCAT_Parameters'
    file = open(Output_Path, "w")
    file.write('Nx= %d ' %Nx)
    file.write('\nOmega_0= %g ' %Omega_0)
    file.write('\nOmega_L= %g ' %Omega_L)
    file.write('\nTime(s)= ')   
    for i in range(len(time_interval)):
        file.write('%g ' %time_interval[i])
    file.close() 
    ####################
    Output_Path='Outputs_TCAT_Ew'
    file = open(Output_Path, "w")
    for i in range(len(time_interval)):
        file.write('Ew= ')
        for j in range(0,Nx):
            file.write('%f ' %Ve_w_Time_Results[i][j] )
        file.write('\n')    
    file.close()  
    ####################
    Output_Path='Outputs_TCAT_Pw'
    file = open(Output_Path, "w")
    for i in range(len(time_interval)):
        file.write('Pw= ')
        for j in range(0,Nx):
            file.write('%f ' %Vp_w_Time_Results[i][j] )
        file.write('\n')    
    file.close()  
    ####################
    Output_Path='Outputs_TCAT_Ewn'
    file = open(Output_Path, "w")
    for i in range(len(time_interval)):
        file.write('Ewn= ')
        for j in range(0,Nx):
            file.write('%f ' %Ve_wn_Time_Results[i][j] )
        file.write('\n')    
    file.close()  
    ####################
    Output_Path='Outputs_TCAT_Jwn'
    file = open(Output_Path, "w")
    for i in range(len(time_interval)):
        file.write('Jwn= ')
        for j in range(0,Nx):
            file.write('%f ' %Vj_wn_Time_Results[i][j] )
        file.write('\n')    
    file.close()  
    ####################
    Output_Path='Outputs_TCAT_Wwn'
    file = open(Output_Path, "w")
    for i in range(len(time_interval)):
        file.write('Wwn= ')
        for j in range(0,Nx):
            file.write('%g ' %Vw_xwn_Time_Results[i][j] )
        file.write('\n')    
    file.close()  
    ####################
    Output_Path='Outputs_TCAT_Kwn'
    file = open(Output_Path, "w")
    for i in range(len(time_interval)):
        file.write('Kwn= ')
        for j in range(0,Nx):
            file.write('%f ' %Vkn_wn_Time_Results[i][j] )
        file.write('\n')    
    file.close()  
main() 
