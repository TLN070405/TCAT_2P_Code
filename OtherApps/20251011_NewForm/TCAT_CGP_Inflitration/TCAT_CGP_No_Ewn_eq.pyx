import petsc4py
import time
import sys
import math
import numpy as np
import cython
import matplotlib.pyplot as plt
petsc4py.init(sys.argv)
from petsc4py import PETSc
from libc.stdlib cimport malloc, free

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
OptDB = PETSc.Options()

Two_by3=2/3
pi=3.141592653589793
############0

time_interval=np.array([1e-8,1e-7,1e-6,1e-5,1e-4,1e-2,1e-1,1])
#time_interval=np.array([1e-3,0.1,1,3])
t0=0.0
######Space domain parameters######
Nx= 101
Omega_0= 0.0
Omega_L= 0.1
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
Rww       = 3.75e7 # 3.75e5
Gamma_wn  = 7.20e-2
C_wn      = 1e-05    #6.398675e-05 
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
MaxTimeSteps=100000

 
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
cdef double *Vz_w     = <double*>malloc(Nx * sizeof(double))
cdef double *Vw_xwn   = <double*>malloc(Nx * sizeof(double))
cdef double *Vkn_wn   = <double*>malloc(Nx * sizeof(double))

################ Arrays for intermediate calculations #############
cdef double *Vrho_w               = <double*>malloc(Nx * sizeof(double))
cdef double *VKrw_rho_w           = <double*>malloc(Nx * sizeof(double))
cdef double *VPw_J_wn               = <double*>malloc(Nx * sizeof(double))
################################################################
 
VEw_bc       = np.zeros([Nx], dtype=np.double) 
VPw_bc       = np.zeros([Nx], dtype=np.double) 
VEwn_bc       = np.zeros([Nx], dtype=np.double) 
VJwn_bc       = np.zeros([Nx], dtype=np.double) 
VZw_bc       = np.zeros([Nx], dtype=np.double) 
VWxwn_bc       = np.zeros([Nx], dtype=np.double) 
VKwn_bc       = np.zeros([Nx], dtype=np.double) 


####################(Arrays to store results)#######################
Ve_w_Time_Results=np.zeros([len(time_interval),Nx], dtype=np.double) 
Vp_w_Time_Results=np.zeros([len(time_interval),Nx], dtype=np.double) 
Ve_wn_Time_Results=np.zeros([len(time_interval),Nx], dtype=np.double) 
Vj_wn_Time_Results=np.zeros([len(time_interval),Nx], dtype=np.double) 
Vz_w_Time_Results=np.zeros([len(time_interval),Nx], dtype=np.double) 
Vw_xwn_Time_Results=np.zeros([len(time_interval),Nx], dtype=np.double) 
Vkn_wn_Time_Results=np.zeros([len(time_interval),Nx], dtype=np.double) 


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
cdef  double  Wx_wn_IC_func(double dew_dx, double Ewn, double Zw):
    cdef  double Wx_wn  
    Wx_wn= -Zw *(dew_dx/(Ewn*Ewn))
    return Wx_wn 
#######################################
cdef double Zw_IC_func( VEwn_bc, VPw_bc, int i):
    cdef double Vrho_w0, Vrho_w1, Vrho_w2
    cdef double VKrw_rho_w0, VKrw_rho_w1, VKrw_rho_w2
    cdef double Kw_plus, Kw_minus, rho_w_plus, rho_w_minus, q_w_plus, q_w_minus, Z_w
    
    Vrho_w0 = rhow_of_Pw(VPw_bc[i-1])
    Vrho_w1 = rhow_of_Pw(VPw_bc[i])
    Vrho_w2 = rhow_of_Pw(VPw_bc[i+1])

    VKrw_rho_w0 = VEwn_bc[i-1] * VEwn_bc[i-1] * Vrho_w0
    VKrw_rho_w1 = VEwn_bc[i] * VEwn_bc[i] * Vrho_w1
    VKrw_rho_w2 = VEwn_bc[i+1] * VEwn_bc[i+1] * Vrho_w2

    Kw_plus = Half_Inv_Rww * (VKrw_rho_w2 + VKrw_rho_w1)
    Kw_minus = Half_Inv_Rww * (VKrw_rho_w0 + VKrw_rho_w1)
    rho_w_plus = 0.5 * (Vrho_w2 + Vrho_w1)
    rho_w_minus = 0.5 * (Vrho_w0 + Vrho_w1)

    q_w_plus = Kw_plus * (Cx * (VPw_bc[i+1] - VPw_bc[i]) + rho_w_plus * g * d)
    q_w_minus = Kw_minus * (Cx * (VPw_bc[i] - VPw_bc[i-1]) + rho_w_minus * g * d)

    Z_w = (1 / Vrho_w1) * Cx * (q_w_plus - q_w_minus)
    return Z_w

########################################################
#########################BC and IC######################
########################################################
######################################################## 

Ew_bc = 0.357
Pw_bc  = -500
Ewn_bc = 102


for i in range(0,Nx):
    x = Omega_0+i*dx
    if(i==0):
        VEw_bc[i]        = 0.359
        VPw_bc[i]        = -400
        VEwn_bc[i]       = 90
        VJwn_bc[i]       = 7800#-VPw_bc[i]/Gamma_wn
    elif(i==Nx-1):
        VEw_bc[i]        = Ew_bc 
        VPw_bc[i]        = Pw_bc 
        VEwn_bc[i]       = Ewn_bc
        VJwn_bc[i]       = -VPw_bc[i]/Gamma_wn 
    else:
        VEw_bc[i]        = Ew_bc
        VPw_bc[i]        = Pw_bc
        VEwn_bc[i]       = Ewn_bc
        VJwn_bc[i]       = -VPw_bc[i]/Gamma_wn
        

for i in range(0,Nx):
    e_ns,e_ws,js_ns,kn_ns,b0D1,b1D2,b2D3= Linear_Geometric_Constants(VEw_bc[i])
    VKwn_bc[i]       = Kwn_Linear_Geometric( VEw_bc[i], VEwn_bc[i]  ,VJwn_bc[i],  b0D1, b1D2, b2D3,  e_ns,  js_ns, kn_ns)

    if(i==0):
        VZw_bc[i]        = 0.0
        dew_dx           = Cx*(VEw_bc[i+1]-VEw_bc[i])
        VWxwn_bc[i]      = 0.0 #Wx_wn_IC_func(dew_dx, VEwn_bc[i] ,  VZw_bc[i] ) 
    elif(i==Nx-1):
        VZw_bc[i]        = 0.0
        dew_dx           = Cx*(VEw_bc[i]-VEw_bc[i-1])
        VWxwn_bc[i]      = 0.0 #Wx_wn_IC_func(dew_dx, VEwn_bc[i] ,  VZw_bc[i] ) 
    else: 
        VZw_bc[i]        = 0#Zw_IC_func( VEw_bc, VPw_bc, i)
        dew_dx           = Cx*(VEw_bc[i]-VEw_bc[i-1])
        VWxwn_bc[i]      = 0.0 #Wx_wn_IC_func(dew_dx, VEwn_bc[i] ,  VZw_bc[i] ) 

##############################################################################
##############################################################################
##############################################################################
##############################################################################
##############################################################################
cdef int Residual(ts,t,U,Udot,F):

    #######
    cdef unsigned int i 
    cdef double  A_00, A_11, A_22, A_32, A_33, A_41, A_42, A_43, A_44
    cdef double  Kw_plus,Kw_minus,rho_w_plus,rho_w_minus,Od_w, Od0,Od1,Od2,Od3,Od4,Od5,Od6
    cdef double  Cdew_dx,e_ns,e_ws,js_ns,kn_ns,b0D1,b1D2,b2D3
    cdef double  alpha, Fe_up, Fj_up, Fe_down, Fj_down, D_ewn_wx_dx, D_ewn_wx_jwn_dx

    #######
    da=ts.getDM()
    #######
    localU=da.getLocalVec()
    da.globalToLocal(U,localU)
    #######
    u=da.getVecArray(localU)
    udot=da.getVecArray(Udot)
    f=da.getVecArray(F)
    #######
    for i in range(0,Nx):
        #store u[i][0,1,2,3,4,5] at Ve_w[i],...Vkn_wn[i], because we will use it multiple times and using u[i][...] makes code slower
        #the 1e-5 term is added to avoid the division by zero, because at equilibrium CPw_J_wn will go to 0.
        Ve_w[i]               = u[i][0]
        Vp_w[i]               = u[i][1]         
        Ve_wn[i]              = u[i][2]
        Vj_wn[i]              = u[i][3]
        Vz_w[i]               = u[i][4]
        Vw_xwn[i]             = u[i][5]
        Vkn_wn[i]             = u[i][6]
        Vrho_w[i]             = rhow_of_Pw(Vp_w[i])
        VKrw_rho_w[i]         = Ve_w[i] * Ve_w[i] * Vrho_w[i]
        VPw_J_wn[i]           = Vp_w[i] + Gamma_wn * Vj_wn[i]

    for i in range(1,Nx-1):
        ######### calculate the A_SPIJWK matrix components (we type only the non zeros terms of the Matrix)
        A_00 = 1.        

        A_11 = Ve_w[i] * Drvt_rhow_of_Pw(Vp_w[i])  

        A_22 = Gamma_wn

        A_32 = Vj_wn[i]
        A_33 = Ve_wn[i]

        A_41 = Vz_w[i] - C_wn * ( 2 * Vp_w[i] + Gamma_wn * Vj_wn[i] )
        A_42 = -k_wn1_hat * Gamma_wn
        A_43 = -C_wn * Gamma_wn * Vp_w[i]
        A_44 = Vp_w[i]

        ######### calculate the spacial descretization operators 
        ##### Lax-Friedrichs 
        alpha = max(abs(Vw_xwn[i]), abs(Vw_xwn[i+1]))
        Fe_up = 0.5 * (Vw_xwn[i] * Ve_wn[i] + Vw_xwn[i+1] * Ve_wn[i+1]) - 0.5 * alpha * (Ve_wn[i+1] - Ve_wn[i])
        Fj_up = 0.5 * (Vw_xwn[i] * Ve_wn[i]* Vj_wn[i] + Vw_xwn[i+1] * Ve_wn[i+1] * Vj_wn[i+1]) - 0.5 * alpha * (Ve_wn[i+1]*Vj_wn[i+1] - Ve_wn[i]*Vj_wn[i])
        alpha = max(abs(Vw_xwn[i-1]), abs(Vw_xwn[i]))
        Fe_down = 0.5 * (Vw_xwn[i] * Ve_wn[i] + Vw_xwn[i-1] * Ve_wn[i-1]) - 0.5 * alpha * (Ve_wn[i] - Ve_wn[i-1])
        Fj_down = 0.5 * (Vw_xwn[i] * Ve_wn[i] * Vj_wn[i] + Vw_xwn[i-1] * Ve_wn[i-1] * Vj_wn[i-1]) - 0.5 * alpha * (Ve_wn[i]*Vj_wn[i] - Ve_wn[i-1]*Vj_wn[i-1])

        D_ewn_wx_dx     = Cx*(Fe_up-Fe_down)
        D_ewn_wx_jwn_dx = Cx*(Fj_up-Fj_down)

        #### Od0 : correspond to equation 1
        Od0 = Vz_w[i]

        #### Od1 : correspond to equation 2
        Kw_plus     = Half_Inv_Rww * (VKrw_rho_w[i+1] + VKrw_rho_w[i])
        Kw_minus    = Half_Inv_Rww * (VKrw_rho_w[i-1] + VKrw_rho_w[i])
        rho_w_plus  = 0.5 * (Vrho_w[i+1] + Vrho_w[i])
        rho_w_minus = 0.5 * (Vrho_w[i-1] + Vrho_w[i])
        Od_w        = Cx*(Kw_plus*(Cx*(Vp_w[i+1] - Vp_w[i]) + rho_w_plus*g*d) - Kw_minus*(Cx*(Vp_w[i] - Vp_w[i-1]) + rho_w_minus*g*d)) 
        Od1 = Od_w - Vrho_w[i]* Vz_w[i]

        #### Od2 : correspond to equation 3
        Od2 = Vp_w[i]* Vz_w[i] - Gamma_wn *D_ewn_wx_dx - C_wn*VPw_J_wn[i]*VPw_J_wn[i]

        #### Od3 : correspond to equation 4
        Od3 = Two_by3_Cx2 * (Vz_w[i+1] - 2 * Vz_w[i] + Vz_w[i-1]) - 2 * Vkn_wn[i] * Vz_w[i] - 2 * D_ewn_wx_jwn_dx

        #### Od4 : correspond to equation 5
        Od4 = 0.

        #### Od5 : correspond to equation 6
        Cdew_dx  = 0.5 * Cx  * ( Ve_w[i+1] - Ve_w[i-1] )
        Od5 =  Cdew_dx * Vz_w[i] + Ve_wn[i] * Ve_wn[i] * Vw_xwn[i] 

        #### Od6 : correspond to equation 7
        e_ns,e_ws,js_ns,kn_ns,b0D1,b1D2,b2D3= Linear_Geometric_Constants(Ve_w[i])
        Od6 = Vkn_wn[i] - Kwn_Linear_Geometric(Ve_w[i], Ve_wn[i]  ,Vj_wn[i],  b0D1, b1D2, b2D3,  e_ns,  js_ns, kn_ns)

        ######### define the  residuals developed in the SPIJWK formulation inside the domain
        f[i][0] = A_00 * udot[i][0]                                                               - Od0      #eq 1
        f[i][1] = A_11 * udot[i][1]                                                               - Od1      #eq 2
        f[i][2] = A_22 * udot[i][2]                                                               - Od2      #eq 3
        f[i][3] = A_32 * udot[i][2] + A_33 * udot[i][3]                                           - Od3      #eq 4
        f[i][4] = A_41 * udot[i][1] + A_42 * udot[i][2] + A_43 * udot[i][3] + A_44 * udot[i][4]   - Od4      #eq 5
        f[i][5] = Od5                                                                                        #eq 6
        f[i][6] = Od6                                                                                        #eq 7

    ######### Boundary points residuals #########    
    f[0][0] = u[0][0]  - VEw_bc[0]      
    f[0][1] = u[0][1]  - VPw_bc[0]    
    f[0][2] = u[0][2]  - VEwn_bc[0]      
    f[0][3] = u[0][3]  - VJwn_bc[0]            
    f[0][4] = u[1][4]  - u[0][4]  #VZw_bc[0]      
    f[0][5] = u[1][5]  - u[0][5]  #VWxwn_bc[0]     
    f[0][6] = u[0][6]  - VKwn_bc[0]                    

    f[Nx-1][0] = u[Nx-1][0]  - VEw_bc[Nx-1]      
    f[Nx-1][1] = u[Nx-1][1]  - VPw_bc[Nx-1]    
    f[Nx-1][2] = u[Nx-1][2]  - VEwn_bc[Nx-1]      
    f[Nx-1][3] = u[Nx-1][3]  - VJwn_bc[Nx-1]            
    f[Nx-1][4] = u[Nx-1][4]  - u[Nx-2][4]  #VZw_bc[Nx-1]      
    f[Nx-1][5] = u[Nx-1][5]  - u[Nx-2][5]  #V VWxwn_bc[Nx-1]     
    f[Nx-1][6] = u[Nx-1][6]  - VKwn_bc[Nx-1]               

#######################################
cdef int Initialization(ts,U):
# a private function that we created to initialize the solution vector at t=t0
    cdef unsigned int i
    #######
    da=ts.getDM()
    u=da.getVecArray(U)
    ####### Initial conditions and boundary conditions
    for i in range(0,Nx):
        u[i][0]   = VEw_bc[i] 
        u[i][1]   = VPw_bc[i] 
        u[i][2]   = VEwn_bc[i]
        u[i][3]   = VJwn_bc[i]
        u[i][4]   = VZw_bc[i]
        u[i][5]   = VWxwn_bc[i]
        u[i][6]   = VKwn_bc[i] 

#######################################
cdef void main():
#######################################    
    # here we created a 1D grid, to let the solver store the jacobian as a sparse matrix
    cdef unsigned int i,j
    comm = PETSc.COMM_WORLD 
    bx= PETSc.DMDA.BoundaryType.NONE
    da=PETSc.DMDA().create(dof= 7,sizes=(Nx, ),boundary_type=bx,stencil_type=None,stencil_width=1, comm=comm ) 
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
       ts.solve(u)                            # call ts solver to solve the system
       #####visualization
       U=da.getVecArray(u)
       ##### store the solution vector at each time_interval value
       for j in range(0,Nx):
           Ve_w_Time_Results[i][j]    = U[j][0]
           Vp_w_Time_Results[i][j]    = U[j][1]
           Ve_wn_Time_Results[i][j]   = U[j][2]
           Vj_wn_Time_Results[i][j]   = U[j][3] 
           Vz_w_Time_Results[i][j]    = U[j][4] 
           Vw_xwn_Time_Results[i][j]  = U[j][5]
           Vkn_wn_Time_Results[i][j]  = U[j][6]
       ##### This condition is to stop looping (the time loop) if the MaxTimeSteps is reached
       if(ts.getStepNumber()==MaxTimeSteps): 
         break


    #####  plot ew
    plt.figure(figsize=(10,6))

    for i in range(len(time_interval)):
        plt.plot(mesh, Ve_w_Time_Results[i, :], label=f"t={time_interval[i]}")

    plt.xlabel("x")
    plt.ylabel("e_w")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

    #####  plot pw
    plt.figure(figsize=(10,6))
    for i in range(len(time_interval)):
        plt.plot(mesh, Vp_w_Time_Results[i, :], label=f"t={time_interval[i]}")

    plt.xlabel("x")
    plt.ylabel("p_w")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

    #####  plot ewn
    plt.figure(figsize=(10,6))
    for i in range(len(time_interval)):
        plt.plot(mesh, Ve_wn_Time_Results[i, :], label=f"t={time_interval[i]}")

    plt.xlabel("x")
    plt.ylabel("e_wn")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

    #####  plot jwn
    plt.figure(figsize=(10,6))
    for i in range(len(time_interval)):
        plt.plot(mesh, Vj_wn_Time_Results[i, :], label=f"t={time_interval[i]}")

    plt.xlabel("x")
    plt.ylabel("j_wn")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

    #####  plot zw
    plt.figure(figsize=(10,6))
    for i in range(len(time_interval)):
        plt.plot(mesh, Vz_w_Time_Results[i, :], label=f"t={time_interval[i]}")

    plt.xlabel("x")
    plt.ylabel("z_w")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

    #####  plot wwn
    plt.figure(figsize=(10,6))
    for i in range(len(time_interval)):
        plt.plot(mesh, Vw_xwn_Time_Results[i, :], label=f"t={time_interval[i]}")

    plt.xlabel("x")
    plt.ylabel("w_wn")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

    #####  plot kwn
    plt.figure(figsize=(10,6))
    for i in range(len(time_interval)):
        plt.plot(mesh, Vkn_wn_Time_Results[i, :], label=f"t={time_interval[i]}")

    plt.xlabel("x")
    plt.ylabel("k_wn")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()



    #free memory
    free(Ve_w)
    free(Vp_w)
    free(Ve_wn)
    free(Vj_wn)
    free(Vz_w)
    free(Vw_xwn)
    free(Vkn_wn)

    free(Vrho_w)
    free(VKrw_rho_w)
main() 


























