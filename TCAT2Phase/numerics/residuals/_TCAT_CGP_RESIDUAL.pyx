import petsc4py,time,sys,math,cython,numpy as np
petsc4py.init(sys.argv)
from petsc4py import PETSc
from libc.stdlib cimport malloc
#####################################################################################################################   
####################################Import our private modules#######################################################
#####################################################################################################################
OptDB      = PETSc.Options()
Test_Case  = OptDB.getString('Test_Case', 'Test_Case4')
filenumber = OptDB.getInt('filenumber', 1)
sys.path.insert(0, './TCAT_CGP_Subcodes/Import_Modules')
import Import_Modules as IM
inp,BIC,CA,SE,SD,ER=IM.ImportModules(Test_Case)
#####################Create arrays###################################
Ve_wn_eq,VEw_bc,VPw_bc,VEwn_bc,VJwn_bc,VWxwn_bc,VKwn_bc,Ve_w_Time_Results,Vp_w_Time_Results,Ve_wn_Time_Results,Vj_wn_Time_Results,Vw_xwn_Time_Results,Vkn_wn_Time_Results,Clock_Array,ftime=CA.CreateArrays(inp.Nx,inp.time_interval)
####################Fill BC and IC arrays to be used in initialization####################################
BIC.BC_IC(inp,SE,VEw_bc,VPw_bc,VEwn_bc,VJwn_bc,VWxwn_bc,VKwn_bc,Ve_wn_eq)
##############################################################################
Nx=inp.Nx
##############################################################################
cdef int Residual(ts,t,U,Udot,F):
    cdef unsigned int i 
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
    SD.SD_SPIJWK_Arrays(inp,SE,u,Ve_wn_eq)
    for i in range(1,Nx-1):
        A_SPIJWK_11,A_SPIJWK_21,A_SPIJWK_22,A_SPIJWK_31,A_SPIJWK_33,A_SPIJWK_41,A_SPIJWK_43,A_SPIJWK_44,Od_pc,Od_w,Od_wn,Od_J,Cdew_dx,e_wn2,w_xwn,e_w,Cewn_ens,Cejwn_ejns,Cekwn_ekns,b0D1,b1D2,b2D3=SD.SD_SPIJWK_A(inp,SE,u,i)
        ######### define the  residuals developed in the SPIJWK formulation inside the domain
        f[i][0] = A_SPIJWK_11 * udot[i][0]                                                        - Od_pc     #eq 2
        f[i][1] = A_SPIJWK_21 * udot[i][0] + A_SPIJWK_22 * udot[i][1]                             - Od_w      #eq 1 
        f[i][2] = A_SPIJWK_31 * udot[i][0] + A_SPIJWK_33 * udot[i][2]                             - Od_wn     #eq 3
        f[i][3] = A_SPIJWK_41 * udot[i][0] + A_SPIJWK_43 * udot[i][2] + A_SPIJWK_44 * udot[i][3]  - Od_J      #eq 4
        f[i][4] = Cdew_dx     * Od_pc      +       e_wn2 * w_xwn                                              #eq 5
        f[i][5] = e_w  - inp.epsilon + b0D1 * Cewn_ens + b1D2 * Cejwn_ejns + b2D3 * Cekwn_ekns                #eq 6
    ######### Boundary points residuals #########    
    f[0][0] = u[0][0]  - VEw_bc[0]      
    f[0][1] = u[0][1]  - VPw_bc[0]    
    f[0][2] = u[0][2]  - VEwn_bc[0] 
    f[0][3] = u[0][3]  - VJwn_bc[0] 
    f[0][4] = u[0][4]  - VWxwn_bc[0]
    f[0][5] = u[0][5]  - VKwn_bc[0] 
    
    f[Nx-1][0] = u[Nx-1][0] - VEw_bc[Nx-1]    
    f[Nx-1][1] = u[Nx-1][1] - VPw_bc[Nx-1]   
    f[Nx-1][2] = u[Nx-1][2] - VEwn_bc[Nx-1] 
    f[Nx-1][3] = u[Nx-1][3] - VJwn_bc[Nx-1]    
    f[Nx-1][4] = u[Nx-1][4] - VWxwn_bc[Nx-1]     
    f[Nx-1][5] = u[Nx-1][5] - VKwn_bc[Nx-1] 
#######################################
cdef int Initialization(ts,U):
    # private function that we created to initialize the solution vectors at t=t0
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
    ts.setMaxSteps(inp.MaxTimeSteps)                # maximum steps allowed to reach final time
    ts.setMaxStepRejections(inp.MaxStepRejections)  # maximum time steps rejections allowed ,       
    ts.setMaxSNESFailures(inp.MaxSNESFailures)
    ts.setTolerances(inp.rtol, inp.atol)                # set tolerances of the the chosen ts (here bdf )
    ts.setTime(inp.t0)                              # set initial time t0 
    ts.setExactFinalTime(ts.ExactFinalTime.MATCHSTEP)  # this option is to match the output time of ts with the output time defined by user.
    ts.setFromOptions()                         # give the user the possibility to personalize the ts solver at running time
#######################################
    Initialization(ts,u)                        # initialization of solution vector (initial conditions)
    U=da.getVecArray(u)                         # get the solution vector right after initialization 
#############Calling the solver and start solving the DAE system ###################### 
    
    for i in range(len(inp.time_interval)):
       ts.setMaxTime(inp.time_interval[i])
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
       if(NTimeSteps==inp.MaxTimeSteps): 
         break     
           
#############Visualization      
    ER.Export_Results(inp,Test_Case,filenumber,Ve_w_Time_Results,Vp_w_Time_Results,Ve_wn_Time_Results,Vj_wn_Time_Results,Vw_xwn_Time_Results,Vkn_wn_Time_Results)    
main() 
