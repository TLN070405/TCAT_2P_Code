import petsc4py,time,sys,math,cython,numpy as np
petsc4py.init(sys.argv)
from petsc4py import PETSc
from libc.stdlib cimport malloc
#####################################################################################################################   
####################################Import our private modules#######################################################
#####################################################################################################################
OptDB = PETSc.Options()
Test_Case = OptDB.getString('Test_Case', 'Test_Case4')
filenumber = OptDB.getInt('filenumber', 1)
sys.path.insert(0, './RE_Subcodes/Import_Modules')
import Import_Modules as IM
inp,BIC,CA,SE,SD,ER=IM.ImportModules(Test_Case)
#####################Create arrays###################################
VPw_bc,Ve_w_Time_Results,Vp_w_Time_Results,Clock_Array,ftime=CA.CreateArrays(inp.Nx,inp.time_interval)
####################Fill BC and IC arrays to be used in initialization####################################
BIC.BC_IC(inp,SE,VPw_bc)
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
    SD.SD_PR_Arrays(inp,SE,u)
    for i in range(1,Nx-1):
        A_PR ,Od_w=SD.SD_APR(inp,SE,u,i)
        ######### define the  residuals developed in the PR formulation inside the domain
        f[i] = A_PR * udot[i]  - Od_w
    f[0]    =  u[0] -VPw_bc[0]
    f[Nx-1] =  u[Nx-1]  -VPw_bc[Nx-1]
        
#######################################
cdef int Initialization(ts,U):
    # private function that we created to initialize the solution vectors at t=t0
    cdef unsigned int i
    #######
    da=ts.getDM()
    u=da.getVecArray(U)
    ####### Initial conditions and boundary conditions 
    for i in range(0,Nx):
        u[i]   = VPw_bc[i] 

#######################################
cdef void main():
#######################################    
    # here we created a 1D grid, to let the solver store the jacobian as a sparse matrix
    cdef unsigned int i,j
    comm = PETSc.COMM_WORLD 
    bx= PETSc.DMDA.BoundaryType.NONE
    da=PETSc.DMDA().create(dof= 1,sizes=(Nx, ),boundary_type=bx,stencil_type=None,stencil_width=1, comm=comm ) 
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
           Ve_w_Time_Results[i][j]    = SE.Epsilon_w_of_Pc(inp,-U[j])
           Vp_w_Time_Results[i][j]    = U[j]
       ##### This condition is to stop looping (the time loop) if the MaxTimeSteps is reached
       NTimeSteps=ts.getStepNumber()
       if(NTimeSteps==inp.MaxTimeSteps): 
         break     
           
#############Visualization      
    ER.Export_Results(inp,Test_Case,filenumber,Ve_w_Time_Results,Vp_w_Time_Results)    
main() 
