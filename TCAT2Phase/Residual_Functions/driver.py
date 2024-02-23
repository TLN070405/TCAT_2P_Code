
import petsc4py
import time
#petsc4py.init(sys.argv)
from petsc4py import PETSc


def main(nodes,time_interval):

    bz= PETSc.DMDA.BoundaryType.NONE
    
    if(Formulation=='SPP'): 
      dof=3
    elif(Formulation=='SP' or Formulation=='PP'):  
      dof=2

    da=PETSc.DMDA().create(dof= dof,sizes=(nodes,),boundary_type=bz,stencil_type=None,stencil_width=1, comm=PETSc.COMM_WORLD ) 
    da.setUp()
    da.setMatType(PETSc.Mat.Type.AIJ)           # create a sparse matrix related to the grid created before (actually this is the type)
    da.createMatrix()                           # create the matrix
    u= da.createGlobalVec() # create a structured array related to the grid created before 

    ts = PETSc.TS().create(PETSc.COMM_WORLD)    # create a time-stepping solver object 
    ts.setProblemType(ts.ProblemType.NONLINEAR) # set type of problem 
    ts.setType(ts.Type.BDF)                     # set type to BDF method	
    ts.setIFunction(Residual)                   # pass the residual function to "ts" so it can use it  
    ts.setDM(da)                                # attach the ts solver to the grid created 
    ts.setMaxSteps(MaxTimeSteps)                # maximum steps allowed to take to reach final time
    ts.setMaxStepRejections(MaxStepRejections)  # maximum steps rejections allowed , this is related to snes functions         
    ts.setMaxSNESFailures(MaxSNESFailures)
    ts.setTolerances(rtol, atol)                # set tolerances of the  chosen ts (here bdf )
    ts.setTime(t0)                              # set initial time t0 (here t0=0)
    ts.setExactFinalTime(ts.ExactFinalTime.MATCHSTEP)  # this option is to match the output time of ts with the output time defined by user.
    ts.setFromOptions()                         # give the user the possibility to personalize the ts solver at running time


    snes=ts.getSNES()                           # get the snes solver
    snes.setFromOptions()                       # give the user the possibility to personalize the "snes" at running time    
    ksp=snes.getKSP()                           # get the ksp solver
    ksp.setFromOptions()                        # give the user the possibility to personalize the "ksp" at running time
    pc=ksp.getPC()                              # get the pc object from ksp
    pc.setFromOptions()                         # give the user the possibility to personalize the "pc" at running time
    
    


    IC.Initial_Conditions(ts,u)                                  # initialization of solution vector (initial conditions)
    U=da.getVecArray(u)                                          # get the solution vector right after initialization 


    for time in time_interval:
       ts.setMaxTime(time) 
       ts.solve(u)                     
       U=da.getVecArray(u)       

       if(Formulation=='SPP'):     
          for k in range(0,Nz):
              VEpsilonw[i][k]=U[k][0]
              VPw[i][k]=U[k][1]
              VPn[i][k]=U[k][2]
       elif(Formulation=='SP'): 
          for k in range(0,Nz):
              VEpsilonw[i][k]=U[k][0]
              VPw[i][k]=U[k][1]
              VPn[i][k]=PSKM.Pc(VEpsilonw[i][k]) + VPw[i][k]
       elif(Formulation=='PP'): 
          for k in range(0,Nz):
              VPw[i][k]=U[k][0]
              VPn[i][k]=U[k][1]
              VEpsilonw[i][k]=PSKM.Epsilon_w_of_Pc(VPn[i][k]-VPw[i][k])
       ##### this if condition is just to avoid looping when the MaxTimeSteps is reached    
       NTimeSteps=ts.getStepNumber() 
       if(NTimeSteps==MaxTimeSteps): 
         break
    
    ts.view()  ## print the ts,snes,ksp,pc solvers parameters on screen       
#######################################
