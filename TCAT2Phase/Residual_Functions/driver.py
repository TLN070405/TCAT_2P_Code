
import petsc4py
import time
#petsc4py.init(sys.argv)
from petsc4py import PETSc
from TCAT2Phase.Residual_Functions import *


class Base_Formulation:
    pass

class SPP(Base_Formulation):
    """
    SPP  Formulation Class
    """
    def __init__(self):
        self.n_equations = 3
        self.residual = two_phase_spp_residual()


class SP(Base_Formulation):
    """
    SPP  Formulation Class
    """
    def __init__(self):
        self.n_equations = 2
        self.residual = two_phase_sp_residual()


class PP(Base_Formulation):
    """
    SPP  Formulation Class
    """
    def __init__(self):
        self.n_equations = 2
        self.residual = two_phase_pp_residual()


def solver(nodes,time_interval,formulation):
    """
    Collect all information and drive the simulation 
    """

    da=PETSc.DMDA().create(dof=formulation.n_equations,
                           sizes=(nodes,),
                           boundary_type=PETSc.DMDA.BoundaryType.NONE,
                           stencil_type=None,
                           stencil_width=1, 
                           comm=PETSc.COMM_WORLD 
                            ) 
    da.setUp()
    da.setMatType(PETSc.Mat.Type.AIJ)           # create a sparse matrix related to the grid created before (actually this is the type)
    da.createMatrix()                           # create the matrix
    u= da.createGlobalVec() # create a structured array related to the grid created before 

    ts = PETSc.TS().create(PETSc.COMM_WORLD)    # create a time-stepping solver object 
    ts.setProblemType(ts.ProblemType.NONLINEAR) # set type of problem 
    ts.setType(ts.Type.BDF)                     # set type to BDF method	
    ts.setIFunction(formulation.residual)       # pass the residual function to "ts" so it can use it  
    ts.setDM(da)                                # attach the ts solver to the grid created 
    ts.setMaxSteps(MaxTimeSteps)                # maximum steps allowed to take to reach final time
    ts.setMaxStepRejections(MaxStepRejections)  # maximum steps rejections allowed , this is related to snes functions         
    ts.setMaxSNESFailures(MaxSNESFailures)
    ts.setTolerances(rtol, atol)                # set tolerances of the  chosen ts (here bdf )
    ts.setTime(time[0])                              # set initial time t0 (here t0=0)
    ts.setExactFinalTime(ts.ExactFinalTime.MATCHSTEP)  # this option is to match the output time of ts with the output time defined by user.
    ts.setFromOptions()                         # give the user the possibility to personalize the ts solver at running time


    snes=ts.getSNES()                           # get the snes solver
    snes.setFromOptions()                       # give the user the possibility to personalize the "snes" at running time    
    
    ksp=snes.getKSP()                           # get the ksp solver
    ksp.setFromOptions()                        # give the user the possibility to personalize the "ksp" at running time
    
    pc=ksp.getPC()                              # get the pc object from ksp
    pc.setFromOptions()                         # give the user the possibility to personalize the "pc" at running time
    
    

    set_initial_conditions(ts,u)
    #IC.Initial_Conditions(ts,u)                                  # initialization of solution vector (initial conditions)
    #U=da.getVecArray(u)                                          # get the solution vector right after initialization 


    for time in time_interval:
        ts.setMaxTime(time) 
        ts.solve(u)                     
        U=da.getVecArray(u)       


    ts.view()  ## print the ts,snes,ksp,pc solvers parameters on screen       
#######################################
