"""solver.py"""
import petsc4py
from petsc4py import PETSc

__all__ = [
    "solver"
]

def solver(model,numerics):
    """
    Collect all information and drive the simulation 
    """

    da=PETSc.DMDA().create(dof=model.model.num_u,
                           sizes=(model.nodes,),
                           boundary_type=PETSc.DMDA.BoundaryType.NONE,
                           stencil_type=None,
                           stencil_width=1, 
                           comm=PETSc.COMM_WORLD 
                            ) 
    da.setUp()
    da.setMatType(PETSc.Mat.Type.AIJ)           # create a sparse matrix related to the grid created before (actually this is the type)
    da.createMatrix()                           # create the matrix
    u= da.createGlobalVec() # create a structured array related to the grid created before 

    # create a time-stepping solver object 
    ts = PETSc.TS().create(PETSc.COMM_WORLD)

    # set type of problem
    ts.setProblemType(ts.ProblemType.NONLINEAR)
    
    # set type to BDF method
    ts.setType(ts.Type.BDF)

    # pass the residual function to "ts" so it can use it
    ts.setIFunction(model.model.residual,args=[model])

    # attach the ts solver to the grid created
    ts.setDM(da)

    # maximum steps allowed to take to reach final time
    ts.setMaxSteps(numerics.max_time_steps)

    # maximum steps rejections allowed , this is related to snes functions
    ts.setMaxStepRejections(numerics.max_step_rejections)
    ts.setMaxSNESFailures(numerics.max_snes_failures)

    # set tolerance
    ts.setTolerances(numerics.r_tol,numerics.a_tol)

    # set initial time
    ts.setTime(model.time[0])

    # this option is to match the output time of ts with the output time defined by user.
    ts.setExactFinalTime(ts.ExactFinalTime.MATCHSTEP)

    # get the snes solver
    snes=ts.getSNES()
    snes.setFromOptions()
    
    # get the ksp solver
    ksp=snes.getKSP()
    ksp.setFromOptions()
    
    # get the pc object from ksp
    pc=ksp.getPC()
    pc.setFromOptions()
    
    model.initial_conditions(ts,u)

    for t in model.time:
        ts.setMaxTime(t)
        ts.solve(u)
        U = da.getVecArray(u)


    #print solvers parameters
    ts.view()  

