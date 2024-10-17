"""solver.py"""

import numpy as np
import petsc4py
from petsc4py import PETSc

__all__ = ["solver"]


def solver(model):
    """
    Collect all information and drive the simulation
    """

    # PETSc arrays and grid

    da, u = _get_da(model)

    # PETSc time-stepper
    ts = _get_time_stepper(model, da)

    # PETSc non-linear solver
    snes = _get_nonlinear_solver(ts)

    # PETSc linear solver
    ksp = _get_linear_solver(snes)

    # PETSc preconditioner
    pc = _get_preconditioner(ksp)

    # Solve initial conditions (ic are not excplicitly expressed in terms of solution variabes e.g:Neumann BC)
    _solve_ic(da, u, model)

    # set the hysteresis function to be called after each successful time step.
    if model.hysteresis is True:
        ts.setPostStep(model.interface.model_update_hysteresis)

    # Loop through Error Tolerance (not time)
    iteration = 1
    max_steps = 100000
    error_threshold = 1e-6
    error = float("inf")
    u_old = u.getArray().copy()

    while error > error_threshold and ts.getStepNumber() < iteration < max_steps:
        ts.setMaxSteps(iteration)
        ts.solve(u)

        u_new = u.getArray().copy()
        error = np.linalg.norm( np.abs(u_new - u_old)/ts.getTimeStep())/len(u_new)
        print(error)
        u_old = u_new
        iteration += 1

    model.clean_solution(u.getArray(), ts.getTime())

    return model.solution


def _get_da(model):
    """
    Create a PETSc da object:
    da - A DM object that is used to manage data for a structured grid in 1, 2, or 3 dimensions.
    """
    da = PETSc.DMDA().create(
        dof=model.model.num_u,
        sizes=(model.nodes,),
        boundary_type=PETSc.DMDA.BoundaryType.NONE,
        stencil_type=None,
        stencil_width=1,
        comm=PETSc.COMM_WORLD,
    )

    da.setUp()
    da.setMatType(PETSc.Mat.Type.AIJ)
    da.createMatrix()
    u = da.createGlobalVec()

    return da, u


def _get_time_stepper(model, da):
    """
    Creatw PETSc time stepper and set options
    """
    ts = PETSc.TS().create(PETSc.COMM_WORLD)
    ts.setProblemType(ts.ProblemType.NONLINEAR)
    ts.setIFunction(model.model.residual, args=[model])
    ts.setDM(da)
    ts.setTime(0)
    ts.setExactFinalTime(ts.ExactFinalTime.MATCHSTEP)
    ts.setFromOptions()

    return ts


def _get_nonlinear_solver(ts):
    """
    Set PETSc nonlinear solver snes and set options
    """
    snes = ts.getSNES()
    snes.setFromOptions()

    return snes


def _get_linear_solver(snes):
    """
    Set PETSc linear solver ksp and set options
    """
    ksp = snes.getKSP()
    ksp.setFromOptions()

    return ksp


def _get_preconditioner(ksp):
    """
    Set PETSc preconditioner  and set options
    """
    pc = ksp.getPC()
    pc.setFromOptions()

    return pc


def _solve_ic(da, u, model):
    """
    Create nonlinear solver to solve initial and boundary conditions
    """
    # setup snes solver
    da_ic, u_ic = _get_da(model)
    r_ic = u_ic.duplicate()
    snes_ic = PETSc.SNES().create(PETSc.COMM_WORLD)
    snes_ic.setFunction(model.model.form_function, r_ic, args=[model])
    snes_ic.setDM(da_ic)

    # provide an initial estimation for the solution of initial conditions+ boundary conditions
    model.initial_conditions(snes_ic, u_ic)

    # solve the system to get initial conditions in terms of solution variables
    snes_ic.solve(None, u_ic)

    # copy solution to the ts array solution.
    U = da.getVecArray(u)
    U_ic = da_ic.getVecArray(u_ic)
    for n in range(0, model.nodes):
        U[n] = U_ic[n]
