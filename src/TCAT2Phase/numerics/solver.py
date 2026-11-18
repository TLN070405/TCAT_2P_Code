"""solver.py"""

import numpy as np
from petsc4py import PETSc
from . import model_types

__all__ = ["setup_solver", "solver_time", "solver_error"]

def solver_time(u,ts,model):
    """
    Collect all information and drive the simulation
    """
    # Loop through time
    for t in model.time:
        ts.setMaxTime(t)
        ts.solve(u)
        model.clean_solution(u.getArray(), t)

    return model.solution

def solver_error(u,ts,model):
    """
    Collect all information and drive the simulation
    """
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
        u_old = u_new
        iteration += 1

    model.clean_solution(u.getArray(), ts.getTime())

    return model.solution

def setup_solver(model):
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
    _get_preconditioner(ksp)

    # Solve initial conditions (ic are not excplicitly expressed in terms of solution variabes e.g:Neumann BC)
    _solve_ic(da, u, model)

    # Start an index-1 DAE (e.g. incompressible two-fluid flow) from consistent values
    if _needs_consistent_initialization(model):
        _consistent_initialization(model, da, u, ts)

    # set the hysteresis function to be called after each successful time step.
    if model.hysteresis is True:
        ts.setPostStep(model.interface.model_update_hysteresis)

    return u, ts

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
    residual_class = model_types.get_residual_class(model)
    residual_instance = residual_class(model)
    ts.setIFunction(residual_instance.compute_residual)
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
    model.pass_initial_conditions(snes_ic, u_ic)

    # solve the system to get initial conditions in terms of solution variables
    snes_ic.solve(None, u_ic)

    # copy solution to the ts array solution.
    U = da.getVecArray(u)
    U_ic = da_ic.getVecArray(u_ic)
    for n in range(0, model.nodes):
        U[n] = U_ic[n]


def _needs_consistent_initialization(model):
    """
    True when the system is an index-1 DAE whose initial conditions need not
    satisfy its algebraic constraint: the two-phase model with both fluids
    incompressible. Then one pressure (the total-flow pressure) has no time
    derivative in any equation and is fixed instantaneously by the boundary
    conditions; for example, Dirichlet saturation at one end and no flow at the
    other gives a pressure that jumps at t = 0+. BDF sees that jump as a local
    truncation error that does not shrink with the step size, and rejects every
    step. model.consistent_initialization = True/False overrides the check.
    """
    flag = getattr(model, "consistent_initialization", None)
    if flag is not None:
        return bool(flag)
    if not model.model.__class__.__name__ == "TwoPhase":
        return False
    p0 = 0.0
    try:
        return (model.w_phase.d_density_function(p0) == 0.0 and
                model.n_phase.d_density_function(p0) == 0.0)
    except Exception:
        return False


def _consistent_initialization(model, da, u, ts, fraction=1.0e-8):
    """
    Project the initial state onto the algebraic constraint with one backward-
    Euler step of size dt0 = fraction * (first output time). Over such a step the
    differential unknowns (saturation) change by O(dt0) while the algebraic ones
    take their consistent values. The main time stepper then starts at t = dt0.
    The step uses its own TS (options prefix 'ci_') so the BDF settings do not
    apply; the Jacobian is by colored finite differences and the linear solve
    is a direct LU, as for the small 1D systems here.
    """
    t_first = float(np.min(model.time)) if model.time is not None else 1.0
    dt0 = fraction * t_first

    opts = PETSc.Options()
    for key, val in [("ci_snes_fd_color", None), ("ci_ksp_type", "preonly"),
                     ("ci_pc_type", "lu"), ("ci_snes_max_it", 100),
                     ("ci_ts_adapt_type", "none")]:
        if not opts.hasName(key):
            opts.setValue(key, val)

    ts0 = PETSc.TS().create(PETSc.COMM_WORLD)
    ts0.setOptionsPrefix("ci_")
    ts0.setProblemType(ts0.ProblemType.NONLINEAR)
    residual_instance = model_types.get_residual_class(model)(model)
    ts0.setIFunction(residual_instance.compute_residual)
    da0, u0 = _get_da(model)   # own DM: TS callbacks are stored on the DM
    u.copy(u0)
    ts0.setDM(da0)
    ts0.setType(PETSc.TS.Type.BEULER)
    ts0.setTime(0.0)
    ts0.setTimeStep(dt0)
    ts0.setMaxTime(dt0)
    ts0.setMaxSteps(1)
    ts0.setExactFinalTime(ts0.ExactFinalTime.MATCHSTEP)
    ts0.setFromOptions()
    ts0.solve(u0)
    if ts0.getConvergedReason() < 0:
        raise RuntimeError("Consistent initialization step failed "
                           f"(TS reason {ts0.getConvergedReason()})")
    u0.copy(u)
    ts.setTime(ts0.getTime())
