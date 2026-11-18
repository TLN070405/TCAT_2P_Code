"""PETSc solver settings shared by the test-case drivers."""
from petsc4py import PETSc


def solver_options(max_time_steps, bdf_order, pc_type, monitor=True):
    """
    Set the PETSc options used by the drivers.

    max_time_steps: maximum number of time steps (ts_max_steps)
    bdf_order:      order of the BDF time integrator (ts_bdf_order)
    pc_type:        preconditioner, e.g. 'sor' or 'ilu' (pc_type)
    monitor:        print 'TS dt ... time ...' after every step (ts_monitor)

    Any other PETSc option can be set the same way with
    PETSc.Options().setValue(name, value) before calling setup_solver.
    """
    opts = PETSc.Options()

    # Time stepper: implicit BDF with adaptive step size
    opts.setValue('snes_fd_color', None)   # Jacobian by colored finite differences
    opts.setValue('ts_type', 'bdf')
    opts.setValue('ts_bdf_order', bdf_order)
    opts.setValue('ts_max_steps', max_time_steps)
    opts.setValue('ts_max_reject', 500)
    if monitor:
        opts.setValue('ts_monitor', None)
    else:
        opts.delValue('ts_monitor')

    # Nonlinear solver
    opts.setValue('ts_max_snes_failures', 10000)

    # Time-step error control
    opts.setValue('ts_rtol', 1.e-8)
    opts.setValue('ts_atol', 1.e-8)

    # Preconditioner
    opts.setValue('pc_type', pc_type)
