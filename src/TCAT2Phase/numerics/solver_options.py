"""PETSc solver settings shared by the test-case drivers."""
from petsc4py import PETSc


def solver_options(max_time_steps, bdf_order, pc_type, monitor=True, rtol=1.e-6, atol=1.e-6):
    """
    Set the PETSc options used by the drivers.

    max_time_steps: maximum number of time steps (ts_max_steps)
    bdf_order:      order of the BDF time integrator (ts_bdf_order)
    pc_type:        preconditioner, e.g. 'sor' or 'ilu' (pc_type)
    monitor:        print 'TS dt ... time ...' after every step (ts_monitor)
    rtol, atol:     local truncation error tolerances for the adaptive time step
                    (ts_rtol, ts_atol). 1e-6 by default. The published runs used
                    1e-8, which takes about 13 times as many steps for a change of
                    about 3e-5 relative in pressure (Test Case 8 RE, first 5 s);
                    the *_defense.py drivers pass 1e-8 to reproduce the slides.

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
    opts.setValue('ts_rtol', rtol)
    opts.setValue('ts_atol', atol)

    # Preconditioner
    opts.setValue('pc_type', pc_type)
