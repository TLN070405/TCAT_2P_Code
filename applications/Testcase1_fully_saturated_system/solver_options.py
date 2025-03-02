"""set the petsc solver options"""

from petsc4py import PETSc

def solver_options(max_time_steps,bdf_order,pc_type):
    # Numerical Parameters
    OptDB = PETSc.Options()

    # Time-Stepper Options
    OptDB.setValue('snes_fd_color', None)
    OptDB.setValue('ts_type', 'bdf')
    OptDB.setValue('ts_bdf_order', bdf_order)
    OptDB.setValue('ts_max_steps',max_time_steps)
    OptDB.setValue('ts_max_reject',500)
    OptDB.setValue('ts_monitor', None)

    # Nonlinear Solver Options
    OptDB.setValue('ts_max_snes_failures',10000)

    # Linear Solver Options
    OptDB.setValue('ts_rtol', 1.e-8)
    OptDB.setValue('ts_atol', 1.e-8)
    # Preconditioner Options
    OptDB.setValue('pc_type', pc_type)
