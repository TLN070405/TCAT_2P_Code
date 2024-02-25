# cython: profile = True
# cython: linetrace = True
# cython: boundscheck = True
# cython: wraparound = False
# cython: cdivision = True

import cython
import sys
import petsc4py
import time
import cython
#petsc4py.init(sys.argv)
from petsc4py import PETSc
from TCAT2Phase.Discretizations import Spacial_Discr

def two_phase_residual(ts,t,U,Udot,F):
    """
    Residual function for two-phase sp formulation
    """

    da = ts.getDM()
    localU = da.getLocalVec() 
    da.globalToLocal(U,localU) 
    
    # Collect solution variables and residual 
    u = da.getVecArray(localU)
    udot = da.getVecArray(Udot)
    f = da.getVecArray(F)

    Spacial_Discr.two_sphase_residual_internal(f,u,udot,formulation,discretization,parameters)


    #########define residual at boundaries    
    f[0][0] = u[0][0] - Ew_bc_0
    f[0][1] = u[0][1] - Pw_bc_0
    f[Nz-1][0] = u[Nz-1][0] - Ew_bc_L
    f[Nz-1][1] = u[Nz-1][1] - Pw_bc_L

