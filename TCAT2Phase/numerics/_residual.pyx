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
from petsc4py import PETSc
from TCAT2Phase.numerics import discretization

def two_phase_residual(ts,t,U,Udot,F,model,parameters):
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

    nodes = model.nodes

    model.two_sphase_residual_internal(f,u,udot,model)

    #Residual at Boundary 1   
    f[0][0] = u[0][0] - model.boundary_conditions[0,0]
    f[0][1] = u[0][1] - model.boundary_conditions[0,1]

    # Residual at Boundary 2
    f[nodes-1][0] = u[nodes-1][0] - model.boundary_conditions[1,0]
    f[nodes-1][1] = u[nodes-1][1] - model.boundary_conditions[1,1]

