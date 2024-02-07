import sys
import petsc4py
petsc4py.init(sys.argv)
from petsc4py import PETSc
########################################################
#########################BC and IC######################
########################################################
########################################################  
def BC_IC(inp,SE,VPw_bc):
	dx      = inp.dx
	Omega_0 = inp.Omega_0
	rho_w0  = inp.rho_w0
	g       = inp.g
	for i in range(0,inp.Nx):
	    x = Omega_0+i*dx
	    if(i==0):
	      VPw_bc[i]        = -1000-rho_w0*g*x 
	    elif(i==inp.Nx-1):
	      VPw_bc[i]        = -1000-rho_w0*g*x
	    else:
	      VPw_bc[i]        = -1000-rho_w0*g*x

