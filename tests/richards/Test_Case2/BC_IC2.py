import sys
import petsc4py
petsc4py.init(sys.argv)
from petsc4py import PETSc
########################################################
#########################BC and IC######################
########################################################
########################################################  
def BC_IC(inp,SE,VPw_bc):
	for i in range(0,inp.Nx):
	    if(i==0):
	      VPw_bc[i]        = -1000
	    elif(i==inp.Nx-1):
	      VPw_bc[i]        = -1000
	    else:
	      VPw_bc[i]        = -1000 

