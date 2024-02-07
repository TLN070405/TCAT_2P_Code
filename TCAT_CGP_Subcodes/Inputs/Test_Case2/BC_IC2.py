import sys
import petsc4py
petsc4py.init(sys.argv)
from petsc4py import PETSc
########################################################
#########################BC and IC######################
########################################################
########################################################  
def BC_IC(inp,SE,VEw_bc,VPw_bc,VEwn_bc,VJwn_bc,VWxwn_bc,VKwn_bc,Ve_wn_eq):
	Nx=inp.Nx
	Cx=inp.Cx
	dx=inp.dx
	Omega_0=inp.Omega_0
	Gamma_wn=inp.Gamma_wn
	for i in range(0,Nx):
	    x = Omega_0+i*dx
	    if(i==0):
	      VPw_bc[i]        = -1000
	      VEw_bc[i]        = 0.3321
	      VEwn_bc[i]       = 100
	      VJwn_bc[i]       = -VPw_bc[i]/Gamma_wn
	      Ve_wn_eq[i]      = 100
	    elif(i==Nx-1):
	      VPw_bc[i]        = -1000
	      VEw_bc[i]        = 0.3321
	      VEwn_bc[i]       = 100
	      VJwn_bc[i]       = -VPw_bc[i]/Gamma_wn
	      Ve_wn_eq[i]      = 100 
	    else:
	      VPw_bc[i]        = -1000 
	      VEw_bc[i]        = 0.3321
	      VEwn_bc[i]       = 100
	      VJwn_bc[i]       = -VPw_bc[i]/Gamma_wn 
	      Ve_wn_eq[i]      = 100 
        ######### We get wx and kwn out of the first loop because we need dew_dx which requires ew to be filled completely first.
	for i in range(0,Nx):
	    e_ns,e_ws,js_ns,kn_ns,b0D1,b1D2,b2D3=SE.Linear_Geometric_Constants(inp,VEw_bc[i])
	    if(i==0):
	      dew_dx           =Cx*(VEw_bc[i+1]-VEw_bc[i])
	      ##### 
	      VWxwn_bc[i]      =SE.Wx_wn_IC_func(inp,dew_dx,Ve_wn_eq[i], VPw_bc[i] , VEwn_bc[i] ,  VJwn_bc[i]) 
	      VKwn_bc[i]       =SE.Kwn_Linear_Geometric(inp,VEw_bc[i], VEwn_bc[i] , VJwn_bc[i] , b0D1, b1D2 , b2D3, e_ns , js_ns, kn_ns)
	    elif(i==Nx-1):
	      dew_dx           =Cx*(VEw_bc[i]-VEw_bc[i-1])
	      #####
	      VWxwn_bc[i]      =SE.Wx_wn_IC_func(inp,dew_dx,Ve_wn_eq[i], VPw_bc[i] , VEwn_bc[i] ,  VJwn_bc[i] ) 
	      VKwn_bc[i]       =SE.Kwn_Linear_Geometric(inp,VEw_bc[i], VEwn_bc[i]  ,VJwn_bc[i],  b0D1, b1D2, b2D3,  e_ns,  js_ns, kn_ns)
	    else: 
	      dew_dx           =Cx*(VEw_bc[i]-VEw_bc[i-1])
	      #####
	      VWxwn_bc[i]      =SE.Wx_wn_IC_func(inp,dew_dx,Ve_wn_eq[i], VPw_bc[i] , VEwn_bc[i] ,  VJwn_bc[i] ) 
	      VKwn_bc[i]       =SE.Kwn_Linear_Geometric(inp,VEw_bc[i], VEwn_bc[i]  ,VJwn_bc[i],  b0D1, b1D2, b2D3,  e_ns,  js_ns, kn_ns)

            
            
