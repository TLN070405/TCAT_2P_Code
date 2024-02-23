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
def cython_test():
  ans = 42
  return ans


cdef int Residual(ts,t,U,Udot,F):
  """
  """
  global Sum_Trap_0,Sum_Trap_L
  cdef unsigned int i
  cdef int Ndt
  Ndt=ts.getStepNumber() # get the number of the current time step (we use that in mass error, exactly to track the successful time steps). because inside the Residual function, the ts solver tries a lot of failed time steps so we need to be sure that we choose only the successful time steps and solutions to calculate qwL and qw0 and their integral over time
  da=ts.getDM() #get the DM structure 
  ####### we need to include those two lines so we can access the vector "u", if we don't include them we won't be able to access "u"
  localU=da.getLocalVec() 
  da.globalToLocal(U,localU) 
  u=da.getVecArray(localU)
  udot=da.getVecArray(Udot)
  f=da.getVecArray(F)
  #######
  if(Formulation=='SPP'):
        ###u[i][0]=Epsilon_w[i]
        ###u[i][1]=P_w[i]
        ###u[i][2]=P_n[i]
        SD.SD_SPP_PKR(PSKM,u)
        for i in range(1,Nz-1):
            A_SPP_11,A_SPP_12,A_SPP_21,A_SPP_23,Od_w,Od_n,Od_dp=SD.SD_SPP_A_Od(PSKM,u,i)
            ######### define the 3 residuals developed in the SPP formulation inside the domain
            f[i][0] = (tau/phi)* udot[i][0] - Od_dp
            f[i][1] = A_SPP_11 * udot[i][0] + A_SPP_12 * udot[i][1] - Od_w
            f[i][2] = A_SPP_21 * udot[i][0] + A_SPP_23 * udot[i][2] - Od_n
        #########define residual at boundaries    
        f[0][0] = u[0][0] - Ew_bc_0
        f[0][1] = u[0][1] - Pw_bc_0
        f[0][2] = u[0][2] - Pn_bc_0
        f[Nz-1][0] = u[Nz-1][0] - Ew_bc_L
        f[Nz-1][1] = u[Nz-1][1] - Pw_bc_L
        f[Nz-1][2] = u[Nz-1][2] - Pn_bc_L
        ######### Call Flux_Integration function to calculate the integral of qw0 and qwL over time #########
        if(Mass_Activation=='ON'):
          Sum_Trap_0,Sum_Trap_L=MC.Flux_Integration_SPP(ts,Ndt) # calculate the integral over time of qwL and qw0
  elif(Formulation=='SP'):
        ###u[i][0]=Epsilon_w[i]
        ###u[i][1]=P_w[i]
        SD.SD_SP_PKR(PSKM,u)
        for i in range(1,Nz-1):
            A_SP_11,A_SP_12,A_SP_21,A_SP_22,Od_w,Od_n=SD.SD_SP_A_Own(PSKM,u,i)
            ######### define the 3 residuals developed in the SPP formulation inside the domain
            f[i][0] = A_SP_11 * udot[i][0] + A_SP_12 * udot[i][1] - Od_w
            f[i][1] = A_SP_21 * udot[i][0] + A_SP_22 * udot[i][1] - Od_n
        #########define residual at boundaries    
        f[0][0] = u[0][0] - Ew_bc_0
        f[0][1] = u[0][1] - Pw_bc_0
        f[Nz-1][0] = u[Nz-1][0] - Ew_bc_L
        f[Nz-1][1] = u[Nz-1][1] - Pw_bc_L
        ######### Call Flux_Integration function to calculate the integral of qw0 and qwL over time #########
        if(Mass_Activation=='ON'):
          Sum_Trap_0,Sum_Trap_L=MC.Flux_Integration_SP(ts,Ndt) # calculate the integral over time of qwL and qw0
  elif(Formulation=='PP'):
        ###u[i][0]=P_w[i]
        ###u[i][1]=P_n[i]
        SD.SD_PP_PKR(PSKM,u)
        for i in range(1,Nz-1):
            A_PP_11,A_PP_12,A_PP_21,A_PP_22,Od_w,Od_n=SD.SD_PP_A_Own(PSKM,u,i)
            ######### define the 3 residuals developed in the SPP formulation inside the domain
            f[i][0] = A_PP_11 * udot[i][0] + A_PP_12 * udot[i][1] - Od_w
            f[i][1] = A_PP_21 * udot[i][0] + A_PP_22 * udot[i][1] - Od_n
        #########define residual at boundaries    
        f[0][0] = u[0][0] - Pw_bc_0
        f[0][1] = u[0][1] - Pn_bc_0
        f[Nz-1][0] = u[Nz-1][0] - Pw_bc_L
        f[Nz-1][1] = u[Nz-1][1] - Pn_bc_L
