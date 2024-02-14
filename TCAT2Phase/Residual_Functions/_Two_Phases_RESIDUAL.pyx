import cython

def cython_test():
  ans = 42
  return ans

# ############################## call necessary libraries ##############################
# import sys # import system so we can initialize petsc4py 
# import petsc4py
# import time
# import cython
# petsc4py.init(sys.argv)
# from petsc4py import PETSc
# #####################################################################################################################   
# ####################################Import our private modules#######################################################
# #####################################################################################################################
# OptDB = PETSc.Options()
# Formulation = OptDB.getString('Formulation', 'SPP')     # get type of formulation used SPP, SP, PP
# Mass_Activation = OptDB.getString('Mass_Activation', 'ON')  # ON or OFF mass calculations
# Paper = OptDB.getString('Paper', 'Helmig') # get paper name from terminal, Helmig paper is default if no paper is precised
# sys.path.insert(0, './2P_Subcodes/Import_Modules')
# import Import_Modules
# inp,PSKM,MC,IC,ER,EI,CA,SD=Import_Modules.ImportModules(Paper)
# #####################################################################################################################   
# ################the parameters needed for the problem, the values are extracted from input.py########################
# #####################################################################################################################
# Nz,t0,time_interval,Omega_0,Omega_L,rtol,atol,MaxStepRejections,MaxSNESFailures,MaxTimeSteps,phi,g,d,tau,K,mu_w,Ew_bc_0,Ew_bc_L,Pw_bc_0,Pw_bc_L,Pn_bc_0,Pn_bc_L,Ew_IC,Pw_IC,Pn_IC,dz ,Cz,k_mun_half,k_muw_half,filenumber=EI.ExtractInputs(inp,OptDB)
# #####################################################################################################################   
# ############################################## Numpy arrays #########################################################
# #####################################################################################################################
# Mw,Sum_rho_Epsilonw,Integral_qw_L,Integral_qw_0,Error_Mass,Vp_w_t0,VEpsilon_w_t0,VEpsilonw,VPw,VPn,Clock_Array=CA.CreateArrays(Nz,time_interval)
# #####################################################################################################################   
# ########################################## Residual  function #######################################################
# #####################################################################################################################
# cdef int Residual(ts,t,U,Udot,F):
# # residual function is like sundials residual, the prototype is fixed but the name of the function can be changed

#     global Sum_Trap_0,Sum_Trap_L
#     cdef unsigned int i
#     cdef int Ndt
#     Ndt=ts.getStepNumber() # get the number of the current time step (we use that in mass error, exactly to track the successful time steps). because inside the Residual function, the ts solver tries a lot of failed time steps so we need to be sure that we choose only the successful time steps and solutions to calculate qwL and qw0 and their integral over time
#     da=ts.getDM() #get the DM structure 
#     ####### we need to include those two lines so we can access the vector "u", if we don't include them we won't be able to access "u"
#     localU=da.getLocalVec() 
#     da.globalToLocal(U,localU) 
#     u=da.getVecArray(localU)
#     udot=da.getVecArray(Udot)
#     f=da.getVecArray(F)
#     #######
#     if(Formulation=='SPP'):
#         ###u[i][0]=Epsilon_w[i]
#         ###u[i][1]=P_w[i]
#         ###u[i][2]=P_n[i]
#         SD.SD_SPP_PKR(PSKM,u)
#         for i in range(1,Nz-1):
#             A_SPP_11,A_SPP_12,A_SPP_21,A_SPP_23,Od_w,Od_n,Od_dp=SD.SD_SPP_A_Od(PSKM,u,i)
#             ######### define the 3 residuals developed in the SPP formulation inside the domain
#             f[i][0] = (tau/phi)* udot[i][0] - Od_dp
#             f[i][1] = A_SPP_11 * udot[i][0] + A_SPP_12 * udot[i][1] - Od_w
#             f[i][2] = A_SPP_21 * udot[i][0] + A_SPP_23 * udot[i][2] - Od_n
#         #########define residual at boundaries    
#         f[0][0] = u[0][0] - Ew_bc_0
#         f[0][1] = u[0][1] - Pw_bc_0
#         f[0][2] = u[0][2] - Pn_bc_0
#         f[Nz-1][0] = u[Nz-1][0] - Ew_bc_L
#         f[Nz-1][1] = u[Nz-1][1] - Pw_bc_L
#         f[Nz-1][2] = u[Nz-1][2] - Pn_bc_L
#         ######### Call Flux_Integration function to calculate the integral of qw0 and qwL over time #########
#         if(Mass_Activation=='ON'):
#           Sum_Trap_0,Sum_Trap_L=MC.Flux_Integration_SPP(ts,Ndt) # calculate the integral over time of qwL and qw0
#     elif(Formulation=='SP'):
#         ###u[i][0]=Epsilon_w[i]
#         ###u[i][1]=P_w[i]
#         SD.SD_SP_PKR(PSKM,u)
#         for i in range(1,Nz-1):
#             A_SP_11,A_SP_12,A_SP_21,A_SP_22,Od_w,Od_n=SD.SD_SP_A_Own(PSKM,u,i)
#             ######### define the 3 residuals developed in the SPP formulation inside the domain
#             f[i][0] = A_SP_11 * udot[i][0] + A_SP_12 * udot[i][1] - Od_w
#             f[i][1] = A_SP_21 * udot[i][0] + A_SP_22 * udot[i][1] - Od_n
#         #########define residual at boundaries    
#         f[0][0] = u[0][0] - Ew_bc_0
#         f[0][1] = u[0][1] - Pw_bc_0
#         f[Nz-1][0] = u[Nz-1][0] - Ew_bc_L
#         f[Nz-1][1] = u[Nz-1][1] - Pw_bc_L
#         ######### Call Flux_Integration function to calculate the integral of qw0 and qwL over time #########
#         if(Mass_Activation=='ON'):
#           Sum_Trap_0,Sum_Trap_L=MC.Flux_Integration_SP(ts,Ndt) # calculate the integral over time of qwL and qw0
#     elif(Formulation=='PP'):
#         ###u[i][0]=P_w[i]
#         ###u[i][1]=P_n[i]
#         SD.SD_PP_PKR(PSKM,u)
#         for i in range(1,Nz-1):
#             A_PP_11,A_PP_12,A_PP_21,A_PP_22,Od_w,Od_n=SD.SD_PP_A_Own(PSKM,u,i)
#             ######### define the 3 residuals developed in the SPP formulation inside the domain
#             f[i][0] = A_PP_11 * udot[i][0] + A_PP_12 * udot[i][1] - Od_w
#             f[i][1] = A_PP_21 * udot[i][0] + A_PP_22 * udot[i][1] - Od_n
#         #########define residual at boundaries    
#         f[0][0] = u[0][0] - Pw_bc_0
#         f[0][1] = u[0][1] - Pn_bc_0
#         f[Nz-1][0] = u[Nz-1][0] - Pw_bc_L
#         f[Nz-1][1] = u[Nz-1][1] - Pn_bc_L
#         ######### Call Flux_Integration function to calculate the integral of qw0 and qwL over time #########
#         if(Mass_Activation=='ON'):
#           Sum_Trap_0,Sum_Trap_L=MC.Flux_Integration_PP(ts,Ndt) # calculate the integral over time of qwL and qw0

# #####################################################################################################################   
# ##########################################         Main code       ##################################################
# #####################################################################################################################
# cdef void main():
#     #####################################################################################################################   
#     #####################    setup the ts, snes, ksp and pc solvers before solving the problem  #########################
#     #####################################################################################################################
#     # here we created a 1D grid, to let the solver store the jacobian as a sparse matrix (this is one of the ways to do that) 
#     cdef unsigned int i,k
#     bz= PETSc.DMDA.BoundaryType.NONE
#     if(Formulation=='SPP'): 
#       dof=3
#     elif(Formulation=='SP' or Formulation=='PP'):  
#       dof=2
#     da=PETSc.DMDA().create(dof= dof,sizes=(Nz, ),boundary_type=bz,stencil_type=None,stencil_width=1, comm=PETSc.COMM_WORLD ) 
#     da.setUp()
#     u= da.createGlobalVec() # create a structured array related to the grid created before 
#     da.setMatType(PETSc.Mat.Type.AIJ)           # create a sparse matrix related to the grid created before (actually this is the type)
#     da.createMatrix()                           # create the matrix
#     ts = PETSc.TS().create(PETSc.COMM_WORLD)    # create a time-stepping solver object 
#     ts.setProblemType(ts.ProblemType.NONLINEAR) # set type of problem 
#     ts.setType(ts.Type.BDF)                     # set type to BDF method	
#     ts.setIFunction(Residual)                   # pass the residual function to "ts" so it can use it  
#     ts.setDM(da)                                # attach the ts solver to the grid created    
#     snes=ts.getSNES()                           # get the snes solver
#     snes.setFromOptions()                       # give the user the possibility to personalize the "snes" at running time    
#     ksp=snes.getKSP()                           # get the ksp solver
#     ksp.setFromOptions()                        # give the user the possibility to personalize the "ksp" at running time
#     pc=ksp.getPC()                              # get the pc object from ksp
#     pc.setFromOptions()                         # give the user the possibility to personalize the "pc" at running time
#     ts.setMaxSteps(MaxTimeSteps)                # maximum steps allowed to take to reach final time
#     ts.setMaxStepRejections(MaxStepRejections)  # maximum steps rejections allowed , this is related to snes functions         
#     ts.setMaxSNESFailures(MaxSNESFailures)
#     ts.setTolerances(rtol, atol)                # set tolerances of the  chosen ts (here bdf )
#     ts.setTime(t0)                              # set initial time t0 (here t0=0)
#     ts.setExactFinalTime(ts.ExactFinalTime.MATCHSTEP)  # this option is to match the output time of ts with the output time defined by user.
#     ts.setFromOptions()                         # give the user the possibility to personalize the ts solver at running time
#     #####################################################################################################################   
#     ########################### initialize the solution vector and calculate initial mass ###############################
#     #####################################################################################################################
#     IC.Initial_Conditions(ts,u)                                  # initialization of solution vector (initial conditions)
#     U=da.getVecArray(u)                                          # get the solution vector right after initialization 
#     Mw0=MC.Initial_Mass_Inside_Domain(U,Vp_w_t0,VEpsilon_w_t0)   # calculate the mass inside domain at t=t0 and Ew(t0) and Pw(t0)
#     #####################################################################################################################   
#     ################################# call the ts solver and start solving the problem ##################################
#     #####################################################################################################################
#     for i in range(len(time_interval)): # loop over output times
#        ts.setMaxTime(time_interval[i])  # set the max time so the ts solver will return the solution each time it reach the values of time_interval
#        CPU_start = time.process_time()  # time before solving
#        ts.solve(u)                      # ts starts solving the problem "u: is the solution vector"
#        CPU_end = time.process_time()    # time after solving
#        Clock_Array[i]=CPU_end-CPU_start # CPU time is difference between start and end
#        U=da.getVecArray(u)              # get the solution vector values (we need to use getVecArray function to access the values of 'u'
#        if(Mass_Activation=='ON'):
#        ##### store the integral over time of qw0 and qwL (to be used later in error mass)
#          Integral_qw_L[i]=Sum_Trap_L
#          Integral_qw_0[i]=Sum_Trap_0 
#        ##### store the solution vector at each time_interval value
#        if(Formulation=='SPP'):     
#           for k in range(0,Nz):
#               VEpsilonw[i][k]=U[k][0]
#               VPw[i][k]=U[k][1]
#               VPn[i][k]=U[k][2]
#        elif(Formulation=='SP'): 
#           for k in range(0,Nz):
#               VEpsilonw[i][k]=U[k][0]
#               VPw[i][k]=U[k][1]
#               VPn[i][k]=PSKM.Pc(VEpsilonw[i][k]) + VPw[i][k]
#        elif(Formulation=='PP'): 
#           for k in range(0,Nz):
#               VPw[i][k]=U[k][0]
#               VPn[i][k]=U[k][1]
#               VEpsilonw[i][k]=PSKM.Epsilon_w_of_Pc(VPn[i][k]-VPw[i][k])
#        ##### this if condition is just to avoid looping when the MaxTimeSteps is reached    
#        NTimeSteps=ts.getStepNumber() 
#        if(NTimeSteps==MaxTimeSteps): 
#          break

#     #####################################################################################################################   
#     ###################################### Mass accumulation and mass balance error #############################################
#     #####################################################################################################################  
#     if(Mass_Activation=='ON'):
#       MC.Mass_Accumulation(Mw,VEpsilonw,VPw,Nz,Mw0) # calculate mass accumulation (in fact, this quantity has been plotted only in Helmig) 
#       MC.Mass_inside_domain(Sum_rho_Epsilonw,VEpsilonw,VPw,Vp_w_t0,VEpsilon_w_t0)  # intermediate step to calculate mass error
#       MC.Error_Mass_Function(Error_Mass,Sum_rho_Epsilonw,Integral_qw_L,Integral_qw_0) # error mass calculation (1-(sum/integral(qw0+qwL)))
#     #####################################################################################################################   
#     ###################################### export results to external files #############################################
#     #####################################################################################################################
#     ER.Export_Outputs(Mw0,Error_Mass,Paper,filenumber,Nz,Omega_0,Omega_L,dz,tau,K,time_interval,Clock_Array,Mw,VEpsilonw)  
    
#     ts.view()  ## print the ts,snes,ksp,pc solvers parameters on screen       
# #######################################
 
# main() 
