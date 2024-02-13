import sys
import petsc4py
petsc4py.init(sys.argv)
from petsc4py import PETSc
########################################################################################
OptDB = PETSc.Options() # OPtDB is used so we can precise the paper used from terminal at running time.
Formulation = OptDB.getString('Formulation', 'SPP')
Paper = OptDB.getString('Paper', 'Helmig')
if(Paper=='Fucik'):
   sys.path.insert(0, '../inputs/inputs_Fucik')
   import inputs as inp # import the inupts.py file as a module so we can access directly the parameters defined inside it
   sys.path.insert(0, '../PSK_Models')
   import Brooks_Corey_Model as PSKM
elif( Paper=='Helmig'):
   sys.path.insert(0, '../inputs/inputs_Helmig')  
   import inputs as inp # import the inupts.py file as a module so we can access directly the parameters defined inside it
   sys.path.insert(0, '../PSK_Models')
   import Van_Genuchten_Model as PSKM  
elif( Paper=='Malgorzata'):
   sys.path.insert(0, '../inputs/inputs_Malgorzata')  
   import inputs as inp # import the inupts.py file as a module so we can access directly the parameters defined inside it
   sys.path.insert(0, '../PSK_Models')
   import Van_Genuchten_Model as PSKM  
########################################################################################
counter=0
Sum_Trap_L=0.0
Sum_Trap_0=0.0
Lx=inp.Lx # length in x direction (used only for mass accumulation, it has no effect on error_mass_balance)
Ly=inp.Ly # length in y direction (used only for mass accumulation, it has no effect on error_mass_balance)
Nz=inp.Nz
Cz=inp.Cz
K=inp.K
mu_w=inp.mu_w 
dz=inp.dz
g=inp.g
########################################################################################
def Initial_Mass_Inside_Domain(U,Vp_w_t0,VEpsilon_w_t0): 
    Mw0=0.0
    if(Formulation=='SPP' or Formulation=='SP'): 
        for k in range(0,Nz):
            VEpsilon_w_t0[k]=U[k][0] # Store Ew at t=t0 inside "VEpsilon_w_t0" 
            Vp_w_t0[k]= U[k][1]      # Store Pw at t=t0 inside "Vp_w_t0" 
        for k in range(0,Nz-1):
            Mw0=Mw0+(0.5*(PSKM.rhow_of_Pw(Vp_w_t0[k])+PSKM.rhow_of_Pw(Vp_w_t0[k+1])))*(0.5*(VEpsilon_w_t0[k]+VEpsilon_w_t0[k+1]))*(dz*Lx*Ly)   #calculate mass inside domain at t=t0
    elif(Formulation=='PP'): 
        for k in range(0,Nz):
            VEpsilon_w_t0[k]=PSKM.Epsilon_w_of_Pc(U[k][1]-U[k][0]) # Store Ew at t=t0 inside "VEpsilon_w_t0"
            Vp_w_t0[k]= U[k][0]                                    # Store Pw at t=t0 inside "Vp_w_t0" 
        for k in range(0,Nz-1):
            Mw0=Mw0+(0.5*(PSKM.rhow_of_Pw(Vp_w_t0[k])+PSKM.rhow_of_Pw(Vp_w_t0[k+1])))*(0.5*(VEpsilon_w_t0[k]+VEpsilon_w_t0[k+1]))*(dz*Lx*Ly)   #calculate mass inside domain at t=t0

            
            
    return Mw0
########################################################################################
def Mass_Accumulation(Mw,VEpsilonw,VPw,Nz,Mw0):
    #this function is to calculate mass accumulation 
    # Mw=Rho_w*Epsilon_w*(volume of domain (m3)); (it can be seen also as the integral of Ew over space)
    # we multiply by Lx*Ly because the depth and width of domain are 1 and 0.1 (in HELMIG PAPER)
    # we used trapezoidal method to calculate Mw ; integral(Mw)=(0.5*Rhow[K+1/2])*(0.5*Ew[K+1/2])*(Volume of domain)
    for i in range(len(Mw)):
        for k in range(0,Nz-1):
            Mw[i]=Mw[i]+(0.5*(PSKM.rhow_of_Pw(VPw[i][k])+PSKM.rhow_of_Pw(VPw[i][k+1])))*(0.5*(VEpsilonw[i][k]+VEpsilonw[i][k+1]))*(dz*Lx*Ly) 
        Mw[i]=Mw[i]-Mw0
########################################################################################
# "Flux_Integration" function calculates only the integral of qw0 and qwL over time (t0 to t_final)
def Flux_Integration_SPP(ts,Ndt):
    global counter,Sum_Trap_L,Sum_Trap_0,qw_L_t_old,qw_0_t_old,dt_old
    if(counter==0):
       Solution=ts.getSolution()   # here we get the solution after ts finds the correct dt
       qw_L_t_old=PSKM.Flux_qw(Solution[3*(Nz-2)],Solution[3*(Nz-2)+1],Solution[3*(Nz-3)+1],-1)
       qw_0_t_old=PSKM.Flux_qw(Solution[3],Solution[4],Solution[7],1)
       counter=counter+1
    if(Ndt == counter):
      # here we keep calculating qw0 and qwL each time the time changes and dt changes
      if (Ndt==1):
         dt_old=ts.getTime()
      Solution=ts.getSolution() 
      dt=ts.getTimeStep()
      qw_L_t=PSKM.Flux_qw(Solution[3*(Nz-2)],Solution[3*(Nz-2)+1],Solution[3*(Nz-3)+1],-1)
      qw_0_t=PSKM.Flux_qw(Solution[3],Solution[4],Solution[7],1)
      Sum_Trap_L=Sum_Trap_L+((qw_L_t+qw_L_t_old)*dt_old)/2.0 #Sum_Trap_L is the integral over time using trap method of qwL
      Sum_Trap_0=Sum_Trap_0+((qw_0_t+qw_0_t_old)*dt_old)/2.0 #Sum_Trap_0 is the integral over time using trap method of qw0
      qw_L_t_old=qw_L_t
      qw_0_t_old=qw_0_t
      dt_old=dt
      counter=counter+1
    return Sum_Trap_0,Sum_Trap_L
########################################################################################
# "Flux_Integration" function calculates only the integral of qw0 and qwL over time (t0 to t_final)
def Flux_Integration_SP(ts,Ndt):
    global counter,Sum_Trap_L,Sum_Trap_0,qw_L_t_old,qw_0_t_old,dt_old
    if(counter==0):
       Solution=ts.getSolution()   #here we get the solution after ts finds the correct dt
       qw_L_t_old=PSKM.Flux_qw(Solution[2*(Nz-2)],Solution[2*(Nz-2)+1],Solution[2*(Nz-3)+1],-1)
       qw_0_t_old=PSKM.Flux_qw(Solution[2],Solution[3],Solution[5],1)
       counter=counter+1
    if(Ndt == counter):
      # here we keep calculating qw0 and qwL each time the time changes and dt changes
      if (Ndt==1):
         dt_old=ts.getTime()
      Solution=ts.getSolution() 
      dt=ts.getTimeStep()
      qw_L_t=PSKM.Flux_qw(Solution[2*(Nz-2)],Solution[2*(Nz-2)+1],Solution[2*(Nz-3)+1],-1)
      qw_0_t=PSKM.Flux_qw(Solution[2],Solution[3],Solution[5],1)
      Sum_Trap_L=Sum_Trap_L+((qw_L_t+qw_L_t_old)*dt_old)/2.0 #Sum_Trap_L is the integral over time using trap method of qwL
      Sum_Trap_0=Sum_Trap_0+((qw_0_t+qw_0_t_old)*dt_old)/2.0 #Sum_Trap_0 is the integral over time using trap method of qw0
      qw_L_t_old=qw_L_t
      qw_0_t_old=qw_0_t
      dt_old=dt
      counter=counter+1
    return Sum_Trap_0,Sum_Trap_L
########################################################################################
# "Flux_Integration" function calculates only the integral of qw0 and qwL over time (t0 to t_final)
def Flux_Integration_PP(ts,Ndt):
    global counter,Sum_Trap_L,Sum_Trap_0,qw_L_t_old,qw_0_t_old,dt_old
    if(counter==0):
       Solution=ts.getSolution()   #here we get the solution after ts finds the correct dt
       qw_L_t_old=PSKM.Flux_qw(PSKM.Epsilon_w_of_Pc(Solution[2*(Nz-2)+1]-Solution[2*(Nz-2)]),Solution[2*(Nz-2)],Solution[2*(Nz-3)],-1)
       qw_0_t_old=PSKM.Flux_qw(PSKM.Epsilon_w_of_Pc(Solution[3]-Solution[2]),Solution[2],Solution[4],1)
       
       counter=counter+1
    if(Ndt == counter):
      # here we keep calculating qw0 and qwL each time the time changes and dt changes
      if (Ndt==1):
         dt_old=ts.getTime()
      Solution=ts.getSolution() 
      dt=ts.getTimeStep()
      qw_L_t=PSKM.Flux_qw(PSKM.Epsilon_w_of_Pc(Solution[2*(Nz-2)+1]-Solution[2*(Nz-2)]),Solution[2*(Nz-2)],Solution[2*(Nz-3)],-1)
      qw_0_t=PSKM.Flux_qw(PSKM.Epsilon_w_of_Pc(Solution[3]-Solution[2]),Solution[2],Solution[4],1)
      Sum_Trap_L=Sum_Trap_L+((qw_L_t+qw_L_t_old)*dt_old)/2.0 #Sum_Trap_L is the integral over time using trap method of qwL
      Sum_Trap_0=Sum_Trap_0+((qw_0_t+qw_0_t_old)*dt_old)/2.0 #Sum_Trap_0 is the integral over time using trap method of qw0
      qw_L_t_old=qw_L_t
      qw_0_t_old=qw_0_t
      dt_old=dt
      counter=counter+1
    return Sum_Trap_0,Sum_Trap_L
    
    
########################################################################################
# "Mass_inside_domain" function calculate: sum of [rhow(t)*Epsilonw(t)-rhow(t0)*Epsilonw(t0)]
# this sum will be used to calculate the error mass balance
def Mass_inside_domain(Sum_rho_Epsilonw,VEpsilonw,VPw,Vp_w_t0,VEpsilon_w_t0):  
    for i in range(len(Sum_rho_Epsilonw)):
        for k in range(0,Nz):
            Sum_rho_Epsilonw[i]=Sum_rho_Epsilonw[i]+(PSKM.rhow_of_Pw(VPw[i][k])*VEpsilonw[i][k] -PSKM.rhow_of_Pw(Vp_w_t0[k])*VEpsilon_w_t0[k])*dz
            
########################################################################################
def Error_Mass_Function(Error_Mass,Sum_rho_Epsilonw,Integral_qw_L,Integral_qw_0):
    for i in range(len(Error_Mass)):
        Error_Mass[i]=(1.0-(Sum_rho_Epsilonw[i]/(Integral_qw_L[i]-Integral_qw_0[i]))) # Sum_Trap_L and  Sum_Trap_0 are the integral of qwL and qw0 and has been already calculated with "Flux_Integration" function 
    

    
