import sys
import petsc4py
import cython
petsc4py.init(sys.argv)
from petsc4py import PETSc
from libc.stdlib cimport malloc # needed to create the C dynamic arrays

#######################################
OptDB = PETSc.Options()
Paper = OptDB.getString('Paper', 'Helmig') # get paper name from terminal so we can decide which inputs and PSK model to use
if(Paper=='Fucik'):
   sys.path.insert(0, './inputs/inputs_Fucik')
   import inputs as inp # import the inupts.py file as a module so we can access directly the parameters defined inside it Genuchten
elif( Paper=='Helmig'):
   sys.path.insert(0, './inputs/inputs_Helmig')  
   import inputs as inp # import the inupts.py file as a module so we can access directly the parameters defined inside it
elif( Paper=='Malgorzata'):
   sys.path.insert(0, './inputs/inputs_Malgorzata')  
   import inputs as inp # import the inupts.py file as a module so we can access directly the parameters defined inside it
#######################################
Nz=inp.Nz
Cz=inp.Cz
k_muw_half=inp.k_muw_half
k_mun_half=inp.k_mun_half
g=inp.g
d=inp.d


#######################################
cdef double *Vp_c = <double*>malloc(Nz * sizeof(double))
cdef double *Vp_w = <double*>malloc(Nz * sizeof(double))
cdef double *Vp_n = <double*>malloc(Nz * sizeof(double))
cdef double *Vrho_w_of_P_w = <double*>malloc(Nz * sizeof(double))
cdef double *Vrho_n_of_P_n = <double*>malloc(Nz * sizeof(double))
cdef double *VKrw_of_P_c = <double*>malloc(Nz * sizeof(double))
cdef double *VKrn_of_P_c = <double*>malloc(Nz * sizeof(double))
cdef double *VKrw_rho_w = <double*>malloc(Nz * sizeof(double))
cdef double *VKrn_rho_n = <double*>malloc(Nz * sizeof(double))
####################################### SPP Formulation  #######################################
def SD_SPP_PKR(PSKM,u):
    for i in range(0,Nz):
        Vp_c[i] = PSKM.Pc(u[i][0]) # store cappilary pressure inside the array Vp_c (V stands for Vector)
        Vp_w[i] = u[i][1]          # store u[i][1] at Vp_w[i], because its used multiple times and using u[i][1] makes code slower
        Vp_n[i] = u[i][2]          # same thing as line before
        Vrho_w_of_P_w[i]=PSKM.rhow_of_Pw(Vp_w[i]) # store Rhow[i] to do not recalculate same quantities again and again
        VKrw_of_P_c[i]=PSKM.Krw_of_Pc(Vp_c[i])
        Vrho_n_of_P_n[i]=PSKM.rhon_of_Pn(Vp_n[i])
        VKrn_of_P_c[i]=PSKM.Krn_of_Pc(Vp_c[i])
        VKrw_rho_w[i]=Vrho_w_of_P_w[i]* VKrw_of_P_c[i]
        VKrn_rho_n[i]=Vrho_n_of_P_n[i]* VKrn_of_P_c[i]
#######################################
def SD_SPP_A_Od(PSKM,u,i): 
    cdef double  CEpsilon_n_of_Pc,CDrvt_rho_n_of_Pn
    cdef double  A_SPP_11,A_SPP_12,A_SPP_21,A_SPP_23,Od_w,Od_n,Od_dp
    cdef double  Kw_plus,Kw_minus,rho_w_plus,rho_w_minus,Kn_plus,Kn_minus,rho_n_plus,rho_n_minus
    ######### calculate the A_SPP matrix components
    CEpsilon_n_of_Pc=PSKM.Epsilon_n_of_Pc(u[i][0])
    CDrvt_rho_n_of_Pn=PSKM.Drvt_rhon_of_Pn(Vp_n[i])
    A_SPP_11 = Vrho_w_of_P_w[i]                  
    A_SPP_12 = u[i][0] *PSKM.Drvt_rhow_of_Pw(Vp_w[i])
    A_SPP_21 = -Vrho_n_of_P_n[i]
    A_SPP_23 = CEpsilon_n_of_Pc*CDrvt_rho_n_of_Pn
    ######### calculate the spacial descretization parts
    Kw_plus=k_muw_half*(VKrw_rho_w[i+1]+VKrw_rho_w[i])
    Kw_minus=k_muw_half*(VKrw_rho_w[i-1] + VKrw_rho_w[i])
    rho_w_plus = 0.5*(Vrho_w_of_P_w[i+1] + Vrho_w_of_P_w[i])
    rho_w_minus = 0.5*(Vrho_w_of_P_w[i-1] + Vrho_w_of_P_w[i])
    Kn_plus = k_mun_half*(VKrn_rho_n[i+1]+ VKrn_rho_n[i])
    Kn_minus = k_mun_half*(VKrn_rho_n[i-1]+VKrn_rho_n[i])
    rho_n_plus = 0.5*(Vrho_n_of_P_n[i+1]+ Vrho_n_of_P_n[i])
    rho_n_minus =0.5*(Vrho_n_of_P_n[i-1] + Vrho_n_of_P_n[i])
    Od_w = Cz * (Kw_plus*(Cz*(Vp_w[i+1] - Vp_w[i]) + rho_w_plus*g*d) - Kw_minus * (Cz*(Vp_w[i] - Vp_w[i-1]) + rho_w_minus*g*d))
    Od_n = Cz * (Kn_plus*(Cz*(Vp_n[i+1] - Vp_n[i]) + rho_n_plus*g*d) - Kn_minus * (Cz*(Vp_n[i] - Vp_n[i-1]) + rho_n_minus*g*d))
    Od_dp=(Vp_w[i]-Vp_n[i]+ Vp_c[i])
    return A_SPP_11,A_SPP_12,A_SPP_21,A_SPP_23,Od_w,Od_n,Od_dp
####################################### SP Formulation  #######################################
def SD_SP_PKR(PSKM,u):
    for i in range(0,Nz):
        Vp_c[i] = PSKM.Pc(u[i][0])    # store cappilary pressure inside the array Vp_c (V stands for Vector)
        Vp_w[i] = u[i][1]             # store u[i][1] at Vp_w[i], because its used multiple times and using u[i][1] makes code slower
        Vp_n[i] = Vp_c[i]+ Vp_w[i]    # same thing as line before
        Vrho_w_of_P_w[i]=PSKM.rhow_of_Pw(Vp_w[i]) # store Rhow[i] to do not recalculate same quantities again and again
        VKrw_of_P_c[i]=PSKM.Krw_of_Pc(Vp_c[i])
        Vrho_n_of_P_n[i]=PSKM.rhon_of_Pn(Vp_n[i])
        VKrn_of_P_c[i]=PSKM.Krn_of_Pc(Vp_c[i])
        VKrw_rho_w[i]=Vrho_w_of_P_w[i]* VKrw_of_P_c[i]
        VKrn_rho_n[i]=Vrho_n_of_P_n[i]* VKrn_of_P_c[i]
#######################################        
def SD_SP_A_Own(PSKM,u,i): 
    cdef double  CEpsilon_n_of_Pc,CDrvt_rho_n_of_Pn
    cdef double  A_SP_11,A_SP_12,A_SP_21,A_SP_22,Od_w,Od_n
    cdef double  Kw_plus,Kw_minus,rho_w_plus,rho_w_minus,Kn_plus,Kn_minus,rho_n_plus,rho_n_minus
    ######### calculate the A_SP matrix components
    CEpsilon_n_of_Pc=PSKM.Epsilon_n_of_Pc(u[i][0])
    CDrvt_rho_n_of_Pn=PSKM.Drvt_rhon_of_Pn(Vp_n[i])
    A_SP_11 = Vrho_w_of_P_w[i]                  
    A_SP_12 = u[i][0] *PSKM.Drvt_rhow_of_Pw(Vp_w[i])
    A_PP_21 = CEpsilon_n_of_Pc*CDrvt_rho_n_of_Pn*PSKM.Drvt_Pc_by_Epsilon_w(u[i][0])-Vrho_n_of_P_n[i]
    A_SP_22 = CEpsilon_n_of_Pc*CDrvt_rho_n_of_Pn
    ######### calculate the spacial descretization parts
    Kw_plus=k_muw_half*(VKrw_rho_w[i+1]+VKrw_rho_w[i])
    Kw_minus=k_muw_half*(VKrw_rho_w[i-1] + VKrw_rho_w[i])
    rho_w_plus = 0.5*(Vrho_w_of_P_w[i+1] + Vrho_w_of_P_w[i])
    rho_w_minus = 0.5*(Vrho_w_of_P_w[i-1] + Vrho_w_of_P_w[i])
    Kn_plus = k_mun_half*(VKrn_rho_n[i+1]+ VKrn_rho_n[i])
    Kn_minus = k_mun_half*(VKrn_rho_n[i-1]+VKrn_rho_n[i])
    rho_n_plus = 0.5*(Vrho_n_of_P_n[i+1]+ Vrho_n_of_P_n[i])
    rho_n_minus =0.5*(Vrho_n_of_P_n[i-1] + Vrho_n_of_P_n[i])
    Od_w = Cz * (Kw_plus*(Cz*(Vp_w[i+1] - Vp_w[i]) + rho_w_plus*g*d) - Kw_minus * (Cz*(Vp_w[i] - Vp_w[i-1]) + rho_w_minus*g*d))
    Od_n = Cz * (Kn_plus*(Cz*(Vp_n[i+1] - Vp_n[i]) + rho_n_plus*g*d) - Kn_minus * (Cz*(Vp_n[i] - Vp_n[i-1]) + rho_n_minus*g*d))
    return A_SP_11,A_SP_12,A_SP_21,A_SP_22,Od_w,Od_n
        
####################################### PP Formulation  #######################################
def SD_PP_PKR(PSKM,u):
    for i in range(0,Nz):
        Vp_w[i] = u[i][0]             # store u[i][1] at Vp_w[i], because its used multiple times and using u[i][1] makes code slower
        Vp_n[i] = u[i][1]                     
        Vp_c[i] = Vp_n[i]-Vp_w[i]     # store cappilary pressure inside the array Vp_c (V stands for Vector)
        Vrho_w_of_P_w[i]=PSKM.rhow_of_Pw(Vp_w[i]) # store Rhow[i] to do not recalculate same quantities again and again
        VKrw_of_P_c[i]=PSKM.Krw_of_Pc(Vp_c[i])
        Vrho_n_of_P_n[i]=PSKM.rhon_of_Pn(Vp_n[i])
        VKrn_of_P_c[i]=PSKM.Krn_of_Pc(Vp_c[i])
        VKrw_rho_w[i]=Vrho_w_of_P_w[i]* VKrw_of_P_c[i]
        VKrn_rho_n[i]=Vrho_n_of_P_n[i]* VKrn_of_P_c[i]
#######################################        
def SD_PP_A_Own(PSKM,u,i): 
    cdef double  CDrvt_Epsilon_w_of_Pc,CDrvt_rho_n_of_Pn
    cdef double  A_PP_11,A_PP_12,A_PP_21,A_PP_22,Od_w,Od_n
    cdef double  Kw_plus,Kw_minus,rho_w_plus,rho_w_minus,Kn_plus,Kn_minus,rho_n_plus,rho_n_minus
    ######### calculate the A_PP matrix components
    CDrvt_Epsilon_w_of_Pc = PSKM.Drvt_Epsilon_w_of_Pc(Vp_c[i])
    CEpsilon_w_of_Pc = PSKM.Epsilon_w_of_Pc(Vp_c[i])
    A_PP_11 = -Vrho_w_of_P_w[i] * CDrvt_Epsilon_w_of_Pc + CEpsilon_w_of_Pc * PSKM.Drvt_rhow_of_Pw(Vp_w[i])
    A_PP_12 =  Vrho_w_of_P_w[i] * CDrvt_Epsilon_w_of_Pc
    A_PP_21 =  Vrho_n_of_P_n[i] * CDrvt_Epsilon_w_of_Pc
    A_PP_22 = -A_PP_21 + PSKM.Epsilon_n_of_Pc(CEpsilon_w_of_Pc) * PSKM.Drvt_rhon_of_Pn(Vp_n[i])
    ######### calculate the spacial descretization parts
    Kw_plus=k_muw_half*(VKrw_rho_w[i+1]+VKrw_rho_w[i])
    Kw_minus=k_muw_half*(VKrw_rho_w[i-1] + VKrw_rho_w[i])
    rho_w_plus = 0.5*(Vrho_w_of_P_w[i+1] + Vrho_w_of_P_w[i])
    rho_w_minus = 0.5*(Vrho_w_of_P_w[i-1] + Vrho_w_of_P_w[i])
    Kn_plus = k_mun_half*(VKrn_rho_n[i+1]+ VKrn_rho_n[i])
    Kn_minus = k_mun_half*(VKrn_rho_n[i-1]+VKrn_rho_n[i])
    rho_n_plus = 0.5*(Vrho_n_of_P_n[i+1]+ Vrho_n_of_P_n[i])
    rho_n_minus =0.5*(Vrho_n_of_P_n[i-1] + Vrho_n_of_P_n[i])
    Od_w = Cz * (Kw_plus*(Cz*(Vp_w[i+1] - Vp_w[i]) + rho_w_plus*g*d) - Kw_minus * (Cz*(Vp_w[i] - Vp_w[i-1]) + rho_w_minus*g*d))
    Od_n = Cz * (Kn_plus*(Cz*(Vp_n[i+1] - Vp_n[i]) + rho_n_plus*g*d) - Kn_minus * (Cz*(Vp_n[i] - Vp_n[i-1]) + rho_n_minus*g*d))
    return A_PP_11,A_PP_12,A_PP_21,A_PP_22,Od_w,Od_n
        
        
    
