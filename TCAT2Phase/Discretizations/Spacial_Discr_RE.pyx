import sys
import petsc4py
import cython
petsc4py.init(sys.argv)
from petsc4py import PETSc
from libc.stdlib cimport malloc # needed to create the C dynamic arrays
OptDB = PETSc.Options()
Test_Case = OptDB.getString('Test_Case', 'Test_Case4')

if(Test_Case=='Test_Case1'):
       sys.path.insert(0, './RE_Subcodes/Inputs/Test_Case1')
       import Inputs1 as inp1 # import the inputs.py file as a module so we can access directly the parameters defined inside it
elif( Test_Case=='Test_Case2'):
       sys.path.insert(0, './RE_Subcodes/Inputs/Test_Case2')  
       import Inputs2 as inp1 # import the inputs.py file as a module so we can access directly the parameters defined inside it
elif( Test_Case=='Test_Case3'):
       sys.path.insert(0, './RE_Subcodes/Inputs/Test_Case3')  
       import Inputs3 as inp1 # import the inputs.py file as a module so we can access directly the parameters defined inside it 
elif( Test_Case=='Test_Case4'):
       sys.path.insert(0, './RE_Subcodes/Inputs/Test_Case4')  
       import Inputs4 as inp1 # import the inputs.py file as a module so we can access directly the parameters defined inside it
       
#######################  
Nx=inp1.Nx      
################# Solution variables arrays#######################
cdef double *Vp_w            = <double*>malloc(Nx * sizeof(double))
################ Arrays for intermediate calculations #############
cdef double *Vp_c            = <double*>malloc(Nx * sizeof(double))
cdef double *Vrho_w_of_P_w   = <double*>malloc(Nx * sizeof(double))
cdef double *VKrw_rho_w      = <double*>malloc(Nx * sizeof(double))
####################################### SPP Formulation  #######################################
def SD_PR_Arrays(inp,SE,u):
    for i in range(0,inp.Nx):
        Vp_w[i] = u[i]
        Vp_c[i] = -Vp_w[i]
        Vrho_w_of_P_w[i]=SE.rhow_of_Pw(inp,Vp_w[i])
        VKrw_rho_w[i]=Vrho_w_of_P_w[i]* SE.Krw_of_Pc(inp,Vp_c[i]) 
#######################################
def SD_APR(inp,SE,u,i): 
        A_PR        =  -Vrho_w_of_P_w[i]*SE.Drvt_Epsilon_w_of_Pc(inp,Vp_c[i])+SE.Epsilon_w_of_Pc(inp,Vp_c[i])*SE.Drvt_rhow_of_Pw(inp,Vp_w[i])     
        #########
        Kw_plus     = inp.k_muw_half*(VKrw_rho_w[i+1] + VKrw_rho_w[i])
        Kw_minus    = inp.k_muw_half*(VKrw_rho_w[i-1] + VKrw_rho_w[i])
        rho_w_plus  = 0.5*(Vrho_w_of_P_w[i+1] + Vrho_w_of_P_w[i])
        rho_w_minus = 0.5*(Vrho_w_of_P_w[i-1] + Vrho_w_of_P_w[i])
        Od_w = inp.Cx * (Kw_plus*(inp.Cx*(Vp_w[i+1] - Vp_w[i]) + rho_w_plus*inp.g*inp.d) - Kw_minus * (inp.Cx*(Vp_w[i] - Vp_w[i-1]) + rho_w_minus*inp.g*inp.d))
        return A_PR ,Od_w

