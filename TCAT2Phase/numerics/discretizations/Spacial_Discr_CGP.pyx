import sys
import petsc4py
import cython
petsc4py.init(sys.argv)
from petsc4py import PETSc
from libc.stdlib cimport malloc # needed to create the C dynamic arrays
OptDB = PETSc.Options()
Test_Case = OptDB.getString('Test_Case', 'Test_Case4')

if(Test_Case=='Test_Case1'):
       sys.path.insert(0, './TCAT_CGP_Subcodes/Inputs/Test_Case1')
       import Inputs1 as inp1 # import the inputs.py file as a module so we can access directly the parameters defined inside it
elif( Test_Case=='Test_Case2'):
       sys.path.insert(0, './TCAT_CGP_Subcodes/Inputs/Test_Case2')  
       import Inputs2 as inp1 # import the inputs.py file as a module so we can access directly the parameters defined inside it
elif( Test_Case=='Test_Case3'):
       sys.path.insert(0, './TCAT_CGP_Subcodes/Inputs/Test_Case3')  
       import Inputs3 as inp1 # import the inputs.py file as a module so we can access directly the parameters defined inside it 
elif( Test_Case=='Test_Case4'):
       sys.path.insert(0, './TCAT_CGP_Subcodes/Inputs/Test_Case4')  
       import Inputs4 as inp1 # import the inputs.py file as a module so we can access directly the parameters defined inside it
       
#######################  
Nx=inp1.Nx      
################# Solution variables arrays#######################
cdef double *Ve_w     = <double*>malloc(Nx * sizeof(double))
cdef double *Vp_w     = <double*>malloc(Nx * sizeof(double))
cdef double *Ve_wn    = <double*>malloc(Nx * sizeof(double))
cdef double *Vj_wn    = <double*>malloc(Nx * sizeof(double))
cdef double *Vw_xwn   = <double*>malloc(Nx * sizeof(double))
cdef double *Vkn_wn   = <double*>malloc(Nx * sizeof(double))
################ Arrays for intermediate calculations #############
cdef double *Vrho_w               = <double*>malloc(Nx * sizeof(double))
cdef double *VKrw_rho_w           = <double*>malloc(Nx * sizeof(double))
cdef double *VDif_ewn_eq_ewn      = <double*>malloc(Nx * sizeof(double))
cdef double *VOd_pc               = <double*>malloc(Nx * sizeof(double))
####################################### SPP Formulation  #######################################
def SD_SPIJWK_Arrays(inp,SE,u,Ve_wn_eq):
    for i in range(0,inp.Nx):
        #the "Tol" term is added to avoid the division by zero, because at equilibrium CPw_J_wn will go to 0.
        Ve_w[i]               = u[i][0]
        Vp_w[i]               = u[i][1]         
        Ve_wn[i]              = u[i][2]
        Vj_wn[i]              = u[i][3]
        Vw_xwn[i]             = u[i][4]
        Vkn_wn[i]             = u[i][5]
        Vrho_w[i]             = SE.rhow_of_Pw(inp,Vp_w[i])      
        VKrw_rho_w[i]         = Ve_w[i] * Ve_w[i] * Vrho_w[i]
        VDif_ewn_eq_ewn[i]    = Ve_wn_eq[i] - Ve_wn[i]
        CPw_J_wn              = Vp_w[i] + inp.Gamma_wn * Vj_wn[i]
        VOd_pc[i]             = inp.C_wn * CPw_J_wn + (inp.Prod_Gamma_K_wn/(CPw_J_wn+inp.Tol)) * VDif_ewn_eq_ewn[i]  
#######################################
def SD_SPIJWK_A(inp,SE,u,i): 
        cdef double  A_SPIJWK_11,A_SPIJWK_12,A_SPIJWK_21,A_SPIJWK_22,A_SPIJWK_31,A_SPIJWK_33,A_SPIJWK_41,A_SPIJWK_43,A_SPIJWK_44
        cdef double  Kw_plus,Kw_minus,rho_w_plus,rho_w_minus,Od_w,Od_wn,Od_J,Od_pc
        cdef double  Cdew_dx, e_wn2, Cewn_ens, Cejwn_ejns, Cekwn_ekns,e_ns,e_ws,js_ns,kn_ns,b0D1,b1D2,b2D3
        ######### calculate the A_SPIJWK matrix components (we type only the non zeros terms of the Matrix)
        A_SPIJWK_11 = 1                 
        A_SPIJWK_21 = Vrho_w[i]
        A_SPIJWK_22 = Ve_w[i] * SE.Drvt_rhow_of_Pw(inp,Vp_w[i])
        A_SPIJWK_31 = Vj_wn[i]
        A_SPIJWK_33 = 1
        A_SPIJWK_41 = 2 * Vkn_wn[i]
        A_SPIJWK_43 = Vj_wn[i]
        A_SPIJWK_44 = Ve_wn[i]
        ######### calculate the spacial descretization operators 
        Od_pc       = VOd_pc[i]
        #########
        Kw_plus     = inp.Half_Inv_Rww * (VKrw_rho_w[i+1] + VKrw_rho_w[i])
        Kw_minus    = inp.Half_Inv_Rww * (VKrw_rho_w[i-1] + VKrw_rho_w[i])
        rho_w_plus  = 0.5 * (Vrho_w[i+1] + Vrho_w[i])
        rho_w_minus = 0.5 * (Vrho_w[i-1] + Vrho_w[i])
        Od_w        = inp.Cx*(Kw_plus*(inp.Cx*(Vp_w[i+1] - Vp_w[i]) + rho_w_plus*inp.g*inp.d) - Kw_minus*(inp.Cx*(Vp_w[i] - Vp_w[i-1]) + rho_w_minus*inp.g*inp.d)) 
        ##### 
        D_ewn_wx_dx = inp.Cx * ( Ve_wn[i] * Vw_xwn[i] - Ve_wn[i-1] * Vw_xwn[i-1] )  
        Od_wn       = inp.k_wn_hat * VDif_ewn_eq_ewn[i] - D_ewn_wx_dx
        #####
        D_ewn_wx_jwn_dx  = inp.Cx * ( Ve_wn[i] * Vj_wn[i] * Vw_xwn[i] - Ve_wn[i-1] * Vj_wn[i-1] * Vw_xwn[i-1] )
        Od_J             = inp.Two_by3_Cx2 * ( VOd_pc[i+1] - 2 * VOd_pc[i] + VOd_pc[i-1] ) - 2 * D_ewn_wx_jwn_dx
        #####
        Cdew_dx          = inp.Cx  * ( Ve_w[i] - Ve_w[i-1] )
        e_wn2            = Ve_wn[i] * Ve_wn[i]
        w_xwn            = Vw_xwn[i]
        e_w              = Ve_w[i]
        #####Calculate necessary vectors for equation 6
        e_ns,e_ws,js_ns,kn_ns,b0D1,b1D2,b2D3 = SE.Linear_Geometric_Constants(inp,Ve_w[i])
        #####
        Cewn_ens         = Ve_wn[i] + e_ns
        Cejwn_ejns       =-Ve_wn[i] * Vj_wn[i]  + e_ns * js_ns
        Cekwn_ekns       = Ve_wn[i] * Vkn_wn[i] + e_ns * kn_ns
        return A_SPIJWK_11,A_SPIJWK_21,A_SPIJWK_22,A_SPIJWK_31,A_SPIJWK_33,A_SPIJWK_41,A_SPIJWK_43,A_SPIJWK_44,Od_pc,Od_w,Od_wn,Od_J,Cdew_dx,e_wn2,w_xwn,e_w,Cewn_ens,Cejwn_ejns,Cekwn_ekns,b0D1,b1D2,b2D3

