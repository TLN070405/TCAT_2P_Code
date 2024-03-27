import numpy as np

from TCAT2Phase.numerics import utils
from TCAT2Phase.numerics import residual
 

def two_phase_formfunction(snes,U,F,model):
    """
    form function for two-phase sp formulation
    """
    da=snes.getDM()
    localU=da.getLocalVec()
    da.globalToLocal(U,localU)
    #######
    u=da.getVecArray(localU)
    f=da.getVecArray(F)
    #######
    u=da.getVecArray(localU)

    two_phase_formfunction_internal(f,u,model)

    two_phase_formfunction_boundary(f,u,model)


def two_phase_formfunction_internal(f,u,model):
    """
    Initial conditions residual at interior of domain
    """
    # Collect discretization parameters
    nodes  = model.nodes
    u_name = model.u_names
    for n in range(1,nodes-1):
        f[n][0] = u[n][0] - model.initial_condition[u_name[0]]   
        f[n][1] = u[n][1] - model.initial_condition[u_name[1]] 


 

def two_phase_formfunction_boundary(f,u,model):
    """
    Boundary points residual
    """
   # Collect discretization parameters
    dx = model.res
    nodes = model.nodes
    form = model.model.name
    if form == 'two-phase-sp':
        e_w,p_w,p_n,p_c = residual.two_phase_sp_primary_variables(u,nodes,model)
    elif form == 'two-phase-pp':
        e_w,p_w,p_n,p_c = residual.two_phase_pp_primary_variables(u,nodes,model)
    else:
        print('Error Model Form is unknown')

    rho_w  = np.zeros_like(e_w)
    rho_n  = np.zeros_like(e_w)
    del_pw = np.zeros_like(e_w)
    del_pn = np.zeros_like(e_w)
    k_rw   = np.zeros_like(e_w)
    k_rn   = np.zeros_like(e_w)
    q_w    = np.zeros_like(e_w)
    q_n    = np.zeros_like(e_w)
    for n in range(0,nodes):
        rho_w[n] = model.w_phase.density_function(p_w[n])
        rho_n[n] = model.n_phase.density_function(p_n[n])
 
        del_pw[n] = utils.gradient(p_w[n],dx)
        del_pn[n] = utils.gradient(p_n[n],dx)

        k_rw[n] = model.w_phase.rel_perm_function(p_c[n])
        k_rn[n] = model.n_phase.rel_perm_function(p_c[n])

    k = model.solid.permeability
    mu_w = model.w_phase.viscosity
    mu_n = model.n_phase.viscosity
    for n in range(0,nodes-1):
        q_w[n] =residual.darcy_velocity(k,0.5*(k_rw[n+1]*rho_w[n+1]+k_rw[n]*rho_w[n]),mu_w,del_pw[n] ,0.5*(rho_w[n+1]+rho_w[n]),g)
        q_n[n] =residual.darcy_velocity(k,0.5*(k_rn[n+1]*rho_n[n+1]+k_rn[n]*rho_n[n]),mu_n,del_pn[n] ,0.5*(rho_n[n+1]+rho_n[n]),g)
    
    q_bc =[[q_w[0],q_n[0]],[q_w[nodes-1],q_n[nodes-1]]]
    u_name= model.u_names
    for n in range(model.model.num_u):
        BC=[model.boundary_type[u_name[n]][0][0],model.boundary_type[u_name[n]][0][1]]
        BV=[model.boundary_value[n][0][0],model.boundary_value[n][0][1]]
        #Residual at Boundary 1 
        f[0][n] = (u[0][n] - BV[0])*(1-BC[0]) + (q_bc[0][n]- BV[0])*  BC[0]
        # Residual at Boundary 2
        f[nodes-1][n] = (u[nodes-1][n] - BV[1])*(1-BC[1]) + (q_bc[1][n] - BV[1] ) * BC[1]


 