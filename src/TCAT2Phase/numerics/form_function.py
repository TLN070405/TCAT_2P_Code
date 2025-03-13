import numpy as np

from TCAT2Phase.numerics import utils
from TCAT2Phase.numerics import residual

def model_formfunction_tm(snes,U,F,model):
    """
    residual function to solve boundary conditions in case we have flux bc
    tm: stands for traditional model.
    """
    da=snes.getDM()
    localU=da.getLocalVec()
    da.globalToLocal(U,localU)
    #######
    u=da.getVecArray(localU)
    f=da.getVecArray(F)
    #######
    u=da.getVecArray(localU)

    model_formfunction_internal(f,u,model)

    model_formfunction_boundary_tm(0,f,u,model)


def model_formfunction_tcat(snes,U,F,model):
    """
    residual function to solve boundary conditions in case we have flux bc
    tcat stands for: THERMODYNAMICALLY CONSTRAINED AVERAGING THEORY
    """
    da=snes.getDM()
    localU=da.getLocalVec()
    da.globalToLocal(U,localU)
    #######
    u=da.getVecArray(localU)
    f=da.getVecArray(F)
    #######
    u=da.getVecArray(localU)

    model_formfunction_internal(f,u,model)

    model_formfunction_boundary_tcat(0,f,u,model)


def model_formfunction_internal(f,u,model):
    """
    Initial conditions residual at interior of domain
    """
    u_name = model.model.u_names
    nodes = model.nodes
    
    for n_u,u_name in enumerate(model.model.u_names):
        for n in range(0,nodes):
            index =(n,) * (len(u.shape) - 1) + (slice(n_u),)
            x = model.domain.domain_size[0,0] + n* model.domain.domain_length[0]/(nodes-1)
            f[index] = u[index] - model.initial_conditions[u_name](x)


def model_formfunction_boundary_tm(t,f,u,model):
    """
    Set Boundary points residual equations in order to solve IC in case we have Neumman BC. 
    """
    e_w, p_w, e_n, p_n, p_c = residual.model_primary_variables_tm(u, model)
    q_w, q_n, rho_w, rho_n = residual.model_phase_flux_tm(p_w, p_n, p_c, model)
    residual.model_residual_boundary_tm(t, f, u, q_w, q_n, model)


def model_formfunction_boundary_tcat(t,f,u,model):
    """
    Set Boundary points residual equations in order to solve IC in case we have Neumman BC. 
    """
    e_w, p_w, e_wn, j_wn, w_wn, k_wn = residual.model_primary_variables_tcat(u, model)
    q_w, rho_w = residual.model_phase_flux_tcat(e_w, p_w, model)
    residual.model_residual_boundary_tcat(t, f, u,q_w, model)
