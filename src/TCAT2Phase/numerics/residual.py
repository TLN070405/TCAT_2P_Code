import numpy as np
from TCAT2Phase import state_equations
from TCAT2Phase.numerics import utils


def two_phase_residual(ts,t,U,Udot,F,model):
    """
    Residual function for two-phase sp formulation
    """

    da = ts.getDM()
    localU = da.getLocalVec()
    da.globalToLocal(U,localU)

    # Collect solution variables and residual
    u = da.getVecArray(localU)
    udot = da.getVecArray(Udot)
    f = da.getVecArray(F)

    nodes = model.nodes

    q_w,q_n=two_phase_residual_internal(f,u,udot,model)

    two_phase_residual_boundary(f,u,q_w,q_n,nodes,model)


def two_phase_residual_internal(f,u,udot,model):
    """
    Not sure
    """
    # Collect discretization parameters
    nodes = model.nodes
    dx = model.res

    form = model.model.name
    if form == 'two-phase-sp':
        e_w,p_w,p_n,p_c = two_phase_sp_primary_variables(u,nodes,model)
    elif form == 'two-phase-pp':
        e_w,p_w,p_n,p_c = two_phase_pp_primary_variables(u,nodes,model)
    else:
        print('Error Model Form is unknown')


    rho_w  = np.zeros_like(e_w)
    rho_n  = np.zeros_like(e_w)
    e_n    = np.zeros_like(e_w)
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
        q_w[n] =darcy_velocity(k,0.5*(k_rw[n+1]*rho_w[n+1]+k_rw[n]*rho_w[n]),mu_w,del_pw[n] ,0.5*(rho_w[n+1]+rho_w[n]),g)
        q_n[n] =darcy_velocity(k,0.5*(k_rn[n+1]*rho_n[n+1]+k_rn[n]*rho_n[n]),mu_n,del_pn[n] ,0.5*(rho_n[n+1]+rho_n[n]),g)

    for n in range(1,nodes-1):

        # Calculate the spacial descretization parts
        Od_w = (1./dx)*(q_w[n] - q_w[n-1])
        Od_n = (1./dx)*(q_n[n] - q_n[n-1])

        # Time derivative matrix
        if form == 'two-phase-sp':
            A = two_phase_sp_A_matrix(e_w[n],p_w[n],p_n[n],p_c[n],rho_w,rho_n,e_n,model)
        elif form == 'two-phase-pp':
            A = two_phase_pp_A_matrix(e_w[n],p_w[n],p_n[n],p_c[n],rho_w,rho_n,e_n,model)

        f[n][0] = A[0,0] * udot[n][0] + A[0,1] * udot[n][1] - Od_w
        f[n][1] = A[1,0] * udot[n][0] + A[1,1] * udot[n][1] - Od_n

    return q_w,q_n

def two_phase_residual_boundary(f,u,q_w,q_n,nodes,model):
    """
    Residual at boundary points
    """
    q_bc =[[q_w[0],q_n[0]],[q_w[nodes-1],q_n[nodes-1]]]
    u_name= model.u_names
    for n in range(model.model.num_u):
        BC=[model.boundary_type[u_name[n]][0][0],model.boundary_type[u_name[n]][0][1]]
        BV=[model.boundary_value[n][0][0],model.boundary_value[n][0][1]]
        #Residual at Boundary 1
        f[0][n] = (u[0][n] - BV[0])*(1-BC[0]) + (q_bc[0][n] -  BV[0])*BC[0]
        # Residual at Boundary 2
        f[nodes-1][n] = (u[nodes-1][n] - BV[1])*(1-BC[1]) + (q_bc[1][n] - BV[1] )*BC[1]

def two_phase_sp_primary_variables(u,nodes,model):
    """
    Collect primary variables from solution variables
    """
    # Solution Variables
    e_w = u[:,0]
    p_w = u[:,1]

    # Primary Variables
    p_c = np.zeros_like(p_w)
    p_n = np.zeros_like(p_w)
    for n in range(0,nodes):
        p_c[n] = model.interface.pc_function(e_w[n])
        p_n[n] = p_w[n] + p_c[n]

    return e_w,p_w,p_n,p_c

def two_phase_sp_A_matrix(e_w,p_w,p_n,p_c,rho_w,rho_n,e_n,model):
    """
    Calculate the A matrix. See Kees_Miller_02 for formulation
    """
    A = np.zeros([2,2])

    d_rho_w_p_w = model.w_phase.d_density_function(p_w)
    d_rho_n_p_n = model.n_phase.d_density_function(p_n)
    d_p_c_e_w   = model.interface.d_pc_function(e_w)

    A[0,0] = rho_w
    A[0,1] = e_w*d_rho_w_p_w
    A[1,0] = e_n*d_rho_n_p_n*d_p_c_e_w - rho_n
    A[1,1] = e_n*d_rho_n_p_n

    return A

def two_phase_pp_primary_variables(u,nodes,model):
    """
    Collect primary variables from solution variables
    """
    # Solution Variables
    p_w = u[:][0]
    p_n = u[:][1]

    # Primary Variables
    p_c = np.zeros_like(p_w)
    e_w = np.zeros_like(p_w)
    for n in range(0,nodes):
        p_c[n] = p_n[n] - p_w[n]
        e_w[n] = model.w_phase.volume_fraction_function(p_c[n])

    return e_w,p_w,p_n,p_c

def two_phase_pp_A_matrix(e_w,p_w,p_n,p_c,rho_w,rho_n,e_n,model):
    """
    Calculate the A matrix. See Kees_Miller_02 for formulation
    """
    A = np.zeros([2,2])
    d_e_w_p_c   = model.w_phase.d_volume_fraction_function(p_c)
    d_rho_w_p_w = model.w_phase.d_density_function(p_w)
    d_rho_n_p_n = model.n_phase.d_density_function(p_n)

    A[0,0] = -rho_w*d_e_w_p_c + e_w*d_rho_w_p_w
    A[0,1] = rho_w*d_e_w_p_c
    A[1,0] = rho_n*d_e_w_p_c
    A[1,1] = -rho_n*d_e_w_p_c + e_n*d_rho_n_p_n

    return A


def darcy_velocity(k,k_r,mu,del_p,rho,g):
    """
    Calaculate the Darcy velocity as:
        q^a = -k*k_r^a/mu^a(\grad p^a - rho^a*g)
    """
    q = k*k_r/mu*(del_p - rho*g)
    return q
