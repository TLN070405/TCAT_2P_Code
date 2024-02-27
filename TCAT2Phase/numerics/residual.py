import numpy as np
from TCAT2Phase import state_equations


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

    two_sphase_residual_internal(f,u,udot,model)

    #Residual at Boundary 1 
    for n in range(model.model.num_u):
        f[0][n] = u[0][n] - model.boundary_conditions[0,n]
        f[0][n] = u[0][n] - model.boundary_conditions[0,n]

        # Residual at Boundary 2
        f[nodes-1][n] = u[nodes-1][n] - model.boundary_conditions[1,n]
        f[nodes-1][n] = u[nodes-1][n] - model.boundary_conditions[1,n]


def two_sphase_residual_internal(f,u,udot,model):
    """
    Not sure
    """
    # Collect discretization parameters
    nodes = model.nodes
    dx = model.res

    form = model.model.name
    if form == 'two-phase-sp':
        e_w,p_w,p_n,p_c = two_phase_sp_primary_variables(u,nodes)
    elif form == 'two-phase-pp':
        e_w,p_w,p_n,p_c = two_phase_pp_primary_variables(u,nodes)
    else:
        print('Error Model From is unknown')

    for n in range(1,nodes-2):
    
        # w density
        rho_w_plus = utils.calc_function_average(state_equations.rhow_of_Pw,p_w[n],p_w[n+1])
        rho_w_minus = utils.calc_function_average(state_equations.rhow_of_Pw,p_w[n],p_w[n-1]) 

        # n density
        rho_n_plus = utils.calc_function_average(state_equations.rhon_of_Pn,p_n[n],p_n[n+1])
        rho_n_minus = utils.calc_function_average(state_equations.rhon_of_Pn,p_n[n],p_n[n-1])

        # Darcy velocity
        q_w_plus = darcy_velocity(p_w,p_c,rho_w_plus,parameters,discretization)
        q_w_minus = darcy_velocity(p_w,p_c,rho_w_minus,parameters,discretization)

        q_n_plus = darcy_velocity(p_n,p_c,rho_n_plus,parameters,discretization)
        q_n_minus = darcy_velocity(p_n,p_c,rho_n_minus,parameters,discretization)

        # Calculate the spacial descretization parts
        Od_w = (1./dx)*(rho_w_plus* q_w_plus - rho_w_minus*q_w_minus)
        Od_n = (1./dx)*(rho_n_plus* q_n_plus - rho_n_minus*q_n_minus) 

        # Time derivative matrix
        
        rho_w = state_equations.rhow_of_Pw(p_w[n]) 
        rho_n = state_equations.rhon_of_Pn(p_n[n])
        e_n = state_equations.Epsilon_n_of_Pc(e_w[n])

        A = formulation.get_A_matrix(e_w[n],p_w[n],p_n[n],p_c[n],rho_w,rho_n,e_n)

        f[n][0] = A[0,0] * udot[n][0] + A[0,1] * udot[n][1] - Od_w
        f[n][1] = A[1,0] * udot[n][0] + A[1,1] * udot[n][1] - Od_n

    return f


def two_phase_sp_primary_variables(u,nodes):
    """
    Collect primary variables from solution variables
    """
    # Solution Variables
    e_w = u[:][0]
    p_w = u[:][1]

    # Primary Variables 
    p_c = np.zeros_like(p_w)
    p_n = np.zeros_like(p_w)
    for n in range(0,nodes):
        p_c[n] = state_equations.Pc(e_w[n])
        p_n[n] = p_w[n] + p_c[n]

    return e_w,p_w,p_n,p_c

def two_phase_sp_A_matrix(e_w,p_w,p_n,p_c,rho_w,rho_n,e_n):
    """
    Calculate the A matrix. See Kees_Miller_02 for formulation
    """
    A = np.zeros([2,2])

    d_rho_w_p_w = state_equations.d_rho_w_p_w(p_w)
    d_rho_n_p_n = state_equations.d_rho_n_p_n(p_n)
    d_p_c_e_w = state_equations.d_p_c_e_w(e_w)

    A[0,0] = rho_w
    A[0,1] = e_w*d_rho_w_p_w
    A[1,0] = e_n*d_rho_n_p_n*d_p_c_e_w - rho_n
    A[1,1] = e_n*d_rho_n_p_n    


def two_phase_pp_primary_variables(u,nodes):
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
        e_w[n] = state_equations.Epsilon_w_of_Pc(p_c[n])

    return e_w,p_w,p_n,p_c

def get_A_matrix(e_w,p_w,p_n,p_c,rho_w,rho_n,e_n):
    """
    Calculate the A matrix. See Kees_Miller_02 for formulation
    """
    A = np.zeros([2,2])
    d_e_w_p_c = state_equations.d_e_w_p_c(p_c)
    d_rho_w_p_w = state_equations.d_rho_w_p_w(p_w)
    d_rho_n_p_n = state_equations.d_rho_n_p_n(p_n)

    A[0,0] = -rho_w*d_e_w_p_c + e_w*d_rho_w_p_w
    A[0,1] = rho_w*d_e_w_p_c
    A[1,0] = rho_n*d_e_w_p_c
    A[1,1] = -rho_n*d_e_w_p_c + e_n*d_rho_n_p_n

    return A    


def darcy_velocity(p_a,p_c,rho,parameters,discretization):
    """
    Calaculate the Darcy velocity as:
        q^a = -k*k_r^a/mu^a(\grad p^a - rho^a*g)
    """
    dx = discretization.dx
    k = parameters.k
    g = parameters.g
    mu = parameters.mu

    grad_pa = utils.calc_gradient(p_a,dx)
    k_ra = utils.calc_function_average(state_equations.k_ra_p_c,p_c)
    q = k*k_ra/mu*(grad_pa - rho*g)

    return q   