import sys
import petsc4py
import numpy as np
from TCAT2Phase import state_equations 
petsc4py.init(sys.argv)

def calc_average(x,y):
    """
    Return the average of a value
    """
    return 0.5*(x+y)

def calc_function_average(func,x,y):
    """
    Calaculate the aveage of a function 
    """
    avg_func = 0.5*(func(x) + func(y))
    return avg_func

def calc_gradient(x,y,dz):
    """
    Return the gradient of a value
    """
    return (x-y)/dz

def darcy_velocity(p_a,p_c,rho,parameters,discretization):
    """
    Calaculate the Darcy velocity as:
        q^a = -k*k_r^a/mu^a(\grad p^a - rho^a*g)
    """
    dx = discretization.dx
    k = parameters.k
    g = parameters.g
    mu = parameters.mu

    grad_pa = calc_gradient(p_a,dx)
    k_ra = calc_function_average(state_equations.k_ra_p_c,p_c)
    q = k*k_ra/mu*(grad_pa - rho*g)
    
    return q

def two_sphase_residual_internal(f,u,udot,formulation,discretization,parameters):
    """
    Not sure
    """
    # Collect discretization parameters
    nodes = discretization.nodes
    dx = discretization.dx

    e_w,p_w,p_n,p_c = formulation.get_primary_variables(u,nodes)

    for n in range(1,nodes-2):
    
        # w density
        rho_w_plus = calc_function_average(state_equations.rhow_of_Pw,p_w[n],p_w[n+1])
        rho_w_minus = calc_function_average(state_equations.rhow_of_Pw,p_w[n],p_w[n-1]) 

        # n density
        rho_n_plus = calc_function_average(state_equations.rhon_of_Pn,p_n[n],p_n[n+1])
        rho_n_minus = calc_function_average(state_equations.rhon_of_Pn,p_n[n],p_n[n-1])

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
