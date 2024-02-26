import numpy as np
from TCAT2Phase import state_equations
from TCAT2Phase.numerics.residual import _residual
class BaseFormulation:
    """
    Base formulation for two-phase models
    """
    pass

class SPP(BaseFormulation):
    """
    SPP  Formulation Class
    """
    def __init__(self):
        pass


class SP(BaseFormulation):
    """
    SPP  Formulation Class
    """
    def __init__(self):
        pass

    def get_primary_variables(self,u,nodes):
        """
        Collect primary variables from solution variables
        """
        # Solution Variables
        e_w = u[:][0]
        p_w = u[:][1]

        # Primary Variables 
        p_c = np.zeros_like(p_w)
        p_n = np.zeors_like(p_w)
        for n in range(0,nodes):
            p_c[n] = state_equations.Pc(e_w[n])
            p_n[n] = p_w[n] + p_c[n]

        return e_w,p_w,p_n,p_c
    
    def get_A_matrix(self,e_w,p_w,p_n,p_c,rho_w,rho_n,e_n):
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


class PP(BaseFormulation):
    """
    SPP  Formulation Class
    """
    def __init__(self):
        pass

    def get_primary_variables(self,u,nodes):
        """
        Collect primary variables from solution variables
        """
        # Solution Variables
        p_w = u[:][0]
        p_n = u[:][1]

        # Primary Variables 
        p_c = np.zeros_like(p_w)
        e_w = np.zeors_like(p_w)
        for n in range(0,nodes):
            p_c[n] = p_n[n] - p_w[n]
            e_w[n] = state_equations.Epsilon_w_of_Pc(p_c[n])

        return e_w,p_w,p_n,p_c
    
    def get_A_matrix(self,e_w,p_w,p_n,p_c,rho_w,rho_n,e_n):
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

