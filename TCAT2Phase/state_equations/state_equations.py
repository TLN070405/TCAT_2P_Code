"""state_equations.py"""
import math

def effective_saturation(ew,Epsilon_wr,Const_Epsilon):
    """
    Effective saturation
    """
    se = (ew - Epsilon_wr) / Const_Epsilon
    return se


def vg_capillary_pressure(ew):
    """
    Van-Genuchten capillary pressure
    """
    se = effective_saturation(ew,Epsilon_w,Const_Epsilon)
    if 0.0  < se < 1.0:
        p_c = (  ( se**(-m_inv)  - 1.0)**n_inv )*alpha_inv
    else:
        p_c = 0.0
    return p_c

def drvt_ew_of_pc(pc):
    """
    this function is dew/dpc
    """
    if pc > 0.0:
        drvt_se = -alpha_m_n*((alpha*pc)**n1)*((1.0 + ((alpha*pc)**n))**m1)
        drvt_ew = Const_Epsilon*drvt_se
    else:
        drvt_ew = 0.0
    return drvt_ew

def krw_of_pc(pc):
    """
    The w phase relative permeability function
    """
    if pc > 0.0:
        se =  (1.0 + ((alpha*pc)**n))**(-m)
        krw =    (se**0.5)  *(  (1.0 - (  (1.0 - se**m_inv)**m  )   )**2.0)
    else:
        krw = 1.0
    return krw

def ew_of_pc(pc):
    """
    Volume fraction from Van-Genuchten model
    """
    if pc > 0.0:
        se =  (1.0 + (alpha*pc)**n)**(-m)
        ew = Const_Epsilon *  se + Epsilon_wr
    else:
        ew = epsilon - Epsilon_nr
    return ew

def density_exp_pressure(p,rho0,beta):
    """
    Density as a exponential function of the pressure
    """
    rho = rho0 * math.exp(beta * p)
    return rho

def d_density_exp_pressure(p,rho0,beta):
    """
    Derivative of the density with respect to pressure
    """
    d_rho = beta * rho0 * math.exp(beta * p)
    return d_rho

def density_linear_pressure(p,rho0,alpha):
    """
    Density as a linear function of the pressure
    """
    rho = rho0 + alpha * p
    return rho

def d_density_linear_pressure(p,rho0,alpha):
    """
    Derivative of the density with respect to pressure
    """
    d_rho = alpha
    return d_rho