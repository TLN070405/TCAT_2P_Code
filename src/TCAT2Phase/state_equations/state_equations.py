"""state_equations.py"""
import math

def effective_saturation(e_w,e_wr,e_nr,porosity):
    """
    Effective saturation
    """
    se = (e_w - e_wr) / (porosity-e_wr-e_nr)
    return se

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

def volume_fraction(e_in,porosity):
    """
    Caluculate volume fraction assuming 2P model
    """
    e_out = porosity - e_in
    return e_out
