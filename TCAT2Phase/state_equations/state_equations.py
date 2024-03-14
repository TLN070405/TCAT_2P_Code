"""state_equations.py"""
import math

def vg_volume_fraction_w(pc,e_wr,e_nr,porosity,alpha,n,m):
    """
    Volume fraction of w phase from Van-Genuchten model
    """
    if pc > 0.0:
        se =  (1.0 + (alpha*pc)**n)**(-m)
        ew = (porosity-e_wr-e_nr) *  se + e_wr
    else:
        ew = porosity - e_nr
    return ew

def d_vg_volume_fraction_w(pc,e_wr,e_nr,porosity,alpha,n,alpha_m_n,n1,m1):
    """
    this function is dew/dpc
    """
    if pc > 0.0:
        drvt_se = -alpha_m_n*((alpha*pc)**n1)*((1.0 + ((alpha*pc)**n))**m1)
        drvt_ew = (porosity-e_wr-e_nr)*drvt_se
    else:
        drvt_ew = 0.0
    return drvt_ew

def vg_rel_perm_w(pc,alpha,n,m,m_inv):
    """
    The w phase relative permeability function
    """
    if pc > 0.0:
        se =  (1.0 + ((alpha*pc)**n))**(-m)
        krw =    (se**0.5)  *(  (1.0 - (  (1.0 - se**m_inv)**m  )   )**2.0)
    else:
        krw = 1.0
    return krw

def vg_rel_perm_n(pc,alpha,n,m,m_inv,m2):
    """
    The nw phase relative permeability function
    """
    if pc > 0.0:
        se =  (1.0 + (alpha*pc)**n)**(-m)
        krn = ((1.0 - se)**0.5)*((1.0 - (se**(m_inv)))**(m2))
    else:
        krn = 0.0
    return krn

def effective_saturation(e_w,e_wr,e_nr,porosity):
    """
    Effective saturation
    """
    se = (e_w - e_wr) / (porosity-e_wr-e_nr)
    return se


def vg_capillary_pressure(e_w,e_wr,e_nr,porosity,m_inv,n_inv,alpha_inv):
    """
    Van-Genuchten capillary pressure
    """
    se = effective_saturation(e_w,e_wr,e_nr,porosity)
    if 0.0 < se < 1.0:
        p_c = (  ( se**(-m_inv)  - 1.0)**n_inv )*alpha_inv
    else:
        p_c = 0.0
    return p_c

def d_vg_capillary_pressure(e_w,e_wr,e_nr,porosity,alpha_m_n,m_inv,n_inv1):
    """
    dpc/dew
    """
    se = effective_saturation(e_w,e_wr,e_nr,porosity)
    if 0.0 <se <1.0:
        c0      = (alpha_m_n*(e_w-e_wr)*(se**m_inv))
        drvt_pc = -(( se**(-m_inv) - 1.0)**n_inv1 )*(1.0/c0)
    else:
        drvt_pc = 0.0
    return drvt_pc

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

def rel_perm_w_of_pc(pc,alpha,n,m,m_inv):
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

def volume_fraction(e_in,porosity):
    """caluculate volume fraction assuming 2P model"""
    e_out = porosity - e_in
    return e_out