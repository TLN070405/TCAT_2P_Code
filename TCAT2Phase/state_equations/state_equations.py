"""state_equations.py"""
import math


def effective_satuaration(ew,Epsilon_wr,Const_Epsilon):
    """
    Effective saturation
    """
    se = (ew - Epsilon_wr) / Const_Epsilon
    return se


def vg_capillary_pressure(ew):
    """
    Van-Genuchten capillary pressure
    """
    se = effective_satuaration(ew,Epsilon_w,Const_Epsilon)
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

def  krw_of_pc(pc):
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
    volume fraction from Van-Genuchten model
    """
    if pc > 0.0:
        se =  (1.0 + (alpha*pc)**n)**(-m)
        ew = Const_Epsilon *  se + Epsilon_wr
    else:
        ew = epsilon - Epsilon_nr
    return ew

def rhow_of_pw(pw):
    """
    density of wetting phase
    """
    rhow = rho_w0 * math.exp(Beta_w * pw)
    return rhow

def  drvt_rhow_of_pw(pw):
    """
    d(rohw)/d(pw)
    """
    drvt_rho_w = Beta_w * rho_w0 * math.exp(Beta_w * pw)
    return drvt_rho_w
