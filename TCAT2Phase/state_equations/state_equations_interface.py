"""state_equations.py"""
import math

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
