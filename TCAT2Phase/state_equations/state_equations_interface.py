"""state_equations.py"""
import dataclasses


@dataclasses.dataclass
class VGdata:
    """
    Node info for set class
    """
    m: float
    n: float
    alpha: float
    m_inv: float = 0
    n_inv: float = 0
    alpha_inv: float = 0
    alpha_m_n: float = 0
    n_inv1: float = 0
    m1: float = 0
    n1: float = 0


class VanGenuchten:
    """
    VanGenuchten Class to hold functions and parameters
    """
    def __init__(self,m,n,alpha,w_phase,n_phase,s_phase):
        """
        Initialize the VanGenuchten parameters
        """
        self.data = VGdata(m,n,alpha)
        self._invert_params()
        self.w = w_phase
        self.n = n_phase
        self.s = s_phase

    def _invert_params(self):
        """
        Calculate the inverse of the parameters
        """ 
        self.data.m_inv = 1./self.data.m
        self.data.n_inv = 1./self.data.n
        self.data.alpha_inv = 1./self.data.alpha

    def capillary_pressure(self,e_w):
        """
        Van-Genuchten capillary pressure
        """
        se = effective_saturation(e_w,self.w.e_wr,self.n.e_nr,self.s.porosity)
        if 0.0 < se < 1.0:
            p_c = self.data.alpha_inv*((se **(-self.data.m_inv)  - 1.0)**self.data.n_inv )
        else:
            p_c = 0.0
        return p_c

    def d_capillary_pressure(self,e_w):
        """
        Derivative of capillary pressure over the w-phase folume fraction
        """
        se = effective_saturation(e_w,self.w.e_wr,self.n.e_nr,self.s.porosity)

        if 0.0 < se < 1.0:
            c0 = self.data.alpha_m_n * (e_w-self.w.e_wr) * ( se ** self.data.m_inv)
            drvt_pc = -1.0/c0 * ((se**(-self.data.m_inv) - 1.0)**self.data.n_inv1 )
        else:
            drvt_pc = 0.0
        return drvt_pc

    def volume_fraction_w(self,p_c):
        """
        Volume fraction of w phase from Van-Genuchten model
        """
        if p_c > 0.0:
            s_e =  self.effective_saturation(p_c)
            e_w = self.w.e_wr + s_e * (self.s.porosity - self.w.e_wr - self.n.e_nr)
        else:
            e_w = self.s.porosity - self.n.e_nr
        return e_w
    
    def d_volume_fraction_w(self,p_c):
        """
        Derivative of w phase volume fraction
        """

        if p_c > 0.0:
            drvt_se = -self.data.alpha_m_n*((self.data.alpha*p_c)**self.data.n1)*((1.0 + ((self.data.alpha*p_c)**self.data.n))**self.data.m1)
            drvt_ew = drvt_se * (self.s.porosity - self.w.e_wr - self.n.e_nr)
        else:
            drvt_ew = 0.0
        return drvt_ew

    def effective_saturation(self,p_c):
        """
        Effective saturation
        """
        return (1.0 + (self.data.alpha * p_c)**self.data.n)**(-self.data.m)




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

