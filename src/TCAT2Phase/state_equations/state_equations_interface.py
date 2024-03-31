"""interface.py"""
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
    m2: float = 0


class VanGenuchten:
    """
    VanGenuchten Class to hold functions and parameters
    """
    def __init__(self,m,n,alpha,porosity,e_wr,e_nr):
        """
        Initialize the VanGenuchten parameters
        """
        self.data = VGdata(m,n,alpha)
        self._invert_params()
        self.porosity = porosity
        self.e_wr = e_wr
        self.e_nr = e_nr

    def _invert_params(self):
        """
        Calculate the inverse of the parameters
        """ 
        self.data.m_inv = 1./self.data.m
        self.data.n_inv = 1./self.data.n
        self.data.alpha_inv = 1./self.data.alpha
        self.data.alpha_m_n = self.data.alpha*self.data.m*self.data.n
        self.data.n_inv1    = self.data.n_inv-1.0
        self.data.m1        = -self.data.m-1.0
        self.data.n1        = self.data.n-1.0
        self.data.m2        =  2.0*self.data.m

    def capillary_pressure(self,e_w):
        """
        Van-Genuchten capillary pressure
        """
        se = effective_saturation(e_w, self.e_wr, self.e_nr, self.porosity)
        if 0.0 < se < 1.0:
            p_c = self.data.alpha_inv*((se **(-self.data.m_inv)  - 1.0)**self.data.n_inv )
        else:
            p_c = 0.0
        return p_c

    def d_capillary_pressure(self,e_w):
        """
        Derivative of capillary pressure over the w-phase folume fraction
        """
        se = effective_saturation(e_w, self.e_wr, self.e_nr, self.porosity)

        if 0.0 < se < 1.0:
            c0 = self.data.alpha_m_n * (e_w - self.e_wr) * ( se ** self.data.m_inv)
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
            e_w = self.e_wr + s_e * (self.porosity - self.e_wr - self.e_nr)
        else:
            e_w = self.porosity - self.e_nr
        return e_w
    
    def d_volume_fraction_w(self,p_c):
        """
        Derivative of w phase volume fraction
        """
        if p_c > 0.0:
            drvt_se = -self.data.alpha_m_n*((self.data.alpha*p_c)**self.data.n1)*((1.0 + ((self.data.alpha*p_c)**self.data.n))**self.data.m1)
            drvt_ew = drvt_se * (self.porosity - self.e_wr - self.e_nr)
        else:
            drvt_ew = 0.0
        return drvt_ew

    def effective_saturation(self,p_c):
        """
        Effective saturation
        """
        return (1.0 + (self.data.alpha * p_c)**self.data.n)**(-self.data.m)

    def w_relative_permability(self,p_c):
        """
        Relative permeability for w-phase
        """
        if p_c > 0.0:
            s_e = self.effective_saturation(p_c)
            k_wr = (s_e**0.5) * (( 1.0 - ((1.0 - s_e**self.data.m_inv)**self.data.m))**2.0)
        else:
            k_wr = 1.0
        return k_wr

    def n_relative_permability(self,p_c):
        """
        Relative permeability for n-phase
        """
        if p_c > 0.0:
            s_e =  self.effective_saturation(p_c)
            k_nr = ((1.0 - s_e)**0.5)*((1.0 - (s_e**(self.data.m_inv)))**(self.data.m2))
        else:
            k_nr = 0.0
        return k_nr


def effective_saturation(e_w,e_wr,e_nr,porosity):
    """
    Effective saturation
    """
    se = (e_w - e_wr) / (porosity-e_wr-e_nr)
    return se