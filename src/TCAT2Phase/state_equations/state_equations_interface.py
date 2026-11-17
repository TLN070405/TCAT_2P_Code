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

@dataclasses.dataclass
class BCdata:
    """
    Node info for set class
    """
    pd: float
    lamda: float

@dataclasses.dataclass
class TCATdata:
    """
    Node info for set class
    """
    b1: float
    b2: float
    b3: float
    D: float = 0

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

    def w_relative_permeability(self,p_c):
        """
        Relative permeability for w-phase
        """
        if p_c > 0.0:
            s_e = self.effective_saturation(p_c)
            k_wr = (s_e**0.5) * (( 1.0 - ((1.0 - s_e**self.data.m_inv)**self.data.m))**2.0)
        else:
            k_wr = 1.0
        return k_wr

    def n_relative_permeability(self,p_c):
        """
        Relative permeability for n-phase
        """
        if p_c > 0.0:
            s_e =  self.effective_saturation(p_c)
            k_nr = ((1.0 - s_e)**0.5)*((1.0 - (s_e**(self.data.m_inv)))**(self.data.m2))
        else:
            k_nr = 0.0
        return k_nr

class BrooksCorey:
    """
    Brooks-Corey Class to hold functions and parameters
    """
    def __init__(self,pd,lamda,porosity,e_wr,e_nr):
        """
        Initialize the Brooks-Corey parameters
        """
        self.data = BCdata(pd,lamda)
        self.porosity = porosity
        self.e_wr = e_wr
        self.e_nr = e_nr

    def capillary_pressure(self,e_w):
        """Calculates the Capillary Pressure as a function of ew"""
        se = effective_saturation(e_w, self.e_wr, self.e_nr, self.porosity)
        if 0.0 <se <1.0:
            p_c = self.data.pd*((se)**(-1.0/self.data.lamda))
        else:
            p_c = self.data.pd
        return p_c

    def d_capillary_pressure(self,e_w):
        """
        Derivative of capillary pressure over the w-phase folume fraction
        """
        se = effective_saturation(e_w, self.e_wr, self.e_nr, self.porosity)
        if 0.0 < se < 1.0:
            drvt_pc = (-self.data.pd/(self.data.lamda*(self.porosity - self.e_wr - self.e_nr)))*(se**((-1/self.data.lamda)-1))
        else:
            drvt_pc = 0.0
        return drvt_pc

    def volume_fraction_w(self,p_c):
        """
        Volume fraction of w phase from Brooks-Corey model
        """
        if p_c > self.data.pd:
            se  = self.effective_saturation(p_c)
            e_w = se * (self.porosity - self.e_wr - self.e_nr) + self.e_wr
        else:
            e_w =  self.porosity - self.e_nr
        return e_w

    def d_volume_fraction_w(self,p_c):
        """
        Derivative of w phase volume fraction
        """
        if p_c > self.data.pd:
            enr_wr  = self.porosity - self.e_wr - self.e_nr
            drvt_ew = enr_wr*(  (-self.data.lamda/self.data.pd)*( (p_c/self.data.pd)**(-self.data.lamda-1) ))
        else:
            drvt_ew = 0
        return drvt_ew

    def effective_saturation(self,p_c):
        """
        Effective saturation
        """
        return (p_c/self.data.pd )**(-self.data.lamda)

    def w_relative_permeability(self,p_c):
        """
        Relative permeability for w-phase
        """
        if p_c > self.data.pd:
            se = self.effective_saturation(p_c)
            k_wr = se**(3+(2/self.data.lamda))
        else:
            k_wr = 1.0
        return k_wr

    def n_relative_permeability(self,p_c):
        """
        Relative permeability for n-phase
        """
        if p_c > self.data.pd:
            se =  self.effective_saturation(p_c)
            k_nr = ((1.0-se)**2)*(1-(se**(1.0+(2/self.data.lamda))))
        else:
            k_nr = 0.0
        return k_nr

class TCAT:
    """
    TCAT Class to hold functions and parameters
    """
    def __init__(self,b1,b2,b3,porosity,e_ss,js_ns,kn_ns):
        """
        Initialize parameters
        """
        self.data = TCATdata(b1,b2,b3)
        self.porosity = porosity
        self.e_ss = e_ss
        self.js_ns = js_ns
        self.kn_ns = kn_ns
        self.D  = 6.0*porosity/e_ss
        self.D2 = self.D*self.D
        self.D3 = self.D2*self.D
        self.b1D  = b1*self.D
        self.b2D2 = b2*self.D2
        self.b3D3 = b3*self.D3

    def volume_fraction_w(self,e_wn,j_wn,k_wn):
        """
        Volume fraction of w phase from the linear geometric state equation
        (dissertation Eqn 3.51), the same relation gaussian_curvature solves:

            porosity - e_w = e_wn * W + e_ns * N
            W = b1 D - b2 D^2 J_wn + b3 D^3 K_wn
            N = b1 D + b2 D^2 J_ns + b3 D^3 K_ns
            e_ns = e_ss * (1 - e_w / porosity)

        Solving for e_w gives e_w = porosity - e_wn * W / (1 - e_ss * N / porosity),
        so this function is the exact inverse of gaussian_curvature.

        Note: the result is sensitive to K_wn when e_ss * N is close to the porosity
        (about 89% of it for the Finney pack parameters).
        """
        wn_term = self.b1D - self.b2D2 * j_wn + self.b3D3 * k_wn
        ns_term = self.b1D + self.b2D2 * self.js_ns + self.b3D3 * self.kn_ns

        e_w = self.porosity - e_wn * wn_term / (1.0 - self.e_ss * ns_term / self.porosity)
        return e_w

    def w_relative_permeability(self,e_w):
        """
        Equivalent Relative permeability for w-phase
        """
        k_wr = e_w * e_w
        return k_wr

    def n_relative_permeability(self,e_n):
        """
        Equivalent Relative permeability for n-phase
        """
        k_nr = e_n * e_n
        return k_nr

    def gaussian_curvature(self,e_w, e_wn,j_wn):
        """
        Gaussian curvature Kwn of the wn interface from linear geometric relation
        """
        e_ns = self.interfacial_area_ns(e_w)
        ewn_term = self.b1D  * (e_wn + e_ns )
        jwn_term = self.b2D2 * ( -e_wn * j_wn + e_ns *self.js_ns )

        k_wn = (((self.porosity - e_w - ewn_term - jwn_term)/self.b3D3) - e_ns * self.kn_ns)/e_wn
        return k_wn

    def interfacial_area_ns(self,e_w):
        """
        interfacial area of non wetting and solid phases
        """
        e_ns = self.e_ss*( 1-e_w/self.porosity )
        return e_ns

    def interfacial_area_ws(self,e_w):
        """
        interfacial area of wetting and solid phases
        """
        e_ws = self.e_ss*(e_w/self.porosity)
        return e_ws

def effective_saturation(e_w,e_wr,e_nr,porosity):
    """
    Effective saturation
    """
    se = (e_w - e_wr) / (porosity-e_wr-e_nr)
    return se
