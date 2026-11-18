"""Interface.py"""
from functools import partial

class Interface:
    """
    The base Interface class shares common objects and methods with both 
    the traditional and tcat interfaces.
    """
    def __init__(self, state_functions):
        # state_functions contains multiple functions necessary to close
        # the system of equations of the model  (2P or TCAT CGP)
        self.volume_fraction_function = self.set_volume_fraction_function(state_functions.volume_fraction_w)
        self.w_relative_permeability = self.set_w_relative_permeability(state_functions.w_relative_permeability)

    def set_volume_fraction_function(self,volume_fraction_function,**parameters):
        """
        Set Volume Fraction
        """
        return partial(volume_fraction_function,**parameters)

    def set_w_relative_permeability(self,w_relative_permeability,**parameters):
        """
        Set relative permeability of wetting phase
        """
        return partial(w_relative_permeability,**parameters)


class TraditionalInterface(Interface):
    """
    Traditional Interface class used for 2P model problems
    """
    def __init__(self, psk):
        super().__init__(psk)
        # psk is Pressure-Saturation-Relative premeability relations, could be Van Genuchten or Brooks-Corey
        # for now the model incorporates only this two psk relations.
        self.pc_function = self.set_pc_function(psk.capillary_pressure)
        self.d_pc_function = self.set_d_pc_function(psk.d_capillary_pressure)
        self.d_volume_fraction_function = self.set_d_volume_fraction_function(psk.d_volume_fraction_w)
        self.n_relative_permeability = self.set_n_relative_permeability(psk.n_relative_permeability)

    def set_pc_function(self,pc_function,**parameters):
        """
        Set capillary pressure
        """
        return partial(pc_function,**parameters)

    def set_d_pc_function(self,d_pc_function,**parameters):
        """
        Set derivative of capillary pressure
        """
        return partial(d_pc_function,**parameters)

    def set_d_volume_fraction_function(self,d_volume_fraction_function,**parameters):
        """
        Set derivative of Volume Fraction 
        """
        return partial(d_volume_fraction_function,**parameters)

    def set_n_relative_permeability(self,n_relative_permeability,**parameters):
        """
        Set relative permeability of non wetting phase
        """
        return partial(n_relative_permeability,**parameters)


class TCATInterface(Interface):
    """
    TCAT Interface class used for tcat cgp problems.
    """
    def __init__(self,tcat,interfacial_dynamics,
                           interfacial_tension,
                           interfacial_area_relaxation,
                           interfacial_area_equilibrium,
                           gaussian_curvature_sign=1.0):
        # tcat in the inputs is similar to psk relations in 2P model and it contains closure relations.
        super().__init__(tcat)
        # Sign s of the Gaussian-curvature term in the mean-curvature equation,
        #   e_wn dJ/dt + J de_wn/dt + s*2K de_w/dt = ...
        # with J the curvature seen from the non-wetting phase (p_c = gamma*J > 0).
        # s = +1 is the theory (Gray, Shepherd & Miller curvature-evolution manuscript,
        # Eqns 87/89 with de_n/dt = -de_w/dt and J_w = -J_n; dissertation Eqn 3.49).
        # s = -1 reproduces the original code behind the dissertation/defense results.
        if gaussian_curvature_sign not in (1, -1):
            raise ValueError("gaussian_curvature_sign must be +1 (theory) or -1 (legacy)")
        self.gaussian_curvature_sign = float(gaussian_curvature_sign)
        self.interfacial_dynamics = interfacial_dynamics
        self.interfacial_tension = interfacial_tension
        self.interfacial_area_relaxation = interfacial_area_relaxation
        self.interfacial_area_equilibrium = interfacial_area_equilibrium
        self.gaussian_curvature = self.set_gaussian_curvature_function(tcat.gaussian_curvature)
        self.n_relative_permeability = self.set_n_relative_permeability(tcat.n_relative_permeability)

    def set_gaussian_curvature_function(self,gaussian_curvature,**parameters):
        """
        Set gaussian curvature function from linear geometric relation
        """
        return partial(gaussian_curvature,**parameters)

    def set_n_relative_permeability(self,n_relative_permeability,**parameters):
        """
        Set relative permeability of non wetting phase
        """
        return partial(n_relative_permeability,**parameters)