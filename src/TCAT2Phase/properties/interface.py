"""Interface.py"""
from functools import partial

class Interface:
    """
    The base Interface class shares common objects and methods with both 
    the traditional and tcat interfaces.
    """
    def __init__(self, model_relations):
        self.volume_fraction_function = None
        self.w_relative_permeability = None
        self.set_volume_fraction_function(model_relations.volume_fraction_w)
        self.set_w_relative_permeability(model_relations.w_relative_permeability)

    def set_volume_fraction_function(self,volume_fraction_function,**parameters):
        """
        Set Volume Fraction
        """
        self.volume_fraction_function = partial(volume_fraction_function,**parameters)

    def set_w_relative_permeability(self,w_relative_permeability,**parameters):
        """
        Set relative permeability of wetting phase
        """
        self.w_relative_permeability = partial(w_relative_permeability,**parameters)


class TraditionalInterface(Interface):
    """
    Traditional Interface class used for 2P model problems
    """
    def __init__(self, psk):
        super().__init__(psk)
        self.pc_function = None
        self.d_pc_function = None
        self.d_volume_fraction_function = None
        self.n_relative_permeability = None
        self.set_pc_function(psk.capillary_pressure)
        self.set_d_pc_function(psk.d_capillary_pressure)
        self.set_d_volume_fraction_function(psk.d_volume_fraction_w)
        self.set_n_relative_permeability(psk.n_relative_permeability)

    def set_pc_function(self,pc_function,**parameters):
        """
        Set capillary pressure
        """
        self.pc_function = partial(pc_function,**parameters)

    def set_d_pc_function(self,d_pc_function,**parameters):
        """
        Set derivative of capillary pressure
        """
        self.d_pc_function = partial(d_pc_function,**parameters)

    def set_d_volume_fraction_function(self,d_volume_fraction_function,**parameters):
        """
        Set derivative of Volume Fraction 
        """
        self.d_volume_fraction_function = partial(d_volume_fraction_function,**parameters)

    def set_n_relative_permeability(self,n_relative_permeability,**parameters):
        """
        Set density
        """
        self.n_relative_permeability = partial(n_relative_permeability,**parameters)

class TCATInterface(Interface):
    """
    TCAT Interface class used for tcat cgp problems.
    """
    def __init__(self,tcat,interfacial_dynamics,
                           interfacial_tension,
                           interfacial_area_relaxation,
                           interfacial_area_equilibrium):
        super().__init__(tcat)
        self.interfacial_dynamics = interfacial_dynamics
        self.interfacial_tension = interfacial_tension
        self.interfacial_area_relaxation = interfacial_area_relaxation
        self.interfacial_area_equilibrium = interfacial_area_equilibrium
