"""Interface.py"""
from functools import partial

class Interface:
    """
    Interface class
    """
    def __init__(self):
        self.interfacial_dynamics  = None
        self.interfacial_tension  = None
        self.interfacial_area_relaxation = None
        self.interfacial_area_equilibrium = None
        self.pc_function = None
        self.d_pc_function = None
        self.volume_fraction_function = None
        self.d_volume_fraction_function = None

    def set_interfacial_dynamics(self,interfacial_dynamics):
        """
        Set interfacial_dynamics (c_wn =porosity/tau)
        """
        self.interfacial_dynamics = interfacial_dynamics

    def set_interfacial_tension(self,interfacial_tension):
        """
        Set interfacial_tension (gamma_wn)
        """
        self.interfacial_tension = interfacial_tension

    def set_interfacial_area_relaxation(self,interfacial_area_relaxation):
        """
        Set interfacial_area_relaxation (k_hat_wn)
        """
        self.interfacial_area_relaxation = interfacial_area_relaxation

    def set_interfacial_area_equilibrium(self,interfacial_area_equilibrium):
        """
        Set interfacial_area_equilibrium e_wn_eq
        """
        self.interfacial_area_equilibrium = interfacial_area_equilibrium

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

    def set_volume_fraction_function(self,volume_fraction_function,**parameters):
        """
        Set Volume Fraction
        """
        self.volume_fraction_function = partial(volume_fraction_function,**parameters)

    def set_d_volume_fraction_function(self,d_volume_fraction_function,**parameters):
        """
        Set derivative of Volume Fraction 
        """
        self.d_volume_fraction_function = partial(d_volume_fraction_function,**parameters)
