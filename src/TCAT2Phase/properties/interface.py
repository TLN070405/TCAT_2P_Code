"""Interface.py"""
from functools import partial

class Interface:
    """
    Interface class
    """
    def __init__(self):
        self.pc_function = None
        self.d_pc_function = None
        self.pc_hysteretic_function = None
        self.d_pc_hysteretic_function = None
        self.volume_fraction_hysteretic_function = None
        
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

    def set_pc_hysteretic_function(self,pc_hysteretic_function,**parameters):
        """
        Set hysteretic capillary pressure
        """
        self.pc_hysteretic_function = partial(pc_hysteretic_function,**parameters)

    def set_d_pc_hysteretic_function(self,d_pc_hysteretic_function,**parameters):
        """
        Set derivative of hysteretic capillary pressure
        """
        self.d_pc_hysteretic_function = partial(d_pc_hysteretic_function,**parameters)

    def set_volume_fraction_hysteretic_function(self,volume_fraction_hysteretic_function,**parameters):
        """
        Set hysteretic volume fraction  
        """
        self.volume_fraction_hysteretic_function = partial(volume_fraction_hysteretic_function,**parameters)
