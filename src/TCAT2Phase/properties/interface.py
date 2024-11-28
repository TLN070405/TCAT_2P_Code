"""Interface.py"""
from functools import partial

class Interface:
    """
    Interface class
    """
    def __init__(self):
        self.c_wn  = None
        self.gamma_wn  = None
        self.k_hat_wn = None
        self.e_wn_eq = None
        self.pc_function = None
        self.d_pc_function = None
        self.volume_fraction_function = None
        self.d_volume_fraction_function = None

    def set_c_wn(self,c_wn):
        """
        Set c_wn
        """
        self.c_wn = c_wn

    def set_gamma_wn(self,gamma_wn):
        """
        Set gamma_wn
        """
        self.gamma_wn = gamma_wn

    def set_k_hat_wn(self,k_hat_wn):
        """
        Set k_hat_wn
        """
        self.k_hat_wn = k_hat_wn

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
