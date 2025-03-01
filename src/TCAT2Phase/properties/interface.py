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
        self.w_relative_permeability = None
        self.n_relative_permeability = None

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

    def set_w_relative_permeability(self,w_relative_permeability,**parameters):
        """
        Set density
        """
        self.w_relative_permeability = partial(w_relative_permeability,**parameters)

    def set_n_relative_permeability(self,n_relative_permeability,**parameters):
        """
        Set density
        """
        self.n_relative_permeability = partial(n_relative_permeability,**parameters)

def initialize_traditional_interface(psk):
    """
    Set up the w-n interphase properties
    This function setup the interface object for traditional 2P problems,
    It setup the necessary function of the PSK relations for applications, to make 
    the preparation of applications easier.
    This function is not a member of the Interface() class because its only role is to 
    create instances of the Interface() calss and preparation of the interface obeject for applications.
    """
    #create an interface object
    wn_interface = Interface()
    # set the PSK relations (VG or BC) provided by the user
    wn_interface.set_pc_function(psk.capillary_pressure)
    wn_interface.set_d_pc_function(psk.d_capillary_pressure)
    wn_interface.set_volume_fraction_function(psk.volume_fraction_w)
    wn_interface.set_d_volume_fraction_function(psk.d_volume_fraction_w)
    wn_interface.set_w_relative_permeability(psk.w_relative_permeability)
    wn_interface.set_n_relative_permeability(psk.n_relative_permeability)

    return wn_interface

def initialize_tcat_interface(tcat,
                              interfacial_dynamics,
                              interfacial_tension,
                              interfacial_area_relaxation,
                              interfacial_area_equilibrium):
    """
    Set up the interphase properties
    This function setup the interface of tcat model, extra parameters related to tcat are needed.
    The traditional and tcat interfaces are quite different, so two different functions are implemented.
    The tcat input is the equivalent of psk input in initialize_traditional_interface function.
    This function is not a member of the Interface() class because its only role is to 
    create instances of the Interface() calss and preparation of the interface obeject for applications.
    """
    #create an interface object
    wn_interface = Interface()
    # set necessary constant parameters for tcat interface
    wn_interface.set_interfacial_dynamics(interfacial_dynamics)
    wn_interface.set_interfacial_tension(interfacial_tension)
    wn_interface.set_interfacial_area_relaxation(interfacial_area_relaxation)
    # set the equilibrium interfacial area which is a function of space e_wn(x) and not necessarly constant
    wn_interface.set_interfacial_area_equilibrium(interfacial_area_equilibrium)
    # Set volume fraction function from tcat relations provided by user as input
    # The volume fraction function for now includes only the linear geometric relation
    wn_interface.set_volume_fraction_function(tcat.volume_fraction_w)
    # Set the relative permeability function. In fact, the tcat model doesn't have a relative premeability
    #function, but by comapraison of tcat first equation to traditional model we named this function as
    # relative permeability because they have the same units.
    wn_interface.set_w_relative_permeability(tcat.w_relative_permability)

    return wn_interface
