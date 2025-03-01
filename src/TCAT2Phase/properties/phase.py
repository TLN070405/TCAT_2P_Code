"""phase.py"""
from functools import partial
from TCAT2Phase.state_equations import state_equations

class Phase:
    """
    Phase class
    """
    def __init__(self,phase_name):
        self.phase = phase_name

class Fluid(Phase):
    """
    Fluid phase class
    """
    def __init__(self,phase_name):
        super().__init__(phase_name)
        self.viscosity = None
        self.momentum_resistance      = None
        self.residual_volume_fraction = None
        self.density_function = None
        self.d_density_function = None
        self.rel_perm_function = None

    def set_viscosity(self,viscosity):
        """
        Set viscosity
        """
        self.viscosity = viscosity

    def set_momentum_resistance(self,momentum_resistance):
        """
        Set momentum_resistance Rw_w
        """
        self.momentum_resistance = momentum_resistance

    def set_residual_volume_fraction(self,residual_volume_fraction):
        """
        Set irreducible phase volume fraction
        """
        self.residual_volume_fraction = residual_volume_fraction

    def set_density_function(self,density_function,**parameters):
        """
        Set density
        """
        self.density_function = partial(density_function,**parameters)

    def set_d_density_function(self,d_density_function,**parameters):
        """
        Set derivative density
        """
        self.d_density_function = partial(d_density_function,**parameters)


class Solid(Phase):
    """
    Solid phase class
    """
    def __init__(self,phase_name):
        super().__init__(phase_name)
        self.porosity = None
        self.permeability = None
        self.gravity = None
        self.dynamic_capillary_pressure_parameter = None

    def set_porosity(self,porosity):
        """
        Set porosity
        """
        self.porosity = porosity

    def set_permeability(self,permeability):
        """
        Set permeability constant
        """
        self.permeability = permeability

    def set_gravity(self,gravity):
        """
        Set gravity constant
        """
        self.gravity = gravity

    def set_dynamic_capillary_pressure_parameter(self,dynamic_capillary_pressure_parameter):
        """
        Set dynamic_capillary_pressure_parameter  constant
        """
        self.dynamic_capillary_pressure_parameter = dynamic_capillary_pressure_parameter


def initialize_wetting_phase(viscosity,
                             momentum_resistance,
                             residual_volume_fraction,
                             w_den_params,
                             density_function_type='exp'):
    """
    Initialize wetting phase
    This function create a wetting phase and set all its necessary parameters.
    It sets also the density functions and its derivative rhow and drhow/dpw.
    This function is just for creating and setting up an instance of the Fluid() class 
    to make create of applications easier.
    """
    w_phase = Fluid('w')
    w_phase.set_viscosity(viscosity)
    w_phase.set_momentum_resistance(momentum_resistance)
    w_phase.set_residual_volume_fraction(residual_volume_fraction)
    if density_function_type=='linear':
        w_phase.set_density_function(state_equations.density_linear_pressure,**w_den_params)
        w_phase.set_d_density_function(state_equations.d_density_linear_pressure,**w_den_params)
    elif density_function_type=='exp':
        w_phase.set_density_function(state_equations.density_exp_pressure,**w_den_params)
        w_phase.set_d_density_function(state_equations.d_density_exp_pressure,**w_den_params)
    else:
        print("density_function_type tooks the values 'linear' or 'exp' only")

    return w_phase

def initialize_nonwetting_phase(viscosity,
                                momentum_resistance,
                                residual_volume_fraction,
                                n_den_params,
                                density_function_type='linear'):
    """
    Initialize non wetting phase
    This function create a nonwetting phase and set all its necessary parameters.
    It sets also the density functions and its derivative rhon and drhon/dpn
    This function is just for creating and setting up an instance of the Fluid() class
    to make create of applications easier.
    """
    n_phase = Fluid('n')
    n_phase.set_viscosity(viscosity)
    n_phase.set_momentum_resistance(momentum_resistance)
    n_phase.set_residual_volume_fraction(residual_volume_fraction)
    if density_function_type=='linear':
        n_phase.set_density_function(state_equations.density_linear_pressure,**n_den_params)
        n_phase.set_d_density_function(state_equations.d_density_linear_pressure,**n_den_params)
    elif density_function_type=='exp':
        n_phase.set_density_function(state_equations.density_exp_pressure,**n_den_params)
        n_phase.set_d_density_function(state_equations.d_density_exp_pressure,**n_den_params)
    else:
        print("density_function_type tooks the values 'linear' or 'exp' only")
    return n_phase


def initialize_solid_phase(porosity,
                           permeability,
                           gravity,
                           dynamic_capillary_pressure_parameter):
    """
    Initialize solid phase
    Create the solid phase and set its parameters. This function is just for creating and setting up
    an instance of the Solid() class to make create of applications easier.
    """
    s_phase = Solid('s')
    s_phase.set_porosity(porosity)
    s_phase.set_permeability(permeability)
    s_phase.set_gravity(gravity)
    s_phase.set_dynamic_capillary_pressure_parameter(dynamic_capillary_pressure_parameter)

    return s_phase
