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

    def set_rel_perm_function(self,rel_perm_function,**parameters):
        """
        Set density
        """
        self.rel_perm_function = partial(rel_perm_function,**parameters)

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
                             w_den_params):
    """
    Initialize wetting phase
    """
    w_phase = Fluid('w')
    w_phase.set_viscosity(viscosity)
    w_phase.set_momentum_resistance(momentum_resistance)
    w_phase.set_residual_volume_fraction(residual_volume_fraction)
    w_phase.set_density_function(state_equations.density_exp_pressure,**w_den_params)
    w_phase.set_d_density_function(state_equations.d_density_exp_pressure,**w_den_params)

    return w_phase

def initialize_nonwetting_phase(viscosity,
                                momentum_resistance,
                                residual_volume_fraction,
                                n_den_params):
    """
    Initialize non wetting phase
    """
    n_phase = Fluid('n')
    n_phase.set_viscosity(viscosity)
    n_phase.set_momentum_resistance(momentum_resistance)
    n_phase.set_residual_volume_fraction(residual_volume_fraction)
    n_phase.set_density_function(state_equations.density_linear_pressure,**n_den_params)
    n_phase.set_d_density_function(state_equations.d_density_linear_pressure,**n_den_params)

    return n_phase


def initialize_solid_phase(porosity,
                           permeability,
                           gravity,
                           dynamic_capillary_pressure_parameter):
    """
    Initialize solid phase
    """
    s_phase = Solid('s')
    s_phase.set_porosity(porosity)
    s_phase.set_permeability(permeability)
    s_phase.set_gravity(gravity)
    s_phase.set_dynamic_capillary_pressure_parameter(dynamic_capillary_pressure_parameter)

    return s_phase
