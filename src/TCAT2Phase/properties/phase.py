"""phase.py"""
from functools import partial

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
    def __init__(self,phase_name,viscosity,
                                 momentum_resistance,
                                 residual_volume_fraction,
                                 density_function,
                                 d_density_function,
                                 **parameters):
        super().__init__(phase_name)
        self.viscosity = viscosity
        self.momentum_resistance = momentum_resistance
        self.residual_volume_fraction = residual_volume_fraction
        self.density_function = self.set_density_function(density_function,**parameters)
        self.d_density_function = self.set_d_density_function(d_density_function,**parameters)
        self.relative_permeability_function = None

    def set_density_function(self,density_function,**parameters):
        """
        Set density
        """
        return partial(density_function,**parameters)

    def set_d_density_function(self,d_density_function,**parameters):
        """
        Set derivative density
        """
        return partial(d_density_function,**parameters)

    def set_relative_permeability_function(self,relative_permeability_function,**parameters):
        """
        Set density
        """
        self.relative_permeability_function = partial(relative_permeability_function,**parameters)

class Solid(Phase):
    """
    Solid phase class
    """
    def __init__(self,phase_name, porosity, permeability,
                                            gravity,
                                            dynamic_capillary_pressure_parameter):
        super().__init__(phase_name)
        self.porosity     = porosity
        self.permeability = permeability
        self.gravity      = gravity
        self.dynamic_capillary_pressure_parameter = dynamic_capillary_pressure_parameter
