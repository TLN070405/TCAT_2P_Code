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
    def __init__(self,phase_name):
        super().__init__(phase_name)
        self.viscosity = 1.
        self.residual_volume_fraction = 0.
        self.density_function = None
        self.d_density_function = None
        self.rel_perm_function = None
        self.volume_fraction_function = None
        self.d_volume_fraction_function = None


    def set_viscosity(self,viscosity):
        """
        Set viscosity
        """
        self.viscosity = viscosity

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

class Solid(Phase):
    """
    Solid phase class
    """
    def __init__(self,phase_name):
        super().__init__(phase_name)
        self.porosity = 0.
        self.permeability = 0.
        self.gravity = 0.
        self.dynamic_capillary_pressure_parameter = 0.
    def set_porosity(self,porosity):
        """
        Set porosity
        """
        self.porosity = porosity

    def set_permeability(self,permeability):
        """
        Set permeability
        """
        self.permeability = permeability

    def set_gravity(self,gravity):
        """
        Set gravity constant
        """
        self.gravity = gravity

    def set_dynamic_capillary_pressure_parameter(self,dynamic_capillary_pressure_parameter):
        """
        Set gravity constant
        """
        self.dynamic_capillary_pressure_parameter = dynamic_capillary_pressure_parameter