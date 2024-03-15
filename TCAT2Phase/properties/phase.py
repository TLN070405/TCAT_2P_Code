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
        self.viscosity = None
        self.e_wr = None
        self.e_nr = None
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

    def set_e_wr(self,e_wr):
        """
        Set e_wr or irreducible wetting phase volume fraction
        """
        self.e_wr = e_wr    

    def set_e_nr(self,e_nr):
        """
        Set e_nr or residual nonwetting phase volume fraction
        """
        self.e_nr = e_nr

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
        self.porosity = None
        self.permeability = None

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
