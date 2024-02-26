"""phase.py"""

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
        self.viscosity = 0
        self.density = 0

    def set_viscosity(self,viscosity):
        """
        Set viscosity
        """
        self.viscosity = viscosity

    def set_density(self,density):
        """
        Set density
        """
        self.density = density

class Solid(Phase):
    """
    Solid phase class
    """
    def __init__(self,phase_name):
        super().__init__(phase_name)
        self.porosity = 0
        self.permeability = 0

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