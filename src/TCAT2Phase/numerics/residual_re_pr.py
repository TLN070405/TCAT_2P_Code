import numpy as np
from TCAT2Phase.numerics.base_residual import BaseResidual

class ResidualRePR(BaseResidual):
    """
    class holds the residual of the system of equations
    Pr: pressure w
    """

    def compute_primary_variables(self,u):
        """
        function to get and compute solution and primary variables for 2 phases PP formulation
        """
        # Solution Variables
        p_w = u[:]
        # Primary Variables
        e_w = np.zeros_like(p_w)
        p_n = np.zeros_like(p_w)
        p_c = -p_w
        for n in range(0, self.nodes):
            self.model.current_cell = n
            e_w[n] = self.model.interface.volume_fraction_function(p_c[n])
        e_n = self.model.solid.porosity - e_w

        return {'e_w':e_w,'e_n':e_n,'p_w':p_w,'p_n':p_n,'p_c':p_c}

    def compute_spatial_operator(self,primary_variables,fluxes,cell):
        """
        function to calculate correspondent time matrix
        """
        n = self.model.current_cell
        od_w = 1.0 / cell.length * (fluxes['q_w'][n + 1] - fluxes['q_w'][n])
        od = od_w[0]
        return od

    def compute_time_matrix(self,primary_variables,fluxes):
        """
        function to calculate correspondent time matrix
        """
        n = self.model.current_cell
        d_rho_w_p_w = self.model.w_phase.d_density_function(primary_variables['p_w'][n])
        d_e_w_p_c   = self.model.interface.d_volume_fraction_function(primary_variables['p_c'][n])

        self.time_mat[0, 0] =  -fluxes['rho_w'][n] * d_e_w_p_c + primary_variables['e_w'][n] * d_rho_w_p_w
