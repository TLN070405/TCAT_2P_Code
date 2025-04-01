import numpy as np
from TCAT2Phase.numerics.base_residual import BaseResidual

class ResidualE2pSPP(BaseResidual):
    """
    class holds the residual of the system of equations
    SPP: saturation, pressure w, pressure n
    """

    def compute_primary_variables(self,u):
        """
        function to get and compute solution and primary variables for 2 phases SPP formulation
        """
        # Solution Variables
        e_w = u[:, 0]
        p_w = u[:, 1]
        p_n = u[:, 2]
        # Primary Variables
        p_c = np.zeros_like(p_w)
        for n in range(0, self.nodes):
            # we assign the current cell to model so it can be used by the hysteresis interface functions.
            self.model.current_cell = n
            p_c[n] = self.model.interface.pc_function(e_w[n])
        e_n = self.model.solid.porosity - e_w
        return {'e_w':e_w,'e_n':e_n,'p_w':p_w,'p_n':p_n,'p_c':p_c}

    def compute_spatial_operator(self,primary_variables,fluxes,cell):
        """
        function to calculate the spatial operator
        """
        n = self.model.current_cell
        od_w = 1.0 / cell.length * (fluxes['q_w'][n + 1] - fluxes['q_w'][n])
        od_n = 1.0 / cell.length * (fluxes['q_n'][n + 1] - fluxes['q_n'][n])
        od_dp = primary_variables['p_w'][n] - primary_variables['p_n'][n] + primary_variables['p_c'][n]
        od = [od_dp,od_w[0], od_n[0]]
        return od

    def compute_time_matrix(self,primary_variables,fluxes):
        """
        function to calculate the time matrix
        """
        n = self.model.current_cell
        d_rho_w_p_w = self.model.w_phase.d_density_function(primary_variables['p_w'][n])
        d_rho_n_p_n = self.model.n_phase.d_density_function(primary_variables['p_n'][n])

        self.time_mat[0, 0] = (self.model.solid.dynamic_capillary_pressure_parameter / self.model.solid.porosity)
        self.time_mat[1, 0] = fluxes['rho_w'][n]
        self.time_mat[1, 1] = primary_variables['e_w'][n] * d_rho_w_p_w
        self.time_mat[2, 0] = -fluxes['rho_n'][n]
        self.time_mat[2, 2] = primary_variables['e_n'][n] * d_rho_n_p_n
