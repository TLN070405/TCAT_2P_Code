import numpy as np
from TCAT2Phase.numerics import utils
from TCAT2Phase.state_equations import state_equations
from TCAT2Phase.numerics.base_residual import BaseResidual

class ResidualPP(BaseResidual):
    """
    class holds the residual of the system of equations
    PP: pressure w, pressure n
    """

    def compute_primary_variables(self,u):
        """
        function to get and compute solution and primary variables for 2 phases PP formulation
        """
        # Solution Variables
        p_w = u[:, 0]
        p_n = u[:, 1]
        # Primary Variables
        p_c = p_n - p_w
        e_w = np.zeros_like(p_w)
        for n in range(0, self.nodes):
            # we assign the current cell to model so it can be used by the hysteresis interface functions.
            self.model.current_cell = n
            e_w[n] = self.model.interface.volume_fraction_function(p_c[n])
        e_n = self.model.solid.porosity - e_w
        return {'e_w':e_w,'e_n':e_n,'p_w':p_w,'p_n':p_n,'p_c':p_c}

    def compute_phases_fluxes(self,primary_variables):
        """
        function to calculate phases fluxes
        """
        mesh = self.model.mesh
        gravity = self.model.solid.gravity
        k = self.model.solid.permeability
        mu_w = self.model.w_phase.viscosity
        mu_n = self.model.n_phase.viscosity

        rho_w = utils.cell_center(primary_variables['p_w'], self.model.w_phase.density_function, mesh)
        rho_n = utils.cell_center(primary_variables['p_n'], self.model.n_phase.density_function, mesh)

        k_rw = utils.cell_center(primary_variables['p_c'], self.model.interface.w_relative_permeability, mesh)
        k_rn = utils.cell_center(primary_variables['p_c'], self.model.interface.n_relative_permeability, mesh)

        del_pw = utils.gradient(primary_variables['p_w'], mesh)
        del_pn = utils.gradient(primary_variables['p_n'], mesh)

        rho_w_face = utils.interpolate(rho_w, mesh)
        rho_n_face = utils.interpolate(rho_n, mesh)

        q_w = state_equations.darcy_velocity(mesh, k, utils.interpolate(k_rw * rho_w, mesh), mu_w, del_pw, rho_w_face, gravity)
        q_n = state_equations.darcy_velocity(mesh, k, utils.interpolate(k_rn * rho_n, mesh), mu_n, del_pn, rho_n_face, gravity)

        return {'q_w':q_w,'q_n':q_n,'rho_w':rho_w,'rho_n':rho_n}

    def compute_spatial_operator(self,primary_variables,fluxes):
        """
        function to calculate correspondent time matrix
        """
        n = self.model.current_cell
        cell = self.model.cell[n]  # to be verified
        od_w = 1.0 / cell.length * (fluxes['q_w'][n + 1] - fluxes['q_w'][n])
        od_n = 1.0 / cell.length * (fluxes['q_n'][n + 1] - fluxes['q_n'][n])
        od = [od_w[0], od_n[0]]
        return od

    def compute_time_matrix(self,primary_variables,fluxes):
        """
        function to calculate correspondent time matrix
        """
        n = self.model.current_cell

        d_rho_w_p_w = self.model.w_phase.d_density_function(primary_variables['p_w'][n])
        d_rho_n_p_n = self.model.n_phase.d_density_function(primary_variables['p_n'][n])
        d_e_w_p_c   = self.model.interface.d_volume_fraction_function(primary_variables['p_c'][n])

        self.time_mat[0, 0] = -fluxes['rho_w'][n] * d_e_w_p_c + primary_variables['e_w'][n] * d_rho_w_p_w
        self.time_mat[0, 1] =  fluxes['rho_w'][n] * d_e_w_p_c
        self.time_mat[1, 0] =  fluxes['rho_n'][n] * d_e_w_p_c
        self.time_mat[1, 1] = -fluxes['rho_n'][n] * d_e_w_p_c + primary_variables['e_n'][n] * d_rho_n_p_n

