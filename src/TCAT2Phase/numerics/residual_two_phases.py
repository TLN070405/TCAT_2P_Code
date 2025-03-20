import numpy as np
import TCAT2Phase
from TCAT2Phase.numerics import utils
from TCAT2Phase.state_equations import state_equations

class Residual:
    """
    class holds the residual of the system of equations
    """
    def __init__(self, model):
        self.model = model
        self.nodes = model.nodes
        self.primary_variables_form = {
            "extended-two-phase-spp":self.primary_variables_two_phase_spp,
            "two-phase-sp":self.primary_variables_two_phase_sp,
            "two-phase-pp":self.primary_variables_two_phase_pp}
        self.phase_fluxes_form= {
            "extended-two-phase-spp":self.phases_fluxes_two_phase,
            "two-phase-sp":self.phases_fluxes_two_phase,
            "two-phase-pp":self.phases_fluxes_two_phase}
        self.spatial_operator_form = {
            "extended-two-phase-spp":self.spatial_operator_two_phase_spp,
            "two-phase-sp":self.spatial_operator_two_phase_sp_pp,
            "two-phase-pp":self.spatial_operator_two_phase_sp_pp}
        self.time_matrix_form = {
            "extended-two-phase-spp":self.time_matrix_two_phase_spp,
            "two-phase-sp":self.time_matrix_two_phase_sp,
            "two-phase-pp":self.time_matrix_two_phase_pp}

    def primary_variables_two_phase_spp(self,u):
        """
        spp: saturation, pressure w, pressure n
        """
        # Solution Variables
        e_w = u[:, 0]
        p_w = u[:, 1]
        p_n = u[:, 2]
        # Primary Variables
        p_c = np.zeros_like(p_w)
        for n in range(0, self.nodes):
            self.model.current_cell = n
            p_c[n] = self.model.interface.pc_function(e_w[n])

        primary_variables={'e_w':e_w,'p_w':p_w,'p_n':p_n,'p_c':p_c}
        return primary_variables

    def primary_variables_two_phase_sp(self,u):
        """
        sp: saturation, pressure w
        """
        # Solution Variables
        e_w = u[:, 0]
        p_w = u[:, 1]
        # Primary Variables
        p_c = np.zeros_like(p_w)
        p_n = np.zeros_like(p_w)
        for n in range(0, self.nodes):
            self.model.current_cell = n
            p_c[n] = self.model.interface.pc_function(e_w[n])
            p_n[n] = p_w[n] + p_c[n]

        primary_variables={'e_w':e_w,'p_w':p_w,'p_n':p_n,'p_c':p_c}
        return primary_variables

    def primary_variables_two_phase_pp(self,u):
        """
        pp: pressure w, pressure n
        """
        # Solution Variables
        p_w = u[:, 0]
        p_n = u[:, 1]
        # Primary Variables
        p_c = np.zeros_like(p_w)
        e_w = np.zeros_like(p_w)
        for n in range(0, self.nodes):
            self.model.current_cell = n
            p_c[n] = p_n[n] - p_w[n]
            e_w[n] = self.model.interface.volume_fraction_function(p_c[n])

        primary_variables={'e_w':e_w,'p_w':p_w,'p_n':p_n,'p_c':p_c}
        return primary_variables

    def phases_fluxes_two_phase(self,primary_variables):
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

        fluxes = {'q_w':q_w,'q_n':q_n,'rho_w':rho_w,'rho_n':rho_n}
        return fluxes

    def spatial_operator_two_phase_sp_pp(self,primary_variables,fluxes):
        """
        function to calculate correspondent time matrix
        """
        n = self.model.current_cell
        cell = self.model.cell[n]  # to be verified
        od_w = 1.0 / cell.length * (fluxes['q_w'][n + 1] - fluxes['q_w'][n])
        od_n = 1.0 / cell.length * (fluxes['q_n'][n + 1] - fluxes['q_n'][n])
        od = [od_w[0], od_n[0]]
        return od

    def spatial_operator_two_phase_spp(self,primary_variables,fluxes):
        """
        function to calculate correspondent time matrix
        """
        n = self.model.current_cell
        od_pp = self.spatial_operator_two_phase_sp_pp(primary_variables,fluxes)
        od_dp = primary_variables['p_w'][n] - primary_variables['p_n'][n] + primary_variables['p_c'][n]
        od = [od_dp, od_pp[0], od_pp[1]]

        return od

    def time_matrix_two_phase_spp(self,mat,primary_variables,fluxes):
        """
        function to calculate correspondent time matrix
        """
        n = self.model.current_cell
        e_n_d_rho_n, e_w_d_rho_w = self.common_time_matrix_components(primary_variables)

        mat[0, 0] = (self.model.solid.dynamic_capillary_pressure_parameter / self.model.solid.porosity)
        mat[1, 0] = fluxes['rho_w'][n]
        mat[1, 1] = e_w_d_rho_w
        mat[2, 0] = -fluxes['rho_n'][n]
        mat[2, 2] = e_n_d_rho_n
        return mat

    def time_matrix_two_phase_sp(self,mat,primary_variables,fluxes):
        """
        function to calculate correspondent time matrix
        """
        n = self.model.current_cell
        e_n_d_rho_n, e_w_d_rho_w = self.common_time_matrix_components(primary_variables)
        d_p_c_e_w  = self.model.interface.d_pc_function(primary_variables['e_w'][n])

        mat[0, 0] = fluxes['rho_w'][n]
        mat[0, 1] = e_w_d_rho_w
        mat[1, 0] = e_n_d_rho_n * d_p_c_e_w -fluxes['rho_n'][n]
        mat[1, 1] = e_n_d_rho_n
        return mat

    def time_matrix_two_phase_pp(self,mat,primary_variables,fluxes):
        """
        function to calculate correspondent time matrix
        """
        n = self.model.current_cell
        e_n_d_rho_n, e_w_d_rho_w = self.common_time_matrix_components(primary_variables)
        d_e_w_p_c   = self.model.interface.d_volume_fraction_function(primary_variables['p_c'][n])

        mat[0, 0] = -fluxes['rho_w'][n] * d_e_w_p_c + e_w_d_rho_w
        mat[0, 1] =  fluxes['rho_w'][n] * d_e_w_p_c
        mat[1, 0] =  fluxes['rho_n'][n] * d_e_w_p_c
        mat[1, 1] = -fluxes['rho_n'][n] * d_e_w_p_c + e_n_d_rho_n
        return mat

    def common_time_matrix_components(self,primary_variables):
        """
        function to calculate correspondent time matrix
        """
        n = self.model.current_cell
        d_rho_w_p_w = self.model.w_phase.d_density_function(primary_variables['p_w'][n])
        d_rho_n_p_n = self.model.n_phase.d_density_function(primary_variables['p_n'][n])

        e_n_d_rho_n = primary_variables['e_n'][n] * d_rho_n_p_n
        e_w_d_rho_w = primary_variables['e_w'][n] * d_rho_w_p_w

        return e_n_d_rho_n, e_w_d_rho_w
