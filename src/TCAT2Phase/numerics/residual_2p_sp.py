import numpy as np
from TCAT2Phase.numerics.base_residual import BaseResidual

class Residual2pSP(BaseResidual):
    """
    class holds the residual of the system of equations
    SP: saturation w, pressure w
    """

    def compute_primary_variables(self,u):
        """
        function to get and compute solution and primary variables for 2 phases SP formulation
        """
        # Solution Variables
        e_w = u[:, 0]
        p_w = u[:, 1]
        # Primary Variables
        p_c = np.zeros_like(p_w)
        for n in range(0, self.nodes):
            # we assign the current cell to model so it can be used by the hysteresis interface functions.
            self.model.current_cell = n
            p_c[n] = self.model.interface.pc_function(e_w[n])
        p_n = p_w + p_c
        e_n = self.model.solid.porosity - e_w
        return {'e_w':e_w,'e_n':e_n,'p_w':p_w,'p_n':p_n,'p_c':p_c}

    def compute_spatial_operator(self,primary_variables,fluxes,cell):
        """
        function to calculate correspondent time matrix
        """
        n = self.model.current_cell
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
        d_p_c_e_w  = self.model.interface.d_pc_function(primary_variables['e_w'][n])

        self.time_mat[0, 0] = fluxes['rho_w'][n]
        self.time_mat[0, 1] = primary_variables['e_w'][n] * d_rho_w_p_w
        self.time_mat[1, 0] = primary_variables['e_n'][n] * d_rho_n_p_n * d_p_c_e_w -fluxes['rho_n'][n]
        self.time_mat[1, 1] = primary_variables['e_n'][n] * d_rho_n_p_n

    def compute_residual_boundary(self, t, f, u):
        """
        Boundary residuals, as in BaseResidual, except where one boundary has no-
        flow-type conditions for both phases: e_w with a w-phase flux (type 1) and
        p_w with an n-phase flux (type 2). The w-phase flux depends on e_w only
        through k_rw times the pressure gradient, so in the e_w row its derivative
        with respect to e_w is zero whenever the gradient is (e.g. a uniform
        initial state), a zero pivot that stops LU and SOR. The n-phase flux
        depends on e_w through p_n = p_w + p_c(e_w), so the two rows are swapped:
        the solution is unchanged and the Jacobian diagonal is nonzero.
        """
        super().compute_residual_boundary(t, f, u)
        names = self.model.model.u_names
        for side, node in enumerate([0, self.model.nodes - 1]):
            types = (self.model.boundary_type[names[0]][0][side], self.model.boundary_type[names[1]][0][side])
            if types == (1, 2):
                row = np.array(f[node], dtype=float)
                f[node] = row[::-1]
