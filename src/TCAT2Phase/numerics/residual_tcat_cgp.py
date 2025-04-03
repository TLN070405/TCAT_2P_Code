import numpy as np
from TCAT2Phase.numerics import utils
from TCAT2Phase.state_equations import state_equations
from TCAT2Phase.numerics.base_residual import BaseResidual

class ResidualTcatCGP(BaseResidual):
    """
    class holds the residual of the system of equations
    """
    def __init__(self, model):
        # Call the __init__ method of BaseResidual
        super().__init__(model)
        self.e_wn_eq = self.get_equilibrium_interfacial_area_array()
        self.od_dp = np.zeros(3)
    
    def compute_primary_variables(self,u):
        """
        function to get and compute solution and primary variables for 2 phases SP formulation
        """
        # Solution Variables
        e_w = u[:, 0]
        p_w = u[:, 1]
        e_wn = u[:, 2]
        j_wn = u[:, 3]
        w_wn = u[:, 4]
        k_wn = u[:, 5]

        return {'e_w':e_w,'e_wn':e_wn,'p_w':p_w,'j_wn':j_wn,'w_wn':w_wn,'k_wn':k_wn}

    def compute_phases_fluxes(self,primary_variables):
        """
        function to calculate phases fluxes
        """
        e_w = primary_variables['e_w']
        p_w = primary_variables['p_w']
        e_wn = primary_variables['e_wn']
        j_wn = primary_variables['j_wn']
        w_wn = primary_variables['w_wn']
        k_wn = primary_variables['k_wn']

        mesh = self.mesh
        gravity = self.model.solid.gravity
        k = 1./self.model.w_phase.momentum_resistance
        mu_w = 1.0

        rho_w = utils.cell_center(primary_variables['p_w'], self.model.w_phase.density_function, mesh)

        k_rw = utils.cell_center(primary_variables['e_w'], self.model.interface.w_relative_permeability, mesh)

        del_pw = utils.gradient(primary_variables['p_w'], mesh)

        rho_w_face = utils.interpolate(rho_w, mesh)

        q_w = state_equations.darcy_velocity(mesh, k, utils.interpolate(k_rw * rho_w, mesh), mu_w, del_pw, rho_w_face, gravity)

        d_e_w_d_x, d_e_wn_d_x, d_j_wn_d_x, d_w_wn_d_x = self.compute_primary_variables_gradient(e_w, e_wn, j_wn, w_wn)

        return {'q_w':q_w,'rho_w':rho_w, 'd_e_w_d_x':d_e_w_d_x, 'd_e_wn_d_x':d_e_wn_d_x, 'd_j_wn_d_x':d_j_wn_d_x, 'd_w_wn_d_x':d_w_wn_d_x}

    def compute_spatial_operator(self,primary_variables,fluxes,cell):
        """
        function to calculate correspondent time matrix
        """
        n = cell.ID
        # model parameters
        c_wn = self.model.interface.interfacial_dynamics
        gamma_wn = self.model.interface.interfacial_tension
        k1_wn_hat = self.model.interface.interfacial_area_relaxation
        e_wn_eq  = self.e_wn_eq

        # spatial operator array
        od_dp = self.od_dp

        # intermediate calculation parameters
        dx2 = cell.length[0]*cell.length[0]

        # primary variables
        e_w = primary_variables['e_w']
        p_w = primary_variables['p_w']
        e_wn = primary_variables['e_wn']
        j_wn = primary_variables['j_wn']
        w_wn = primary_variables['w_wn']
        k_wn = primary_variables['k_wn']

        # fluxes
        q_w = fluxes['q_w']

        # primary variables gradients
        d_e_w_d_x = fluxes['d_e_w_d_x']
        d_e_wn_d_x = fluxes['d_e_wn_d_x']
        d_j_wn_d_x = fluxes['d_j_wn_d_x']
        d_w_wn_d_x = fluxes['d_w_wn_d_x']

        for idx,node in enumerate ([n-1, n, n+1]):
            pw_jwn = p_w[node] + gamma_wn * j_wn[node]
            od_dp[idx]= c_wn * (pw_jwn)  - (gamma_wn * k1_wn_hat/p_w[node]) * (e_wn_eq[node] - e_wn[node])

        d2_od_dp = ( od_dp[2] - 2 * od_dp[1] + od_dp[0] )/dx2

        # d(ewn*wwn)/dx
        d_ewn_wwn_dx = w_wn[n] * d_e_wn_d_x[n] + e_wn[n] * d_w_wn_d_x[n]

        #d(jwn*ewn*wwn)/dx = jwn*d(ewn*wwn)/dx + ewn*wwn* djwn/dx
        d_ewn_jwn_wwn_dx =j_wn[n] * d_ewn_wwn_dx  + e_wn[n] * w_wn[n] * d_j_wn_d_x[n]

        # spatial operators
        od_w  = (1.0 / cell.length) * (q_w[n + 1] - q_w[n])
        od_ia = -(k1_wn_hat *( p_w[n] + gamma_wn*j_wn[n] )/p_w[n] ) * (e_wn_eq[n] - e_wn[n]) - d_ewn_wwn_dx
        od_j  = (2/3) * d2_od_dp - 2 * d_ewn_jwn_wwn_dx
        od_v  = e_w[n] * e_wn[n] * w_wn[n] + od_dp[1] * d_e_w_d_x[n]
        od_k  = k_wn[n] - self.model.interface.gaussian_curvature(e_w[n], e_wn[n], j_wn[n])
        #od_k  = (e_w[n] - self.model.interface.volume_fraction_function(e_wn[n], j_wn[n], k_wn[n]))

        od =[od_dp[1], od_w[0], od_ia, od_j, od_v, od_k]
        return od

    def compute_time_matrix(self,primary_variables,fluxes):
        """
        function to calculate correspondent time matrix
        """
        n = self.model.current_cell
        d_rho_w_p_w = self.model.w_phase.d_density_function(primary_variables['p_w'][n])

        self.time_mat[0, 0] = 1
        self.time_mat[1, 0] = fluxes['rho_w'][n]
        self.time_mat[1, 1] = primary_variables['e_w'][n] * d_rho_w_p_w
        self.time_mat[2, 0] = primary_variables['j_wn'][n]
        self.time_mat[2, 2] = 1
        self.time_mat[3, 0] = -2*primary_variables['k_wn'][n]
        self.time_mat[3, 2] = primary_variables['j_wn'][n]
        self.time_mat[3, 3] = primary_variables['e_wn'][n]

    def compute_primary_variables_gradient(self,e_w, e_wn, j_wn, w_wn):
        """
        gradient of primary variables
        """
        d_e_w_d_x = utils.gradient(e_w, self.mesh)
        d_e_wn_d_x = utils.gradient(e_wn, self.mesh)
        d_j_wn_d_x = utils.gradient(j_wn, self.mesh)
        d_w_wn_d_x = utils.gradient(w_wn, self.mesh)

        return d_e_w_d_x, d_e_wn_d_x, d_j_wn_d_x, d_w_wn_d_x

    def get_equilibrium_interfacial_area_array(self):
        """
        Function to transform the e_wn_eq(x) function to an array to be used directly
        in discretization of spatial operator.
        """
        nodes = self.model.nodes
        e_wn_eq  = np.zeros([nodes])

        for n in range(0,nodes):
            x = self.model.domain.domain_size[0,0] + n* self.model.domain.domain_length[0]/(nodes-1)
            e_wn_eq[n] = self.model.interface.interfacial_area_equilibrium(x)
        return e_wn_eq
