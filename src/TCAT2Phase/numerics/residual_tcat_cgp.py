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

        mesh = self.mesh
        gravity = self.model.solid.gravity
        kw = 1./self.model.w_phase.momentum_resistance

        rho_w = utils.cell_center(primary_variables['p_w'], self.model.w_phase.density_function, mesh)

        k_rw = utils.cell_center(primary_variables['e_w'], self.model.interface.w_relative_permeability, mesh)

        del_pw = utils.gradient(primary_variables['p_w'], mesh)

        rho_w_face = utils.interpolate(rho_w, mesh)

        q_w = state_equations.darcy_velocity(mesh, kw, utils.interpolate(k_rw * rho_w, mesh), 1., del_pw, rho_w_face, gravity)

        return {'q_w':q_w,'rho_w':rho_w}

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
        cx= 1.0 / cell.length[0]
        cx2 = cx*cx

        # primary variables
        e_w = primary_variables['e_w']
        p_w = primary_variables['p_w']
        e_wn = primary_variables['e_wn']
        j_wn = primary_variables['j_wn']
        w_wn = primary_variables['w_wn']
        k_wn = primary_variables['k_wn']

        qe_up,qj_up,qe_down,qj_down =self.compute_primary_variables_gradient_lax_friedrichs(e_wn, j_wn, w_wn)

        for idx,node in enumerate ([n-1, n, n+1]):
            pw_jwn = p_w[node] + gamma_wn * j_wn[node]
            od_dp[idx]= c_wn * (pw_jwn)  - (gamma_wn * k1_wn_hat/p_w[node]) * (e_wn_eq[node] - e_wn[node])

        d2_od_dp = cx2*( od_dp[2] - 2. * od_dp[1] + od_dp[0] )

        # d(ewn*wwn)/dx
        d_ewn_wwn_dx =  cx*(qe_up-qe_down)

        #d(jwn*ewn*wwn)/dx = jwn*d(ewn*wwn)/dx + ewn*wwn* djwn/dx
        d_ewn_jwn_wwn_dx = cx*(qj_up-qj_down)

        #d(e_w)/dx
        d_e_w_d_x = 0.5*cx*(e_w[n+1]-e_w[n-1])

        # spatial operators
        od_w  = cx * (fluxes['q_w'][n + 1] - fluxes['q_w'][n])
        od_ia = -(k1_wn_hat *( p_w[n] + gamma_wn*j_wn[n] )/p_w[n] ) * (e_wn_eq[n] - e_wn[n]) - d_ewn_wwn_dx
        od_j  = (2./3.) * d2_od_dp - 2. * d_ewn_jwn_wwn_dx
        od_v  = e_w[n] * e_wn[n] * w_wn[n] + od_dp[1] * d_e_w_d_x
        od_k  = k_wn[n] - self.model.interface.gaussian_curvature(e_w[n], e_wn[n], j_wn[n])

        od =[od_dp[1], od_w, od_ia, od_j, od_v, od_k]
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

    def get_equilibrium_interfacial_area_array(self):
        """
        Function to transform the e_wn_eq(x) function to an array to be used directly
        in discretization of spatial operator.
        """
        nodes = self.model.nodes
        e_wn_eq  = np.zeros([nodes])

        for cell in self.mesh.cells:
            x = self.model.domain.domain_size[0,0] + cell.ID * cell.length[0]
            e_wn_eq[cell.ID] = self.model.interface.interfacial_area_equilibrium(x)

        return e_wn_eq

    def compute_primary_variables_gradient_lax_friedrichs(self, e_wn, j_wn, w_wn):
        """
        Function to compute gradient of primary variables and limiting the slope.
        """
        n = self.model.current_cell

        alpha = max(abs(w_wn[n]), abs(w_wn[n+1]))
        qe_up   = 0.5 * (w_wn[n] * e_wn[n] + w_wn[n+1] * e_wn[n+1]) - 0.5 * alpha * (e_wn[n+1] - e_wn[n])
        qj_up = 0.5 * (w_wn[n] * e_wn[n]* j_wn[n] + w_wn[n+1] * e_wn[n+1] * j_wn[n+1]) - 0.5 * alpha * (e_wn[n+1]*j_wn[n+1] - e_wn[n]*j_wn[n])
        alpha = max(abs(w_wn[n-1]), abs(w_wn[n]))
        qe_down = 0.5 * (w_wn[n] * e_wn[n] + w_wn[n-1] * e_wn[n-1]) - 0.5 * alpha * (e_wn[n] - e_wn[n-1])
        qj_down = 0.5 * (w_wn[n] * e_wn[n] * j_wn[n] + w_wn[n-1] * e_wn[n-1] * j_wn[n-1]) - 0.5 * alpha * (e_wn[n]*j_wn[n] - e_wn[n-1]*j_wn[n-1])
        
        return qe_up,qj_up,qe_down,qj_down

    def compute_primary_variables_gradient_upwind(self, e_wn, j_wn, w_wn):
        """
        Function to compute gradient of primary variables and limiting the slope.
        """
        n = self.model.current_cell
        w_wn_up   = 0.5*(w_wn[n] + w_wn[n+1])
        w_wn_down = 0.5*(w_wn[n] + w_wn[n-1])

        if w_wn_up>=0:
            qe_up = w_wn_up*e_wn[n]
            qj_up = w_wn_up*e_wn[n]*j_wn[n]
        else:
            qe_up = w_wn_up*e_wn[n+1]
            qj_up = w_wn_up*e_wn[n+1]*j_wn[n+1]

        if w_wn_down>=0:
            qe_down = w_wn_down*e_wn[n-1]
            qj_down = w_wn_down*e_wn[n-1]*j_wn[n-1]
        else:
            qe_down  = w_wn_down*e_wn[n]
            qj_down = w_wn_down*e_wn[n]*j_wn[n]

        return qe_up,qj_up,qe_down,qj_down
