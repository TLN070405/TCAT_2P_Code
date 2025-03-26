import numpy as np
from TCAT2Phase.numerics import utils
from TCAT2Phase.state_equations import state_equations

class BaseResidual:
    """
    Base class for residual computation.
    Other residual classes will inherit from this.
    """
    def __init__(self, model):
        self.model = model
        self.nodes = model.nodes
        self.mesh = model.mesh
        self.time_mat = np.zeros([model.model.num_u, model.model.num_u])

    def compute_residual(self,ts, t, U, Udot, F):
        """
        function to compute the whole system equations residuals
        """
        da = ts.getDM()
        localU = da.getLocalVec()
        da.globalToLocal(U, localU)

        u = da.getVecArray(localU)
        udot = da.getVecArray(Udot)
        f = da.getVecArray(F)

        # calculate the residual at interior nodes
        self.compute_internal_residual(f, u, udot)

        # calculate the residual at boundaries
        self.compute_residual_boundary(t, f, u)

    def compute_internal_residual(self, f, u, udot):
        """
        function to compute residuals at interior nodes
        """
        primary_variables = self.compute_primary_variables(u)

        fluxes = self.compute_phases_fluxes(primary_variables)

        for cell in self.mesh.cells:
            if cell.boundary:
                continue
            n = cell.ID
            self.model.current_cell = n

            od       = self.compute_spatial_operator(primary_variables,fluxes,cell)
            self.compute_time_matrix(primary_variables,fluxes)
            f[n]     = self.compute_residual_equations (udot, od)

    def compute_primary_variables(self,u):
        """
        function to get and compute solution and primary variables depending on model and formulation
        """

        raise NotImplementedError("This function ")

    def compute_phases_fluxes(self,primary_variables):
        """
        function to compute the phases fluxes depending on the model and formualtion
        """
        raise NotImplementedError("This function ")

    def compute_spatial_operator(self,primary_variables,fluxes,cell):
        """
        function to compute the spatial operator of the system of equations depending on model and formulation
        [time_matrix].udot =spatial_operator
        """

        raise NotImplementedError("This function ")

    def compute_time_matrix(self,primary_variables,fluxes):
        """
        function to compute the time matrix of the system of equations depending on model and formulation
        [time_matrix].udot =spatial_operator
        """
        raise NotImplementedError("This function ")

    def compute_residual_equations(self, udot, od):
        """
        function to combine the spatial operator and time matrix to get the residual equations 
        """
        n = self.model.current_cell
        f = np.dot(self.time_mat, udot[n]) - od
        return f

    def compute_residual_boundary(self,t, f, u):
        """
        function to compute boundary residuals
        """
        boundary_type = self.model.boundary_type
        boundary_conditions = self.model.boundary_conditions
        u_names = self.model.model.u_names
        bc_array = np.zeros(self.model.model.num_u)
        fluxes_bc = np.zeros(self.model.model.num_u)
        bc_type0  = np.zeros(self.model.model.num_u)

        for i, j in enumerate([0, self.model.nodes-1]):
            # Residual at Boundaries left and right
            for n in range(self.model.model.num_u):
                # check if bc are time dependent or just constant bc
                bc_array[n] = self.check_callable_boundary_conditions(boundary_conditions[n][0][i],t)
                if boundary_type[u_names[n]][0][i] == 0:
                    fluxes_bc[n] = 0
                    bc_type0[n]  = 1
                elif boundary_type[u_names[n]][0][i] == 1:
                    fluxes_bc[n] = self.compute_boundary_flux(j,self.model.w_phase)
                    bc_type0[n]  = 0
                elif boundary_type[u_names[n]][0][i] == 2:
                    fluxes_bc[n] = self.compute_boundary_flux(j,self.model.n_phase)
                    bc_type0[n]  = 0
                else:
                    print("Boundary type non supported")
            f[j]= bc_type0 * u[:][j] +(1-bc_type0) * fluxes_bc - bc_array


    def compute_boundary_flux(self,phase,u):
        """
        function to compute phases fluxes at boundary points
        """
        # get primary variables
        primary_variables  = self.compute_primary_variables(u)

        # Cell Centered Variables
        if phase.phase =='w':
            rho = utils.cell_center(primary_variables['p_w'], self.model.w_phase.density_function, self.mesh.boundary)
            k_r = utils.cell_center(primary_variables['p_c'], self.model.interface.w_relative_permeability, self.mesh.boundary)
            del_p = utils.gradient(primary_variables['p_w'], self.mesh.boundary)
        elif phase.phase =='n':
            rho = utils.cell_center(primary_variables['p_n'], self.model.n_phase.density_function, self.mesh.boundary)
            k_r = utils.cell_center(primary_variables['p_c'], self.model.interface.n_relative_permeability, self.mesh.boundary)
            del_p = utils.gradient(primary_variables['p_n'], self.mesh.boundary)
        else:
            print("phase should be 'n' or 'w'")
        # Flux
        q =  state_equations.darcy_velocity(
                self.mesh.boundary,
                self.model.solid.permeability,
                k_r*rho,
                self.model.phase.viscosity,
                del_p,
                rho,
                self.model.solid.gravity)
        return  q

    def check_callable_boundary_conditions(self,boundary_conditions,t):
        """
        Function to check if the boundary condition is time dependent function or just constant.
        """
        return (boundary_conditions(t) if callable(boundary_conditions) else boundary_conditions)
