import numpy as np
import TCAT2Phase
from TCAT2Phase.numerics import utils
from TCAT2Phase.state_equations import state_equations

class BaseResidual:
    """
    class holds the residual of the system of equations
    """
    def __init__(self, ts, model):
        self.ts = ts
        self.model = model
        self.mesh = model.mesh
        self.da = ts.getDM()
        self.primary_variables_form = 1
        self.phase_fluxes_form= 1
        self.spatial_operator_form = 1
        self.time_matrix_form = 1

    def compute_residual(self, t, U, Udot, F):
        """
        function to compute the whole system equations residuals
        """
        localU = self.da.getLocalVec()
        self.da.globalToLocal(U, localU)

        u = self.da.getVecArray(localU)
        udot = self.da.getVecArray(Udot)
        f = self.da.getVecArray(F)

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

            od       = self.compute_spatial_operator(primary_variables,fluxes)
            time_mat = self.compute_time_matrix(primary_variables,fluxes)
            f[n]     = self.compute_residual_equations(time_mat, udot, od)

    def compute_primary_variables(self,u):
        """
        function to get and compute solution and primary variables depending on model and formulation
        """
        if self.model.form in self.primary_variables_form:
            form_primary_variables = self.primary_variables_form[self.model.form](u)
        else:
            print("Error Model Form is unknown")
        e_n = self.model.solid.porosity - form_primary_variables['e_w']

        primary_variables = {'e_n': e_n, **form_primary_variables}
        return primary_variables

    def compute_phases_fluxes(self,primary_variables):
        """
        function to compute the phases fluxes depending on the model and formualtion
        """
        if self.model.form in self.phase_fluxes_form:
            fluxes = self.phase_fluxes_form[self.model.form](primary_variables)
        else:
            print("Error Model Form is unknown")
        return fluxes

    def compute_spatial_operator(self,primary_variables,fluxes):
        """
        function to compute the spatial operator of the system of equations depending on model and formulation
        [time_matrix].udot =spatial_operator
        """
        if self.model.form in self.spatial_operator_form:
            od = self.spatial_operator_form[self.model.form](primary_variables,fluxes)
        else:
            print("Error Model Form is unknown")
        return od

    def compute_time_matrix(self,primary_variables,fluxes):
        """
        function to compute the time matrix of the system of equations depending on model and formulation
        [time_matrix].udot =spatial_operator
        """
        if self.model.form in self.time_matrix_form:
            time_mat = self.time_matrix_form[self.model.form](primary_variables,fluxes)
        else:
            print("Error Model Form is unknown")
        return time_mat

    def compute_residual_equations(self,time_mat, udot, od):
        """
        function to combine the spatial operator and time matrix to get the residual equations 
        """
        n = self.model.current_cell
        f = np.dot(time_mat, udot[n]) - od
        return f

    def compute_residual_boundary(self,t, f, u):
        """
        function to compute boundary residuals
        """
        boundary_type = self.model.boundary_type
        boundary_conditions = self.model.boundary_conditions
        u_names = self.model.model.u_names

        for n in range(self.model.model.num_u):
            # Residual at Boundaries left and right
            for i, j in enumerate([0, self.model.nodes-1]):
                bc_value = self.check_callable_boundary_conditions(boundary_conditions[n][0][i],t)
                index =(j,) * (len(u.shape) - 1) + (slice(n),)
                if boundary_type[u_names[n]][0][i] == 0:
                    f[index] = u[index] - bc_value
                elif boundary_type[u_names[n]][0][i] == 1:
                    q = self.compute_boundary_flux(j,self.model.w_phase)
                    f[index]  = q -  bc_value
                elif boundary_type[u_names[n]][0][i] == 2:
                    q = self.compute_boundary_flux(j,self.model.n_phase)
                    f[index]  = q -  bc_value
                else:
                    print("Boundary type non supported")

    def compute_boundary_flux(self,phase,u):
        """
        function to compute phases fluxes at boundary points
        """
        # get primary variables
        primary_variables  = self.compute_primary_variables(u)

        #create a boundary mesh consisting of one cell

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
