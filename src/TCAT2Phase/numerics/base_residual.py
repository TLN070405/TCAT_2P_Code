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

        self.model.interface.primary_variables = primary_variables

        fluxes = self.compute_phases_fluxes(primary_variables)

        for cell in self.mesh.cells:
            if cell.boundary:
                continue
            n = cell.ID
            self.model.current_cell = n

            od   = self.compute_spatial_operator(primary_variables,fluxes,cell)
            self.compute_time_matrix(primary_variables,fluxes)
            f[n] = self.compute_residual_equations(udot, od)

    def compute_primary_variables(self,u):
        """
        function to get and compute solution and primary variables depending on model and formulation
        """

        raise NotImplementedError("This function should be implemented in subclass residual")

    def compute_phases_fluxes(self,primary_variables):
        """
        function to compute the phases fluxes depending on the model and formualtion
        """
        mesh = self.model.mesh
        gravity = self.model.solid.gravity
        k = self.model.solid.permeability
        mu_w = self.model.w_phase.viscosity
        mu_n = self.model.n_phase.viscosity

        rho_w = utils.cell_center(primary_variables['p_w'], self.model.w_phase.density_function, mesh)
        rho_n = utils.cell_center(primary_variables['p_n'], self.model.n_phase.density_function, mesh)

        k_rw = utils.cell_center(primary_variables['p_c'], self.model.w_phase.relative_permeability_function, mesh)
        k_rn = utils.cell_center(primary_variables['p_c'], self.model.n_phase.relative_permeability_function, mesh)

        del_pw = utils.gradient(primary_variables['p_w'], mesh)
        del_pn = utils.gradient(primary_variables['p_n'], mesh)

        rho_w_face = utils.interpolate(rho_w, mesh)
        rho_n_face = utils.interpolate(rho_n, mesh)

        q_w = state_equations.darcy_velocity(mesh, k, utils.interpolate(k_rw * rho_w, mesh), mu_w, del_pw, rho_w_face, gravity)
        q_n = state_equations.darcy_velocity(mesh, k, utils.interpolate(k_rn * rho_n, mesh), mu_n, del_pn, rho_n_face, gravity)

        return {'q_w':q_w,'q_n':q_n,'rho_w':rho_w,'rho_n':rho_n}

    def compute_spatial_operator(self,primary_variables,fluxes,cell):
        """
        function to compute the spatial operator of the system of equations depending on model and formulation
        [time_matrix].udot =spatial_operator
        """

        raise NotImplementedError("This function should be implemented in subclass residual")

    def compute_time_matrix(self,primary_variables,fluxes):
        """
        function to compute the time matrix of the system of equations depending on model and formulation
        [time_matrix].udot =spatial_operator
        """
        raise NotImplementedError("This function should be implemented in subclass residual")

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
        # bc_vlaues is an array of size (number of variables of the formulation)
        # it is used to store all bc_values of one specific boundary node (0 or nodes-1)
        # It is used to define the residual as a whole and not one by one.
        bc_values = np.zeros(self.model.model.num_u)
        # fluxes_bc is used to store all fluxes bc, calculated using "u"
        fluxes_bc = np.zeros(self.model.model.num_u)
        # bc_type_0 is used to know which one of the bc_values is flux and which one is dirichlet
        # it is 0 for DIrchlet and 0 for Neumman (q_w or q_n)
        bc_type0  = np.zeros(self.model.model.num_u)

        for i, j in enumerate([0, self.model.nodes-1]):
            # Residual at Boundaries left and right
            for n in range(self.model.model.num_u):
                # check if bc are time dependent or just constant bc and store all bc values for node "j"
                bc_values[n] = self.check_callable_boundary_conditions(boundary_conditions[n][0][i],t)
                if boundary_type[u_names[n]][0][i] == 0:
                    # if bc type is Dirichlet than no flux
                    fluxes_bc[n] = 0
                    # if bc type is Dirichlet than bc_type0 is 1
                    bc_type0[n]  = 1
                elif boundary_type[u_names[n]][0][i] == 1:
                    # If bc type is 1 or 2 (Neumman) than compute the correspendant flux
                    fluxes_bc[n] = self.compute_boundary_flux(self.model.w_phase,u)
                    # bc type0 is 0 here because its Neumman BC
                    bc_type0[n]  = 0
                elif boundary_type[u_names[n]][0][i] == 2:
                    # Same as line above, just compute q_n flux instead of q_w
                    fluxes_bc[n] = self.compute_boundary_flux(self.model.n_phase,u)
                    bc_type0[n]  = 0
                else:
                    print("Boundary type non supported")
            # here we combine both the Dirichlet and Neumman bc into one code line
            # this line computes the residual of all solution variables at boundary node "j"
            # If we print f[j] we get an array(number of variables) and not one single value
            #  bc_type0 * u[:][j] +(1-bc_type0) * fluxes_bc : this formula combines Dirichlet and Neumman
            # And it allows to detect which variable has Dirichlet and which one is Neumman.
            f[j]= bc_type0 * u[:][j] +(1-bc_type0) * fluxes_bc - bc_values

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
