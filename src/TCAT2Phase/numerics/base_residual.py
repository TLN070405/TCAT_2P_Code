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
        self._time = 0.0

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
        self._time = t
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
            source = getattr(self.model, "source", None)
            if source is not None:
                # model.source(x, t): one value per equation, added to the spatial
                # operator, so that time_mat . udot = od + source
                od = od + np.asarray(source(self.mesh.points[n][0], self._time), dtype=float)
            self.compute_time_matrix(primary_variables,fluxes)
            f[n] = utils.petsc_value(self.compute_residual_equations(udot, od))

        # boundary nodes with mass-conservative flux conditions (types 4 and 5)
        for side in (0, 1):
            if self.conservative_flux_boundary(side):
                self.compute_conservative_boundary_residual(f, udot, primary_variables, fluxes, side)

    def conservative_flux_boundary(self, side):
        """
        True when the boundary on this side (0 left, 1 right) uses the mass-
        conservative flux condition: boundary type 4 (w-phase mass flux) on one
        unknown and type 5 (n-phase mass flux) on the other.
        """
        u_names = self.model.model.u_names
        types = sorted(self.model.boundary_type[u][0][side] for u in u_names)
        if 4 not in types and 5 not in types:
            return False
        if types != [4, 5]:
            raise ValueError("Boundary types 4 and 5 (mass-conservative fluxes) must be used together, "
                             "one per unknown, at the same boundary; got " + str(types))
        return True

    def compute_conservative_boundary_residual(self, f, udot, primary_variables, fluxes, side):
        """
        Boundary types 4 and 5: the boundary node gets the same two mass balances
        as an interior node, over its half cell (length dx/2), with the
        prescribed mass fluxes on the outer face. So the boundary node stores
        mass, as in the cell-centered schemes of Kees & Miller (2002) and
        Szymkiewicz (2007), instead of the algebraic rows of types 1 and 2
        (zero storage, flux set on the face next to the boundary).

        Prescribed values use the code's flux convention,
        q = rho k k_r / mu (dp/dx + rho g) = -(rho * Darcy velocity in +x), so 0
        is no flow; at x = 0 the value equals the outward normal mass flux, and
        at x = L it is minus the outward normal mass flux.
        """
        u_names = self.model.model.u_names
        n = 0 if side == 0 else self.nodes - 1
        self.model.current_cell = n
        cell = self.mesh.cells[n]
        half = 0.5 * float(np.ravel(cell.length)[0])
        q_bc = {}
        for k, u in enumerate(u_names):
            btype = self.model.boundary_type[u][0][side]
            key = 'q_w' if btype == 4 else 'q_n'
            q_bc[key] = self.check_callable_boundary_conditions(self.model.boundary_conditions[k][0][side], self._time)
        if side == 0:
            od = [(fluxes['q_w'][1] - q_bc['q_w']) / half, (fluxes['q_n'][1] - q_bc['q_n']) / half]
        else:
            od = [(q_bc['q_w'] - fluxes['q_w'][n]) / half, (q_bc['q_n'] - fluxes['q_n'][n]) / half]
        self.compute_time_matrix(primary_variables, fluxes)
        f[n] = utils.petsc_value(self.compute_residual_equations(udot, od))

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

        if getattr(self.model, "mobility_weighting", "arithmetic") == "upwind":
            # phase-wise upstream weighting on the sign of the potential gradient
            mob_w = utils.upwind(k_rw * rho_w, del_pw + rho_w_face * gravity, mesh)
            mob_n = utils.upwind(k_rn * rho_n, del_pn + rho_n_face * gravity, mesh)
        else:
            mob_w = utils.interpolate(k_rw * rho_w, mesh)
            mob_n = utils.interpolate(k_rn * rho_n, mesh)

        q_w = state_equations.darcy_velocity(mesh, k, mob_w, mu_w, del_pw, rho_w_face, gravity)
        q_n = state_equations.darcy_velocity(mesh, k, mob_n, mu_n, del_pn, rho_n_face, gravity)

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

        bc_values = np.zeros(self.model.model.num_u)
        bc_flux = np.zeros(self.model.model.num_u)
        bc_type  = np.zeros(self.model.model.num_u)

        for i, j in enumerate([0, self.model.nodes-1]):
            # Residual at Boundaries left and right
            if self.conservative_flux_boundary(i):
                continue   # mass balance set in compute_internal_residual
            for n in range(self.model.model.num_u):
                # check if bc are time dependent or just constant bc and store all bc values for node "j"
                bc_values[n] = self.check_callable_boundary_conditions(boundary_conditions[n][0][i],t)
                if boundary_type[u_names[n]][0][i] == 0:
                    # if bc type is Dirichlet than bc_type[n]  = 0
                    bc_type[n]  = 0
                elif boundary_type[u_names[n]][0][i] == 1:
                    # If bc type is 1 or 2 (Neumman) than compute the correspendant flux
                    bc_flux[n] = self.compute_boundary_flux(self.model.w_phase, u, i)
                    bc_type[n]  = 1
                elif boundary_type[u_names[n]][0][i] == 2:
                    # Same as line above, just compute q_n flux instead of q_w
                    bc_flux[n] = self.compute_boundary_flux(self.model.n_phase, u, i)
                    bc_type[n]  = 1
                elif boundary_type[u_names[n]][0][i] == 3:
                    # Same as line above, just compute q_n flux instead of q_w
                    bc_flux[n] =  (u[j+(1-i)][n] -  u[j-i][n])
                    bc_type[n]  = 1
                else:
                    print("Boundary type non supported")
            f[:][j]= utils.petsc_value((1-bc_type) * u[:][j] + bc_type * bc_flux - bc_values)

    def compute_boundary_flux(self, phase, u, side):
        """
        Mass flux of a phase next to a boundary, for flux (Neumann) boundary conditions.

        side: 0 for the left boundary (x = x0), 1 for the right boundary (x = xL).

        Returns the flux at the face between the boundary node and its neighbor,
        computed exactly as in the interior residual (compute_phases_fluxes), so
        it uses the same sign convention: q = rho k k_r / mu (dp/dx + rho g).
        A prescribed value of 0 gives a no-flow boundary.
        """
        primary_variables = self.compute_primary_variables(u)
        fluxes = self.compute_phases_fluxes(primary_variables)

        key = 'q_' + phase.phase
        if key not in fluxes:
            raise ValueError(f"This model does not compute a flux for phase '{phase.phase}'")

        face = 1 if side == 0 else self.model.nodes - 1
        return fluxes[key][face]

    def check_callable_boundary_conditions(self,boundary_conditions,t):
        """
        Function to check if the boundary condition is time dependent function or just constant.
        """
        return (boundary_conditions(t) if callable(boundary_conditions) else boundary_conditions)
