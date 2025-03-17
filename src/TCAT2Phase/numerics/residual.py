import numpy as np
import TCAT2Phase
from TCAT2Phase.numerics import utils
from TCAT2Phase.state_equations import state_equations


class Residual:
    """
    class holds the residual of the system of equations
    """
    def __init__(self, ts, model):
        self.ts = ts
        self.model = model
        self.da = ts.getDM()

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
        ResidualInternal(f, u, udot, self.model).compute()
        # calculate the residual at boundaries
        ResidualBoundary(t, f, u, self.model).compute()

class ResidualInternal:
    """
    class that holds informations and methods to compute residuals at interior domain
    """
    def __init__(self, f, u, udot, model):
        self.f = f
        self.u = u
        self.udot = udot
        self.model = model

    def compute(self):
        """
        function to compute residuals at interior nodes
        """
        e_w, p_w, e_n, p_n, p_c = PrimaryVariables(self.u, self.model).compute()

        self.model.interface.update(e_w, p_w, p_n, p_c)
        q_w, q_n, rho_w, rho_n = PhaseFlux(p_w, p_n, p_c, self.model).compute()

        for cell in self.model.mesh.cells:
            if cell.boundary:
                continue

            n = cell.ID
            self.model.current_cell = n
            od = SpatialOperator(p_w, p_n, p_c, q_w, q_n, cell).compute()
            A = TimeMatrix(e_w[n], p_w[n], p_n[n], p_c[n], rho_w[n], rho_n[n], e_n[n], self.model).compute()

            self.f[n] = ResidualEquations(A, self.udot, od, self.model).compute()

class PrimaryVariables:
    """
    Calss to hold information and methods to get primary variables depending on model and formulation 
    """
    def __init__(self, u, model):
        self.u = u
        self.model = model
        self.nodes = model.nodes
        self.form  = model.model.name
        self.model_form_mapping={
            "extended-two-phase-spp":self.set_extended_two_phase_spp,
            "two-phase-sp":self.set_two_phase_sp,
            "two-phase-pp":self.set_two_phase_pp,
            "richards-pr":self.set_richards_pr,
            "extended-richards-sp":self.set_extended_richards_sp,
        }
    def compute(self):
        """
        function to get and compute solution and primary variables 
        """
        if self.form in self.model_form_mapping:
            e_w, p_w, p_n, p_c = self.model_form_mapping[self.form]()
        else:
            print("Error Model Form is unknown")
        e_n = self.model.solid.porosity - e_w
        return e_w, p_w, e_n, p_n, p_c

    def set_extended_two_phase_spp(self):
        """
        spp: saturation, pressure w, pressure n
        """
        # Solution Variables
        e_w = self.u[:, 0]
        p_w = self.u[:, 1]
        p_n = self.u[:, 2]
        # Primary Variables
        p_c = np.zeros_like(p_w)
        for n in range(0, self.nodes):
            self.model.current_cell = n
            p_c[n] = self.model.interface.pc_function(e_w[n])
        return e_w, p_w, p_n, p_c

    def set_two_phase_sp(self):
        """
        sp: saturation, pressure w
        """
        # Solution Variables
        e_w = self.u[:, 0]
        p_w = self.u[:, 1]
        # Primary Variables
        p_c = np.zeros_like(p_w)
        p_n = np.zeros_like(p_w)
        for n in range(0, self.nodes):
            self.model.current_cell = n
            p_c[n] = self.model.interface.pc_function(e_w[n])
            p_n[n] = p_w[n] + p_c[n]
        return e_w, p_w, p_n, p_c

    def set_two_phase_pp(self):
        """
        pp: pressure w, pressure n
        """
        # Solution Variables
        p_w = self.u[:, 0]
        p_n = self.u[:, 1]
        # Primary Variables
        p_c = np.zeros_like(p_w)
        e_w = np.zeros_like(p_w)
        for n in range(0, self.nodes):
            self.model.current_cell = n
            p_c[n] = p_n[n] - p_w[n]
            e_w[n] = self.model.interface.volume_fraction_function(p_c[n])
        return e_w, p_w, p_n, p_c

    def set_richards_pr(self):
        """
        pr: pressure w
        """
        # Solution Variables
        p_w = self.u[:]
        # Primary Variables
        p_c = np.zeros_like(p_w)
        e_w = np.zeros_like(p_w)
        p_n = np.zeros_like(p_w)
        for n in range(0, self.nodes):
            self.model.current_cell = n
            p_c[n] = -p_w[n]
            e_w[n] = self.model.interface.volume_fraction_function(p_c[n])
        return e_w, p_w, p_n, p_c

    def set_extended_richards_sp(self):
        """
        sp: saturation, pressure w
        """
        # Solution Variables
        e_w = self.u[:, 0]
        p_w = self.u[:, 1]
        # Primary Variables
        p_c = np.zeros_like(p_w)
        p_n = np.zeros_like(p_w)
        for n in range(0, self.nodes):
            self.model.current_cell = n
            p_c[n] = self.model.interface.pc_function(e_w[n])
        return e_w, p_w, p_n, p_c


class PhaseFlux:
    """
    calss to hold information and methods for phases fluxes
    """
    def __init__(self, p_w, p_n, p_c, model):
        self.p_w = p_w
        self.p_n = p_n
        self.p_c = p_c
        self.model = model

    def compute(self):
        """
        function to calculate phases fluxes
        """
        mesh = self.model.mesh
        gravity = self.model.solid.gravity
        k = self.model.solid.permeability
        mu_w = self.model.w_phase.viscosity
        mu_n = self.model.n_phase.viscosity

        rho_w = utils.cell_center(self.p_w, self.model.w_phase.density_function, mesh)
        rho_n = utils.cell_center(self.p_n, self.model.n_phase.density_function, mesh)

        k_rw = utils.cell_center(self.p_c, self.model.interface.w_relative_permeability, mesh)
        k_rn = utils.cell_center(self.p_c, self.model.interface.n_relative_permeability, mesh)

        del_pw = utils.gradient(self.p_w, mesh)
        del_pn = utils.gradient(self.p_n, mesh)

        rho_w_face = utils.interpolate(rho_w, mesh)
        rho_n_face = utils.interpolate(rho_n, mesh)

        q_w = state_equations.darcy_velocity(mesh, k, utils.interpolate(k_rw * rho_w, mesh), mu_w, del_pw, rho_w_face, gravity)
        q_n = state_equations.darcy_velocity(mesh, k, utils.interpolate(k_rn * rho_n, mesh), mu_n, del_pn, rho_n_face, gravity)

        return q_w, q_n, rho_w, rho_n


class SpatialOperator:
    """
    calss to hold information and methods for spatial operator calculation
    """
    def __init__(self, p_w, p_n, p_c, q_w, q_n, cell):
        self.p_w = p_w
        self.p_n = p_n
        self.p_c = p_c
        self.q_w = q_w
        self.q_n = q_n
        self.cell= cell

    def compute(self):
        """
        function to compute the spatial operator at "cell".
        """
        n = self.cell.ID
        od_w = 1.0 / self.cell.length * (self.q_w[n + 1] - self.q_w[n])
        od_n = 1.0 / self.cell.length * (self.q_n[n + 1] - self.q_n[n])
        od_dp = self.p_w[n] - self.p_n[n] + self.p_c[n]
        od = [od_dp, od_w[0], od_n[0]]
        return od

class TimeMatrix:
    """
    calss to hold informations and methods for time matrix.
    """
    def __init__(self, e_w, p_w, p_n, p_c, rho_w, rho_n, e_n, model):
        self.e_w = e_w
        self.p_w = p_w
        self.p_n = p_n
        self.p_c = p_c
        self.rho_w = rho_w
        self.rho_n = rho_n
        self.e_n = e_n
        self.model = model
        self.d_rho_w_p_w = model.w_phase.d_density_function(p_w)
        self.d_p_c_e_w   = model.interface.d_pc_function(e_w)
        self.d_e_w_p_c   = model.interface.d_volume_fraction_function(p_c)
        self.d_rho_n_p_n = model.n_phase.d_density_function(p_n)
        self.model_form_mapping={
            "extended-two-phase-spp":self.set_extended_two_phase_spp,
            "two-phase-sp":self.set_two_phase_sp,
            "two-phase-pp":self.set_two_phase_pp,
            "richards-pr":self.set_richards_pr,
            "extended-richards-sp":self.set_extended_richards_sp,
        }
    def compute(self):
        """
        function to calculate time matrix "A" components for each formulation
        """
        form = self.model.model.name
        A = np.zeros([self.model.model.num_u, self.model.model.num_u])

        if form in self.model_form_mapping:
            mat = self.model_form_mapping[form](A)
        else:
            print("Error Model Form is unknown")
        return mat

    def set_extended_two_phase_spp(self,A):
        """
        function to calculate correspondent time matrix
        """
        A[0, 0] = (self.model.solid.dynamic_capillary_pressure_parameter / self.model.solid.porosity)
        A[1, 0] = self.rho_w
        A[1, 1] = self.e_w * self.d_rho_w_p_w
        A[2, 0] = -self.rho_n
        A[2, 2] = self.e_n * self.d_rho_n_p_n
        return A

    def set_two_phase_sp(self,A):
        """
        function to calculate correspondent time matrix
        """
        A[0, 0] = self.rho_w
        A[0, 1] = self.e_w * self.d_rho_w_p_w
        A[1, 0] = self.e_n * self.d_rho_n_p_n * self.d_p_c_e_w - self.rho_n
        A[1, 1] = self.e_n * self.d_rho_n_p_n
        return A

    def set_two_phase_pp(self,A):
        """
        function to calculate correspondent time matrix
        """
        A[0, 0] = -self.rho_w * self.d_e_w_p_c + self.e_w * self.d_rho_w_p_w
        A[0, 1] = self.rho_w * self.d_e_w_p_c
        A[1, 0] = self.rho_n * self.d_e_w_p_c
        A[1, 1] = -self.rho_n * self.d_e_w_p_c + self.e_n * self.d_rho_n_p_n
        return A

    def set_richards_pr(self,A):
        """
        function to calculate correspondent time matrix
        """
        A[0, 0] = -self.rho_w * self.d_e_w_p_c + self.e_w * self.d_rho_w_p_w
        return A

    def set_extended_richards_sp(self,A):
        """
        function to calculate correspondent time matrix
        """
        A[0, 0] = (self.model.solid.dynamic_capillary_pressure_parameter / self.model.solid.porosity)
        A[1, 0] = self.rho_w
        A[1, 1] = self.e_w * self.d_rho_w_p_w
        return A


class ResidualEquations:
    """
    calss to hold information and methods for spatial operator calculation
    """
    def __init__(self, A,udot,od,model):
        self.A = A
        self.udot = udot
        self.od = od
        self.model = model

    def compute(self):
        """
        function to compute the spatial operator at "cell".
        """
        form = self.model.model.name
        n = self.model.current_cell
        if form == "extended-two-phase-spp":
            f = np.dot(self.A, self.udot[n]) - self.od[0:3]
        elif form in ("two-phase-sp", "two-phase-pp"):
            f = np.dot(self.A, self.udot[n]) - self.od[1:3]
        elif form == "richards-pr":
            f = np.dot(self.A, self.udot[n]) - self.od[1:2]
        elif form == "extended-richards-sp":
            f = np.dot(self.A, self.udot[n]) - self.od[0:2]
        else:
            print("Error Model Form is unknown")
        return f


class ResidualBoundary:
    """
    class that holds informations and methods to compute residuals at boundaries
    """
    def __init__(self, t, f, u, model):
        self.t = t
        self.f = f
        self.u = u
        self.model = model

    def compute(self):
        """
        function to compute boundary residuals
        The first index (zero in the pic) is the number of variables (it loops from 0 to n_variables-1); the number of variables is related to the formulation used.
        The second index is for more than one dimension (we set it to zero because we are only working with 1d for now).
        The last index is 0 or 1, reflecting the x=0 
        """
        boundary_type = self.model.boundary_type
        boundary_conditions = self.model.boundary_conditions
        u_names = self.model.model.u_names

        for n in range(self.model.model.num_u):
            # Residual at Boundaries left and right
            for i, j in enumerate([0, self.model.nodes-1]):
                bc_value =(boundary_conditions[n][0][i](self.t) if callable(boundary_conditions[n][0][i]) else boundary_conditions[n][0][i])
                index =(j,) * (len(self.u.shape) - 1) + (slice(n),)
                if boundary_type[u_names[n]][0][i] == 0:
                    self.f[index] = self.u[index] - bc_value
                elif boundary_type[u_names[n]][0][i] == 1:
                    q = self.compute_boundary_flux(j,self.model.w_phase)
                    self.f[index]  = q -  bc_value
                elif boundary_type[u_names[n]][0][i] == 2:
                    q = self.compute_boundary_flux(j,self.model.n_phase)
                    self.f[index]  = q -  bc_value
                else:
                    print("Boundary type non supported")

    def compute_boundary_flux(self,cell_id,phase):
        """
        function to compute phases fluxes at boundary points
        """
        # get primary variables
        e_w, p_w, e_n, p_n, p_c = PrimaryVariables(self.u, self.model).compute()

        #create a boundary mesh consisting of one cell
        mesh_bc = TCAT2Phase.mesh.grid.create_submesh(self.model.mesh,cell_id)

        # Cell Centered Variables
        if phase.phase =='w':
            rho = utils.cell_center(p_w, self.model.w_phase.density_function, mesh_bc)
            k_r = utils.cell_center(p_c, self.model.interface.w_relative_permeability, mesh_bc)
            del_p = utils.gradient(p_w, mesh_bc)
        elif phase.phase =='n':
            rho = utils.cell_center(p_n, self.model.n_phase.density_function, mesh_bc)
            k_r = utils.cell_center(p_c, self.model.interface.n_relative_permeability, mesh_bc)
            del_p = utils.gradient(p_n, mesh_bc)
        else:
            print("phase should be 'n' or 'w'")
        # Fluxe
        q =  state_equations.darcy_velocity(
            mesh_bc,
            self.model.solid.permeability,
            k_r*rho,
            self.model.w_phase.viscosity,
            del_p,
            rho,
            self.model.solid.gravity,
        )
        return  q
