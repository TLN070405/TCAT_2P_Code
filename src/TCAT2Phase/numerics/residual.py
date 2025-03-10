import numpy as np
from TCAT2Phase.numerics import utils


def model_residual_tm(ts, t, U, Udot, F, model):
    """
    Residual function for the selected traditional model.
    """

    da = ts.getDM()
    localU = da.getLocalVec()
    da.globalToLocal(U, localU)

    # Collect solution variables and residual
    u = da.getVecArray(localU)
    udot = da.getVecArray(Udot)
    f = da.getVecArray(F)
    q_w, q_n =model_residual_internal_tm(f, u, udot, model)
    model_residual_boundary_tm(t, f, u,q_w, q_n, model)

def model_residual_tcat(ts, t, U, Udot, F, model):
    """
    Residual function for the selected traditional model.
    """

    da = ts.getDM()
    localU = da.getLocalVec()
    da.globalToLocal(U, localU)

    # Collect solution variables and residual
    u = da.getVecArray(localU)
    udot = da.getVecArray(Udot)
    f = da.getVecArray(F)
    q_w = model_residual_internal_tcat(f, u, udot, model)
    model_residual_boundary_tcat(t, f, u,q_w, model)

def model_residual_internal_tm(f, u, udot, model):
    """
    residual at internal nodes.
    """
    # Create primary variables
    e_w, p_w, e_n, p_n, p_c = model_primary_variables_tm(u, model)

    model.interface.e_w = e_w
    model.interface.p_w = p_w
    model.interface.p_n = p_n
    model.interface.p_c = p_c

    q_w, q_n, rho_w, rho_n = model_phase_flux_tm(p_w, p_n, p_c, model)

    mesh = model.mesh
    for cell in mesh.cells:
        if cell.boundary:
            continue
        n = cell.ID
        model.current_cell = n
        od = model_spatial_operator_tm(p_w,p_n,p_c,q_w,q_n,cell )

        # Time derivative matrix
        A = model_time_matrix_tm(e_w[n], p_w[n], p_n[n], p_c[n], rho_w[n], rho_n[n], e_n[n], model)

        # Residuals at interior points
        f[n] = model_residual_equations(A, udot, od, model)
    return q_w, q_n

def model_residual_internal_tcat(f, u, udot, model):
    """
    residual at internal nodes.
    """
    # Create primary variables
    e_w, p_w, e_wn, j_wn, w_wn, k_wn = model_primary_variables_tcat(u, model)

    # Calculate fluxes
    q_w, rho_w = model_phase_flux_tcat(e_w, p_w, model)

    # Calculate gradient of primary variables
    del_ew, del_ewn, del_jwn, del_wwn = model_primary_variables_gradient_tcat(e_w, e_wn, j_wn, w_wn, model)

    mesh = model.mesh
    for cell in mesh.cells:
        if cell.boundary:
            continue
        n = cell.ID
        model.current_cell = n
        od = model_spatial_operator_tcat(
            e_w,
            p_w,
            e_wn,
            j_wn,
            w_wn,
            k_wn,
            q_w,
            del_ew,
            del_ewn,
            del_jwn,
            del_wwn,
            cell,
            model,
        )
        # Time derivative matrix
        A = model_time_matrix_tcat(e_w[n], p_w[n], e_wn[n], j_wn[n], k_wn[n],rho_w[n], model)

        # Residuals at interior points
        f[n] = model_residual_equations(A, udot, od, model)

    return q_w

def model_primary_variables_tm(u, model):
    """
    primary variables for 2P traditional model formulation
    """
    nodes = model.nodes
    form = model.model.name
    if form == "extended-two-phase-spp":
        # Solution Variables
        e_w = u[:, 0]
        p_w = u[:, 1]
        p_n = u[:, 2]
        # Primary Variables
        p_c = np.zeros_like(p_w)
        for n in range(0, nodes):
            model.current_cell = n
            p_c[n] = model.interface.pc_function(e_w[n])
    #sp: saturation, pressure w
    elif form == "two-phase-sp":
        # Solution Variables
        e_w = u[:, 0]
        p_w = u[:, 1]
        # Primary Variables
        p_c = np.zeros_like(p_w)
        p_n = np.zeros_like(p_w)
        for n in range(0, nodes):
            model.current_cell = n
            p_c[n] = model.interface.pc_function(e_w[n])
            p_n[n] = p_w[n] + p_c[n]
    #pp: pressure w, pressure n
    elif form == "two-phase-pp":
        # Solution Variables
        p_w = u[:, 0]
        p_n = u[:, 1]
        # Primary Variables
        p_c = np.zeros_like(p_w)
        e_w = np.zeros_like(p_w)
        for n in range(0, nodes):
            model.current_cell = n
            p_c[n] = p_n[n] - p_w[n]
            e_w[n] = model.interface.volume_fraction_function(p_c[n])
    #pr: pressure w
    elif form == "richards-pr":
        # Solution Variables
        p_w = u[:]
        # Primary Variables
        p_c = np.zeros_like(p_w)
        e_w = np.zeros_like(p_w)
        p_n = np.zeros_like(p_w)
        for n in range(0, nodes):
            model.current_cell = n
            p_c[n] = -p_w[n]
            e_w[n] = model.interface.volume_fraction_function(p_c[n])
    #sp: saturation, pressure w
    elif form == "extended-richards-sp":
        # Solution Variables
        e_w = u[:, 0]
        p_w = u[:, 1]
        # Primary Variables
        p_c = np.zeros_like(p_w)
        p_n = np.zeros_like(p_w)
        for n in range(0, nodes):
            model.current_cell = n
            p_c[n] = model.interface.pc_function(e_w[n])
    else:
        print("Error Model Form is unknown")
    e_n = model.solid.porosity - e_w

    return e_w, p_w, e_n, p_n, p_c

def model_primary_variables_tcat(u, model):
    """
    primary variables for tcat formulations
    """
    form = model.model.name
    # the spijvk stands for the primary variables: saturation, pressure, interfacial area,
    # mean curvature,velocity and Gaussian curvature
    if form == "cgp-TCAT-spijvk":
        # Solution Variables
        e_w = u[:, 0]
        p_w = u[:, 1]
        e_wn = u[:, 2]
        j_wn = u[:, 3]
        w_wn = u[:, 4]
        k_wn = u[:, 5]
    else:
        print("Error Model Form is unknown")

    return e_w, p_w, e_wn, j_wn, w_wn, k_wn

def model_primary_variables_gradient_tcat(e_w, e_wn, j_wn, w_wn, model):
    """
    gradient of primary variables
    """
    mesh = model.mesh
    d_e_w_d_x = utils.gradient(e_w, mesh)
    d_e_wn_d_x = utils.gradient(e_wn, mesh)
    d_j_wn_d_x = utils.gradient(j_wn, mesh)
    d_w_wn_d_x = utils.gradient(w_wn, mesh)

    return d_e_w_d_x, d_e_wn_d_x, d_j_wn_d_x, d_w_wn_d_x

def model_phase_flux_tm( p_w, p_n, p_c, model):
    """
    phases fluxes calculations
    """
    mesh = model.mesh
    # Constant Pameters
    gravity = model.solid.gravity

    # Constant Pameters
    k = model.solid.permeability
    mu_w = model.w_phase.viscosity
    mu_n = model.n_phase.viscosity

    # Cell Centered Variables
    rho_w = utils.cell_center(p_w, model.w_phase.density_function, mesh)
    rho_n = utils.cell_center(p_n, model.n_phase.density_function, mesh)
    k_rw = utils.cell_center(p_c, model.interface.w_relative_permeability, mesh)
    k_rn = utils.cell_center(p_c, model.interface.n_relative_permeability, mesh)

    # Derivatives Values
    del_pw = utils.gradient(p_w, mesh)
    del_pn = utils.gradient(p_n, mesh)

    # Face Centered Values
    rho_w_face = utils.interpolate(rho_w, mesh)
    rho_n_face = utils.interpolate(rho_n, mesh)

    # Fluxes
    q_w = darcy_velocity(
        mesh,
        k,
        utils.interpolate(k_rw * rho_w, mesh),
        mu_w,
        del_pw,
        rho_w_face,
        gravity,
    )

    q_n = darcy_velocity(
        mesh,
        k,
        utils.interpolate(k_rn * rho_n, mesh),
        mu_n,
        del_pn,
        rho_n_face,
        gravity,
    )
    return q_w, q_n, rho_w, rho_n

def model_phase_flux_tcat(e_w, p_w, model):
    """
    phases fluxes calculations
    """
    mesh = model.mesh

    # Constant Pameters
    gravity = model.solid.gravity

    # Cell Centered Variables
    rho_w = utils.cell_center(p_w, model.w_phase.density_function, mesh)
    k_rw = utils.cell_center(e_w, model.interface.w_relative_permeability, mesh)

    # Derivatives Values
    del_pw = utils.gradient(p_w, mesh)

    # Face Centered Values
    rho_w_face = utils.interpolate(rho_w, mesh)

    # Fluxes
    q_w = darcy_velocity(
        mesh,
        1./model.w_phase.momentum_resistance,
        utils.interpolate(k_rw * rho_w, mesh),
        1.0,
        del_pw,
        rho_w_face,
        gravity,
    )

    return q_w, rho_w

def model_spatial_operator_tm(p_w,p_n,p_c,q_w,q_n,cell):
    """
    Spatial operator for each formulation
    """
    n = cell.ID
    od_w = 1.0 / cell.length * (q_w[n + 1] - q_w[n])
    od_n = 1.0 / cell.length * (q_n[n + 1] - q_n[n])
    od_dp = p_w[n] - p_n[n] + p_c[n]
    od = [od_dp, od_w[0], od_n[0]]
    return od

def model_spatial_operator_tcat(
    e_w,
    p_w,
    e_wn,
    j_wn,
    w_wn,
    k_wn,
    q_w,
    del_ew,
    del_ewn,
    del_jwn,
    del_wwn,
    cell,
    model
):
    """
    Spatial operator for each formulation
    """
    n = cell.ID
    c_wn = model.interface.interfacial_dynamics
    gamma_wn = model.interface.interfacial_tension
    k_hat_wn = model.interface.interfacial_area_relaxation
    e_wn_eq  = model.interface.interfacial_area_equilibrium

    od_w  = 1.0 / cell.length * (q_w[n + 1] - q_w[n])
    od_dp = c_wn * (p_w[n] + gamma_wn * j_wn[n])  + gamma_wn * k_hat_wn/(p_w[n] + gamma_wn * j_wn[n] + 1e-3) * (e_wn_eq[n] - e_wn[n])
    od_dp_forward  = c_wn * (p_w[n+1] + gamma_wn * j_wn[n+1])  + gamma_wn * k_hat_wn/(p_w[n+1] + gamma_wn * j_wn[n+1] + 1e-3) * (e_wn_eq[n+1] - e_wn[n+1])
    od_dp_backward = c_wn * (p_w[n-1] + gamma_wn * j_wn[n-1])  + gamma_wn * k_hat_wn/(p_w[n-1] + gamma_wn * j_wn[n-1] + 1e-3) * (e_wn_eq[n-1] - e_wn[n-1])
    od_ia = k_hat_wn * (e_wn_eq[n] - e_wn[n]) - ( w_wn[n] * del_ewn[n] + e_wn[n] * del_wwn[n])
    od_j = 2/(3*cell.length[0]*cell.length[0]) * (od_dp_forward-2*od_dp+od_dp_backward)-2 * (w_wn[n] * j_wn[n] * del_ewn[n] + e_wn[n] * w_wn[n] * del_jwn[n] + e_wn[n] * j_wn[n]* del_wwn[n])
    od_v = e_w[n] * e_wn[n] * w_wn[n] + od_dp * del_ew[n]
    od_k = e_w[n] - model.interface.volume_fraction_function(e_wn[n], j_wn[n], k_wn[n])
    od   = [od_dp, od_w[0], od_ia, od_j, od_v, od_k]
    return od

def model_time_matrix_tm(e_w, p_w, p_n, p_c, rho_w, rho_n, e_n, model):
    """
    A matrix components for each formulation
    """
    form = model.model.name
    d_rho_w_p_w = model.w_phase.d_density_function(p_w)
    d_p_c_e_w   = model.interface.d_pc_function(e_w)
    d_e_w_p_c   = model.interface.d_volume_fraction_function(p_c)
    d_rho_n_p_n = model.n_phase.d_density_function(p_n)

    A = np.zeros([model.model.num_u, model.model.num_u])
    if form == "extended-two-phase-spp":
        A[0, 0] = (model.solid.dynamic_capillary_pressure_parameter / model.solid.porosity)
        A[1, 0] = rho_w
        A[1, 1] = e_w * d_rho_w_p_w
        A[2, 0] = -rho_n
        A[2, 2] = e_n * d_rho_n_p_n
    elif form == "two-phase-sp":
        A[0, 0] = rho_w
        A[0, 1] = e_w * d_rho_w_p_w
        A[1, 0] = e_n * d_rho_n_p_n * d_p_c_e_w - rho_n
        A[1, 1] = e_n * d_rho_n_p_n
    elif form == "two-phase-pp":
        A[0, 0] = -rho_w * d_e_w_p_c + e_w * d_rho_w_p_w
        A[0, 1] = rho_w * d_e_w_p_c
        A[1, 0] = rho_n * d_e_w_p_c
        A[1, 1] = -rho_n * d_e_w_p_c + e_n * d_rho_n_p_n
    elif form == "richards-pr":
        A[0, 0] = -rho_w * d_e_w_p_c + e_w * d_rho_w_p_w
    elif form == "extended-richards-sp":
        A[0, 0] = (model.solid.dynamic_capillary_pressure_parameter / model.solid.porosity)
        A[1, 0] = rho_w
        A[1, 1] = e_w * d_rho_w_p_w
    else:
        print("Error Model Form is unknown")

    return A

def model_time_matrix_tcat(e_w, p_w, e_wn, j_wn, k_wn, rho_w, model):
    """
    A matrix components for each formulation
    """
    form = model.model.name
    d_rho_w_p_w = model.w_phase.d_density_function(p_w)

    A = np.zeros([model.model.num_u, model.model.num_u])
    if form == "cgp-TCAT-spijvk":
        A[0, 0] = 1
        A[1, 0] = rho_w
        A[1, 1] = e_w * d_rho_w_p_w
        A[2, 0] = j_wn
        A[2, 2] = 1
        A[3, 0] = -2 * k_wn
        A[3, 2] = j_wn
        A[3, 3] = e_wn
    else:
        print("Error Model Form is unknown")

    return A

def model_residual_equations(A, udot, od, model):
    """
    Residual equations
    """
    form = model.model.name
    n = model.current_cell
    if form == "cgp-TCAT-spijvk":
        f = np.dot(A, udot[n]) - od
    elif form == "extended-two-phase-spp":
        f = np.dot(A, udot[n]) - od[0:3]
    elif form in ("two-phase-sp", "two-phase-pp"):
        f = np.dot(A, udot[n]) - od[1:3]
    elif form == "richards-pr":
        f = np.dot(A, udot[n]) - od[1:2]
    elif form == "extended-richards-sp":
        f = np.dot(A, udot[n]) - od[0:2]
    else:
        print("Error Model Form is unknown")
    return f


def model_residual_boundary_tm(t, f, u, q_w, q_n, model):
    """
    boundary residuals
    """
    nodes = model.nodes
    u_names = model.model.u_names
    boundary_type = model.boundary_type
    boundary_conditions = model.boundary_conditions

    if model.model.num_u == 1:
        # boundary conditions for one variable formulations

        # Residual at Boundary 1
        if boundary_type[u_names[0]][0][0] == 0:
            # when bc==0 then its Dirichlet BC
            f[0] = u[0] - boundary_conditions[0][0][0](t)
        elif boundary_type[u_names[0]][0][0] == 1:
            # when bc==1 then its Neumman BC with qw flux
            f[0] = q_w[0] - boundary_conditions[0][0][0](t)
        elif boundary_type[u_names[0]][0][0] == 2:
            # when bc==2 then its Neumman BC with qn flux
            f[0] = q_n[0] - boundary_conditions[0][0][0](t)
        else:
            print("Boundary type non supported")
        # Residual at Boundary 2
        if boundary_type[u_names[0]][0][1] == 0:
            f[nodes - 1] = u[nodes - 1] - boundary_conditions[0][0][1](t)
        elif boundary_type[u_names[0]][0][1] == 1:
            f[nodes - 1] = q_w[nodes - 1] - boundary_conditions[0][0][1](t)
        elif boundary_type[u_names[0]][0][1] == 2:
            f[nodes - 1] = q_n[nodes - 1] - boundary_conditions[0][0][1](t)
        else:
            print("Boundary type non supported")

    else:
        # boundary conditions for two or more variables formulations

        for n in range(model.model.num_u):
            # Residual at Boundary 1
            if boundary_type[u_names[n]][0][0] == 0:
                f[0][n] = u[0][n] - boundary_conditions[n][0][0](t)
            elif boundary_type[u_names[n]][0][0] == 1:
                f[0][n] = q_w[0] - boundary_conditions[n][0][0](t)
            elif boundary_type[u_names[n]][0][0] == 2:
                f[0][n] = q_n[0] - boundary_conditions[n][0][0](t)
            else:
                print("Boundary type non supported")
            # Residual at Boundary 2
            if boundary_type[u_names[n]][0][1] == 0:
                f[nodes - 1][n] = u[nodes - 1][n] - boundary_conditions[n][0][1](t)
            elif boundary_type[u_names[n]][0][1] == 1:
                f[nodes - 1][n] = q_w[nodes - 1] - boundary_conditions[n][0][1](t)
            elif boundary_type[u_names[n]][0][1] == 2:
                f[nodes - 1][n] = q_n[nodes - 1] - boundary_conditions[n][0][1](t)
            else:
                print("Boundary type non supported")

def model_residual_boundary_tcat(t, f, u,q_w, model):
    """
    boundary residuals
    """
    mesh = model.mesh
    nodes = model.nodes
    u_names = model.model.u_names
    boundary_type = model.boundary_type
    boundary_conditions = model.boundary_conditions

    # boundary conditions
    for n in range(model.model.num_u):
        # Residual at Boundary 1
        if boundary_type[u_names[n]][0][0] == 0:
            f[0][n] = u[0][n] - boundary_conditions[n][0][0](t)
        elif boundary_type[u_names[n]][0][0] == 1:
            f[0][n] = q_w[0] - boundary_conditions[n][0][0](t)
        elif boundary_type[u_names[n]][0][0] == 2:
            del_u = utils.gradient_cell(u[:][n],0,mesh)
            f[0][n] = del_u - boundary_conditions[n][0][0](t)
        else:
            print("Boundary type non supported")
        # Residual at Boundary 2
        if boundary_type[u_names[n]][0][1] == 0:
            f[nodes - 1][n] = u[nodes - 1][n] - boundary_conditions[n][0][1](t)
        elif boundary_type[u_names[n]][0][1] == 1:
            f[nodes - 1][n] = q_w[nodes - 1] - boundary_conditions[n][0][1](t)
        elif boundary_type[u_names[n]][0][1] == 2:
            del_u = utils.gradient_cell(u[:][n],nodes - 2,mesh)
            f[nodes - 1][n] = del_u - boundary_conditions[n][0][1](t)
        else:
            print("Boundary type non supported")

def darcy_velocity(mesh, k, k_r, mu, del_p, rho, g):
    """
    Calculate the Darcy velocity as:
        q^a = -k*k_r^a/mu^a(\grad p^a - rho^a*g)
    """
    q = np.zeros(mesh.num_faces)
    for face in mesh.faces:
        if face.boundary:
            continue
        q[face.ID] = k * k_r[face.ID] / mu * (del_p[face.ID] - rho[face.ID] * g)
    return q
