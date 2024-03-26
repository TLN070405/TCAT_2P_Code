import numpy as np
from TCAT2Phase import state_equations
from TCAT2Phase.numerics import utils


def two_phase_residual(ts,t,U,Udot,F,model):
    """
    Residual function for two-phase sp formulation
    """

    da = ts.getDM()
    localU = da.getLocalVec() 
    da.globalToLocal(U,localU) 
    
    # Collect solution variables and residual 
    u = da.getVecArray(localU)
    udot = da.getVecArray(Udot)
    f = da.getVecArray(F)

    nodes = model.nodes

    two_sphase_residual_internal(f,u,udot,model)

    for n in range(model.model.num_u):

        #Residual at Boundary 1 
        f[0][n] = u[0][n] - model.boundary_value[n][0][0]

        # Residual at Boundary 2
        f[nodes-1][n] = u[nodes-1][n] - model.boundary_value[n][0][1]


def two_sphase_residual_internal(f,u,udot,model):
    """
    Not sure
    """
    # Collect discretization parameters
    nodes = model.nodes
    dx = model.res

    form = model.model.name
    if form == 'two-phase-sp':
        e_w,p_w,p_n,p_c = two_phase_sp_primary_variables(u,nodes,model)
    elif form == 'two-phase-pp':
        e_w,p_w,p_n,p_c = two_phase_pp_primary_variables(u,nodes,model)
    else:
        print('Error Model From is unknown')

    mesh = model.mesh

    # Constant Pameters
    k = model.solid.permeability
    gravity = 9.81
    mu_w = model.w_phase.viscosity
    mu_n = model.n_phase.viscosity
    porosity = model.solid.porosity

    # Cell Centered Variables
    rho_w = utils.cell_center(p_w,model.w_phase.density_function,mesh)
    rho_n = utils.cell_center(p_n,model.n_phase.density_function,mesh)
    e_n  = porosity - e_w
    k_rw = utils.cell_center(p_c,model.w_phase.rel_perm_function,mesh)
    k_rn = utils.cell_center(p_c,model.n_phase.rel_perm_function,mesh)

    # Face Centered Values
    del_pw = utils.gradient(p_w,mesh)
    del_pn = utils.gradient(p_n,mesh)

    rho_w_face = utils.interpolate(rho_w,mesh)
    rho_n_face = utils.interpolate(rho_n,mesh)

    q_w = darcy_velocity(
                   mesh,
                   k,
                   utils.interpolate(k_rw,mesh),
                   mu_w,
                   del_pw,
                   rho_w_face,
                   gravity)

    q_n = darcy_velocity(
                   mesh,
                   k,
                   utils.interpolate(k_rn,mesh),
                   mu_n,
                   del_pn,
                   rho_n_face,
                   gravity)

    for cell in mesh.cells:
        if cell.boundary:
            continue

        n = cell.ID

        od_w = 1./cell.length*(rho_w_face[n]*q_w[n] - rho_w_face[n-1]*q_w[n-1])
        od_n = 1./cell.length*(rho_n_face[n]*q_n[n] - rho_n_face[n-1]*q_n[n-1])
        
        # Time derivative matrix
        if form == 'two-phase-sp':
            A = two_phase_sp_A_matrix(e_w[n],p_w[n],p_n[n],p_c[n],rho_w[n],rho_n[n],e_n[n],model)
        elif form == 'two-phase-pp':
            A = two_phase_pp_A_matrix(e_w[n],p_w[n],p_n[n],p_c[n],rho_w[n],rho_n[n],e_n[n])

        f[n][0] = A[0,0] * udot[n][0] + A[0,1] * udot[n][1] - od_w
        f[n][1] = A[1,0] * udot[n][0] + A[1,1] * udot[n][1] - od_n

    return f


def two_phase_sp_primary_variables(u,nodes,model):
    """
    Collect primary variables from solution variables
    """
    # Solution Variables
    e_w = u[:,0]
    p_w = u[:,1]

    # Primary Variables 
    p_c = np.zeros_like(p_w)
    p_n = np.zeros_like(p_w)
    for n in range(0,nodes):
        p_c[n] = model.interface.pc_function(e_w[n])
        p_n[n] = p_w[n] + p_c[n]

    return e_w,p_w,p_n,p_c

def two_phase_sp_A_matrix(e_w,p_w,p_n,p_c,rho_w,rho_n,e_n,model):
    """
    Calculate the A matrix. See Kees_Miller_02 for formulation
    """
    A = np.zeros([2,2])

    d_rho_w_p_w = model.w_phase.d_density_function(p_w)
    d_rho_n_p_n = model.n_phase.d_density_function(p_n)
    d_p_c_e_w = model.interface.d_pc_function(e_w)

    A[0,0] = rho_w
    A[0,1] = e_w*d_rho_w_p_w
    A[1,0] = e_n*d_rho_n_p_n*d_p_c_e_w - rho_n
    A[1,1] = e_n*d_rho_n_p_n

    return A

def two_phase_pp_primary_variables(u,nodes,model):
    """
    Collect primary variables from solution variables
    """
    # Solution Variables
    p_w = u[:][0]
    p_n = u[:][1]

    # Primary Variables 
    p_c = np.zeros_like(p_w)
    e_w = np.zeros_like(p_w)
    for n in range(0,nodes):
        p_c[n] = p_n[n] - p_w[n]
        e_w[n] = state_equations.Epsilon_w_of_Pc(p_c[n])

    return e_w,p_w,p_n,p_c

def two_phase_pp_A_matrix(e_w,p_w,p_n,p_c,rho_w,rho_n,e_n):
    """
    Calculate the A matrix. See Kees_Miller_02 for formulation
    """
    A = np.zeros([2,2])
    d_e_w_p_c = state_equations.d_e_w_p_c(p_c)
    d_rho_w_p_w = state_equations.d_rho_w_p_w(p_w)
    d_rho_n_p_n = state_equations.d_rho_n_p_n(p_n)

    A[0,0] = -rho_w*d_e_w_p_c + e_w*d_rho_w_p_w
    A[0,1] = rho_w*d_e_w_p_c
    A[1,0] = rho_n*d_e_w_p_c
    A[1,1] = -rho_n*d_e_w_p_c + e_n*d_rho_n_p_n

    return A    


def darcy_velocity(mesh,k,k_r,mu,del_p,rho,g):
    """
    Calaculate the Darcy velocity as:
        q^a = -k*k_r^a/mu^a(\grad p^a - rho^a*g)
    """
    q = np.zeros(mesh.num_faces)
    for face in mesh.faces:
        if face.boundary:
            continue
        q[face.ID] = k*k_r[face.ID]/mu*(del_p[face.ID] - rho[face.ID]*g)
    return q   