import numpy as np
import pdb
from TCAT2Phase import state_equations
from TCAT2Phase.numerics import utils


def model_residual(ts,t,U,Udot,F,model):
    """
    Residual function for the selected model. 
    """
    da = ts.getDM()
    localU = da.getLocalVec() 
    da.globalToLocal(U,localU) 
    
    # Collect solution variables and residual 
    u = da.getVecArray(localU)
    udot = da.getVecArray(Udot)
    f = da.getVecArray(F)

    model_residual_internal(f,u,udot,model)
    model_residual_boundary(t,f,u,udot,model)

def model_residual_internal(f,u,udot,model):
    """
    residual at internal nodes.
    """
    # Collect discretization parameters
    e_w,p_w,p_n,p_c = model_primary_variables(u,model)
    mesh = model.mesh

    # Constant Pameters
    k = model.solid.permeability
    gravity =model.solid.gravity
    mu_w = model.w_phase.viscosity
    mu_n = model.n_phase.viscosity
    porosity = model.solid.porosity

    # Cell Centered Variables
    rho_w = utils.cell_center(p_w,model.w_phase.density_function,mesh)
    rho_n = utils.cell_center(p_n,model.n_phase.density_function,mesh)
    e_n   = porosity - e_w
    k_rw  = utils.cell_center(p_c,model.w_phase.rel_perm_function,mesh)
    k_rn  = utils.cell_center(p_c,model.n_phase.rel_perm_function,mesh)

    # Face Centered Values
    del_pw = utils.gradient(p_w,mesh)
    del_pn = utils.gradient(p_n,mesh)

    rho_w_face = utils.interpolate(rho_w,mesh)
    rho_n_face = utils.interpolate(rho_n,mesh)

    q_w = darcy_velocity(
                   mesh,
                   k,
                   utils.interpolate(k_rw*rho_w,mesh),
                   mu_w,
                   del_pw,
                   rho_w_face,
                   gravity)

    q_n = darcy_velocity(
                   mesh,
                   k,
                   utils.interpolate(k_rn*rho_n,mesh),
                   mu_n,
                   del_pn,
                   rho_n_face,
                   gravity)

    num_u = model.model.num_u

    for cell in mesh.cells:
        if cell.boundary:
            continue
           
        n = cell.ID
        od_w  = 1./cell.length*(q_w[n+1] - q_w[n])
        od_n  = 1./cell.length*(q_n[n+1] - q_n[n])
        od_dp = p_w[n]-p_n[n]+ p_c[n]
        od    = [od_dp,od_w[0],od_n[0]]

        # Time derivative matrix
        A = model_A_matrix(e_w[n],p_w[n],p_n[n],p_c[n],rho_w[n],rho_n[n],e_n[n],model)
        # residuals
        f[n] = np.dot(A, udot[n]) -  od[ int(num_u*(-0.5*num_u+1.5)):int(num_u*(-0.5*num_u+2.5))]

    return f

def model_primary_variables(u,model):
    """
    primary variables for each formulation
    """
    nodes = model.nodes
    form  = model.model.name
    if form == 'extended-two-phase-spp':
        # Solution Variables
        e_w,p_w,p_n= u[:,0],u[:,1],u[:,2]
        # Primary Variables 
        p_c = np.zeros_like(p_w)
        for n in range(0,nodes):
            p_c[n] = model.interface.pc_function(e_w[n])
    elif form == 'two-phase-sp':
        # Solution Variables
        e_w ,p_w= u[:,0],u[:,1]
        # Primary Variables 
        p_c = np.zeros_like(p_w)
        p_n = np.zeros_like(p_w)
        for n in range(0,nodes):
            p_c[n] = model.interface.pc_function(e_w[n])
            p_n[n] = p_w[n] + p_c[n]
    elif form == 'two-phase-pp':
        # Solution Variables
        p_w,p_n = u[:,0],u[:,1]
        # Primary Variables 
        p_c = np.zeros_like(p_w)
        e_w = np.zeros_like(p_w)
        for n in range(0,nodes):
            p_c[n] = p_n[n] - p_w[n]
            e_w[n] = model.w_phase.volume_fraction_function(p_c[n])
    elif form == 'richards-pr':
        # Solution Variables
        p_w = u[:]
        # Primary Variables 
        p_c = np.zeros_like(p_w)
        e_w = np.zeros_like(p_w)
        p_n = np.zeros_like(p_w)
        for n in range(0,nodes):
            p_c[n] = - p_w[n]
            e_w[n] = model.w_phase.volume_fraction_function(p_c[n])
    else:
        print('Error Model Form is unknown')

    return e_w,p_w,p_n,p_c

def model_A_matrix(e_w,p_w,p_n,p_c,rho_w,rho_n,e_n,model):
    """
    A matrix components for each formulation
    """
    d_rho_n_p_n = model.n_phase.d_density_function(p_n)
    d_rho_w_p_w = model.w_phase.d_density_function(p_w)
    d_p_c_e_w   = model.interface.d_pc_function(e_w)
    d_e_w_p_c   = model.w_phase.d_volume_fraction_function(p_c)
    
    A = np.zeros([model.model.num_u,model.model.num_u])

    form = model.model.name

    if form == 'extended-two-phase-spp':
        A[0,0] = model.solid.dynamic_capillary_pressure_parameter/model.solid.porosity
        A[1,0] = rho_w                   
        A[1,1] = e_w*d_rho_w_p_w     
        A[2,0] = -rho_n  
        A[2,2] = e_n*d_rho_n_p_n
    elif form == 'two-phase-sp':
        A[0,0] = rho_w
        A[0,1] = e_w*d_rho_w_p_w
        A[1,0] = e_n*d_rho_n_p_n*d_p_c_e_w - rho_n
        A[1,1] = e_n*d_rho_n_p_n
    elif form == 'two-phase-pp':
        A[0,0] = -rho_w*d_e_w_p_c + e_w*d_rho_w_p_w
        A[0,1] = rho_w*d_e_w_p_c
        A[1,0] = rho_n*d_e_w_p_c
        A[1,1] = -rho_n*d_e_w_p_c + e_n*d_rho_n_p_n
    elif form == 'richards-pr':
        A[0,0] = -rho_w*d_e_w_p_c + e_w*d_rho_w_p_w
    else:
        print('Error Model Form is unknown')

    return A    

def model_residual_boundary(t,f,u,udot,model):
    nodes = model.nodes
    """
    boundary residuals 
    """
    model.boundary_name
    if model.model.num_u == 1:
       #Residual at Boundary 1 
       f[0] = u[0] - model.boundary_value[0][0][0](t)
       # Residual at Boundary 2
       f[nodes-1] = u[nodes-1] - model.boundary_value[0][0][1](t)
    else:
       for n in range(model.model.num_u):
           #Residual at Boundary 1 
           f[0][n] = u[0][n] - model.boundary_value[n][0][0](t)
           # Residual at Boundary 2
           f[nodes-1][n] = u[nodes-1][n] - model.boundary_value[n][0][1](t)

    return f


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