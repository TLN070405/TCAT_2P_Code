import numpy as np

from TCAT2Phase.numerics import utils
from TCAT2Phase.numerics import residual

def model_formfunction(snes,U,F,model):
    """
    residual function to solve boundary conditions in case we have flux bc
    """
    da=snes.getDM()
    localU=da.getLocalVec()
    da.globalToLocal(U,localU)
    #######
    u=da.getVecArray(localU)
    f=da.getVecArray(F)
    #######
    u=da.getVecArray(localU)

    model_formfunction_internal(f,u,model)

    model_formfunction_boundary(0,f,u,model)


def model_formfunction_internal(f,u,model):
    """
    Initial conditions residual at interior of domain
    """

    nodes  = model.nodes
    u_name = model.model.u_names

    for n_u,u_name in enumerate(model.model.u_names):
        if model.model.num_u==1:
            for n in range(1,nodes-1):
                f[n]      = u[n]      - model.initial_condition[u_name][n]
        else:
            for n in range(1,nodes-1):
                f[n][n_u] = u[n][n_u] - model.initial_condition[u_name][n]


def model_formfunction_boundary(t,f,u,model):
    """
    Boundary points residual
    """
    # Collect discretization parameters
    e_w,p_w,p_n,p_c = residual.model_primary_variables(u,model)
    mesh = model.mesh
    nodes=model.nodes
    # Constant Pameters
    k = model.solid.permeability
    gravity =model.solid.gravity
    mu_w = model.w_phase.viscosity
    mu_n = model.n_phase.viscosity

    # Cell Centered Variables
    rho_w = utils.cell_center(p_w,model.w_phase.density_function,mesh)
    rho_n = utils.cell_center(p_n,model.n_phase.density_function,mesh)

    k_rw  = utils.cell_center(p_c,model.w_phase.rel_perm_function,mesh)
    k_rn  = utils.cell_center(p_c,model.n_phase.rel_perm_function,mesh)

    # Face Centered Values
    del_pw = utils.gradient(p_w,mesh)
    del_pn = utils.gradient(p_n,mesh)

    rho_w_face = utils.interpolate(rho_w,mesh)
    rho_n_face = utils.interpolate(rho_n,mesh)

    q_w = residual.darcy_velocity(
                   mesh,
                   k,
                   utils.interpolate(k_rw*rho_w,mesh),
                   mu_w,
                   del_pw,
                   rho_w_face,
                   gravity)

    q_n = residual.darcy_velocity(
                   mesh,
                   k,
                   utils.interpolate(k_rn*rho_n,mesh),
                   mu_n,
                   del_pn,
                   rho_n_face,
                   gravity)

    flux_bc ={'q_w':[q_w[0],q_w[nodes-1]],
              'q_n':[q_n[0],q_n[nodes-1]]}

    u_names= model.model.u_names
    boundary_type  = model.boundary_type
    boundary_name  = model.boundary_name
    boundary_value = model.boundary_value

    if model.model.num_u == 1:
        # boundary conditions for one variable formulations
        #Residual at Boundary 1
        if boundary_type[u_names[0]][0][0]==0:
            f[0]    = u[0] - boundary_value[0][0][0](t)
        elif boundary_type[u_names[0]][0][0]==1:
            f[0]    = flux_bc[boundary_name[0][0][0]][0] - boundary_value[0][0][0](t)
        else:
            print('Boundary type non supported')

        # Residual at Boundary 2
        if boundary_type[u_names[0]][0][1]==0:
            f[nodes-1] = u[nodes-1] - boundary_value[0][0][1](t)
        elif boundary_type[u_names[0]][0][1]==1:
            f[nodes-1] = flux_bc[boundary_name[0][0][1]][1] - boundary_value[0][0][1](t)
        else:
            print('Boundary type non supported')

    else:
        # boundary conditions for two or more variables formulations
        for n in range(model.model.num_u):

           #Residual at Boundary 1
            if boundary_type[u_names[n]][0][0]==0:
                f[0][n] = u[0][n] - boundary_value[n][0][0](t)
            elif boundary_type[u_names[n]][0][0]==1:
                f[0][n] = flux_bc[boundary_name[n][0][0]][0] - boundary_value[n][0][0](t)
            else:
                print('Boundary type non supported')

           # Residual at Boundary 2
            if boundary_type[u_names[n]][0][1]==0:
                f[nodes-1][n] = u[nodes-1][n] - boundary_value[n][0][1](t)
            elif boundary_type[u_names[n]][0][1]==1:
                f[nodes-1][n] =  flux_bc[boundary_name[n][0][1]][1] - boundary_value[n][0][1](t)
            else:
                print('Boundary type non supported')
