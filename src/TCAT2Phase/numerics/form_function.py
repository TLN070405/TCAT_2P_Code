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
    u_name = model.model.u_names
    nodes = model.nodes

    for n_u,u_name in enumerate(model.model.u_names):
        if model.model.num_u==1:
            for n in range(0,nodes):
                x = model.domain.domain_size[0,0] + n* model.domain.domain_length[0]/(nodes-1)
                f[n] = u[n] - model.initial_conditions[u_name](x)
        else:
            for n in range(0,nodes):
                x = model.domain.domain_size[0,0] + n* model.domain.domain_length[0]/(nodes-1)
                f[n][n_u] = u[n][n_u] - model.initial_conditions[u_name](x)


def model_formfunction_boundary(t,f,u,model):
    """
    Set Boundary points residual equations in order to solve IC in case we have Neumman BC. 
    """
    residual.model_residual_boundary(t, f, u, model)
