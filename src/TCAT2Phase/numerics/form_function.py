import numpy as np
from . import model_types
from TCAT2Phase.numerics import utils


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
    initial_conditions = np.zeros(model.model.num_u)

    for n in range(0,nodes):
        x = model.domain.domain_size[0,0] + n* model.domain.domain_length[0]/(nodes-1)
        for n_u,u_name in enumerate(model.model.u_names):
            initial_conditions[n_u] = model.initial_conditions[u_name](x)

        f[n] = utils.petsc_value(u[:][n] - initial_conditions)



def model_formfunction_boundary(t,f,u,model):
    """
    Set Boundary points residual equations in order to solve IC in case we have Neumman BC. 
    """
    residual_class = model_types.get_residual_class(model)
    residual_instance = residual_class(model)
    residual_instance.compute_residual_boundary(t, f, u)
