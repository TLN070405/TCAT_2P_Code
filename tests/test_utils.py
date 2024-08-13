import numpy as np
import TCAT2Phase
from TCAT2Phase.numerics import utils
from numpy import linalg
# Domain
domain_size = np.array([[0.0,1.0]])
nodes = 11

domain =  TCAT2Phase.properties.domain.initialize_domain(
                            dims = 1,
                            domain_size = domain_size,
                            )

mesh = TCAT2Phase.mesh.grid.Mesh()
mesh.uniform_mesh_1d(domain,nodes)

allowable_error = 1.e-12

# create y(x)=x^2, derivative and interpolation
y = np.zeros([nodes])
dy = np.zeros([nodes-2])
y_face = np.zeros([nodes-2])
dx = 1./(nodes-1)

for i in range(0,nodes):
    y[i]=(i*dx)**2

for i in range(0,nodes-2):
    dy[i]=1./dx*(y[i+1]-y[i])
    y_face[i]=0.5*(y[i+1]+y[i])


def test_gradient_function():
    """
    test of gradient function
    """
    error=np.subtract(utils.gradient(y,mesh)[1:nodes-1],dy)

    assert linalg.norm(error) <= allowable_error

def test_interpolate_function():
    """
    test of interpolation function
    """
    error=np.subtract(utils.interpolate(y,mesh)[1:nodes-1],y_face)

    assert linalg.norm(error) <= allowable_error

test_gradient_function()
test_interpolate_function()
