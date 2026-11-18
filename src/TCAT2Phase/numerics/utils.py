import sys
import petsc4py
import numpy as np
petsc4py.init(sys.argv)

def calc_average(x,y):
    """
    Return the average of a value
    """
    return 0.5*(x+y)

def calc_function_average(func,x,y):
    """
    Calaculate the aveage of a function 
    """
    avg_func = 0.5*(func(x) + func(y))
    return avg_func


def cell_center(x,function,mesh):
    """
    Calculate the value of the provided function at cell-centers
    """
    value = np.zeros([mesh.num_cells])
    for cell in mesh.cells:
        value[cell.ID] = function(x[cell.ID])
    return value

def _scalar(value):
    """Mesh distances are stored as 1-element arrays; return the plain float."""
    return float(np.ravel(value)[0])

def petsc_value(value):
    """
    Prepare a residual value for assignment into a PETSc DMDA array entry.
    Single-unknown models (e.g. Richards) have scalar entries, so a 1-element
    array is unwrapped; multi-unknown models keep the array.
    """
    value = np.asarray(value)
    return value.item() if value.size == 1 else value

def gradient(x,mesh):
    """
    Return the gradient of a value. Should these be at points or cell-centers?=
    """
    gradient = np.zeros([mesh.num_faces])
    for cell in mesh.cells:
        for face,neigh,dist in zip(cell.faces,cell.neighbors,cell.distance):
            if mesh.faces[face].boundary or mesh.faces[face].owner != cell.ID:
                continue
            gradient[face] = 1/_scalar(dist)*( x[neigh] - x[cell.ID] )

    return gradient

def interpolate(x,mesh):
    """
    Interpolate cell-centered values to faces
    """
    value = np.zeros(mesh.num_faces)
    for cell in mesh.cells:
        for face,neigh,dist in zip(cell.faces,cell.neighbors,cell.distance):

            if mesh.faces[face].boundary or mesh.faces[face].owner != cell.ID:
                continue

            value[face] = x[cell.ID] + _scalar(mesh.faces[face].distance) / _scalar(dist) * (x[neigh] - x[cell.ID])

    return value
