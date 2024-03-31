
import sys
import petsc4py
import numpy as np
from TCAT2Phase import state_equations
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

def gradient(x,mesh):
    """
    Return the gradient of a value. Should these be at points or cell-centers?=
    """
    gradient = np.zeros([mesh.num_faces])
    for cell in mesh.cells:
        for face,neigh,dist in zip(cell.faces,cell.neighbors,cell.distance):
            if mesh.faces[face].boundary or mesh.faces[face].owner != cell.ID:
                continue
            #gradient[face] = 1/dist*(x[cell.ID] - x[neigh] )
            gradient[face] = 1/dist*(x[neigh] - x[cell.ID] )
    
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
            #value[face] = x[cell.ID] + mesh.faces[face].distance/dist*(x[cell.ID] - x[neigh])
            value[face] = 0.5*(x[cell.ID] + x[neigh])
    return value
