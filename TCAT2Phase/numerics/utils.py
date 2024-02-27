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

def calc_gradient(x,y,dz):
    """
    Return the gradient of a value
    """
    return (x-y)/dz
