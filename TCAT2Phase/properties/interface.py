"""Interface.py"""
from functools import partial

class Interface:
    """
    Interface class
    """
    def __init__(self,):
        self.pc_function = None


    def set_pc_function(self,pc_function,**parameters):
        """
        Set density
        """
        self.pc_function = partial(pc_function,**parameters)