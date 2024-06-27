"""domain.py"""
import numpy as np

# TODO: add checks for form dims and valid inputs

__all__ = [
    'initialize_domain'
]

class Domain:
    """
    Domain class for numerics
    """
    def __init__(self,dims,domain_size):
        """
        domain_size = [[dom00 dom01]
                       [dom10 dom11]
                       [dom20 dom21]]

        where domX0 is the start point of the domain in the X dimension
              domX1 is the end point of the domain in the X dimension

        """
        self.dims = dims
        self.domain_size = domain_size
        self.get_domain_length()

    def get_domain_length(self):
        """
        Calculate the domain length
        """
        self.domain_length = np.zeros(self.dims)
        for d in range(self.dims):
            self.domain_length[d] = self.domain_size[d,1] - self.domain_size[d,0]

def initialize_domain(dims,domain_size):
    """
    Initialize domain class
    """
    return Domain(dims,domain_size)
