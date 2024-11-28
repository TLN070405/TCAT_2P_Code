"""model_types.py"""

from TCAT2Phase.numerics import residual
from TCAT2Phase.numerics import form_function

class BaseModel:
    """
    BaseModel class
    """
    def __init__(self):
        self.name = None
        self.u_names = []
        self.num_u = 0

class CgpTCAT(BaseModel):
    """
    Extended two phase model formulation
    """
    def __init__(self,form):
        if form == 'spijvk':
            self.name = 'cgp-TCAT-spijvk'
            self.u_names = ['e_w','p_w','e_wn','j_wn','w_wn','k_wn']
            self.num_u = 6
            self.residual = residual.model_residual
            self.form_function = form_function.model_formfunction

class ExtendedTwoPhase(BaseModel):
    """
    Extended two phase model formulation
    """
    def __init__(self,form):
        if form == 'spp':
            self.name = 'extended-two-phase-spp'
            self.u_names = ['e_w','p_w','p_n']
            self.num_u = 3
            self.residual = residual.model_residual
            self.form_function = form_function.model_formfunction

class TwoPhase(BaseModel):
    """
    Standard two phase model formulation
    """
    def __init__(self,form):
        if form == 'sp':
            self.name = 'two-phase-sp'
            self.u_names = ['e_w','p_w']
            self.num_u = 2
            self.residual = residual.model_residual
            self.form_function = form_function.model_formfunction

        if form == 'pp':
            self.name = 'two-phase-pp'
            self.u_names = ['p_w','p_n']
            self.num_u = 2
            self.residual = residual.model_residual
            self.form_function = form_function.model_formfunction

class ExtendedRichards(BaseModel):
    """
    Extended Richards model formulation
    """
    def __init__(self,form):
        if form == 'sp':
            self.name = 'extended-richards-sp'
            self.u_names = ['e_w','p_w']
            self.num_u = 2
            self.residual = residual.model_residual
            self.form_function = form_function.model_formfunction

class Richards(BaseModel):
    """
    Richards model formulation
    """
    def __init__(self,form):
        if form == 'pr':
            self.name = 'richards-pr'
            self.u_names = ['p_w']
            self.num_u = 1
            self.residual = residual.model_residual
            self.form_function = form_function.model_formfunction
