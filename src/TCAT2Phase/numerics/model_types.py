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

class TwoPhase(BaseModel):
    """
    Standard two phase model formulation
    """
    def __init__(self,form):
        if form == 'sp':
            self.name = 'two-phase-sp'
            self.u_names = ['e_w','p_w']
            self.num_u = 2
            self.residual = residual.two_phase_residual
            self.form_function = form_function.two_phase_formfunction
        if form == 'pp':
            self.name = 'two-phase-pp'
            self.u_names = ['p_w','p_n']
            self.num_u = 2
            self.residual = residual.two_phase_residual
            self.form_function = form_function.two_phase_formfunction
