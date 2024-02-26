"""model_types.py"""

class BaseModel:
    """
    BaseModel class
    """
    def __init__(self):
        self.u_names = []
        self.num_u = 0 

class TwoPhase(BaseModel):
    """
    Standard two phase model formulation
    """
    def __init__(self,form):
        if form == 'sp':
            self.u_names = ['e_w','p_w']
            self.num_u = 2

        if form == 'pp':
            self.u_names = ['p_w','p_n']
            self.num_u = 2











