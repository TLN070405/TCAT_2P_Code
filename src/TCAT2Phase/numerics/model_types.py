"""model_types.py"""

from TCAT2Phase.numerics import form_function
from TCAT2Phase.numerics.residual_2p_pp import Residual2pPP
from TCAT2Phase.numerics.residual_2p_sp import Residual2pSP
from TCAT2Phase.numerics.residual_e2p_spp import ResidualE2pSPP
from TCAT2Phase.numerics.residual_re_pr import ResidualRePR
from TCAT2Phase.numerics.residual_ere_sp import ResidualEReSP
from TCAT2Phase.numerics.residual_tcat_cgp import ResidualTcatCGP
from TCAT2Phase.numerics.residual_full_tcat import ResidualTcatFULL

class BaseModel:
    """
    BaseModel class
    """
    def __init__(self):
        self.name = None
        self.u_names = []
        self.num_u = 0

class FullTCAT(BaseModel):
    """
    Extended two phase model formulation
    """
    def __init__(self,form):
        if form == 'sppijgvk':
            self.name = 'full-TCAT-sppijgvk'
            self.u_names = ['e_w','p_w','p_n','e_wn','j_wn','g_wn','w_wn','k_wn']
            self.num_u = 8
            self.form_function = form_function.model_formfunction

class CgpTCAT(BaseModel):
    """
    Extended two phase model formulation
    """
    def __init__(self,form):
        if form == 'spijvk':
            self.name = 'cgp-TCAT-spijvk'
            self.u_names = ['e_w','p_w','e_wn','j_wn','w_wn','k_wn']
            self.num_u = 6
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
            self.form_function = form_function.model_formfunction

        if form == 'pp':
            self.name = 'two-phase-pp'
            self.u_names = ['p_w','p_n']
            self.num_u = 2
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
            self.form_function = form_function.model_formfunction


def get_residual_class(model):
    """
    function to get the correct residual depending on model and formulation
    """
    form  = model.model.name
    if form=="full-TCAT-sppijgvk":
        return ResidualTcatFULL
    elif form=="cgp-TCAT-spijvk":
        return ResidualTcatCGP
    elif form=="extended-two-phase-spp":
        return ResidualE2pSPP
    elif form=="two-phase-pp":
        return Residual2pPP
    elif form=="two-phase-sp":
        return Residual2pSP
    elif form=="extended-richards-sp":
        return ResidualEReSP
    elif form=="richards-pr":
        return ResidualRePR
