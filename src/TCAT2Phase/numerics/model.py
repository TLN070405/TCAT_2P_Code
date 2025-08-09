"""discretization.py"""
import numpy as np
from . import model_types

__all__ = [
    "initialize_model"
]

model_options = [
        'two_phase',
        'extended_two_phase',
        'Richards',
        'extended_Richards',
        'cgp-TCAT',
        'full-TCAT'
        ]

form_options = {
        'full-TCAT' : {'sppijgvk': model_types.FullTCAT('sppijgvk')
                      },
        'cgp-TCAT' : {'spijvk': model_types.CgpTCAT('spijvk')
                      },
        'extended_two_phase' : {'spp': model_types.ExtendedTwoPhase('spp')
                      },
        'two_phase': {'pp': model_types.TwoPhase('pp'),
                      'sp': model_types.TwoPhase('sp')
                      },
        'Richards' : {'pr': model_types.Richards('pr')
                      },
        'extended_Richards' : {'sp': model_types.ExtendedRichards('sp')
                      }
               }


class Model:
    """
    Class for discretizations
    """
    def __init__(self, model_type, form, domain, nodes, mesh):
        """
        domain is instance to domain class
        nodes = [X,X,X] integer for total number of nodes in each dimension

        boundary_types = [[bc00 bc01]
                          [bc10 bc11]
                          [bc20 bc21]]

        where bcXX = 0 for Dirchlet
                     1 for Neumann 
        """
        self.select_model(model_type, form)
        self.domain = domain
        self.nodes = nodes
        self.mesh = mesh
        self.current_cell = 0
        self.res = np.zeros_like(self.nodes)
        self.boundary_type = {}
        self.boundary_conditions = {}
        self.initial_conditions = {}
        self.solution = {}
        self.time = None
        self.solid = None
        self.w_phase = None
        self.n_phase = None
        self.interface = None
        self.hysteresis = False

    def select_model(self, model_type, form):
        """
        Select the model class
        """
        if model_type in model_options:
            if form in form_options[model_type]:
                self.model = form_options[model_type][form]
            else:
                print(f"Not a valid formulation. Options are {form_options[model_type]}")
        else:
            print(f"Not a valid model. Options are {model_options}")

    def get_resolution(self):
        """
        Calculate the resolution
        """
        for d in range(self.domain.dims):
            self.res[d] = self.domain.domain_length[d]/(self.nodes[d]-1)

    def set_boundary_type(self,boundary_type):
        """
        Set the boundary type for each boundary and solution variable
        """
        for u in self.model.u_names:
            self.boundary_type[u] = boundary_type[u]

    def set_boundary_conditions(self,boundary_conditions):
        """
        Set the boundary conditions for each boundary and solution variable
        """
        for n_u,u in enumerate(self.model.u_names):
            self.boundary_conditions[n_u] = boundary_conditions[u]

    def set_initial_conditions(self,initial_conditions):
        """
        Set the initial condition for each solution variable
        """
        for u in self.model.u_names:
            self.initial_conditions[u] = initial_conditions[u]

    def pass_initial_conditions(self,snes,U):
        """
        Pass the initial conditions from the model object to PETSc solution arrays
        """
        da = snes.getDM()
        u = da.getVecArray(U)
        for n_u,u_name in enumerate(self.model.u_names):
            if self.model.num_u==1:
                for n in range(0,self.nodes):
                    x = self.domain.domain_size[0,0] + n* self.domain.domain_length[0]/(self.nodes-1)
                    u[n] = self.initial_conditions[u_name](x)
            else:
                for n in range(0,self.nodes):
                    x = self.domain.domain_size[0,0] + n* self.domain.domain_length[0]/(self.nodes-1)
                    u[n][n_u] = self.initial_conditions[u_name](x)

    def set_simulation_time(self,time):
        """
        Collect the simulation output times where time = [t0,t1,t2,...,tn]
        """
        self.time = time

    def clean_solution(self,solution,time):
        """
        Take in output PETSc array of solutions and store in self.solutions
        """
        self.solution[time] = {}
        for n_u,u_name in enumerate(self.model.u_names):
            self.solution[time][u_name] = np.copy(solution[n_u::self.model.num_u])

    def set_model_phases(self,s_phase,w_phase,n_phase,interface):
        """
        Set the  model phases and interfaces
        """
        self.solid = s_phase
        self.w_phase = w_phase
        self.n_phase = n_phase
        self.interface = interface
