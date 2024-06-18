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
        'cgp-TCAT'
                ]

form_options = {
        'extended_two_phase' : {'spp': model_types.ExtendedTwoPhase('spp')
                      },
        'two_phase': {'pp': model_types.TwoPhase('pp'),
                      'sp': model_types.TwoPhase('sp')
                      },
        'Richards' : {'pr': model_types.Richards('pr')
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
                     1 for Neumann (not currently supported)
        """
        self.select_model(model_type, form)
        self.domain = domain
        self.nodes = nodes
        self.mesh = mesh
        self.current_cell = 0
        self.res = np.zeros_like(self.nodes)
        self.boundary_type = {}
        self.boundary_value = {}
        self.boundary_name = {}
        self.initial_condition = {}
        self.solution = {}
        self.time = None
        self.solid = None
        self.w_phase = None
        self.n_phase = None
        self.interface = None
        self.hysteresis = False
        self.data_hysteresis = {}
        self.counter_hysteresis = 1


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

    def set_boundary_value(self,boundary_value):
        """
        Set the boundary value for each boundary and solution variable
        """
        for n_u,u in enumerate(self.model.u_names):
            self.boundary_value[n_u] = boundary_value[u]

    def set_boundary_name(self,boundary_name):
        """
        Set the quantity name for each boundary and solution variable
        """
        for n_u,u in enumerate(self.model.u_names):
            self.boundary_name[n_u] = boundary_name[u]

    def set_initial_conditions(self,initial_conditions):
        """
        Set the initial condition for each solution variable
        """
        for u in self.model.u_names:
            self.initial_condition[u] = initial_conditions[u]

    def initial_conditions(self,ts,U):
        """
        Set the initial conditions in the PETSc arrays
        """
        da = ts.getDM()
        u = da.getVecArray(U)

        for n_u,u_name in enumerate(self.model.u_names):
            if self.model.num_u==1:
                for n in range(0,self.nodes): 
                    u[n] = self.initial_condition[u_name][n]
            else:
                for n in range(0,self.nodes): 
                    u[n][n_u] = self.initial_condition[u_name][n]

    def boundary_conditions(self,ts,U):
        """
        Set the boundary conditions in the PETSc arrays
        """
        da = ts.getDM()
        u = da.getVecArray(U)

        if self.model.num_u==1:
            u[0] = self.boundary_value[0][0][0](0)
            u[self.nodes-1] = self.boundary_value[0][0][1](0)
        else:
            for n in range(self.model.num_u):
                u[0][n] = self.boundary_value[n][0][0](0)
                u[self.nodes-1][n] = self.boundary_value[n][0][1](0)

    def set_simulation_time(self,time):
        """
        Collect the simulation output times
        where time = [t0,t1,t2,...,tn]
        """
        self.time = time

    def clean_solution(self,solution,time):
        """
        Take in output PETSc array of solutions and store in self.solutions
        """
        self.solution[time] = {}
        for n_u,u_name in enumerate(self.model.u_names):
            self.solution[time][u_name] = np.copy(solution[n_u::self.model.num_u])

    def set_data_hysteresis(self,data_hysteresis):
        """
        Set the hysteresis arrays
        """
        self.data_hysteresis = data_hysteresis

def initialize_model(model_type,form,domain,nodes,mesh):
    """
    Initialize model class
    """
    return Model(model_type,form,domain,nodes,mesh)