import numpy as np
import TCAT2Phase

# Domain
domain_size = np.array([[0.0,1.0]])
domain =  TCAT2Phase.properties.domain.initialize_domain(
                            dims = 1,
                            domain_size = domain_size,
                            )

# Discretization
nodes = 101
boundary_type = [0,0]
model = TCAT2Phase.model.initialize_model(
                                model_type = 'two_phase',
                                form = 'sp',
                                domain = domain,
                                nodes = nodes
                                    )

# Set boundary types: 0 - Dirchlet
#                     1 - Neumann (but not supported yet)
boundary_type = {'e_w': [[0, 0]],
                 'p_w': [[0, 0]]}

model.set_boundary_type(boundary_type = boundary_type)

# Set boundary values
boundary_value = {'e_w': [[0.12, 0.008]],
                  'p_w': [[-2113.152788450121, -13706.965861446095]]}

model.set_boundary_value(boundary_value = boundary_value)


# Set initial conditions: Only set up for constant values!
initial_conditions = {'e_w': 0.008,
                      'p_w': -13706.965861446095}
model.set_initial_conditions(initial_conditions)


# Set solid phase properties
s_phase = TCAT2Phase.properties.phase.Solid('s')
s_phase.set_porosity(0.4)
s_phase.set_permeability(1e-10)


# Set wetting-phase properties
w_phase = TCAT2Phase.properties.phase.Fluid('w')
w_phase.set_viscosity(1.e-3)
w_den_params = {'rho0':1.e3,
                'beta':4.9e-10}
w_phase.set_density_function(TCAT2Phase.state_equations.density_exp_pressure,**w_den_params)
w_phase.set_d_density_function(TCAT2Phase.state_equations.d_density_exp_pressure,**w_den_params)

# Set nonwetting-phase properties
n_phase = TCAT2Phase.properties.phase.Fluid('n')
n_phase.set_viscosity(1.78e-5)
n_den_params = {'rho0':1.2,
                'alpha':1.189e-5
                }
n_phase.set_density_function(TCAT2Phase.state_equations.density_linear_pressure,**n_den_params)
n_phase.set_d_density_function(TCAT2Phase.state_equations.d_density_linear_pressure,**n_den_params)

# Set w-n interphase properties
nw_interface = TCAT2Phase.properties.interface.Interface() 
pc_params = {'Epsilon_w': 0.,
             'Const_Epsilon': 0.,
             'm_inv': 0.,
             'n_inv': 0.,
             'alpha_inv': 0.
            }
nw_interface.set_pc_function(TCAT2Phase.state_equations.vg_capillary_pressure,**pc_params)


model.solid = s_phase
model.w_phase = w_phase
model.n_phase = n_phase
model.interface = nw_interface



# Time
time_interval = np.array([0.,864.,1728.,2592.,3456.,6048.])
model.set_simulation_time(time_interval)

# Numerical Parameters
numerics = TCAT2Phase.numerics.Numerics()

# Solve model
TCAT2Phase.numerics.solver(model,numerics)




# Archive

# # Physical parameters of the problem and van genuchten model parameters
# phi = 0.4
# epsilon_nr = 0.0
# epsilon_wr = 0.0
# n = 2.494
# rho_w0 = 1e3  
# rho_n0 = 1.2 
# mu_w = 1e-3 
# mu_n = 1.78e-5
# g = 9.8066
# d = -1  
# alpha = 1e-3 
# a = 4.9e-10 
# b = 1.189e-5 
# K = 1e-10
# tau = 0.0  

# # Boundary and initial conditions of the problem
# #S0= 0.02
# #SL= 0.3
# Ew_bc_0 = 0.12                         # phi*SL
# Ew_bc_L = 0.008                        # phi*S0
# Pw_bc_0 = -2113.152788450121           #-Pc(phi*SL)
# Pw_bc_L = -13706.965861446095          #-Pc(phi*S0)
# Pn_bc_0 = 0.0 
# Pn_bc_L = 0.0
# Ew_IC = 0.008                          #phi*S0
# Pw_IC = -13706.965861446095            #-Pc(phi*S0)
# Pn_IC = 0.0

# # Solver parameters
# r_tol = 1e-8
# a_tol = 1e-8
# MaxStepRejections = 500
# MaxSNESFailures = 10000  
# MaxTimeSteps = 20000