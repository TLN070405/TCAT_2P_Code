import numpy as np
import TCAT2Phase


# Domain
domain_size = np.array([[0.0,1.0]])
domain =  TCAT2Phase.initialize_domain(
                            dims = 1,
                            domain_size = domain_size,
                            )

# Discretization
nodes = 101
boundary_type = [0,0]
model = TCAT2Phase.initialize_model(
                                model_type = 'two_phase',
                                form = 'sp',
                                domain = domain,
                                nodes = nodes
                                    )

boundary_type = {'e_w': [[1, 1]],
                  'p_w': [[1, 1]]}

model.set_boundary_type(boundary_type = boundary_type)

boundary_value = {'e_w': [[0.12, 0.008]],
                  'p_w': [[-2113.152788450121, -13706.965861446095]]}

model.set_boundary_value(boundary_value = boundary_value)




# Time
time_interval = np.array([864.,1728.,2592.,3456.,6048.])

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