import numpy as np
from numpy import linalg 
from petsc4py import PETSc
import TCAT2Phase
from TCAT2Phase.numerics import utils
from TCAT2Phase.state_equations import state_equations
from TCAT2Phase.state_equations import state_equations_interface
import matplotlib.pyplot as plt
# Domain
domain_size = np.array([[0.0,0.8]])
domain =  TCAT2Phase.properties.domain.initialize_domain(
                            dims = 1,
                            domain_size = domain_size,
                            )

# Discretization
nodes = 101

# Mesh
mesh = TCAT2Phase.mesh.grid.Mesh()
mesh.uniform_mesh_1d(domain,nodes)

# Boundaries
boundary_type = [0,0]
model = TCAT2Phase.numerics.model.initialize_model(
                                model_type = 'two_phase',
                                form = 'sp',
                                domain = domain,
                                nodes = nodes,
                                mesh = mesh
                                    )

# Set boundary types: 0 - Dirchlet
#                     1 - Neumann (but not supported yet)
boundary_type = {'e_w': [[0, 1]],
                 'p_w': [[0, 1]]}

model.set_boundary_type(boundary_type = boundary_type)



# Set boundary names
boundary_name = {'e_w': [['e_w', 'q_w']],
                 'p_w': [['p_w', 'q_n']]}

model.set_boundary_name(boundary_name = boundary_name)

# Set boundary values
def ew0_bc(t): return 0.27 
def pw0_bc(t): return -1049.67
def qwL_bc(t): return 0.0 
def qnL_bc(t): return 0.0 

boundary_value = {'e_w': [[ ew0_bc, qwL_bc ]],
                 'p_w': [[ pw0_bc , qnL_bc  ]]}


model.set_boundary_value(boundary_value = boundary_value)

# Set initial conditions: Only set up for constant values!
ew_ic= np.zeros([nodes] )
pw_ic= np.zeros([nodes] )
pw_ic[0] = -1049.67 
ew_ic[0] =  0.27 
for i in range(1,nodes):
    pw_ic[i] = -9996.39 
    ew_ic[i] =   0.003 


initial_conditions = {'e_w': ew_ic ,
                      'p_w': pw_ic  }
model.set_initial_conditions(initial_conditions)

# Set solid phase properties
s_phase = TCAT2Phase.properties.phase.Solid('s')
s_phase.set_porosity(0.3 )
s_phase.set_permeability(9.770000000000002e-11 )
s_phase.set_gravity( 0.0)

# Set wetting-phase properties
w_phase = TCAT2Phase.properties.phase.Fluid('w')
w_phase.set_residual_volume_fraction(0.0 )
w_phase.set_viscosity(1.e-3 )
w_den_params = {'rho0':1000 ,
                'beta':4.9e-10 }
w_phase.set_density_function(state_equations.density_exp_pressure,**w_den_params)
w_phase.set_d_density_function(state_equations.d_density_exp_pressure,**w_den_params)

# Set nonwetting-phase properties
n_phase = TCAT2Phase.properties.phase.Fluid('n')
n_phase.set_residual_volume_fraction(0.0 )
n_phase.set_viscosity(1.57e-5 )
n_den_params = {'rho0':1.204 ,
                'alpha':1.189e-5 
                }
n_phase.set_density_function(state_equations.density_linear_pressure,**n_den_params)
n_phase.set_d_density_function(state_equations.d_density_linear_pressure,**n_den_params)
n_en_params = {'porosity':s_phase.porosity
                }
n_phase.set_volume_fraction_function(state_equations.volume_fraction,**n_en_params)

# Set w-n interphase properties
wn_interface = TCAT2Phase.properties.interface.Interface()

pd=  1000.0 
lamda = 2.0
bc = state_equations_interface.BrooksCorey(
    pd = pd,
    lamda = lamda,
    porosity = s_phase.porosity,
    e_wr = w_phase.residual_volume_fraction,
    e_nr = n_phase.residual_volume_fraction,
    )

wn_interface.set_pc_function(bc.capillary_pressure)
wn_interface.set_d_pc_function(bc.d_capillary_pressure)


w_phase.set_rel_perm_function(bc.w_relative_permability)
n_phase.set_rel_perm_function(bc.n_relative_permability)

w_phase.set_volume_fraction_function(bc.volume_fraction_w)
w_phase.set_d_volume_fraction_function(bc.d_volume_fraction_w)


model.solid = s_phase
model.w_phase = w_phase
model.n_phase = n_phase
model.interface = wn_interface

# Time
time_interval = np.array([ 10,100,1000,3000])
model.set_simulation_time(time_interval)
 
# Numerical Parameters
OptDB = PETSc.Options()

# Time-Stepper Options
OptDB.setValue('snes_fd_color', None)
OptDB.setValue('ts_type', 'bdf')
OptDB.setValue('ts_bdf_order', 5)
OptDB.setValue('ts_max_steps',100000)
OptDB.setValue('ts_max_reject',500)
OptDB.setValue('ts_monitor', None)

# Nonlinear Solver Options
OptDB.setValue('ts_max_snes_failures',10000)

# Linear Solver Options
OptDB.setValue('ts_rtol', 1.e-8)
OptDB.setValue('ts_atol', 1.e-8)
# Preconditioner Options 
OptDB.setValue('pc_type', 'sor')
 


 
# Solve model
# Solve model and return model.solutions
# which is a dictionary where
# solutions[time][u_name] = np.array
solutions = TCAT2Phase.numerics.solver.solver(model)

for time,variable in solutions.items():
    plt.plot(mesh.points,variable['e_w'],label=f't={time}')
    plt.xlabel("x(m)")
    plt.ylabel("e_w")
    plt.title("Szymkiewicz Solution")
    plt.legend()
plt.show()  


for time,variable in solutions.items():
    print(time,variable)
    plt.plot(mesh.points,variable['p_w'],label=f't={time}')
    plt.xlabel('x(m)')
    plt.ylabel('Pw')
    plt.title('Szymkiewicz Solution')
    plt.legend()
plt.show()    

 