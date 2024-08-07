import numpy as np
from numpy import linalg 
from petsc4py import PETSc
import TCAT2Phase
from TCAT2Phase.numerics import utils
from TCAT2Phase.state_equations import state_equations
from TCAT2Phase.state_equations import state_equations_interface
import matplotlib.pyplot as plt
# Domain
domain_size = np.array([[0.0,0.6]])
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
boundary_type = {'e_w': [[0, 0]],
                 'p_w': [[0, 0]]}

model.set_boundary_type(boundary_type = boundary_type)
# Set boundary names

boundary_name = {'e_w': [['e_w', 'e_w']],
                 'p_w': [['p_w', 'p_w']]}

model.set_boundary_name(boundary_name = boundary_name)

# Set boundary values
def ew0_bc(t): return 0.1998
def ewL_bc(t): return 0.111
def pw0_bc(t): return 92642.5
def pwL_bc(t): return 1900.


boundary_value = {'e_w': [[ ew0_bc, ewL_bc ]],
                  'p_w': [[ pw0_bc, pwL_bc  ]]}


model.set_boundary_value(boundary_value = boundary_value)


# Set initial conditions
ew_ic= np.zeros([nodes] )
pw_ic= np.zeros([nodes] )
for i in range(0,nodes):
    ew_ic[i] = 0.111
    pw_ic[i] = 1900

initial_conditions = {'e_w': ew_ic,
                      'p_w': pw_ic }

model.set_initial_conditions(initial_conditions)

# Set solid phase properties
s_phase = TCAT2Phase.properties.phase.Solid('s')
s_phase.set_porosity(0.37)
s_phase.set_permeability(9.4e-10)
s_phase.set_gravity(9.8066)

# Set wetting-phase properties
w_phase = TCAT2Phase.properties.phase.Fluid('w')
w_phase.set_residual_volume_fraction(0.1036)
w_phase.set_viscosity(1.e-3)
w_den_params = {'rho0':997.8,
                'beta':4.9e-10}
w_phase.set_density_function(state_equations.density_exp_pressure,**w_den_params)
w_phase.set_d_density_function(state_equations.d_density_exp_pressure,**w_den_params)

# Set nonwetting-phase properties
n_phase = TCAT2Phase.properties.phase.Fluid('n')
n_phase.set_residual_volume_fraction(0.0)
n_phase.set_viscosity(1.7e-5)
n_den_params = {'rho0':1.205 ,
                'alpha':1.189e-5
                }
n_phase.set_density_function(state_equations.density_linear_pressure,**n_den_params)
n_phase.set_d_density_function(state_equations.d_density_linear_pressure,**n_den_params)
n_en_params = {'porosity':s_phase.porosity
                }
n_phase.set_volume_fraction_function(state_equations.volume_fraction,**n_en_params)

# Set w-n interphase properties
wn_interface = TCAT2Phase.properties.interface.Interface()

m = 0.5
n = 2.0 
alpha = 3.4e-4
 
vg = state_equations_interface.VanGenuchten(
    m = m,
    n = n,
    alpha = alpha,
    porosity = s_phase.porosity,
    e_wr = w_phase.residual_volume_fraction,
    e_nr = n_phase.residual_volume_fraction,
    )

wn_interface.set_pc_function(vg.capillary_pressure)
wn_interface.set_d_pc_function(vg.d_capillary_pressure)
wn_interface.set_volume_fraction_function(vg.volume_fraction_w)
wn_interface.set_d_volume_fraction_function(vg.d_volume_fraction_w)


w_phase.set_rel_perm_function(vg.w_relative_permability)
n_phase.set_rel_perm_function(vg.n_relative_permability)

w_phase.set_volume_fraction_function(vg.volume_fraction_w)
w_phase.set_d_volume_fraction_function(vg.d_volume_fraction_w)


model.solid = s_phase
model.w_phase = w_phase
model.n_phase = n_phase
model.interface = wn_interface

# Time
time_interval = np.array([1000])
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
    plt.title("Helmig Solution")
    plt.legend()
plt.show()  


for time,variable in solutions.items():
    print(time,variable)
    plt.plot(mesh.points,variable['p_w'],label=f't={time}')
    plt.xlabel('x(m)')
    plt.ylabel('Pw')
    plt.title('Helmig Solution')
    plt.legend()
plt.show()    

 