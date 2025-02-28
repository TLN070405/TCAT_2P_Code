import numpy as np
from numpy import linalg 
from petsc4py import PETSc
import TCAT2Phase
from TCAT2Phase.numerics import utils
from TCAT2Phase.state_equations import state_equations
from TCAT2Phase.state_equations import state_equations_interface
import matplotlib.pyplot as plt
# Domain
domain_size = np.array([[0.0,1.0]])
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
                                model_type = 'cgp-TCAT',
                                form = 'spijvk',
                                domain = domain,
                                nodes = nodes,
                                mesh = mesh
                                    )

# Set boundary types: 0 - Dirchlet
#                     1 - Neumann 
boundary_type = {'e_w': [[0, 0]],
                 'p_w': [[0, 0]],
                 'e_wn': [[0, 0]],
                 'j_wn': [[0, 0]],
                 'w_wn': [[0, 0]],
                 'k_wn': [[0, 0]]}

model.set_boundary_type(boundary_type = boundary_type)

# Set boundary names
boundary_name = {'e_w': [['e_w', 'e_w']],
                 'p_w': [['p_w', 'p_w']],
                 'e_wn': [['e_wn', 'e_wn']],
                 'j_wn': [['j_wn', 'j_wn']],
                 'w_wn': [['w_wn', 'w_wn']],
                 'k_wn': [['k_wn', 'k_wn']]}

model.set_boundary_name(boundary_name = boundary_name)



# Set solid phase properties
s_phase = TCAT2Phase.properties.phase.Solid('s')
s_phase.set_porosity(0.369)
s_phase.set_gravity(0.0)

# Set wetting-phase properties
w_phase = TCAT2Phase.properties.phase.Fluid('w')
w_phase.set_momentum_resistance(1.5e6)
w_den_params = {'rho0':1.e3,
                'beta':1e-8}
w_phase.set_density_function(state_equations.density_exp_pressure,**w_den_params)
w_phase.set_d_density_function(state_equations.d_density_exp_pressure,**w_den_params)

# Set w-n interphase properties
wn_interface = TCAT2Phase.properties.interface.Interface()

b1 = 1.47e-1
b2 = -2.08e-3
b3 = 5.33e-2/(4.0 * 3.141592653589793)

tcat = state_equations_interface.TCAT(
    b1 = b1,
    b2 = b2,
    b3 = b3,
    porosity = s_phase.porosity,
    e_ns = 1. ,
    e_ws = 1. ,
    js_ns = 1.,
    kn_ns = 1.167425480799784
    )
wn_interface.set_interfacial_dynamics(3.42e-7)
wn_interface.set_interfacial_tension(35e-3)
wn_interface.set_interfacial_area_relaxation(-1.41e-2)
wn_interface.set_volume_fraction_function(tcat.volume_fraction_w)

w_phase.set_rel_perm_function(tcat.w_relative_permability)
w_phase.set_volume_fraction_function(tcat.volume_fraction_w)

model.solid = s_phase
model.w_phase = w_phase
model.interface = wn_interface



ewn_eq = np.zeros([nodes])
for i in range(0,nodes):
    ewn_eq[i] = 0.1
wn_interface.set_interfacial_area_equilibrium(ewn_eq)


# Set boundary values
def ew0_bc(t): return  0.9 * s_phase.porosity
def pw0_bc(t): return -1000
def ewn0_bc(t): return 0.1
def jwn0_bc(t): return 1000/wn_interface.interfacial_tension
def wwn0_bc(t): return  0.0
def kwn0_bc(t): return -12257.385200908866

def ewL_bc(t): return  0.9 * s_phase.porosity
def pwL_bc(t): return -1000
def ewnL_bc(t): return 0.1
def jwnL_bc(t): return 1000/wn_interface.interfacial_tension
def wwnL_bc(t): return  0.0
def kwnL_bc(t): return -12257.385200908866


boundary_value = {'e_w': [[ ew0_bc,ewL_bc ]],
                  'p_w': [[ pw0_bc,pwL_bc ]],
                  'e_wn': [[ ewn0_bc,ewnL_bc ]],
                  'j_wn': [[ jwn0_bc,jwnL_bc ]],
                  'w_wn': [[ wwn0_bc,wwnL_bc ]],
                  'k_wn': [[ kwn0_bc,kwnL_bc ]]}

model.set_boundary_value(boundary_value = boundary_value)





# Set initial conditions
ew_ic= np.zeros([nodes])
pw_ic= np.zeros([nodes])
ewn_ic= np.zeros([nodes])
jwn_ic= np.zeros([nodes])
wwn_ic= np.zeros([nodes])
kwn_ic= np.zeros([nodes])

i=0
ew_ic[i] =  0.9 * s_phase.porosity
pw_ic[i] = -1000
ewn_ic[i] = 0.1
jwn_ic[i] = 1000/wn_interface.interfacial_tension
wwn_ic[i] = 0.0
kwn_ic[i] = -12257.385200908866

i=nodes-1
ew_ic[i] =  0.9 * s_phase.porosity
pw_ic[i] = -1000
ewn_ic[i] = 0.1
jwn_ic[i] = 1000/wn_interface.interfacial_tension
wwn_ic[i] = 0.0
kwn_ic[i] = -12257.385200908866


for i in range(1,nodes-1):
    ew_ic[i] =  0.9 * s_phase.porosity
    pw_ic[i] = -1000+1e-6
    ewn_ic[i] = 0.1
    jwn_ic[i] = 1000/wn_interface.interfacial_tension
    wwn_ic[i] = 0.0
    kwn_ic[i] = -12257.385200908866


initial_conditions = {'e_w': ew_ic,
                      'p_w': pw_ic,
                      'e_wn': ewn_ic,
                      'j_wn': jwn_ic,
                      'w_wn': wwn_ic,
                      'k_wn': kwn_ic}

model.set_initial_conditions(initial_conditions)


# Time
time_interval = np.array([1e-5,1e-4,1e-3,1e-2,1e-1])
model.set_simulation_time(time_interval)

# Numerical Parameters
OptDB = PETSc.Options()

# Time-Stepper Options
OptDB.setValue('snes_fd_color', None)
OptDB.setValue('ts_type', 'bdf')
OptDB.setValue('ts_bdf_order', 1)
OptDB.setValue('ts_max_steps',100000)
OptDB.setValue('ts_max_reject',500)
OptDB.setValue('ts_monitor', None)

# Nonlinear Solver Options
OptDB.setValue('ts_max_snes_failures',10000)

# Linear Solver Options
OptDB.setValue('ts_rtol', 1.e-8)
OptDB.setValue('ts_atol', 1.e-8)
# Preconditioner Options
OptDB.setValue('pc_type', 'ilu')


# Solve model and return model.solutions
# which is a dictionary where
# solutions[time][u_name] = np.array
u, ts = TCAT2Phase.numerics.solver.setup_solver(model)
solutions = TCAT2Phase.numerics.solver.solver_time(u,ts,model)


for time,variable in solutions.items():
    plt.plot(mesh.points,variable['e_w'],label=f't={time}')
    plt.xlabel(r'$x\, (m)$')
    plt.ylabel(r'$\epsilon^w$')
    plt.title('tcat cgp')
    plt.legend()
plt.show()


for time,variable in solutions.items():
    print(time,variable)
    plt.plot(mesh.points,variable['p_w'],label=f't={time}')
    plt.xlabel(r'$x\, (m)$')
    plt.ylabel(r'$p_w$')
    plt.title('tcat cgp')
    plt.legend()
plt.show()

for time,variable in solutions.items():
    print(time,variable)
    plt.plot(mesh.points,variable['e_wn'],label=f't={time}')
    plt.xlabel(r'$x\, (m)$')
    plt.ylabel(r'$\epsilon^{wn}$')
    plt.title('tcat cgp')
    plt.legend()
plt.show()

for time,variable in solutions.items():
    print(time,variable)
    plt.plot(mesh.points,variable['j_wn'],label=f't={time}')
    plt.xlabel(r'$x\, (m)$')
    plt.ylabel(r'$J^{wn}$')
    plt.title('tcat cgp')
    plt.legend()
plt.show()

for time,variable in solutions.items():
    print(time,variable)
    plt.plot(mesh.points,variable['w_wn'],label=f't={time}')
    plt.xlabel(r'$x\, (m)$')
    plt.ylabel(r'$w^{wn}$')
    plt.title('tcat cgp')
    plt.legend()
plt.show()

for time,variable in solutions.items():
    print(time,variable)
    plt.plot(mesh.points,variable['k_wn'],label=f't={time}')
    plt.xlabel(r'$x\, (m)$')
    plt.ylabel(r'$K^{wn}$')
    plt.title('tcat cgp')
    plt.legend()
plt.show()