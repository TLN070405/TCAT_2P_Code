""" tcat cgp fully saturated domain test case """

import TCAT2Phase
from TCAT2Phase.state_equations import state_equations_interface
from TCAT2Phase.state_equations import state_equations
import solver_options
import numpy as np
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
model = TCAT2Phase.numerics.model.Model(
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

# Set solid phase properties
s_phase = TCAT2Phase.properties.phase.Solid('s',porosity = 0.36,
                                                permeability = None,
                                                gravity = 0.0,
                                                dynamic_capillary_pressure_parameter = None)

# Set wetting-phase properties
w_den_params = {'rho0':1.e3, 'beta':1e-8}
w_phase=TCAT2Phase.properties.phase.Fluid('w',viscosity = None,
                                              momentum_resistance = 1.5e6,
                                              residual_volume_fraction = 0.0,
                                              density_function = state_equations.density_exp_pressure,
                                              d_density_function = state_equations.d_density_exp_pressure,
                                              **w_den_params)

# Set w-n interphase properties
tcat = state_equations_interface.TCAT(b1 = 1.47e-1,
                                      b2 = -2.08e-3,
                                      b3 = 5.33e-2/(4.0 * 3.141592653589793),
                                      porosity = s_phase.porosity,
                                      e_ns = 1. ,
                                      e_ws = 1. ,
                                      js_ns = 1.,
                                      kn_ns = 1.)

wn_interface = TCAT2Phase.properties.interface.TCATInterface(tcat,
                                                             interfacial_dynamics = 3.42e-7,
                                                             interfacial_tension = 35e-3,
                                                             interfacial_area_relaxation = -1.41e-2,
                                                             interfacial_area_equilibrium = np.full((nodes), 1e-8))

TCAT2Phase.numerics.model.Model.set_model_phases(
    model, s_phase, w_phase, None, wn_interface
)

# Set boundary values
def ew0_bc(t): return  0.9 * s_phase.porosity
def pw0_bc(t): return 2000
def ewn0_bc(t): return 1e-8
def jwn0_bc(t): return -2000/wn_interface.interfacial_tension
def wwn0_bc(t): return  0.0
def kwn0_bc(t): return -2242533027.870139

def ewL_bc(t): return  0.9 * s_phase.porosity
def pwL_bc(t): return 2000
def ewnL_bc(t): return 1e-8
def jwnL_bc(t): return -2000/wn_interface.interfacial_tension
def wwnL_bc(t): return  0.0
def kwnL_bc(t): return -2242533027.870139


boundary_conditions= {'e_w': [[ ew0_bc,ewL_bc ]],
                  'p_w': [[ pw0_bc,pwL_bc ]],
                  'e_wn': [[ ewn0_bc,ewnL_bc ]],
                  'j_wn': [[ jwn0_bc,jwnL_bc ]],
                  'w_wn': [[ wwn0_bc,wwnL_bc ]],
                  'k_wn': [[ kwn0_bc,kwnL_bc ]]}

model.set_boundary_conditions(boundary_conditions)

# Set initial conditions
def ew_ic(x): return 0.9*s_phase.porosity
def pw_ic(x): return 2000
def ewn_ic(x): return 1e-8
def jwn_ic(x): return -2000/wn_interface.interfacial_tension
def wwn_ic(x): return 0.0
def kwn_ic(x): return -2242533027.870139

initial_conditions = {'e_w': ew_ic,
                      'p_w': pw_ic,
                      'e_wn': ewn_ic,
                      'j_wn': jwn_ic,
                      'w_wn': wwn_ic,
                      'k_wn': kwn_ic}

model.set_initial_conditions(initial_conditions)

# Time
time_interval = np.array([50.])
model.set_simulation_time(time_interval)

# Set petsc solver options
solver_options.solver_options(max_time_steps=100000,bdf_order=1,pc_type='sor')

# Solve model and return model.solutions which is a dictionary where
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
