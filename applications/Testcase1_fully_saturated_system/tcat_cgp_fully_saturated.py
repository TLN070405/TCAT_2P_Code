""" tcat cgp fully saturated domain test case """

import TCAT2Phase
from TCAT2Phase.state_equations import state_equations_interface
from TCAT2Phase.state_equations import state_equations
import solver_options
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from TCAT2Phase.io.hdf5_io import save_solutions_to_hdf5
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
model = TCAT2Phase.numerics.model.Model(
                                model_type = 'cgp-TCAT',
                                form = 'spijvk',
                                domain = domain,
                                nodes = nodes,
                                mesh = mesh
                                    )

# Set boundary types: 0 - Dirchlet
#                     1 - Neumann qw
#                     2 - Neumann qn
boundary_type = {'e_w': [[0, 0]],
                 'p_w': [[0, 0]],
                 'e_wn': [[0, 0]],
                 'j_wn': [[0, 0]],
                 'w_wn': [[0, 0]],
                 'k_wn': [[0, 0]]}

model.set_boundary_type(boundary_type = boundary_type)

# Set solid phase properties
s_phase = TCAT2Phase.properties.phase.Solid('s',porosity = 0.369,
                                                permeability = None,
                                                gravity = 0.0,
                                                dynamic_capillary_pressure_parameter = None)

# Set wetting-phase properties
w_den_params = {'rho0':1.e3, 'beta':4.5e-10}
w_phase=TCAT2Phase.properties.phase.Fluid('w',viscosity = None,
                                              momentum_resistance = 5.10e4,
                                              residual_volume_fraction = 0.0,
                                              density_function = state_equations.density_exp_pressure,
                                              d_density_function = state_equations.d_density_exp_pressure,
                                              **w_den_params)

# Set w-n interphase properties
tcat = state_equations_interface.TCAT(b1 = 1.47e-1,
                                      b2 = -2.08e-3,
                                      b3 = 5.33e-2/(4.0 * 3.141592653589793),
                                      porosity = s_phase.porosity,
                                      e_ss = 1.89e3 ,
                                      js_ns = 2.00e3,
                                      kn_ns = 1.00e6)

def interfacial_area_equilibrium(x): return 1e-3

wn_interface = TCAT2Phase.properties.interface.TCATInterface(tcat,
                                                             interfacial_dynamics = 7.35e-10,
                                                             interfacial_tension = 7.20e-2,
                                                             interfacial_area_relaxation = 3.64e-3,
                                                             interfacial_area_equilibrium = interfacial_area_equilibrium)

TCAT2Phase.numerics.model.Model.set_model_phases(
    model, s_phase, w_phase, None, wn_interface
)

# Set boundary values
ew0_bc  = 0.369
pw0_bc  = 2500
ewn0_bc = 1e-3
jwn0_bc = -pw0_bc/wn_interface.interfacial_tension
wwn0_bc = 0.0
kwn0_bc = model.interface.gaussian_curvature(ew0_bc, ewn0_bc,jwn0_bc)

ewL_bc = 0.369
pwL_bc = 2000
ewnL_bc = 1e-3
jwnL_bc = -pwL_bc/wn_interface.interfacial_tension
wwnL_bc = 0.0
kwnL_bc = model.interface.gaussian_curvature(ewL_bc, ewnL_bc,jwnL_bc)

boundary_conditions= {'e_w': [[ ew0_bc,ewL_bc ]],
                  'p_w': [[ pw0_bc,pwL_bc ]],
                  'e_wn': [[ ewn0_bc,ewnL_bc ]],
                  'j_wn': [[ jwn0_bc,jwnL_bc ]],
                  'w_wn': [[ wwn0_bc,wwnL_bc ]],
                  'k_wn': [[ kwn0_bc,kwnL_bc ]]}

model.set_boundary_conditions(boundary_conditions)

# Set initial conditions
def ew_ic(x):
    """
    volume fraction w initial conditions
    """
    return 0.369
def pw_ic(x):
    """
    pressure w initial conditions
    """
    return ((2000-2500)/(1.-0.))*x + 2500
def ewn_ic(x):
    """
    interfacial area of wn interface initial conditions
    """
    return 1e-3
def jwn_ic(x):
    """
    Mean curvature of wn interface initial conditions
    """
    return -pw_ic(x)/wn_interface.interfacial_tension
def wwn_ic(x):
    """
    velocity of wn interface initial conditions
    """
    return 0.0
def kwn_ic(x):
    """
    Gaussian curvature of wn interface initial conditions
    """
    return model.interface.gaussian_curvature(ew_ic(x), ewn_ic(x),jwn_ic(x))

initial_conditions = {'e_w': ew_ic,
                      'p_w': pw_ic,
                      'e_wn': ewn_ic,
                      'j_wn': jwn_ic,
                      'w_wn': wwn_ic,
                      'k_wn': kwn_ic}

model.set_initial_conditions(initial_conditions)

# Time
time_interval = np.array([1,5,10])
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


# Get the directory where the current script (re_test_case_1.py) is located
script_dir = Path(__file__).resolve().parent

# Ensure we join the script directory with the filename
filename = "tcat_cgp_test_case_1.h5"
file_path = script_dir   # This gives you the full path to the file

# Save the data to the correct file path
save_solutions_to_hdf5(str(file_path), filename,mesh, solutions)