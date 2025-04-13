import numpy as np
import TCAT2Phase
from TCAT2Phase.state_equations import state_equations
from TCAT2Phase.state_equations import state_equations_interface
import matplotlib.pyplot as plt
import solver_options
from pathlib import Path
from TCAT2Phase.io.hdf5_io import save_solutions_to_hdf5

# Domain
domain_size = np.array([[0.0,0.1]])
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
                                              momentum_resistance = 3.75e5,#5.10e4,
                                              residual_volume_fraction = 0.0687*0.369,
                                              density_function = state_equations.density_exp_pressure,
                                              d_density_function = state_equations.d_density_exp_pressure,
                                              **w_den_params)

# Set w-n interphase properties
tcat = state_equations_interface.TCAT(b1 = 1.47e-1,
                                      b2 = -2.08e-3,
                                      b3 = 5.33e-2/(4.0 * 3.141592653589793),
                                      porosity = s_phase.porosity,
                                      e_ss = 1.89e3,
                                      js_ns = 2.00e3,
                                      kn_ns = 1.00e6)


def interfacial_area_equilibrium(x):
    """
    Equilibrium specific interfacial area function of space
    """
    e_wn_eq = 100.
    return e_wn_eq


wn_interface = TCAT2Phase.properties.interface.TCATInterface(tcat,
                                                             interfacial_dynamics = 6.398675e-05, #7.35e-10,
                                                             interfacial_tension = 7.20e-2,
                                                             interfacial_area_relaxation = 1.003597e-10 ,#3.64e-3,
                                                             interfacial_area_equilibrium = interfacial_area_equilibrium)

TCAT2Phase.numerics.model.Model.set_model_phases(model, s_phase, w_phase, None, wn_interface)


# Set initial conditions
def ew_ic(x): return 0.35701796706321004
def pw_ic(x): return -402
def ewn_ic(x): return 100.
def jwn_ic(x): return -pw_ic(x)/wn_interface.interfacial_tension
def wwn_ic(x): return 0.0
def kwn_ic(x): return model.interface.gaussian_curvature(ew_ic(x), ewn_ic(x), jwn_ic(x))

initial_conditions = {'e_w': ew_ic,
                      'p_w': pw_ic,
                      'e_wn': ewn_ic,
                      'j_wn': jwn_ic,
                      'w_wn': wwn_ic,
                      'k_wn': kwn_ic}

model.set_initial_conditions(initial_conditions)


# Set boundary conditions
def ew0_bc(t): return 0.35901615483439936
def pw0_bc(t): return -400
def ewn0_bc(t): return 100.
def jwn0_bc(t): return -pw0_bc(t)/wn_interface.interfacial_tension
def wwn0_bc(t): return 0.0
def kwn0_bc(t): return model.interface.gaussian_curvature(ew0_bc(t), ewn0_bc(t), jwn0_bc(t))

def ewL_bc(t): return ew_ic(domain_size[0][1])
def pwL_bc(t): return pw_ic(domain_size[0][1])
def ewnL_bc(t): return ewn_ic(domain_size[0][1])
def jwnL_bc(t): return jwn_ic(domain_size[0][1])
def wwnL_bc(t): return wwn_ic(domain_size[0][1])
def kwnL_bc(t): return model.interface.gaussian_curvature(ew0_bc(t), ewn0_bc(t), jwn0_bc(t))

boundary_conditions = {'e_w': [[ ew0_bc,ewL_bc ]],
                  'p_w': [[ pw0_bc,pwL_bc ]],
                  'e_wn': [[ ewn0_bc,ewnL_bc ]],
                  'j_wn': [[ jwn0_bc,jwnL_bc ]],
                  'w_wn': [[ wwn0_bc,wwnL_bc ]],
                  'k_wn': [[ kwn0_bc,kwnL_bc ]]}

model.set_boundary_conditions(boundary_conditions)


# Time
time_interval = np.array([1e-7,1e-5,1e-4,1e-3,1e-2,1e-1, 1,4])
model.set_simulation_time(time_interval)

# Set petsc solver options
solver_options.solver_options(max_time_steps=100000,bdf_order=5,pc_type='sor')

# Solve model and return model.solutions
# which is a dictionary where
# solutions[time][u_name] = np.array
u, ts = TCAT2Phase.numerics.solver.setup_solver(model)
solutions = TCAT2Phase.numerics.solver.solver_time(u,ts,model)


# Chart 1: Volume Fraction (ε^w)
plt.figure(figsize=(10, 6))
for time, variable in solutions.items():
    plt.plot(mesh.points, variable['e_w'], label=f'TCAT CGP, t={time}')
plt.xlabel(r'$x\,[m]$', fontsize=20)
plt.ylabel(r'$\epsilon^w\,[-]$', fontsize=20)
plt.legend(fontsize=12)
plt.show()

# Chart 2: Pressure (p^w)
plt.figure(figsize=(10, 6))
for time, variable in solutions.items():
    plt.plot(mesh.points, variable['p_w'], label=f'TCAT CGP, t={time}')
plt.xlabel(r'$x\,[m]$', fontsize=20)
plt.ylabel(r'$p^w\,[Pa]$', fontsize=20)
plt.legend(fontsize=12)
plt.show()

# Chart 3: ε^{wn} with y-axis scale [-2,2]
plt.figure(figsize=(10, 6))
for time, variable in solutions.items():
    plt.plot(mesh.points, variable['e_wn'], label=f'TCAT CGP, t={time}')
plt.xlabel(r'$x\,[m]$', fontsize=20)
plt.ylabel(r'$\epsilon^{wn}\,[-]$', fontsize=20)
plt.ylim([0, 200])
plt.legend(fontsize=12)
plt.show()

# Chart 4: J^{wn}_n with units [1/m]
plt.figure(figsize=(10, 6))
for time, variable in solutions.items():
    plt.plot(mesh.points, variable['j_wn'], label=f'TCAT CGP, t={time}')
plt.xlabel(r'$x\,[m]$', fontsize=20)
plt.ylabel(r'$J^{wn}_n\,[1/m]$', fontsize=20)
plt.legend(fontsize=12)
plt.show()

# Chart 5: w^{wn} with y-axis scale [-2,2]
plt.figure(figsize=(10, 6))
for time, variable in solutions.items():
    plt.plot(mesh.points, variable['w_wn'], label=f'TCAT CGP, t={time}')
plt.xlabel(r'$x\,[m]$', fontsize=20)
plt.ylabel(r'$w^{wn}\,[-]$', fontsize=20)
plt.ylim([-2, 2])
plt.legend(fontsize=12)
plt.show()

# Chart 6: K^{wn}_n with units [1/m^2] overwritten to be zero for all space and time
plt.figure(figsize=(10, 6))
for time, variable in solutions.items():
    plt.plot(mesh.points, variable['k_wn'], label=f'TCAT CGP, t={time}')
plt.xlabel(r'$x\,[m]$', fontsize=20)
plt.ylabel(r'$K^{wn}_n\,[1/m^2]$', fontsize=20)
plt.legend(fontsize=12)
plt.show()


# Export results
filename = "tcat_cgp_test_case_4.h5"
file_path = Path(__file__).resolve().parent/ 'outputs'

# Save the data to the correct file path
save_solutions_to_hdf5(str(file_path), filename,mesh, solutions)