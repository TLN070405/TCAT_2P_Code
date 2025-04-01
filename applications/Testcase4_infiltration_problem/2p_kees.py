import numpy as np
import math
from petsc4py import PETSc
import TCAT2Phase
from TCAT2Phase.state_equations import state_equations
from TCAT2Phase.state_equations import state_equations_interface
import matplotlib.pyplot as plt
import solver_options
# Domain
domain_size = np.array([[5.0,10.0]])
domain =  TCAT2Phase.properties.domain.initialize_domain(
                            dims = 1,
                            domain_size = domain_size,
                            )

# Discretization
nodes = 51

# Mesh
mesh = TCAT2Phase.mesh.grid.Mesh()
mesh.uniform_mesh_1d(domain,nodes)

# Boundaries

# Boundaries
model = TCAT2Phase.numerics.model.Model(
                                        model_type="two_phase", form="pp", domain=domain, nodes=nodes, mesh=mesh
                                        )

# Set boundary types: 0 - Dirchlet
#                     1 - Neumann (but not supported yet)
boundary_type = {'p_w': [[0, 0]],
                 'p_n': [[0, 0]]}

model.set_boundary_type(boundary_type = boundary_type)

# Set solid phase properties
s_phase = TCAT2Phase.properties.phase.Solid(
    "s",
    porosity=0.301,
    permeability=5.957035950473679e-12,
    gravity=-9.81,
    dynamic_capillary_pressure_parameter=None,
)

# Set wetting-phase properties
w_den_params = {"rho0": 9.982e2, "beta": 4.9e-10}
w_phase = TCAT2Phase.properties.phase.Fluid(
    "w",
    viscosity=1.0e-3,
    momentum_resistance=0.0,
    residual_volume_fraction= 9.300e-2,
    density_function=state_equations.density_exp_pressure,
    d_density_function=state_equations.d_density_exp_pressure,
    **w_den_params,
)

# Set nonwetting-phase properties
n_den_params = {"rho0": 1.205, "alpha": 1.189e-5}
n_phase = TCAT2Phase.properties.phase.Fluid(
    "n",
    viscosity=1.8e-4,
    momentum_resistance=0.0,
    residual_volume_fraction=0.0,
    density_function=state_equations.density_linear_pressure,
    d_density_function=state_equations.d_density_linear_pressure,
    **n_den_params,
)

# Set w-n interphase properties
vg = state_equations_interface.VanGenuchten(
    m= 0.7654784240150094,
    n=4.264e0,
    alpha=0.0005585997711272747,
    porosity=s_phase.porosity,
    e_wr=w_phase.residual_volume_fraction,
    e_nr=n_phase.residual_volume_fraction,
)

wn_interface = TCAT2Phase.properties.interface.TraditionalInterface(vg)

# Set model phases and interfaces
TCAT2Phase.numerics.model.Model.set_model_phases(
    model, s_phase, w_phase, n_phase, wn_interface
)

# Set initial conditions
def pw_ic(x):
    """volume fraction ic at t=0""" 
    return -x*9.982e2*9.81

def pn_ic(x):
    """wetting phase pressure ic at t=0"""
    return 0.0

initial_conditions = {'p_w': pw_ic, 'p_n': pn_ic }
model.set_initial_conditions(initial_conditions)

# Set boundary values
def pw0_bc(t): return -5.0*9.982e2*9.81 
def pn0_bc(t): return 0.
def pwL_bc(t): return -10.0*(1.0 - 1.01*math.exp(-(t)/(24.0*3600.0)))*(9.982e2*9.81 )
def pnL_bc(t): return 0.

boundary_conditions = {'p_w': [[ pw0_bc,pwL_bc ]], 'p_n': [[ pn0_bc,pnL_bc  ]]}
model.set_boundary_conditions(boundary_conditions)

# Time
time_interval = np.array([ 864.0,1728.0,2592.0,3456.0,6048.0])
model.set_simulation_time(time_interval)

# Set petsc solver options
solver_options.solver_options(max_time_steps=100000, bdf_order=5, pc_type="lu")

# Solve model and return model.solutions
# which is a dictionary where
# solutions[time][u_name] = np.array
u,ts = TCAT2Phase.numerics.solver.setup_solver(model)
solutions = TCAT2Phase.numerics.solver.solver_time(u,ts,model)

for time,variable in solutions.items():
    plt.plot(variable['p_w'],mesh.points,label=f't={time}')
    plt.xlabel(r'$p_w$')
    plt.ylabel(r'$x\, (m)$')
    plt.title("Kees Solution")
    plt.legend()
plt.show()


for time,variable in solutions.items():
    print(time,variable)
    plt.plot(mesh.points,variable['p_n'],label=f't={time}')
    plt.xlabel(r'$x\, (m)$')
    plt.ylabel(r'$p_n$')
    plt.title('Kees Solution')
    plt.legend()
plt.show()
