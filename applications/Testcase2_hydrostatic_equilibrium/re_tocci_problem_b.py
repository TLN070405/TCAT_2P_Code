import numpy as np
import TCAT2Phase
from TCAT2Phase.state_equations import state_equations
from TCAT2Phase.state_equations import state_equations_interface
import matplotlib.pyplot as plt
import solver_options

# Domain
domain_size = np.array([[0. , 10]])
domain =  TCAT2Phase.properties.domain.initialize_domain(
                            dims = 1,
                            domain_size = domain_size,
                            )

# Discretization
nodes = 51

# Mesh
mesh = TCAT2Phase.mesh.grid.Mesh()
mesh.uniform_mesh_1d(domain,nodes)

# Initialize model
model = TCAT2Phase.numerics.model.Model(
                                        model_type="Richards", form="pr", domain=domain, nodes=nodes, mesh=mesh
                                        )

# Set boundary types: 0 - Dirchlet
#                     1 - Neumann (but not supported yet)
boundary_type = {'p_w': [[0, 0]]}

model.set_boundary_type(boundary_type)


# Set solid phase properties
s_phase = TCAT2Phase.properties.phase.Solid(
    "s",
    porosity=0.301,
    permeability=5.946313285762827e-12,
    gravity=-9.81,
    dynamic_capillary_pressure_parameter=None,
)

# Set wetting-phase properties
w_den_params = {"rho0": 1.0e3, "beta":  4.9e-10}
w_phase = TCAT2Phase.properties.phase.Fluid(
    "w",
    viscosity=1.e-3,
    momentum_resistance=0.0,
    residual_volume_fraction=0.093 ,
    density_function=state_equations.density_exp_pressure,
    d_density_function=state_equations.d_density_exp_pressure,
    **w_den_params,
)

# Set nonwetting-phase properties
n_den_params = {"rho0": 1.205, "alpha": 1.189866e-5}
n_phase = TCAT2Phase.properties.phase.Fluid(
    "n",
    viscosity= 1.8e-4,
    momentum_resistance=0.0,
    residual_volume_fraction=0.0,
    density_function=state_equations.density_linear_pressure,
    d_density_function=state_equations.d_density_linear_pressure,
    **n_den_params,
)

# Set w-n interphase properties
vg = state_equations_interface.VanGenuchten(
    m=0.7652582159624413,
    n=4.26,
    alpha=0.0005575942915392457,
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
    """wetting phase pressure ic at t=0"""
    return -1000*9.81*x

initial_conditions = {"p_w": pw_ic}
model.set_initial_conditions(initial_conditions)

# Set boundary conditions
pw0_bc = 0.0
pwL_bc = 981.

# Set boundary conditions
boundary_conditions = {"p_w": [[pw0_bc, pwL_bc]]}

model.set_boundary_conditions(boundary_conditions)


# Time
time_interval = np.array([17280.])
model.set_simulation_time(time_interval)

# Set petsc solver options
solver_options.solver_options(max_time_steps=100000, bdf_order=5, pc_type="sor")


# Solve model
u, ts = TCAT2Phase.numerics.solver.setup_solver(model)
solutions = TCAT2Phase.numerics.solver.solver_time(u,ts,model)


for time,variable in solutions.items():
    plt.plot(mesh.points,variable['p_w'],label=f't={time}')
    plt.xlabel(r'$x\, (m)$')
    plt.ylabel(r'$p_w$')
    plt.title('Richards Solution')
    plt.legend()
plt.show()
