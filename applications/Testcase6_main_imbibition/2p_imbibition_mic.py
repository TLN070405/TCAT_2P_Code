import TCAT2Phase
from TCAT2Phase.state_equations import state_equations_interface
from TCAT2Phase.state_equations import state_equations
import solver_options
import numpy as np
import matplotlib.pyplot as plt
# Domain
domain_size = np.array([[0.0,1.]])
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
                                        model_type="two_phase", form="sp", domain=domain, nodes=nodes, mesh=mesh
                                        )

# Set boundary types: 0 - Dirchlet
#                     1 - Neumann (but not supported yet)
boundary_type = {'e_w': [[0, 0]], 'p_w': [[0, 0]]}
model.set_boundary_type(boundary_type)

# Set solid phase properties
s_phase = TCAT2Phase.properties.phase.Solid(
    "s",
    porosity=0.369,
    permeability=2.67e-9,
    gravity= 0.0,
    dynamic_capillary_pressure_parameter=None,
)

# Set wetting-phase properties
w_den_params = {"rho0": 1.0e3, "beta": 1e-8}
w_phase = TCAT2Phase.properties.phase.Fluid(
    "w",
    viscosity=1.0e-3,
    momentum_resistance=0.0,
    residual_volume_fraction=0.0687*0.369,
    density_function=state_equations.density_exp_pressure,
    d_density_function=state_equations.d_density_exp_pressure,
    **w_den_params,
)

# Set nonwetting-phase properties
n_den_params = {"rho0": 1.2, "alpha": 1.189e-5}
n_phase = TCAT2Phase.properties.phase.Fluid(
    "n",
    viscosity=1.78e-5,
    momentum_resistance=0.0,
    residual_volume_fraction= (1.-0.915)*0.369,
    density_function=state_equations.density_linear_pressure,
    d_density_function=state_equations.d_density_linear_pressure,
    **n_den_params,
)

# Set w-n interphase properties
vg = state_equations_interface.VanGenuchten(
    m=1.0 - (1.0 /15.6),
    n=15.6,
    alpha=2.73e-3,
    porosity=s_phase.porosity,
    e_wr=w_phase.residual_volume_fraction,
    e_nr=n_phase.residual_volume_fraction,
)

wn_interface = TCAT2Phase.properties.interface.TraditionalInterface(vg)

#set relative permeabilities
w_phase.set_relative_permeability_function(vg.w_relative_permeability)
n_phase.set_relative_permeability_function(vg.n_relative_permeability)

# Set model phases and interfaces
TCAT2Phase.numerics.model.Model.set_model_phases(model, s_phase, w_phase, n_phase, wn_interface)

# Set initial conditions

def ew_ic(x):
    """ volume fraction ic at t=0"""
    return w_phase.residual_volume_fraction +1e-5

def pw_ic(x):
    """wetting phase pressure ic at t=0"""
    return -wn_interface.pc_function(ew_ic(x))

initial_conditions = { 'e_w': ew_ic, 'p_w': pw_ic }
model.set_initial_conditions(initial_conditions)

# Set boundary conditions

pw0_bc = -300.
pwL_bc = -300.
ew0_bc = model.interface.volume_fraction_function(-pw0_bc)
ewL_bc = model.interface.volume_fraction_function(-pwL_bc)

boundary_conditions = {'e_w': [[ ew0_bc, ewL_bc  ]], 'p_w': [[ pw0_bc, pwL_bc  ]] }
model.set_boundary_conditions(boundary_conditions)


# Time
time_interval = np.array([1,10,100])
model.set_simulation_time(time_interval)

# Numerical Parameters
solver_options.solver_options(max_time_steps=100000, bdf_order=5, pc_type="sor")


# Solve model
u,ts = TCAT2Phase.numerics.solver.setup_solver(model)
solutions = TCAT2Phase.numerics.solver.solver_time(u,ts,model)


for time,variable in solutions.items():
    plt.plot(mesh.points,variable['p_w'],label=f't={time}')
    plt.xlabel(r'$x\, (m)$')
    plt.ylabel(r'$p_w$')
    plt.legend()
plt.show()

for time,variable in solutions.items():
    plt.plot(mesh.points,variable['e_w'],label=f't={time}')
    plt.xlabel(r'$x\, (m)$')
    plt.ylabel(r'$\epsilon^w$')
    plt.legend()
plt.show()
