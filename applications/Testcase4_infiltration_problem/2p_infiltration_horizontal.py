import numpy as np
import TCAT2Phase
from TCAT2Phase.state_equations import state_equations
from TCAT2Phase.state_equations import state_equations_interface
from TCAT2Phase.numerics import solver_options
from pathlib import Path
from TCAT2Phase.io.hdf5_io import save_solutions_to_hdf5

# Domain
domain_size = np.array([[0.0, 10.]])  # L = 10 m, Table 4.13
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
    permeability=2.67e-9,  # Table 4.3
    gravity= 0.0,  # horizontal domain
    dynamic_capillary_pressure_parameter=None,
)

# Set wetting-phase properties
w_den_params = {"rho0": 1.0e3, "beta": 4.5e-10}  # Table 4.2
w_phase = TCAT2Phase.properties.phase.Fluid(
    "w",
    viscosity=1.0e-3,
    momentum_resistance=0.0,
    residual_volume_fraction=0.0687*0.369,  # Table 4.3
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
    residual_volume_fraction=0.0,
    density_function=state_equations.density_linear_pressure,
    d_density_function=state_equations.d_density_linear_pressure,
    **n_den_params,
)

# Set w-n interphase properties
vg = state_equations_interface.VanGenuchten(
    m=1.0 - (1.0 /15.6),  # MIC parameters, Table 4.3
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
TCAT2Phase.numerics.model.Model.set_model_phases(
    model, s_phase, w_phase, n_phase, wn_interface
)

# Set initial conditions
def pw_ic(x):
    """wetting phase pressure ic at t=0"""
    return -1000

def ew_ic(x):
    """volume fraction ic at t=0""" 
    return  model.interface.volume_fraction_function(-pw_ic(x))

initial_conditions = {"e_w": ew_ic, "p_w": pw_ic}
model.set_initial_conditions(initial_conditions)

# Set boundary values
pw0_bc = -900.
pwL_bc = -1000.
ew0_bc = model.interface.volume_fraction_function(-pw0_bc)
ewL_bc = model.interface.volume_fraction_function(-pwL_bc)

boundary_conditions = {'e_w': [[ ew0_bc, ewL_bc ]], 'p_w': [[ pw0_bc, pwL_bc  ]]}

model.set_boundary_conditions(boundary_conditions)

# Time
time_interval = np.array([1e1, 1e2, 1e3, 1e4, 1e5, 1.5e5])  # same output times as the RE driver
model.set_simulation_time(time_interval)

# Set petsc solver options
solver_options.solver_options(max_time_steps=100000, bdf_order=5, pc_type="sor")

# Solve model and return model.solutions
# which is a dictionary where solutions[time][u_name] = np.array
u, ts = TCAT2Phase.numerics.solver.setup_solver(model)
solutions = TCAT2Phase.numerics.solver.solver_time(u,ts,model)

for time,variable in solutions.items():
    p_w = variable['p_w']
    e_w = variable['e_w']
    p_n=np.zeros([nodes] )
    for n in range(0,nodes):
        p_n[n]= model.interface.pc_function(e_w[n]) + p_w[n]
    variable["p_n"] = p_n



# Export results
filename = "2P_test_case_4.h5"
file_path =  Path(__file__).resolve().parent.parent.parent / 'outputs'/ 'Testcase4_infiltration_problem'/'outputs'

# Save the data to the correct file path
metadata  = { "nodes":nodes}
save_solutions_to_hdf5(str(file_path), filename,mesh, solutions,metadata)
