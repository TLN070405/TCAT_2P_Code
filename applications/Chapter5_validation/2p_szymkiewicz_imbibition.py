"""
2P-TM validation, imbibition: Example 1 of Szymkiewicz (2007), dissertation
Chapter 5, Test Case 6 (Tables 5.4-5.5, Fig. 5.6).

Countercurrent imbibition: water enters a horizontal 0.8 m column of nearly dry
porous medium at x = 0 while air leaves through the same boundary; x = L is
closed to both fluids. With incompressible fluids the McWhorter & Sunada (1990)
semi-analytic solution applies; outputs/Chapter5_validation/postprocessing/
compare_mcwhorter_sunada.py compares this run with it.

Parameters are Szymkiewicz's Table 1 (Example 1). Tables 5.4-5.5 restate them
with a few entries that disagree with the paper or with each other; this driver
follows the paper:
  - initial e_w = 0.003 (S_e = 0.01 with e_wr = 0), not 0.008, which would need
    a residual volume fraction that neither table lists;
  - p_d = rho_w g 0.102 m = 1000.6 Pa, not 1020 Pa;
  - k_i = K_s mu_w / (rho_w g) = 1.0e-10 m^2 from K_s = 9.81e-4 m/s, not 9.77e-11;
  - rho_n = 1.204 kg/m^3;
  - x = L is impermeable to both fluids (Szymkiewicz Fig. 1 and the
    dissertation text); Table 5.5 lists Dirichlet values there, equal to the
    initial state, which the front does not reach before t = 5000 s.
Outputs at 3000 s (Szymkiewicz's final time) and 5000 s (the dissertation's).
"""
import TCAT2Phase
from TCAT2Phase.state_equations import state_equations_interface
from TCAT2Phase.state_equations import state_equations
from TCAT2Phase.numerics import solver_options
import numpy as np
from petsc4py import PETSc
from pathlib import Path
from TCAT2Phase.io.hdf5_io import save_solutions_to_hdf5

# Domain
domain_size = np.array([[0.0, 0.8]])
domain = TCAT2Phase.properties.domain.initialize_domain(dims=1, domain_size=domain_size)

# Discretization (Szymkiewicz used 10 to 80 grid blocks)
nodes = 81

# Mesh
mesh = TCAT2Phase.mesh.grid.Mesh()
mesh.uniform_mesh_1d(domain, nodes)

model = TCAT2Phase.numerics.model.Model(model_type="two_phase", form="sp", domain=domain, nodes=nodes, mesh=mesh)

# Boundary types: 0 - Dirichlet
#                 4 - mass flux of the wetting phase,     with half-cell storage (use with 5)
#                 5 - mass flux of the non-wetting phase, with half-cell storage (use with 4)
boundary_type = {'e_w': [[0, 4]], 'p_w': [[0, 5]]}
model.set_boundary_type(boundary_type)

rho_w, rho_n, g = 1.0e3, 1.204, 9.81
mu_w, mu_n = 1.0e-3, 1.57e-5
K_s = 9.81e-4                               # m/s
permeability = K_s * mu_w / (rho_w * g)     # 1.0e-10 m^2

s_phase = TCAT2Phase.properties.phase.Solid(
    "s",
    porosity=0.3,
    permeability=permeability,
    gravity=0.0,
    dynamic_capillary_pressure_parameter=None,
)

w_phase = TCAT2Phase.properties.phase.Fluid(
    "w",
    viscosity=mu_w,
    momentum_resistance=0.0,
    residual_volume_fraction=0.0,
    density_function=state_equations.density_exp_pressure,
    d_density_function=state_equations.d_density_exp_pressure,
    rho0=rho_w, beta=0.0,                   # incompressible
)

n_phase = TCAT2Phase.properties.phase.Fluid(
    "n",
    viscosity=mu_n,
    momentum_resistance=0.0,
    residual_volume_fraction=0.0,
    density_function=state_equations.density_linear_pressure,
    d_density_function=state_equations.d_density_linear_pressure,
    rho0=rho_n, alpha=0.0,                  # incompressible
)

# Brooks-Corey-Burdine closure relations
bc = state_equations_interface.BrooksCorey(
    pd=rho_w * g * 0.102,
    lamda=2.0,
    porosity=s_phase.porosity,
    e_wr=w_phase.residual_volume_fraction,
    e_nr=n_phase.residual_volume_fraction,
)
wn_interface = TCAT2Phase.properties.interface.TraditionalInterface(bc)
w_phase.set_relative_permeability_function(bc.w_relative_permeability)
n_phase.set_relative_permeability_function(bc.n_relative_permeability)
TCAT2Phase.numerics.model.Model.set_model_phases(model, s_phase, w_phase, n_phase, wn_interface)

# Initial conditions: S_e = 0.01, p_n = 0
ew_init = 0.01 * s_phase.porosity
pw_init = -wn_interface.pc_function(ew_init)
model.set_initial_conditions({'e_w': lambda x: ew_init, 'p_w': lambda x: pw_init})

# Boundary conditions: x = 0, S_e = 0.9 and p_n = 0; x = L, no flow of either fluid
ew0_bc = 0.9 * s_phase.porosity
pw0_bc = -wn_interface.pc_function(ew0_bc)
boundary_conditions = {'e_w': [[ew0_bc, 0.0]], 'p_w': [[pw0_bc, 0.0]]}
model.set_boundary_conditions(boundary_conditions)

# Time
time_interval = np.array([1000.0, 2000.0, 3000.0, 4000.0, 5000.0])
model.set_simulation_time(time_interval)

# Numerical parameters: a direct solve suits the incompressible (index-1) system
solver_options.solver_options(max_time_steps=100000, bdf_order=2, pc_type="lu")
# Time-error tolerance 1e-6 instead of the drivers' 1e-8: the solution is self-similar
# (dt grows in proportion to t), and 1e-6 keeps the temporal error well below the
# spatial error at this grid while taking about ten times fewer steps.
PETSc.Options().setValue('ts_rtol', 1.0e-6)
PETSc.Options().setValue('ts_atol', 1.0e-6)

# Solve model
u, ts = TCAT2Phase.numerics.solver.setup_solver(model)
solutions = TCAT2Phase.numerics.solver.solver_time(u, ts, model)

for time, variable in solutions.items():
    variable["p_n"] = np.array([model.interface.pc_function(e) for e in variable['e_w']]) + variable['p_w']

# Export results
filename = "2P_szymkiewicz_imbibition.h5"
file_path = Path(__file__).resolve().parent.parent.parent / 'outputs' / 'Chapter5_validation' / 'outputs'
metadata = {"nodes": nodes}
save_solutions_to_hdf5(str(file_path), filename, mesh, solutions, metadata)
