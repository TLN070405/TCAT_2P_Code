"""
2P-TM validation, primary drainage: Problem HD of Kees & Miller (2002),
dissertation Chapter 5, Test Case 5 (Tables 5.2-5.3, Fig. 5.1).

A non-wetting fluid is injected at x = 0 into a horizontal 10 m column that is
initially water saturated; water leaves at x = L. The McWhorter & Sunada (1990)
semi-analytic solution applies; outputs/Chapter5_validation/postprocessing/
compare_mcwhorter_sunada.py compares this run with it.

Parameters are Kees & Miller's Table 1 and 2, which Table 5.2 restates, except
two Table 5.2 entries that disagree with them and with the table's own K_s:
  - k_i = 5.96e-12 m^2 is the VIR-problem value of Kees & Miller; here
    k_i = K_s mu_w / (rho_w g) = 1.0005e-10 m^2 from K_s = 42.4 m/day
    (Table 5.2 gives the unit of K_s as m/s; Kees & Miller give m/day).
  - mu_w = 2e-3 Pa s, mu_n = 1e-3 Pa s as in Table 5.2 (only the ratio, 2, matters).
Both fluids are incompressible (beta_w = beta_n = 0).

Boundary conditions (Table 5.3):
  x = 0: no water flux; non-wetting influx A t^(-1/2) with A = 0.1967 m/day^(1/2),
         replaced on [0, delta_a] (delta_a = 1e-3 day) by Kees & Miller's linear
         ramp, which injects the same volume. Imposed with the mass-conservative
         flux boundary (types 4 and 5): the boundary node keeps its half-cell
         storage, so it can desaturate as the non-wetting fluid enters.
  x = L: e_w = porosity (S_e = 1), p_w = 0.
Initial conditions: e_w = porosity, psi_w = 3 A delta_a^(-1/2) (L - x) / K_s.
"""
import TCAT2Phase
from TCAT2Phase.state_equations import state_equations_interface
from TCAT2Phase.state_equations import state_equations
from TCAT2Phase.numerics import solver_options
import numpy as np
from petsc4py import PETSc
from pathlib import Path
from TCAT2Phase.io.hdf5_io import save_solutions_to_hdf5

DAY = 86400.0

# Domain
domain_size = np.array([[0.0, 10.0]])
domain = TCAT2Phase.properties.domain.initialize_domain(dims=1, domain_size=domain_size)

# Discretization
nodes = 101

# Mesh
mesh = TCAT2Phase.mesh.grid.Mesh()
mesh.uniform_mesh_1d(domain, nodes)

model = TCAT2Phase.numerics.model.Model(model_type="two_phase", form="sp", domain=domain, nodes=nodes, mesh=mesh)

# Boundary types: 0 - Dirichlet
#                 4 - mass flux of the wetting phase,     with half-cell storage (use with 5)
#                 5 - mass flux of the non-wetting phase, with half-cell storage (use with 4)
boundary_type = {'e_w': [[4, 0]], 'p_w': [[5, 0]]}
model.set_boundary_type(boundary_type)

# Fluid and solid properties (Tables 5.2 and Kees & Miller Table 1)
rho_w, rho_n, g = 1.0e3, 1.21, 9.81
mu_w, mu_n = 2.0e-3, 1.0e-3
K_s = 42.4 / DAY                           # saturated hydraulic conductivity [m/s]
permeability = K_s * mu_w / (rho_w * g)    # 1.0005e-10 m^2

s_phase = TCAT2Phase.properties.phase.Solid(
    "s",
    porosity=0.35,
    permeability=permeability,
    gravity=0.0,
    dynamic_capillary_pressure_parameter=None,
)

w_phase = TCAT2Phase.properties.phase.Fluid(
    "w",
    viscosity=mu_w,
    momentum_resistance=0.0,
    residual_volume_fraction=1.75e-2,
    density_function=state_equations.density_exp_pressure,
    d_density_function=state_equations.d_density_exp_pressure,
    rho0=rho_w, beta=0.0,                  # incompressible
)

n_phase = TCAT2Phase.properties.phase.Fluid(
    "n",
    viscosity=mu_n,
    momentum_resistance=0.0,
    residual_volume_fraction=0.0,
    density_function=state_equations.density_linear_pressure,
    d_density_function=state_equations.d_density_linear_pressure,
    rho0=rho_n, alpha=0.0,                 # incompressible
)

# Brooks-Corey closure relations, p_d = 2000 Pa (psi_D = 0.204 m), lambda = 2
bc = state_equations_interface.BrooksCorey(
    pd=2000.0,
    lamda=2.0,
    porosity=s_phase.porosity,
    e_wr=w_phase.residual_volume_fraction,
    e_nr=n_phase.residual_volume_fraction,
)
wn_interface = TCAT2Phase.properties.interface.TraditionalInterface(bc)
w_phase.set_relative_permeability_function(bc.w_relative_permeability)
n_phase.set_relative_permeability_function(bc.n_relative_permeability)
TCAT2Phase.numerics.model.Model.set_model_phases(model, s_phase, w_phase, n_phase, wn_interface)

# Non-wetting injection
A = 0.1967 / np.sqrt(DAY)                  # m/s^(1/2)
delta_a = 1.0e-3 * DAY                     # s


def injection_velocity(t):
    """Volumetric non-wetting influx at x = 0 [m/s], with the start-up ramp on [0, delta_a]."""
    if t < delta_a:
        return 3.0 * A / np.sqrt(delta_a) - 2.0 * A * delta_a ** -1.5 * t
    return A / np.sqrt(t)


# Initial conditions
def ew_ic(x):
    """water saturated"""
    return s_phase.porosity


def pw_ic(x):
    """steady head that carries the initial influx: psi = 3 A delta_a^(-1/2) (L - x) / K_s"""
    return rho_w * g * 3.0 * A / np.sqrt(delta_a) * (10.0 - x) / K_s


model.set_initial_conditions({'e_w': ew_ic, 'p_w': pw_ic})


# Boundary conditions. Flux values use the code's convention q = -(rho * velocity in +x),
# so an influx at x = 0 is negative.
def qw0_bc(t):
    """no water flux at x = 0"""
    return 0.0


def qn0_bc(t):
    """non-wetting mass influx at x = 0"""
    return -rho_n * injection_velocity(t)


boundary_conditions = {'e_w': [[qw0_bc, s_phase.porosity]], 'p_w': [[qn0_bc, 0.0]]}
model.set_boundary_conditions(boundary_conditions)

# Time: Kees & Miller Fig. 1 and dissertation Fig. 5.1 show t = 2.5 days
time_interval = np.array([0.5, 1.0, 1.5, 2.0, 2.5]) * DAY
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
filename = "2P_kees_miller_drainage.h5"
file_path = Path(__file__).resolve().parent.parent.parent / 'outputs' / 'Chapter5_validation' / 'outputs'
metadata = {"nodes": nodes}
save_solutions_to_hdf5(str(file_path), filename, mesh, solutions, metadata)
