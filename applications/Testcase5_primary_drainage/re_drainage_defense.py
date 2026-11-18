"""
Test Case 5, drainage scenario from the May 2025 defense (slides 9-10) and
dissertation Fig. 4.20: Richards' equation.

Horizontal 1 m column at e_w = 0.3372 (p_w ~ -413 Pa). At t = 0 both ends are
set to e_w = 0.2402 (p_w ~ -433 Pa) and the column drains for 8000 s.

Reproduces the standalone code OtherApps/testcases5_6_04MAY25/
Testcase5_cgp_RE_comparison/RE_PR_Pressure_TCAT.pyx with the TCAT2Phase package.
"""
import numpy as np
import TCAT2Phase
from TCAT2Phase.state_equations import state_equations
from TCAT2Phase.state_equations import state_equations_interface
import solver_options
from pathlib import Path
from TCAT2Phase.io.hdf5_io import save_solutions_to_hdf5

# The defense code evaluated k_rw(p_c) at the volume fraction instead of the
# capillary pressure, i.e. k_rw(e_w(p_c)). Because e_w << 1/alpha, that makes
# k_rw = 1 in practice. Keep True to reproduce the slides; set False to use the
# Van Genuchten k_rw(p_c) instead.
REPRODUCE_DEFENSE_KRW = True

# Node whose time history is recorded for the normalized plot (x = 0.5 m)
MONITOR_NODE = 100

# Domain and mesh
domain_size = np.array([[0.0, 1.0]])
domain = TCAT2Phase.properties.domain.initialize_domain(dims=1, domain_size=domain_size)
nodes = 201
mesh = TCAT2Phase.mesh.grid.Mesh()
mesh.uniform_mesh_1d(domain, nodes)

model = TCAT2Phase.numerics.model.Model(
    model_type="Richards", form="pr", domain=domain, nodes=nodes, mesh=mesh)
model.set_boundary_type({"p_w": [[0, 0]]})

# Solid phase: k/mu_w = 1/R_ww with R_ww = 3.75e5 N s/m^4
s_phase = TCAT2Phase.properties.phase.Solid(
    "s", porosity=0.369, permeability=1.0 / 3.75e5, gravity=0.0,
    dynamic_capillary_pressure_parameter=None)

w_phase = TCAT2Phase.properties.phase.Fluid(
    "w", viscosity=1.0, momentum_resistance=0.0,
    residual_volume_fraction=0.0687 * 0.369,
    density_function=state_equations.density_exp_pressure,
    d_density_function=state_equations.d_density_exp_pressure,
    rho0=1.0e3, beta=4.5e-10)

n_phase = TCAT2Phase.properties.phase.Fluid(
    "n", viscosity=1.78e-5, momentum_resistance=0.0, residual_volume_fraction=0.0,
    density_function=state_equations.density_linear_pressure,
    d_density_function=state_equations.d_density_linear_pressure,
    rho0=1.2, alpha=1.189e-5)

# Primary drainage curve
vg = state_equations_interface.VanGenuchten(
    m=1.0 - 1.0 / 37.8, n=37.8, alpha=2.28e-3, porosity=s_phase.porosity,
    e_wr=w_phase.residual_volume_fraction, e_nr=n_phase.residual_volume_fraction)
wn_interface = TCAT2Phase.properties.interface.TraditionalInterface(vg)

if REPRODUCE_DEFENSE_KRW:
    w_phase.set_relative_permeability_function(lambda p_c: vg.w_relative_permeability(vg.volume_fraction_w(p_c)))
else:
    w_phase.set_relative_permeability_function(vg.w_relative_permeability)
n_phase.set_relative_permeability_function(vg.n_relative_permeability)

model.set_model_phases(s_phase, w_phase, n_phase, wn_interface)

# Initial and boundary conditions
EW_INITIAL, EW_BOUNDARY = 0.33723630773638963, 0.2401501293333221
pw_initial = -vg.capillary_pressure(EW_INITIAL)
pw_boundary = -vg.capillary_pressure(EW_BOUNDARY)

model.set_initial_conditions({"p_w": lambda x: pw_initial})
model.set_boundary_conditions({"p_w": [[pw_boundary, pw_boundary]]})

# Output times
model.set_simulation_time(np.array([1e-5, 1, 10, 100, 500, 1000, 3000, 5000, 8000]))

solver_options.solver_options(max_time_steps=100000, bdf_order=5, pc_type="sor")
u, ts = TCAT2Phase.numerics.solver.setup_solver(model)

# Record the monitor node after every time step, for the normalized plot (slide 10)
history = {"time": [], "p_w": []}
def record(ts):
    history["time"].append(ts.getTime())
    history["p_w"].append(ts.getSolution().getArray(readonly=True)[MONITOR_NODE])
ts.setPostStep(record)  # read-only access: a writable getArray() here resets BDF history

solutions = TCAT2Phase.numerics.solver.solver_time(u, ts, model)

for variables in solutions.values():
    variables["e_w"] = np.array([vg.volume_fraction_w(-p) for p in variables["p_w"]])

history["e_w"] = [vg.volume_fraction_w(-p) for p in history["p_w"]]

filename = "RE_test_case_5_defense.h5"
file_path = Path(__file__).resolve().parent.parent.parent / "outputs" / "Testcase5_primary_drainage" / "outputs"
metadata = {"nodes": nodes, "monitor_node": MONITOR_NODE, "reproduce_defense_krw": REPRODUCE_DEFENSE_KRW}
metadata.update({f"history_{k}": np.array(v) for k, v in history.items()})
save_solutions_to_hdf5(str(file_path), filename, mesh, solutions, metadata)
