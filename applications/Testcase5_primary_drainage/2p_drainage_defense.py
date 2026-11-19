"""
Test Case 5, drainage scenario from the May 2025 defense (slides 16-17):
traditional two-phase model (2P-TM), saturation-pressure (SP) formulation.

Horizontal 1 m column at e_w = 0.3372 (p_w ~ -413 Pa). At t = 0 both ends are
set to e_w = 0.2402 (p_w ~ -433 Pa) and the column drains for 8000 s.

Reproduces the standalone code OtherApps/Testcase5_tcat2p/Two_phases_SP_TCAT.pyx
with the TCAT2Phase package.
"""
import numpy as np
import TCAT2Phase
from TCAT2Phase.state_equations import state_equations
from TCAT2Phase.state_equations import state_equations_interface
from TCAT2Phase.numerics import solver_options
from pathlib import Path
from TCAT2Phase.io.hdf5_io import save_solutions_to_hdf5

# The defense code recorded its time history at array index 100 of the
# interleaved [e_w, p_w, ...] solution, which is node 50 (x = 0.25 m), not the
# midpoint. Keep 50 to reproduce slide 17; use 100 for the midpoint.
MONITOR_NODE = 50

domain_size = np.array([[0.0, 1.0]])
domain = TCAT2Phase.properties.domain.initialize_domain(dims=1, domain_size=domain_size)
nodes = 201
mesh = TCAT2Phase.mesh.grid.Mesh()
mesh.uniform_mesh_1d(domain, nodes)

model = TCAT2Phase.numerics.model.Model(
    model_type="two_phase", form="sp", domain=domain, nodes=nodes, mesh=mesh)
model.set_boundary_type({"e_w": [[0, 0]], "p_w": [[0, 0]]})

# permeability = 1 and viscosity = R_aa gives k/mu_a = 1/R_aa, as in the TCAT models
s_phase = TCAT2Phase.properties.phase.Solid(
    "s", porosity=0.369, permeability=1.0, gravity=0.0,
    dynamic_capillary_pressure_parameter=None)

w_phase = TCAT2Phase.properties.phase.Fluid(
    "w", viscosity=3.75e5, momentum_resistance=0.0,
    residual_volume_fraction=0.0687 * 0.369,
    density_function=state_equations.density_exp_pressure,
    d_density_function=state_equations.d_density_exp_pressure,
    rho0=1.0e3, beta=4.5e-10)

n_phase = TCAT2Phase.properties.phase.Fluid(
    "n", viscosity=9.07e2, momentum_resistance=0.0, residual_volume_fraction=0.0,
    density_function=state_equations.density_linear_pressure,
    d_density_function=state_equations.d_density_linear_pressure,
    rho0=1.2, alpha=1.189e-5)

vg = state_equations_interface.VanGenuchten(
    m=1.0 - 1.0 / 37.8, n=37.8, alpha=2.28e-3, porosity=s_phase.porosity,
    e_wr=w_phase.residual_volume_fraction, e_nr=n_phase.residual_volume_fraction)
wn_interface = TCAT2Phase.properties.interface.TraditionalInterface(vg)
w_phase.set_relative_permeability_function(vg.w_relative_permeability)
n_phase.set_relative_permeability_function(vg.n_relative_permeability)

model.set_model_phases(s_phase, w_phase, n_phase, wn_interface)

EW_INITIAL, EW_BOUNDARY = 0.33723630773638963, 0.2401501293333221
pw_initial = -vg.capillary_pressure(EW_INITIAL)
pw_boundary = -vg.capillary_pressure(EW_BOUNDARY)

model.set_initial_conditions({"e_w": lambda x: EW_INITIAL, "p_w": lambda x: pw_initial})
model.set_boundary_conditions({"e_w": [[EW_BOUNDARY, EW_BOUNDARY]],
                               "p_w": [[pw_boundary, pw_boundary]]})

model.set_simulation_time(np.array([1e-5, 1, 10, 100, 500, 1000, 3000, 5000, 8000]))

solver_options.solver_options(max_time_steps=100000, bdf_order=5, pc_type="sor", rtol=1.e-8, atol=1.e-8)  # as for the slides
u, ts = TCAT2Phase.numerics.solver.setup_solver(model)

history = {"time": [], "e_w": [], "p_w": []}
def record(ts):
    values = ts.getSolution().getArray(readonly=True)
    history["time"].append(ts.getTime())
    history["e_w"].append(values[2 * MONITOR_NODE])
    history["p_w"].append(values[2 * MONITOR_NODE + 1])
ts.setPostStep(record)  # read-only access: a writable getArray() here resets BDF history

solutions = TCAT2Phase.numerics.solver.solver_time(u, ts, model)

for variables in solutions.values():
    variables["p_n"] = np.array([vg.capillary_pressure(e) for e in variables["e_w"]]) + variables["p_w"]
history["p_n"] = [vg.capillary_pressure(e) + p for e, p in zip(history["e_w"], history["p_w"])]

filename = "2P_test_case_5_defense.h5"
file_path = Path(__file__).resolve().parent.parent.parent / "outputs" / "Testcase5_primary_drainage" / "outputs"
metadata = {"nodes": nodes, "monitor_node": MONITOR_NODE}
metadata.update({f"history_{k}": np.array(v) for k, v in history.items()})
save_solutions_to_hdf5(str(file_path), filename, mesh, solutions, metadata)
