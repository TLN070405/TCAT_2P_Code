"""
Test Case 5, drainage scenario from the May 2025 defense (slides 9-10) and
dissertation Fig. 4.20: TCAT constant gas-phase pressure (CGP) model.

Horizontal 1 m column at e_w = 0.3372 (p_w ~ -413 Pa, e_wn = 6980 1/m). At
t = 0 both ends are set to e_w = 0.2402 (p_w ~ -433 Pa, e_wn = 7556 1/m) and
the column drains for 8000 s.

Reproduces the standalone code OtherApps/testcases5_6_04MAY25/
Testcase5_cgp_RE_comparison/TCAT_CGP_Cell_Centered.pyx with the TCAT2Phase package.
"""
import numpy as np
import TCAT2Phase
from TCAT2Phase.state_equations import state_equations
from TCAT2Phase.state_equations import state_equations_interface
from TCAT2Phase.numerics import solver_options
from pathlib import Path
from TCAT2Phase.io.hdf5_io import save_solutions_to_hdf5

# Node whose time history is recorded for the normalized plot (x = 0.5 m)
MONITOR_NODE = 100

domain_size = np.array([[0.0, 1.0]])
domain = TCAT2Phase.properties.domain.initialize_domain(dims=1, domain_size=domain_size)
nodes = 201
mesh = TCAT2Phase.mesh.grid.Mesh()
mesh.uniform_mesh_1d(domain, nodes)

model = TCAT2Phase.numerics.model.Model(
    model_type="cgp-TCAT", form="spijvk", domain=domain, nodes=nodes, mesh=mesh)

# 0 = Dirichlet, 3 = zero gradient
model.set_boundary_type({"e_w": [[0, 0]], "p_w": [[0, 0]], "e_wn": [[0, 0]],
                         "j_wn": [[0, 0]], "w_wn": [[3, 3]], "k_wn": [[0, 0]]})

s_phase = TCAT2Phase.properties.phase.Solid(
    "s", porosity=0.369, permeability=None, gravity=0.0,
    dynamic_capillary_pressure_parameter=None)

w_phase = TCAT2Phase.properties.phase.Fluid(
    "w", viscosity=None, momentum_resistance=3.75e5,
    residual_volume_fraction=0.0687 * 0.369,
    density_function=state_equations.density_exp_pressure,
    d_density_function=state_equations.d_density_exp_pressure,
    rho0=1.0e3, beta=4.5e-10)

tcat = state_equations_interface.TCAT(
    b1=1.47e-1, b2=-2.08e-3, b3=5.33e-2 / (4.0 * np.pi),
    porosity=s_phase.porosity, e_ss=1.89e3, js_ns=2.00e3, kn_ns=1.00e6)

EWN_BOUNDARY, EWN_INITIAL = 7556.0, 6980.0

wn_interface = TCAT2Phase.properties.interface.TCATInterface(
    tcat,
    interfacial_dynamics=6.398675e-05,
    interfacial_tension=7.20e-2,
    interfacial_area_relaxation=1.003597e-10,
    interfacial_area_equilibrium=lambda x: EWN_BOUNDARY,
    # Legacy -2K sign of the Gaussian-curvature term, as in the code behind the
    # defense results; the library default is the theory value +1 (README, section 8).
    gaussian_curvature_sign=-1)

model.set_model_phases(s_phase, w_phase, None, wn_interface)

# Primary drainage curve, used only to turn e_w into p_w for the IC/BCs
vg = state_equations_interface.VanGenuchten(
    m=1.0 - 1.0 / 37.8, n=37.8, alpha=2.28e-3, porosity=s_phase.porosity,
    e_wr=w_phase.residual_volume_fraction, e_nr=0.0)

gamma = wn_interface.interfacial_tension
EW_INITIAL, EW_BOUNDARY = 0.33723630773638963, 0.2401501293333221
pw_initial = -vg.capillary_pressure(EW_INITIAL)
pw_boundary = -vg.capillary_pressure(EW_BOUNDARY)

# The defense code initialized w_wn from the velocity equation; with
# j_wn = -p_w/gamma that expression is ~1e-19, so w_wn starts at 0.
model.set_initial_conditions({
    "e_w": lambda x: EW_INITIAL,
    "p_w": lambda x: pw_initial,
    "e_wn": lambda x: EWN_INITIAL,
    "j_wn": lambda x: -pw_initial / gamma,
    "w_wn": lambda x: 0.0,
    "k_wn": lambda x: tcat.gaussian_curvature(EW_INITIAL, EWN_INITIAL, -pw_initial / gamma),
})

k_boundary = tcat.gaussian_curvature(EW_BOUNDARY, EWN_BOUNDARY, -pw_boundary / gamma)
both = lambda value: [[value, value]]
model.set_boundary_conditions({
    "e_w": both(EW_BOUNDARY),
    "p_w": both(pw_boundary),
    "e_wn": both(EWN_BOUNDARY),
    "j_wn": both(-pw_boundary / gamma),
    "w_wn": both(0.0),
    "k_wn": both(k_boundary),
})

model.set_simulation_time(np.array([1e-5, 1, 10, 100, 500, 1000, 3000, 5000, 8000]))

solver_options.solver_options(max_time_steps=100000, bdf_order=5, pc_type="sor", rtol=1.e-8, atol=1.e-8)  # as for the slides
u, ts = TCAT2Phase.numerics.solver.setup_solver(model)

names = model.model.u_names
history = {"time": []} | {name: [] for name in names}
def record(ts):
    values = ts.getSolution().getArray(readonly=True)
    history["time"].append(ts.getTime())
    for k, name in enumerate(names):
        history[name].append(values[MONITOR_NODE * len(names) + k])
ts.setPostStep(record)  # read-only access: a writable getArray() here resets BDF history

solutions = TCAT2Phase.numerics.solver.solver_time(u, ts, model)

filename = "tcat_cgp_test_case_5_defense.h5"
file_path = Path(__file__).resolve().parent.parent.parent / "outputs" / "Testcase5_primary_drainage" / "outputs"
metadata = {"nodes": nodes, "monitor_node": MONITOR_NODE}
metadata.update({f"history_{k}": np.array(v) for k, v in history.items()})
save_solutions_to_hdf5(str(file_path), filename, mesh, solutions, metadata)
