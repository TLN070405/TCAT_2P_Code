"""
Countercurrent imbibition with the TCAT2Phase 2P-TM solver: Example 1 of
Szymkiewicz (2007), "Numerical simulation of one-dimensional two-phase flow in
porous media", Arch. Hydro-Eng. Environ. Mech. 54(2), 117-136 (Table 1), which
has the semi-analytic solution of McWhorter & Sunada (1990).

Horizontal air-water column, L = 0.8 m, incompressible fluids, Brooks-Corey
closures. x = 0: Se fixed at 0.9 (Dirichlet p_w, p_n). x = L: no flow of either
phase (flux boundary types 1 and 2 with value 0), so q_t = 0. Initially Se = 0.01.
"""
import numpy as np
from petsc4py import PETSc

import TCAT2Phase
from TCAT2Phase.state_equations import state_equations
from TCAT2Phase.state_equations.state_equations_interface import BrooksCorey

RHO_W, RHO_N, G = 1000.0, 1.204, 9.81
PHI, MU_W, MU_N = 0.3, 1.0e-3, 1.57e-5
K = 9.81e-4 * MU_W / (RHO_W * G)          # from K_s = 9.81e-4 m/s: 1.0e-10 m^2
PD, LAMBDA, S_WR = RHO_W * G * 0.102, 2.0, 0.0
S_I, S_0 = 0.01, 0.9                      # effective = actual saturation (S_wr = 0)
LENGTH, T_FINAL = 0.8, 3000.0

# Szymkiewicz (2007) Table 2: error in cumulative infiltrated water [%] vs the
# semi-analytic solution, arithmetic interblock conductivity, omega = 1 (implicit)
SZYMKIEWICZ_TM_ERR = {10: 10.00, 20: 4.42, 40: 1.88, 80: 0.76}


def closures():
    return BrooksCorey(pd=PD, lamda=LAMBDA, porosity=PHI, e_wr=S_WR * PHI, e_nr=0.0)


def diffusivity(S, bc=None):
    """McWhorter-Sunada capillary diffusivity D(S) for these closures."""
    bc = bc or closures()
    e_w = PHI * S
    p_c = bc.capillary_pressure(e_w)
    dpc_ds = PHI * bc.d_capillary_pressure(e_w)
    k_rw, k_rn = bc.w_relative_permeability(p_c), bc.n_relative_permeability(p_c)
    f = 1.0 / (1.0 + k_rn * MU_W / (k_rw * MU_N))
    return -(K * k_rn * f / MU_N) * dpc_ds


def run(nodes, t_end=T_FINAL, bdf_order=2, form="pp", monitor=False, air_compressibility=0.0, options=None,
        no_flow="conservative", mobility="arithmetic"):
    """Solve with the 2P-TM pp or sp formulation; return (x, S_w) at t_end.

    options:  extra PETSc options {name: value}, applied after the defaults below.
    no_flow:  how the x = L no-flow condition is imposed: "conservative" (boundary
              types 4/5: half-cell mass balance with zero outer flux) or
              "algebraic" (types 1/2: zero flux on the face next to the boundary).
    mobility: face mobility weighting, "arithmetic" or "upwind".
    """
    domain = TCAT2Phase.properties.domain.initialize_domain(dims=1, domain_size=np.array([[0.0, LENGTH]]))
    mesh = TCAT2Phase.mesh.grid.Mesh()
    mesh.uniform_mesh_1d(domain, nodes)
    model = TCAT2Phase.numerics.model.Model(model_type="two_phase", form=form, domain=domain, nodes=nodes, mesh=mesh)
    first = "p_w" if form == "pp" else "e_w"
    second = "p_n" if form == "pp" else "p_w"
    # x = L: no flow of water (type 1 or 4) or of the non-wetting phase (type 2 or 5)
    tw, tn = (4, 5) if no_flow == "conservative" else (1, 2)
    model.set_boundary_type({first: [[0, tw]], second: [[0, tn]]})
    model.mobility_weighting = mobility

    solid = TCAT2Phase.properties.phase.Solid("s", porosity=PHI, permeability=K, gravity=0.0,
                                               dynamic_capillary_pressure_parameter=None)
    # incompressible fluids, as in the exact solution
    w = TCAT2Phase.properties.phase.Fluid("w", viscosity=MU_W, momentum_resistance=0.0,
                                         residual_volume_fraction=S_WR * PHI,
                                         density_function=state_equations.density_linear_pressure,
                                         d_density_function=state_equations.d_density_linear_pressure,
                                         rho0=RHO_W, alpha=0.0)
    n = TCAT2Phase.properties.phase.Fluid("n", viscosity=MU_N, momentum_resistance=0.0,
                                         residual_volume_fraction=0.0,
                                         density_function=state_equations.density_linear_pressure,
                                         d_density_function=state_equations.d_density_linear_pressure,
                                         rho0=RHO_N, alpha=air_compressibility)
    bc = closures()
    w.set_relative_permeability_function(bc.w_relative_permeability)
    n.set_relative_permeability_function(bc.n_relative_permeability)
    model.set_model_phases(solid, w, n, TCAT2Phase.properties.interface.TraditionalInterface(bc))

    pc_i, pc_0 = bc.capillary_pressure(PHI * S_I), bc.capillary_pressure(PHI * S_0)
    if form == "pp":
        model.set_initial_conditions({"p_w": lambda x: -pc_i, "p_n": lambda x: 0.0})
        model.set_boundary_conditions({"p_w": [[-pc_0, 0.0]], "p_n": [[0.0, 0.0]]})
    else:  # sp: unknowns e_w, p_w; p_n = p_w + p_c(e_w) = 0 at x = 0 and initially
        model.set_initial_conditions({"e_w": lambda x: PHI * S_I, "p_w": lambda x: -pc_i})
        model.set_boundary_conditions({"e_w": [[PHI * S_0, 0.0]], "p_w": [[-pc_0, 0.0]]})
    model.set_simulation_time(np.array([t_end]))

    opts = PETSc.Options()
    for key, val in [("snes_fd_color", None), ("ts_type", "bdf"), ("ts_bdf_order", bdf_order),
                     ("ts_max_steps", 100000), ("ts_max_reject", 500), ("ts_max_snes_failures", 10000),
                     ("ts_rtol", 1.0e-6), ("ts_atol", 1.0e-6), ("pc_type", "lu")]:
        opts.setValue(key, val)
    for key, val in (options or {}).items():
        opts.setValue(key, val)
    if monitor:
        opts.setValue("ts_monitor", None)
    else:
        opts.delValue("ts_monitor")

    u, ts = TCAT2Phase.numerics.solver.setup_solver(model)
    sol = TCAT2Phase.numerics.solver.solver_time(u, ts, model)[t_end]
    if form == "pp":
        S = np.array([bc.volume_fraction_w(pc) for pc in sol["p_n"] - sol["p_w"]]) / PHI
    else:
        S = sol["e_w"] / PHI
    return mesh.points[:, 0], S


def infiltrated_volume(x, S):
    """Water volume per unit area that has entered: int_0^L phi (S - S_i) dx (trapezoid)."""
    return np.trapz(PHI * (S - S_I), x)
