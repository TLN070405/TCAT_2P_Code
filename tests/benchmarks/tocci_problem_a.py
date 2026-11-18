"""
Tocci, Kelley & Miller (1997), "Accurate and economical solution of the pressure-
head form of Richards' equation by the method of lines", Adv. Water Resour.
20(1), 1-14, Problem A (Table 1); the soil is the Van Genuchten example of
Celia, Bouloutas & Zarba (1990), Water Resour. Res. 26(7), 1483-1496, Eq. (13).

Vertical column z in [0, 0.3] m (z up), psi(z, 0) = -10 m, psi(0, t) = -10 m,
psi(0.3, t) = -0.75 m; theta_r = 0.102, theta_s = 0.368, alpha = 3.35 1/m,
n = 2, K_s = 7.97 m/day, S_s = 0.
"""
import numpy as np
from petsc4py import PETSc

import TCAT2Phase
from TCAT2Phase.state_equations import state_equations
from TCAT2Phase.state_equations.state_equations_interface import VanGenuchten

RHO, G, MU = 1000.0, 9.81, 1.0e-3
THETA_R, THETA_S, ALPHA_M, N_VG = 0.102, 0.368, 3.35, 2.0
K_S = 7.97 / 86400.0                     # m/s
PERMEABILITY = K_S * MU / (RHO * G)      # 9.40e-12 m^2
LENGTH = 0.3
PSI_BOTTOM, PSI_TOP, PSI_INIT = -10.0, -0.75, -10.0


def closures():
    return VanGenuchten(m=1.0 - 1.0 / N_VG, n=N_VG, alpha=ALPHA_M / (RHO * G),
                        porosity=THETA_S, e_wr=THETA_R, e_nr=0.0)


def run(nodes, times):
    """Solve RE; return (z, solutions, model) with solutions[t]['p_w'] in Pa."""
    domain = TCAT2Phase.properties.domain.initialize_domain(dims=1, domain_size=np.array([[0.0, LENGTH]]))
    mesh = TCAT2Phase.mesh.grid.Mesh()
    mesh.uniform_mesh_1d(domain, nodes)
    model = TCAT2Phase.numerics.model.Model(model_type="Richards", form="pr", domain=domain, nodes=nodes, mesh=mesh)
    model.set_boundary_type({"p_w": [[0, 0]]})
    solid = TCAT2Phase.properties.phase.Solid("s", porosity=THETA_S, permeability=PERMEABILITY, gravity=G,
                                               dynamic_capillary_pressure_parameter=None)
    # S_s = 0: incompressible water
    w = TCAT2Phase.properties.phase.Fluid("w", viscosity=MU, momentum_resistance=0.0,
                                         residual_volume_fraction=THETA_R,
                                         density_function=state_equations.density_linear_pressure,
                                         d_density_function=state_equations.d_density_linear_pressure,
                                         rho0=RHO, alpha=0.0)
    n = TCAT2Phase.properties.phase.Fluid("n", viscosity=1.78e-5, momentum_resistance=0.0,
                                         residual_volume_fraction=0.0,
                                         density_function=state_equations.density_linear_pressure,
                                         d_density_function=state_equations.d_density_linear_pressure,
                                         rho0=1.2, alpha=0.0)
    vg = closures()
    w.set_relative_permeability_function(vg.w_relative_permeability)
    n.set_relative_permeability_function(vg.n_relative_permeability)
    model.set_model_phases(solid, w, n, TCAT2Phase.properties.interface.TraditionalInterface(vg))

    model.set_initial_conditions({"p_w": lambda z: RHO * G * PSI_INIT})
    model.set_boundary_conditions({"p_w": [[RHO * G * PSI_BOTTOM, RHO * G * PSI_TOP]]})
    model.set_simulation_time(np.asarray(times, dtype=float))

    opts = PETSc.Options()
    for key, val in [("snes_fd_color", None), ("ts_type", "bdf"), ("ts_bdf_order", 2),
                     ("ts_max_steps", 1000000), ("ts_max_reject", 500), ("ts_max_snes_failures", 10000),
                     ("ts_rtol", 1.0e-6), ("ts_atol", 1.0e-6), ("pc_type", "lu")]:
        opts.setValue(key, val)
    opts.delValue("ts_monitor")

    u, ts = TCAT2Phase.numerics.solver.setup_solver(model)
    solutions = TCAT2Phase.numerics.solver.solver_time(u, ts, model)
    return mesh.points[:, 0], solutions, model


def mass_balance(solutions, model):
    """
    Celia et al. (1990) Eq. (11): MB(t) = (mass gained in the interior cells) /
    (time-integrated net boundary inflow), using the solver's own face fluxes.
    The flux integral uses the trapezoid rule over the output times.
    """
    residual = TCAT2Phase.numerics.residual_re_pr.ResidualRePR(model)
    times = sorted(solutions)
    dx = LENGTH / (model.nodes - 1)
    mass, inflow = [], []
    for t in times:
        pv = residual.compute_primary_variables(solutions[t]["p_w"])
        q = residual.compute_phases_fluxes(pv)["q_w"]
        rho = np.array([model.w_phase.density_function(p) for p in pv["p_w"]])
        mass.append(np.sum(rho[1:-1] * pv["e_w"][1:-1]) * dx)
        # residual convention: d(rho e_w)/dt = (q[n+1] - q[n]) / dx, so interior gain rate = q[N-1] - q[1]
        inflow.append(q[model.nodes - 1] - q[1])
    mass, inflow, t = np.array(mass), np.array(inflow), np.array(times)
    cumulative_inflow = np.concatenate([[0.0], np.cumsum(0.5 * (inflow[1:] + inflow[:-1]) * np.diff(t))])
    return t, (mass - mass[0]), cumulative_inflow


def front_position(z, p_w, psi_level=-5.0):
    """Height z where psi first rises above psi_level, by linear interpolation."""
    psi = p_w / (RHO * G)
    i = np.argmax(psi > psi_level)
    return z[i - 1] + (psi_level - psi[i - 1]) * (z[i] - z[i - 1]) / (psi[i] - psi[i - 1])
