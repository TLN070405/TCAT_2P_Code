"""
Method of manufactured solutions (MMS) for Richards' equation (model_type="Richards",
form="pr") and the two-phase traditional model (model_type="two_phase", form="sp").

Pick smooth exact fields, compute the source terms that make them satisfy the continuum
equations the residuals discretize,

    d(rho_a e_a)/dt = dq_a/dx + S_a,   q_a = rho_a k k_ra(p_c) / mu_a (dp_a/dx + rho_a g),

(RE: water only, p_c = -p_w; 2P sp: e_w and p_w given, p_n = p_w + p_c(e_w),
e_n = porosity - e_w), and solve with those sources and the exact boundary and initial
values (model.source). The error then comes from the discretization only, so it must fall
at the scheme's order as the grid is refined: 2 in space for this finite-volume scheme on
a uniform grid with arithmetic face mobility.

The sources are evaluated from the code's own closure functions (densities, van Genuchten
retention and relative permeabilities) by central differences in t and x; their error is
far below the discretization error on the grids used. Both fluids are compressible, and
gravity is on, so every term of the residual is exercised.
"""
import numpy as np
from petsc4py import PETSc

import TCAT2Phase
from TCAT2Phase.state_equations import state_equations
from TCAT2Phase.state_equations.state_equations_interface import VanGenuchten

LENGTH = 1.0
POROSITY, E_WR = 0.4, 0.04
PERMEABILITY, MU_W, MU_N, GRAVITY = 1.0e-11, 1.0e-3, 1.78e-5, 9.81
RHO_W0, BETA_W = 1.0e3, 4.5e-10
RHO_N0, ALPHA_N = 1.2, 1.189e-5
TAU, T_END = 200.0, 400.0


def closures():
    return VanGenuchten(m=0.5, n=2.0, alpha=1.0 / 1500.0, porosity=POROSITY, e_wr=E_WR, e_nr=0.0)


def rho_w(p):
    return state_equations.density_exp_pressure(p, rho0=RHO_W0, beta=BETA_W)


def rho_n(p):
    return state_equations.density_linear_pressure(p, rho0=RHO_N0, alpha=ALPHA_N)


def _ramp(t):
    return 1.0 - np.exp(-t / TAU)


# --- Exact solutions ----------------------------------------------------------

def exact_p_w_richards(x, t):
    """Unsaturated everywhere (p_w between -2300 and -1200 Pa)."""
    return -2000.0 - 300.0 * x / LENGTH + 800.0 * np.sin(np.pi * x / LENGTH) * _ramp(t)


def exact_e_w_two_phase(x, t):
    """Between 0.15 and 0.30; changes in time at both boundaries."""
    return 0.2 + 0.05 * np.cos(np.pi * x / LENGTH) * (1.0 + 0.5 * _ramp(t)) + 0.04 * np.sin(np.pi * x / LENGTH) * _ramp(t)


def exact_p_w_two_phase(x, t):
    return -3000.0 - 400.0 * x / LENGTH + 500.0 * np.sin(np.pi * x / LENGTH) * _ramp(t)


# --- Source terms -------------------------------------------------------------

def _d_dt(f, x, t, h=1e-3):
    return (f(x, t + h) - f(x, t - h)) / (2 * h)


def _d_dx(f, x, t, h=1e-4):
    return (f(x + h, t) - f(x - h, t)) / (2 * h)


def _darcy(rho, k_r, mu, p, dpdx):
    return rho * PERMEABILITY * k_r / mu * (dpdx + rho * GRAVITY)


def richards_source(vg):
    def mass(x, t):
        p = exact_p_w_richards(x, t)
        return rho_w(p) * vg.volume_fraction_w(-p)

    def flux(x, t):
        p = exact_p_w_richards(x, t)
        return _darcy(rho_w(p), vg.w_relative_permeability(-p), MU_W, p, _d_dx(exact_p_w_richards, x, t, 1e-6))

    return lambda x, t: _d_dt(mass, x, t) - _d_dx(flux, x, t)


def two_phase_source(vg):
    p_c = lambda x, t: vg.capillary_pressure(exact_e_w_two_phase(x, t))
    p_n = lambda x, t: exact_p_w_two_phase(x, t) + p_c(x, t)

    def mass_w(x, t):
        return rho_w(exact_p_w_two_phase(x, t)) * exact_e_w_two_phase(x, t)

    def mass_n(x, t):
        return rho_n(p_n(x, t)) * (POROSITY - exact_e_w_two_phase(x, t))

    def flux_w(x, t):
        p = exact_p_w_two_phase(x, t)
        return _darcy(rho_w(p), vg.w_relative_permeability(p_c(x, t)), MU_W, p, _d_dx(exact_p_w_two_phase, x, t, 1e-6))

    def flux_n(x, t):
        p = p_n(x, t)
        return _darcy(rho_n(p), vg.n_relative_permeability(p_c(x, t)), MU_N, p, _d_dx(p_n, x, t, 1e-6))

    return lambda x, t: np.array([_d_dt(mass_w, x, t) - _d_dx(flux_w, x, t),
                                  _d_dt(mass_n, x, t) - _d_dx(flux_n, x, t)])


# --- Solver -------------------------------------------------------------------

def _model(model_type, form, nodes):
    domain = TCAT2Phase.properties.domain.initialize_domain(dims=1, domain_size=np.array([[0.0, LENGTH]]))
    mesh = TCAT2Phase.mesh.grid.Mesh()
    mesh.uniform_mesh_1d(domain, nodes)
    model = TCAT2Phase.numerics.model.Model(model_type=model_type, form=form, domain=domain, nodes=nodes, mesh=mesh)
    solid = TCAT2Phase.properties.phase.Solid("s", porosity=POROSITY, permeability=PERMEABILITY, gravity=GRAVITY,
                                               dynamic_capillary_pressure_parameter=None)
    w = TCAT2Phase.properties.phase.Fluid("w", viscosity=MU_W, momentum_resistance=0.0, residual_volume_fraction=E_WR,
                                         density_function=state_equations.density_exp_pressure,
                                         d_density_function=state_equations.d_density_exp_pressure,
                                         rho0=RHO_W0, beta=BETA_W)
    n = TCAT2Phase.properties.phase.Fluid("n", viscosity=MU_N, momentum_resistance=0.0, residual_volume_fraction=0.0,
                                         density_function=state_equations.density_linear_pressure,
                                         d_density_function=state_equations.d_density_linear_pressure,
                                         rho0=RHO_N0, alpha=ALPHA_N)
    vg = closures()
    w.set_relative_permeability_function(vg.w_relative_permeability)
    n.set_relative_permeability_function(vg.n_relative_permeability)
    model.set_model_phases(solid, w, n, TCAT2Phase.properties.interface.TraditionalInterface(vg))
    return model, mesh, vg


def _solve(model, mesh, t_end, tol):
    model.set_simulation_time(np.array([t_end]))
    opts = PETSc.Options()
    for key, val in [("snes_fd_color", None), ("ts_type", "bdf"), ("ts_bdf_order", 2),
                     ("ts_max_steps", 1000000), ("ts_max_reject", 500), ("ts_max_snes_failures", 10000),
                     ("ts_rtol", tol), ("ts_atol", tol), ("pc_type", "lu")]:
        opts.setValue(key, val)
    opts.delValue("ts_monitor")
    u, ts = TCAT2Phase.numerics.solver.setup_solver(model)
    solutions = TCAT2Phase.numerics.solver.solver_time(u, ts, model)
    return mesh.points[:, 0], solutions[max(solutions)]


def richards_errors(nodes, t_end=T_END, tol=1e-9):
    """Max error in p_w [Pa] at t_end."""
    model, mesh, vg = _model("Richards", "pr", nodes)
    model.set_boundary_type({"p_w": [[0, 0]]})
    model.source = richards_source(vg)
    model.set_initial_conditions({"p_w": lambda x: exact_p_w_richards(x, 0.0)})
    model.set_boundary_conditions({"p_w": [[lambda t: exact_p_w_richards(0.0, t),
                                            lambda t: exact_p_w_richards(LENGTH, t)]]})
    x, final = _solve(model, mesh, t_end, tol)
    return {"p_w": float(np.max(np.abs(np.asarray(final["p_w"]) - exact_p_w_richards(x, t_end))))}


def two_phase_errors(nodes, t_end=T_END, tol=1e-9):
    """Max errors in e_w [-] and p_w [Pa] at t_end."""
    model, mesh, vg = _model("two_phase", "sp", nodes)
    model.set_boundary_type({"e_w": [[0, 0]], "p_w": [[0, 0]]})
    model.source = two_phase_source(vg)
    model.set_initial_conditions({"e_w": lambda x: exact_e_w_two_phase(x, 0.0),
                                  "p_w": lambda x: exact_p_w_two_phase(x, 0.0)})
    model.set_boundary_conditions({
        "e_w": [[lambda t: exact_e_w_two_phase(0.0, t), lambda t: exact_e_w_two_phase(LENGTH, t)]],
        "p_w": [[lambda t: exact_p_w_two_phase(0.0, t), lambda t: exact_p_w_two_phase(LENGTH, t)]]})
    x, final = _solve(model, mesh, t_end, tol)
    return {"e_w": float(np.max(np.abs(np.asarray(final["e_w"]) - exact_e_w_two_phase(x, t_end)))),
            "p_w": float(np.max(np.abs(np.asarray(final["p_w"]) - exact_p_w_two_phase(x, t_end))))}
