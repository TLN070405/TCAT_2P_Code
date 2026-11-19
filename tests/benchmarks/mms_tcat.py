"""
Method of manufactured solutions for the TCAT CGP model (model_type="cgp-TCAT",
form="spijvk"; residual_tcat_cgp.py). Unknowns e_w, p_w, e_wn, J_wn, w_wn, K_wn.

The residual discretizes (time_mat . udot = od + S), with
D = c_wn (p_w + gamma J) - (gamma k1 / p_w)(e_wn_eq - e_wn):

  0: d(e_w)/dt                                = D
  1: d(rho e_w)/dt                            = dq/dx,  q = rho k_rw(e_w) / R (dp_w/dx + rho g)
  2: J d(e_w)/dt + d(e_wn)/dt                 = -k1 (p_w + gamma J)/p_w (e_wn_eq - e_wn) - d(e_wn w_wn)/dx
  3: 2 s K d(e_w)/dt + J d(e_wn)/dt + e_wn dJ/dt = (2/3) d2D/dx2 - 2 d(e_wn J w_wn)/dx
  4: 0                                        = e_wn^2 w_wn + D d(e_w)/dx
  5: 0                                        = K - K_geom(e_w, e_wn, J)

e_w, p_w, e_wn and J are chosen freely; w_wn and K are then defined by rows 4 and 5, so
their sources are zero. The other sources come from the code's own state functions by
central differences. Rows 2 and 3 use a Lax-Friedrichs flux for the w_wn advection,
which adds numerical diffusion of order |w_wn| dx. Here w_wn is only about 5e-9 m/s, so
that term is negligible and all six unknowns converge at second order (observed
2.00-2.32 on 11-81 nodes); this test does not exercise the Lax-Friedrichs limiter.

The closed CGP equations amplify a mismatch in p_w + gamma J at a rate of about
c_wn gamma (J^2 - 2 s K) / e_wn, where s is the Gaussian-curvature sign (README, Known
issues). The fields are therefore chosen where J^2 - 2 s K < 0 with the legacy sign
s = -1 (p_c 170-250 Pa, e_w 0.29-0.33, e_wn 135-200 1/m), as in the stable Test Case 5/6
runs; with s = +1, J^2 - 2K > 0 for any real surface and the errors grow instead.
"""
import numpy as np
from petsc4py import PETSc

import TCAT2Phase
from TCAT2Phase.state_equations import state_equations, state_equations_interface

LENGTH = 0.1
POROSITY = 0.369
RHO_W0, BETA_W, GRAVITY = 1.0e3, 4.5e-10, 9.81
R_HAT = 5.10e4
C_WN, GAMMA, K1 = 6.398675e-05, 7.20e-2, 1.003597e-10
TAU, T_END = 20.0, 40.0
NAMES = ("e_w", "p_w", "e_wn", "j_wn", "w_wn", "k_wn")


def tcat_state():
    return state_equations_interface.TCAT(b1=1.47e-1, b2=-2.08e-3, b3=5.33e-2 / (4.0 * np.pi), porosity=POROSITY,
                                          e_ss=1.89e3, js_ns=2.00e3, kn_ns=1.00e6)


def _r(t):
    return 1.0 - np.exp(-t / TAU)


def _s(x):
    return np.sin(np.pi * x / LENGTH)


def _c(x):
    return np.cos(np.pi * x / LENGTH)


def e_wn_eq(x):
    return 160.0 + 10.0 * x / LENGTH


# free fields
def e_w(x, t):
    return 0.31 + 0.012 * _c(x) * (1.0 + 0.5 * _r(t))


def p_w(x, t):
    return -200.0 - 50.0 * x / LENGTH + 30.0 * _s(x) * _r(t)


def e_wn(x, t):
    return 160.0 + 25.0 * _c(x) + 15.0 * _s(x) * _r(t)


def j_wn(x, t):
    return -p_w(x, t) / GAMMA * (1.0 + 0.05 * _s(x) * _r(t) + 0.02)


def _dx(f, x, t, h=1e-6):
    return (f(x + h, t) - f(x - h, t)) / (2 * h)


def _dxx(f, x, t, h=1e-4):
    return (f(x + h, t) - 2.0 * f(x, t) + f(x - h, t)) / (h * h)


def _dt(f, x, t, h=1e-4):
    return (f(x, t + h) - f(x, t - h)) / (2 * h)


def D(x, t):
    p = p_w(x, t)
    return C_WN * (p + GAMMA * j_wn(x, t)) - (GAMMA * K1 / p) * (e_wn_eq(x) - e_wn(x, t))


# fields defined by the algebraic rows
def w_wn(x, t):
    return -D(x, t) * _dx(e_w, x, t) / e_wn(x, t) ** 2


def make_k_wn(state):
    return lambda x, t: state.gaussian_curvature(e_w(x, t), e_wn(x, t), j_wn(x, t))


def stability_margin(state, sign=-1.0, nx=41, nt=21, t_end=T_END):
    """max of J^2 - 2 s K over the space-time domain; must be negative for a stable run."""
    k_wn = make_k_wn(state)
    return max(j_wn(x, t) ** 2 - 2.0 * sign * k_wn(x, t)
               for x in np.linspace(0.0, LENGTH, nx) for t in np.linspace(0.0, t_end, nt))


def exact_fields(state):
    return {"e_w": e_w, "p_w": p_w, "e_wn": e_wn, "j_wn": j_wn, "w_wn": w_wn, "k_wn": make_k_wn(state)}


def make_source(state, sign):
    k_wn = make_k_wn(state)
    rho = lambda p: state_equations.density_exp_pressure(p, rho0=RHO_W0, beta=BETA_W)

    def flux(x, t):
        p = p_w(x, t)
        return rho(p) * state.w_relative_permeability(e_w(x, t)) / R_HAT * (_dx(p_w, x, t) + rho(p) * GRAVITY)

    mass = lambda x, t: rho(p_w(x, t)) * e_w(x, t)
    adv_e = lambda x, t: e_wn(x, t) * w_wn(x, t)
    adv_j = lambda x, t: e_wn(x, t) * j_wn(x, t) * w_wn(x, t)

    def source(x, t):
        dew = _dt(e_w, x, t)
        dewn = _dt(e_wn, x, t)
        p, j, a = p_w(x, t), j_wn(x, t), e_wn(x, t)
        s0 = dew - D(x, t)
        s1 = _dt(mass, x, t) - _dx(flux, x, t, 1e-5)
        s2 = j * dew + dewn + K1 * (p + GAMMA * j) / p * (e_wn_eq(x) - a) + _dx(adv_e, x, t, 1e-5)
        s3 = (2.0 * sign * k_wn(x, t) * dew + j * dewn + a * _dt(j_wn, x, t)
              - (2.0 / 3.0) * _dxx(D, x, t) + 2.0 * _dx(adv_j, x, t, 1e-5))
        return np.array([s0, s1, s2, s3, 0.0, 0.0])

    return source


def errors(nodes, t_end=T_END, tol=1e-8, sign=-1.0, monitor=False):
    """Max error of each unknown at t_end, relative to its max magnitude."""
    domain = TCAT2Phase.properties.domain.initialize_domain(dims=1, domain_size=np.array([[0.0, LENGTH]]))
    mesh = TCAT2Phase.mesh.grid.Mesh()
    mesh.uniform_mesh_1d(domain, nodes)
    model = TCAT2Phase.numerics.model.Model(model_type="cgp-TCAT", form="spijvk", domain=domain, nodes=nodes, mesh=mesh)
    model.set_boundary_type({name: [[0, 0]] for name in NAMES})
    solid = TCAT2Phase.properties.phase.Solid("s", porosity=POROSITY, permeability=None, gravity=GRAVITY,
                                               dynamic_capillary_pressure_parameter=None)
    w = TCAT2Phase.properties.phase.Fluid("w", viscosity=None, momentum_resistance=R_HAT,
                                         residual_volume_fraction=0.0687 * POROSITY,
                                         density_function=state_equations.density_exp_pressure,
                                         d_density_function=state_equations.d_density_exp_pressure,
                                         rho0=RHO_W0, beta=BETA_W)
    state = tcat_state()
    interface = TCAT2Phase.properties.interface.TCATInterface(
        state, interfacial_dynamics=C_WN, interfacial_tension=GAMMA, interfacial_area_relaxation=K1,
        interfacial_area_equilibrium=e_wn_eq, gaussian_curvature_sign=sign)
    model.set_model_phases(solid, w, None, interface)

    exact = exact_fields(state)
    model.source = make_source(state, sign)
    model.set_initial_conditions({name: (lambda f: (lambda x: f(x, 0.0)))(exact[name]) for name in NAMES})
    model.set_boundary_conditions({name: [[(lambda f: (lambda t: f(0.0, t)))(exact[name]),
                                           (lambda f: (lambda t: f(LENGTH, t)))(exact[name])]] for name in NAMES})
    model.set_simulation_time(np.array([t_end]))

    opts = PETSc.Options()
    for key, val in [("snes_fd_color", None), ("ts_type", "bdf"), ("ts_bdf_order", 2),
                     ("ts_max_steps", 1000000), ("ts_max_reject", 500), ("ts_max_snes_failures", 10000),
                     ("ts_rtol", tol), ("ts_atol", tol), ("pc_type", "lu")]:
        opts.setValue(key, val)
    if monitor:
        opts.setValue("ts_monitor", None)
        opts.setValue("ts_adapt_monitor", None)
    else:
        opts.delValue("ts_monitor")
    u, ts = TCAT2Phase.numerics.solver.setup_solver(model)
    solutions = TCAT2Phase.numerics.solver.solver_time(u, ts, model)
    final = solutions[max(solutions)]
    x = mesh.points[:, 0]
    out = {}
    for name in NAMES:
        ex = np.array([exact[name](xi, t_end) for xi in x])
        out[name] = float(np.max(np.abs(np.asarray(final[name]) - ex)) / np.max(np.abs(ex)))
    return out
