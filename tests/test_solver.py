"""
End-to-end solver tests: small Richards' equation and two-phase problems run
through PETSc, checked against hydrostatic equilibrium.

The column is vertical with x measured upward, so hydrostatic pressure is
p(x) = p(0) - rho g x. These tests exercise the full chain: mesh, closure
relations, residual assembly, boundary conditions, and PETSc time stepping.
"""
import numpy as np
import pytest
from petsc4py import PETSc

import TCAT2Phase
from TCAT2Phase.state_equations import state_equations_interface

from conftest import air, build_mesh, water

LENGTH, NODES, G, RHO_W = 0.1, 11, 9.81, 1000.0


def hydrostatic(x, p0=-1.0):
    return p0 - RHO_W * G * x


def set_petsc_options():
    """Same settings as applications/*/solver_options.py, minus the monitor output."""
    opts = PETSc.Options()
    opts.setValue("snes_fd_color", None)
    opts.setValue("ts_type", "bdf")
    opts.setValue("ts_bdf_order", 2)
    opts.setValue("ts_max_steps", 100000)
    opts.setValue("ts_max_reject", 500)
    opts.setValue("ts_max_snes_failures", 10000)
    opts.setValue("ts_rtol", 1.0e-8)
    opts.setValue("ts_atol", 1.0e-8)
    opts.setValue("pc_type", "sor")
    opts.delValue("ts_monitor")


def build_model(model_type, form, p_w_ic, boundary_conditions, boundary_type):
    domain, mesh = build_mesh(NODES, length=LENGTH)
    model = TCAT2Phase.numerics.model.Model(
        model_type=model_type, form=form, domain=domain, nodes=NODES, mesh=mesh)
    model.set_boundary_type(boundary_type)

    solid = TCAT2Phase.properties.phase.Solid(
        "s", porosity=0.369, permeability=2.67e-9, gravity=G, dynamic_capillary_pressure_parameter=None)
    w_phase, n_phase = water(0.0687 * 0.369), air(0.0)
    vg = state_equations_interface.VanGenuchten(
        m=1.0 - 1.0 / 37.8, n=37.8, alpha=2.28e-3, porosity=solid.porosity,
        e_wr=w_phase.residual_volume_fraction, e_nr=n_phase.residual_volume_fraction)
    w_phase.set_relative_permeability_function(vg.w_relative_permeability)
    n_phase.set_relative_permeability_function(vg.n_relative_permeability)
    model.set_model_phases(solid, w_phase, n_phase, TCAT2Phase.properties.interface.TraditionalInterface(vg))

    ics = {"p_w": p_w_ic}
    if "p_n" in model.model.u_names:
        ics["p_n"] = lambda x: 0.0
    model.set_initial_conditions(ics)
    model.set_boundary_conditions(boundary_conditions)
    return model


def solve(model, times):
    set_petsc_options()
    model.set_simulation_time(np.array(times))
    u, ts = TCAT2Phase.numerics.solver.setup_solver(model)
    return TCAT2Phase.numerics.solver.solver_time(u, ts, model)


def x_points():
    return np.linspace(0.0, LENGTH, NODES)


RE_BCS = {"p_w": [[hydrostatic(0.0), hydrostatic(LENGTH)]]}


def test_re_preserves_hydrostatic_equilibrium():
    model = build_model("Richards", "pr", hydrostatic, RE_BCS, {"p_w": [[0, 0]]})
    solutions = solve(model, [100.0])
    np.testing.assert_allclose(solutions[100.0]["p_w"], hydrostatic(x_points()), atol=1e-2)


def test_re_relaxes_to_hydrostatic_equilibrium():
    """
    Start a fully saturated column (p_w > -air-entry pressure everywhere) away from
    hydrostatic and check it relaxes back. A wrong flux or gravity sign would not.
    """
    p0 = 600.0  # top of column at about -381 Pa, below the ~440 Pa air-entry pressure

    def perturbed(x):
        return hydrostatic(x, p0) + 50.0 * np.sin(np.pi * x / LENGTH)

    bcs = {"p_w": [[hydrostatic(0.0, p0), hydrostatic(LENGTH, p0)]]}
    model = build_model("Richards", "pr", perturbed, bcs, {"p_w": [[0, 0]]})
    solutions = solve(model, [100.0])
    np.testing.assert_allclose(solutions[100.0]["p_w"], hydrostatic(x_points(), p0), atol=1e-2)


def test_re_time_dependent_boundary_condition_is_applied():
    """Boundary values may be callables of time (Test Case 8 uses this)."""
    bcs = {"p_w": [[lambda t: hydrostatic(0.0), lambda t: hydrostatic(LENGTH)]]}
    model = build_model("Richards", "pr", hydrostatic, bcs, {"p_w": [[0, 0]]})
    solutions = solve(model, [10.0])
    np.testing.assert_allclose(solutions[10.0]["p_w"], hydrostatic(x_points()), atol=1e-2)


def test_two_phase_pp_preserves_hydrostatic_equilibrium():
    rho_n = 1.2
    bcs = {"p_w": RE_BCS["p_w"], "p_n": [[0.0, -rho_n * G * LENGTH]]}
    model = build_model("two_phase", "pp", hydrostatic, bcs, {"p_w": [[0, 0]], "p_n": [[0, 0]]})
    model.initial_conditions["p_n"] = lambda x: -rho_n * G * x
    solutions = solve(model, [1000.0])
    np.testing.assert_allclose(solutions[1000.0]["p_w"], hydrostatic(x_points()), atol=1e-2)
    np.testing.assert_allclose(solutions[1000.0]["p_n"], -rho_n * G * x_points(), atol=1e-2)


@pytest.mark.xfail(strict=True, raises=(AttributeError, IndexError), reason=(
    "Flux boundary types 1 and 2 are broken: BaseResidual.compute_boundary_flux applies "
    "full-mesh indexing to the two-cell boundary mesh and reads self.model.phase (no such "
    "attribute). No driver uses these types; Dirichlet (0) and zero-gradient (3) work. "
    "Called directly because inside a PETSc solve the exception aborts the interpreter."))
def test_re_flux_boundary_condition():
    model = build_model("Richards", "pr", hydrostatic, RE_BCS, {"p_w": [[0, 1]]})
    residual = TCAT2Phase.numerics.residual_re_pr.ResidualRePR(model)
    residual.compute_boundary_flux(model.w_phase, hydrostatic(x_points()))
