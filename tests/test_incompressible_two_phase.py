"""
Incompressible two-phase flow with no flow of either fluid at one boundary
(the Szymkiewicz (2007) Example 1 setup in tests/benchmarks/countercurrent_2p.py).

Both fluids incompressible makes the 2P-TM an index-1 DAE: one pressure has no
time derivative and jumps at t = 0+ to satisfy the boundary conditions. Before
the consistent-initialization step in numerics/solver.py, BDF rejected every
first step (its error estimate saw the jump and did not shrink with dt), and
the sp formulation also hit a zero LU pivot in the algebraic no-flow rows.
These tests take a few time steps of each formulation and boundary treatment.
"""
import numpy as np
import pytest
from petsc4py import PETSc

from benchmarks import countercurrent_2p as cc
from TCAT2Phase.numerics import solver


@pytest.mark.parametrize("form", ["pp", "sp"])
@pytest.mark.parametrize("no_flow", ["algebraic", "conservative"])
def test_first_steps_succeed(form, no_flow):
    try:
        x, S = cc.run(11, t_end=1.0, form=form, no_flow=no_flow, options={"ts_max_steps": 4})
    finally:
        PETSc.Options().setValue("ts_max_steps", 100000)
    assert np.all(np.isfinite(S))
    assert S[0] == pytest.approx(cc.S_0)
    assert S[1] > cc.S_I            # water has started to enter
    assert np.all(S[3:] == pytest.approx(cc.S_I, abs=1e-6))


def test_consistent_initialization_is_automatic_only_when_both_fluids_incompressible():
    class Phase:
        def __init__(self, c):
            self.d_density_function = lambda p: c

    class Model:
        consistent_initialization = None
        model = type("TwoPhase", (), {})()

    m = Model()
    m.w_phase, m.n_phase = Phase(0.0), Phase(0.0)
    assert solver._needs_consistent_initialization(m)
    m.n_phase = Phase(1.2e-5)
    assert not solver._needs_consistent_initialization(m)
    m.consistent_initialization = True
    assert solver._needs_consistent_initialization(m)
