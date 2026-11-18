"""
Literature benchmarks.

Fast tests check the McWhorter & Sunada (1990) semi-analytic solution itself
(tests/benchmarks/mcwhorter_sunada.py). Slow tests (pytest -m slow) run the
TCAT2Phase solvers on published problems:

- Szymkiewicz (2007) Example 1: countercurrent air-water imbibition with the
  2P-TM solver, compared with the McWhorter-Sunada solution and with the
  cumulative-infiltration errors Szymkiewicz reports for the same grids.
- Tocci, Kelley & Miller (1997) Problem A: Richards' equation infiltration,
  compared with the front position in their Fig. 1 and with Celia et al. (1990)
  global mass balance.

See each module in tests/benchmarks/ for the full problem definitions.
"""
import numpy as np
import pytest

from benchmarks import countercurrent_2p as cc
from benchmarks import mcwhorter_sunada as ms
from benchmarks import tocci_problem_a as tp


@pytest.fixture(scope="module")
def exact():
    """McWhorter-Sunada solution for Szymkiewicz (2007) Example 1."""
    return ms.solve(cc.diffusivity, cc.S_I, cc.S_0, cc.PHI)


# --- The semi-analytic solution itself (fast) ----------------------------------

def test_fractional_flow_function_endpoints_and_monotonicity(exact):
    s, F, dF, _ = exact
    assert F[0] == pytest.approx(0.0, abs=1e-12)
    assert F[-1] == pytest.approx(1.0, abs=1e-12)
    assert np.all(np.diff(F) >= -1e-12)
    assert np.all(dF >= 0) and dF[-1] == pytest.approx(0.0, abs=1e-12)


def test_fractional_flow_function_satisfies_integral_equation(exact):
    """F must reproduce itself under the right-hand side of Eq. (23)."""
    s, F, _, _ = exact
    D = np.array([cc.diffusivity(v) for v in s])
    q = np.where(F > 0, D / np.maximum(F, 1e-300), 0.0)
    q[0] = q[1]
    G = ms._cumtrapz_from_right(q, s)
    H = ms._cumtrapz_from_right(q * s, s)
    num = H - s * G
    np.testing.assert_allclose(1.0 - num / num[0], F, atol=1e-10)


def test_solution_is_converged_in_its_own_quadrature():
    a_coarse = ms.solve(cc.diffusivity, cc.S_I, cc.S_0, cc.PHI, n=1000)[3]
    a_fine = ms.solve(cc.diffusivity, cc.S_I, cc.S_0, cc.PHI, n=8000)[3]
    assert a_coarse == pytest.approx(a_fine, rel=1e-5)


def test_cumulative_infiltration_equals_2_A_sqrt_t(exact):
    """Integrating the profile from Eq. (10) gives back the closed form 2 A t^(1/2)."""
    s, _, dF, A = exact
    x = np.linspace(0.0, cc.LENGTH, 20001)
    S = ms.saturation_profile(x, cc.T_FINAL, s, dF, A, cc.PHI)
    assert cc.infiltrated_volume(x, S) == pytest.approx(2.0 * A * np.sqrt(cc.T_FINAL), rel=2e-3)


def _thomas(a, b, c, r):
    n = len(b)
    cp, rp = np.zeros(n), np.zeros(n)
    cp[0], rp[0] = c[0] / b[0], r[0] / b[0]
    for i in range(1, n):
        m = b[i] - a[i] * cp[i - 1]
        cp[i], rp[i] = c[i] / m, (r[i] - a[i] * rp[i - 1]) / m
    out = np.zeros(n)
    out[-1] = rp[-1]
    for i in range(n - 2, -1, -1):
        out[i] = rp[i] - cp[i] * out[i + 1]
    return out


def test_matches_independent_finite_difference_solution(exact):
    """
    With q_t = 0 the problem is phi dS/dt = d/dx(D(S) dS/dx). An implicit
    finite-difference solve of that equation, written independently of both the
    semi-analytic solution and TCAT2Phase, must agree with it.
    """
    s, _, dF, A = exact
    t_end, nodes, steps = 300.0, 101, 300
    x = np.linspace(0.0, 0.2, nodes)
    dx, dt = x[1] - x[0], t_end / steps
    D = np.vectorize(cc.diffusivity)
    S = np.full(nodes, cc.S_I)
    S[0] = cc.S_0
    for _ in range(steps):
        S_old = S.copy()
        for _ in range(3):  # Picard iterations on D
            Df = D(0.5 * (S[1:] + S[:-1]))
            a, b, c, r = np.zeros(nodes), np.ones(nodes), np.zeros(nodes), S_old.copy()
            coef = dt / (cc.PHI * dx * dx)
            a[1:-1], c[1:-1] = -coef * Df[:-1], -coef * Df[1:]
            b[1:-1] = 1.0 - a[1:-1] - c[1:-1]
            r[0] = cc.S_0
            a[-1], r[-1] = -1.0, 0.0  # zero gradient at the far end
            S = _thomas(a, b, c, r)
    reference = ms.saturation_profile(x, t_end, s, dF, A, cc.PHI)
    assert np.trapz(np.abs(S - reference), x) < 2e-3


# --- TCAT2Phase vs the literature (slow) ---------------------------------------

@pytest.fixture(scope="module")
def szymkiewicz_runs():
    """2P-TM solutions of Szymkiewicz (2007) Example 1 on 10 and 20 grid blocks."""
    runs = {}
    for blocks in (10, 20):
        # Air gets its physical compressibility; with both fluids incompressible the
        # pp system is a DAE whose far-end no-flow rows make the first step fail.
        runs[blocks] = cc.run(blocks + 1, air_compressibility=1.189e-5)
    return runs


@pytest.mark.slow
@pytest.mark.parametrize("blocks", [10, 20])
def test_szymkiewicz_cumulative_infiltration_error(exact, szymkiewicz_runs, blocks):
    """
    Error in cumulative infiltrated water vs the exact solution, compared with
    Szymkiewicz (2007) Table 2 (same grid, arithmetic interblock averaging).
    """
    _, _, _, A = exact
    x, S = szymkiewicz_runs[blocks]
    err = 100.0 * (cc.infiltrated_volume(x, S) / (2.0 * A * np.sqrt(cc.T_FINAL)) - 1.0)
    published = cc.SZYMKIEWICZ_TM_ERR[blocks]
    assert 0.0 < err < 1.5 * published


@pytest.mark.slow
def test_szymkiewicz_profile_converges_to_exact(exact, szymkiewicz_runs):
    s, _, dF, A = exact
    errors = {}
    for blocks, (x, S) in szymkiewicz_runs.items():
        reference = ms.saturation_profile(x, cc.T_FINAL, s, dF, A, cc.PHI)
        errors[blocks] = np.trapz(np.abs(S - reference), x)
    assert errors[20] < 0.75 * errors[10]
    assert errors[20] < 0.03


@pytest.fixture(scope="module")
def tocci_run():
    times = np.linspace(0.0, 0.2 * 86400.0, 241)[1:]
    z, solutions, model = tp.run(61, times)
    return z, solutions, model, times[-1]


@pytest.mark.slow
def test_tocci_problem_a_front_position(tocci_run):
    """Tocci et al. (1997) Fig. 1: at t = 0.20 days the front (psi = -5 m) is near z = 0.045 m."""
    z, solutions, _, t_end = tocci_run
    assert tp.front_position(z, solutions[t_end]["p_w"]) == pytest.approx(0.045, abs=0.01)


@pytest.mark.slow
def test_tocci_problem_a_mass_balance(tocci_run):
    """Celia et al. (1990) mass balance measure: (mass gained)/(net inflow) = 1."""
    _, solutions, model, _ = tocci_run
    _, gained, inflow = tp.mass_balance(solutions, model)
    assert gained[-1] / inflow[-1] == pytest.approx(1.0, abs=1e-3)
