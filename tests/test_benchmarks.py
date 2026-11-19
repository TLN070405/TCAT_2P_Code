"""
Literature benchmarks.

Fast tests check the McWhorter & Sunada (1990) semi-analytic solution itself
(tests/benchmarks/mcwhorter_sunada.py). Slow tests (pytest -m slow) run the
TCAT2Phase solvers on published problems:

- Szymkiewicz (2007) Example 1: countercurrent air-water imbibition with the
  2P-TM solver, compared with the McWhorter-Sunada solution and with the
  cumulative-infiltration errors Szymkiewicz reports for the same grids.
- Kees & Miller (2002) Problem HD: the McWhorter-Sunada solution with R = 1
  (fast check of the semi-analytic solution against Kees & Miller's Fig. 1;
  the 2P-TM run is applications/Chapter5_validation/2p_kees_miller_drainage.py).
- Tocci, Kelley & Miller (1997) Problem A: Richards' equation infiltration,
  compared with the front position in their Fig. 1 and with Celia et al. (1990)
  global mass balance.
- Method of manufactured solutions (tests/benchmarks/mms.py): RE and 2P-TM (sp)
  with source terms that make chosen smooth fields exact; the error must fall
  at second order under grid refinement.

See each module in tests/benchmarks/ for the full problem definitions.
"""
import numpy as np
import pytest

from benchmarks import countercurrent_2p as cc
from benchmarks import mcwhorter_sunada as ms
from benchmarks import mms
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
        # both fluids incompressible, as in the exact solution (index-1 DAE; the
        # solver starts it with a consistent-initialization step)
        runs[blocks] = cc.run(blocks + 1)
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
    # Tocci et al. (1997) Table 1 and Sec. 3.2 text: Problem A runs to t = 0.25 days,
    # the time of Fig. 1 (the Fig. 1 caption's "0.20 days" disagrees with both; at
    # 0.20 days the front here is at z = 0.072 m, at 0.25 days at z = 0.043 m).
    # 251 equally spaced outputs give an even number of intervals for Simpson's rule.
    times = np.linspace(0.0, 0.25 * 86400.0, 252)[1:]
    z, solutions, model = tp.run(61, times)
    return z, solutions, model, times[-1]


@pytest.mark.slow
def test_tocci_problem_a_front_position(tocci_run):
    """Tocci et al. (1997) Fig. 1: at t = 0.25 days the front (psi = -5 m) is near z = 0.045 m."""
    z, solutions, _, t_end = tocci_run
    assert tp.front_position(z, solutions[t_end]["p_w"]) == pytest.approx(0.045, abs=0.01)


@pytest.mark.slow
def test_tocci_problem_a_mass_balance(tocci_run):
    """
    Celia et al. (1990) mass balance measure: (mass gained)/(net inflow) = 1.

    The net inflow is a time integral of the boundary fluxes sampled at the
    output times. With the trapezoid rule that quadrature dominated the measure
    (ratio 0.99898 over 250 outputs, 0.99948 over 960 at t = 0.20 days), which is
    why this test first failed; with Simpson's rule the ratio is 0.99996.
    """
    _, solutions, model, _ = tocci_run
    _, gained, inflow = tp.mass_balance(solutions, model)
    assert gained[-1] / inflow[-1] == pytest.approx(1.0, abs=1e-3)


# --- McWhorter-Sunada with R = 1: Kees & Miller (2002) Problem HD (fast) ------

@pytest.fixture(scope="module")
def hd_exact():
    from benchmarks import kees_miller_hd as hd
    return hd, hd.exact_solution(n=1000)


def test_hd_solution_reproduces_prescribed_injection(hd_exact):
    """S0 is found so that A(S0) is the prescribed A = 0.1967 m/day^(1/2)."""
    hd, (s, F, dF, A, S0) = hd_exact
    assert A == pytest.approx(hd.A, rel=1e-6)
    assert F[0] == pytest.approx(0.0, abs=1e-12) and F[-1] == pytest.approx(1.0)
    assert np.all(np.diff(F) >= -1e-12)


def test_hd_solution_matches_kees_miller_fig1(hd_exact):
    """
    Kees & Miller (2002) Fig. 1, t = 2.5 days: theta_w is about 0.18 at x = 0 and
    the non-wetting front, where theta_w jumps to theta_s = 0.35, is near x = 4.8 m.
    """
    hd, exact = hd_exact
    s, _, dF, A, _ = exact
    assert hd.theta_w_profile(np.array([0.0]), hd.T_FINAL, exact)[0] == pytest.approx(0.18, abs=0.01)
    assert ms.front_position(s, dF, A, hd.PHI, hd.T_FINAL)[0] == pytest.approx(4.8, abs=0.15)


def test_hd_injected_volume_equals_2_A_sqrt_t(hd_exact):
    hd, exact = hd_exact
    x = np.linspace(0.0, hd.LENGTH, 40001)
    theta = hd.theta_w_profile(x, hd.T_FINAL, exact)
    assert np.trapz(hd.PHI - theta, x) == pytest.approx(2.0 * hd.A * np.sqrt(hd.T_FINAL), rel=2e-3)


# --- Method of manufactured solutions ---------------------------------------------

MMS_NODES = (11, 21, 41)


def observed_orders(errors):
    e = np.asarray(errors)
    return np.log2(e[:-1] / e[1:])


def test_mms_sources_vanish_for_a_steady_state_solution(monkeypatch):
    # With the exact solution frozen in time and hydrostatic, both terms of S vanish up to
    # the finite-difference round-off (~1e-9; the manufactured sources are ~0.1 kg/m^3/s).
    monkeypatch.setattr(mms, "exact_p_w_richards", lambda x, t: -2000.0 - mms.RHO_W0 * mms.GRAVITY * x)
    source = mms.richards_source(mms.closures())
    assert abs(source(0.3, 50.0)) < 1e-7


@pytest.mark.slow
def test_mms_richards_second_order_in_space():
    errors = [mms.richards_errors(n, tol=1e-8)["p_w"] for n in MMS_NODES]
    assert errors[-1] < 0.2   # Pa, on 41 nodes
    assert np.all(observed_orders(errors) > 1.85)


@pytest.mark.slow
def test_mms_two_phase_sp_second_order_in_space():
    runs = [mms.two_phase_errors(n, tol=1e-8) for n in MMS_NODES]
    for key, limit in (("e_w", 3e-5), ("p_w", 4.0)):
        errors = [r[key] for r in runs]
        assert errors[-1] < limit
        assert np.all(observed_orders(errors) > 1.85), (key, errors)
