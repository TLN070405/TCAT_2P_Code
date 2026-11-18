"""
Semi-analytic solution of McWhorter & Sunada (1990), "Exact integral solutions
for two-phase flow", Water Resour. Res. 26(3), 399-413, for COUNTERCURRENT
imbibition (R = q_t/q_0 = 0) in a horizontal, semi-infinite column of two
incompressible fluids:

    S_w(0, t) = S0,  S_w(x, 0) = S_w(inf, t) = Si,  q_t = 0.

With R = 0 the paper's Eqs. (23), (19), (17) and (10) become

    F(S)  = 1 - int_S^S0 (b - S) D/F db / I            (23)
    F'(S) = int_S^S0 D/F db / I                         (19)
    I     = int_Si^S0 (b - Si) D/F db,   A^2 = phi I / 2  (17)
    x(S, t) = (2 A / phi) F'(S) t^(1/2)                 (10)

where D(S) = -(k k_rn f / mu_n) dPc/dS and f = (1 + k_rn mu_w / (k_rw mu_n))^-1.
F is found by fixed-point iteration starting from F = 1, as in the paper.
"""
import numpy as np


def _cumtrapz_from_right(y, s):
    """int_s[i]^s[-1] y ds by the trapezoid rule, for every i."""
    seg = 0.5 * (y[1:] + y[:-1]) * np.diff(s)
    out = np.zeros_like(s)
    out[:-1] = np.cumsum(seg[::-1])[::-1]
    return out


def solve(diffusivity, s_i, s_0, porosity, n=4000, tol=1e-12, max_iter=500):
    """
    diffusivity: callable D(S) [m^2/s], > 0 on (s_i, s_0]
    Returns (s, F, dF, A): saturation grid from s_i to s_0, F(S), F'(S), and A.
    The grid is clustered near s_i, where D/F varies fastest.
    """
    u = np.linspace(0.0, 1.0, n)
    s = s_i + (s_0 - s_i) * u**2
    D = np.array([diffusivity(v) for v in s])
    F = np.ones_like(s)
    for _ in range(max_iter):
        q = np.zeros_like(s)
        q[1:] = D[1:] / F[1:]
        # D/F ~ D(Si)/(F'(Si)(S - Si)) near Si: drop the (integrable) endpoint value
        q[0] = q[1]
        G = _cumtrapz_from_right(q, s)                    # int_S^S0 D/F db
        H = _cumtrapz_from_right(q * s, s)                # int_S^S0 b D/F db
        num = H - s * G                                   # int_S^S0 (b - S) D/F db
        I = num[0]
        F_new = 1.0 - num / I
        F_new[0] = 0.0
        F_new = np.maximum(F_new, 1e-300)
        converged = np.max(np.abs(F_new - F)) < tol
        F = F_new
        if converged:
            break
    dF = G / I
    A = np.sqrt(0.5 * porosity * I)
    return s, F, dF, A


def front_position(s, dF, A, porosity, t):
    """x(S, t) from Eq. (10)."""
    return 2.0 * A / porosity * dF * np.sqrt(t)


def saturation_profile(x, t, s, dF, A, porosity):
    """Invert x(S, t) to S(x, t) on the given x points (x(S) decreases with S)."""
    xs = front_position(s, dF, A, porosity, t)
    order = np.argsort(xs)
    return np.interp(x, xs[order], s[order], left=s[-1], right=s[0])
