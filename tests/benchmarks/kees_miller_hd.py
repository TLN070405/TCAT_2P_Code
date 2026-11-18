"""
Problem HD of Kees & Miller (2002), "Higher order time integration methods for
two-phase flow", Adv. Water Resour. 25, 159-177 (Tables 1-2), the dissertation's
Chapter 5 drainage case (Tables 5.2-5.3): horizontal displacement of water by a
non-wetting fluid injected at x = 0 into an initially water-saturated column.

Both fluids incompressible, Brooks-Corey closures, L = 10 m. x = 0: no water
flux, non-wetting volumetric influx A t^(-1/2) (replaced on [0, delta_a] by the
linear ramp of Kees & Miller with the same injected volume). x = L: theta_w =
theta_s, psi_w = 0. Initially theta_w = theta_s and psi_w = 3 A delta_a^(-1/2)
(L - x) / K_s, the steady head that carries the initial influx.

With q_w(0, t) = 0 the total flux equals the injected flux (R = 1), so the
McWhorter & Sunada (1990) solution applies in terms of the non-wetting
saturation S_n, with S_n(0, t) = S0 fixed by A (exact_solution below).
"""
import numpy as np

from benchmarks import mcwhorter_sunada as ms

DAY = 86400.0
RHO_W, RHO_N, G = 1000.0, 1.21, 9.81
MU_W, MU_N = 2.0e-3, 1.0e-3            # mu_w / mu_n = 2 (Kees & Miller Table 1)
K_S = 42.4 / DAY                       # m/s (Kees & Miller Table 1; dissertation Table 5.2)
K = K_S * MU_W / (RHO_W * G)           # 1.0005e-10 m^2
PHI, THETA_R = 0.35, 0.0175            # theta_s = porosity
PD, LAMBDA = 2000.0, 2.0               # psi_D = 0.204 m
A_DAY = 0.1967                         # m/day^(1/2)
A = A_DAY / np.sqrt(DAY)               # m/s^(1/2)
DELTA_A = 1.0e-3 * DAY                 # s
LENGTH, T_FINAL = 10.0, 2.5 * DAY


def effective_saturation(s_n):
    return 1.0 - PHI * s_n / (PHI - THETA_R)


def capillary_pressure(s_n):
    return PD * effective_saturation(s_n) ** (-1.0 / LAMBDA)


def k_rw(s_n):
    return effective_saturation(s_n) ** ((2.0 + 3.0 * LAMBDA) / LAMBDA)


def k_rn(s_n):
    se = effective_saturation(s_n)
    return (1.0 - se) ** 2 * (1.0 - se ** ((2.0 + LAMBDA) / LAMBDA))


def fractional_flow_n(s_n):
    lw, ln = k_rw(s_n) / MU_W, k_rn(s_n) / MU_N
    return ln / (lw + ln) if ln > 0 else 0.0


def diffusivity(s_n):
    """D(S_n) = k (k_rn/mu_n) f_w dPc/dS_n [m^2/s], the McWhorter-Sunada capillary diffusivity."""
    se = effective_saturation(s_n)
    dpc_dsn = (PD / LAMBDA) * se ** (-1.0 / LAMBDA - 1.0) * PHI / (PHI - THETA_R)
    return K * k_rn(s_n) / MU_N * (1.0 - fractional_flow_n(s_n)) * dpc_dsn


def influx(t):
    """Volumetric non-wetting influx at x = 0 [m/s], with the Kees & Miller start-up ramp."""
    if t < DELTA_A:
        return 3.0 * A / np.sqrt(DELTA_A) - 2.0 * A * DELTA_A ** -1.5 * t
    return A / np.sqrt(t)


def exact_solution(n=4000):
    """
    McWhorter-Sunada solution for S_n with R = 1. S0 = S_n(0, t) is the value
    whose A(S0) equals the prescribed A (bisection). Returns (s, F, dF, A_check, S0).
    """
    lo, hi = 1e-4, 1.0 - THETA_R / PHI - 1e-6
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        a = ms.solve(diffusivity, 0.0, mid, PHI, n=n, R=1.0, fractional_flow=fractional_flow_n)[3]
        lo, hi = (mid, hi) if a < A else (lo, mid)
    s, F, dF, a = ms.solve(diffusivity, 0.0, 0.5 * (lo + hi), PHI, n=n, R=1.0, fractional_flow=fractional_flow_n)
    return s, F, dF, a, 0.5 * (lo + hi)


def theta_w_profile(x, t, exact):
    """theta_w(x, t) from the exact solution."""
    s, _, dF, a, _ = exact
    s_n = ms.saturation_profile(x, t, s, dF, a, PHI)
    return PHI * (1.0 - s_n)
