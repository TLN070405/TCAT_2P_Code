"""
Compare the Chapter 5 validation runs with the McWhorter & Sunada (1990)
semi-analytic solution:

  - 2p_kees_miller_drainage.py   (Kees & Miller 2002, Problem HD; dissertation Fig. 5.1)
  - 2p_szymkiewicz_imbibition.py (Szymkiewicz 2007, Example 1; dissertation Fig. 5.6)

Run the drivers first. Prints, for each output time, the L1 error of e_w, the
front position (where e_w crosses 0.32 for drainage, 0.03 for imbibition), and
the error in the volume of the invading fluid, and plots the profiles. The semi-analytic solutions live in tests/benchmarks/.
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from TCAT2Phase.io.hdf5_io import load_solutions_from_hdf5

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "tests"))
from benchmarks import mcwhorter_sunada as ms          # noqa: E402
from benchmarks import kees_miller_hd as hd            # noqa: E402
from benchmarks import countercurrent_2p as cc         # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "outputs"


def front(x, e_w, level):
    """First x where e_w crosses `level` (linear interpolation)."""
    above = e_w > level
    i = np.argmax(above != above[0])
    return x[i - 1] + (level - e_w[i - 1]) * (x[i] - x[i - 1]) / (e_w[i] - e_w[i - 1])


def drainage():
    solutions, x, _ = load_solutions_from_hdf5(str(OUT), "2P_kees_miller_drainage.h5")
    x = np.asarray(x).reshape(-1)
    exact = hd.exact_solution()
    print(f"Kees & Miller HD: S_n(0) = {exact[4]:.4f} (e_w(0) = {hd.PHI * (1 - exact[4]):.4f}), "
          f"A = {exact[3] * np.sqrt(hd.DAY):.4f} m/day^0.5")
    print(f"{'t [day]':>8} {'L1(e_w) [m]':>12} {'front num':>10} {'front exact':>12} {'injected vol err [%]':>21}")
    fig, ax = plt.subplots(figsize=(8, 5))
    for t in sorted(solutions):
        e_num = solutions[t]["e_w"]
        e_ex = hd.theta_w_profile(x, t, exact)
        x_fine = np.linspace(0.0, hd.LENGTH, 20001)
        e_fine = hd.theta_w_profile(x_fine, t, exact)
        # injected non-wetting volume vs 2 A t^(1/2) (the ramp injects the same volume)
        vol = np.trapz(hd.PHI - e_num, x)
        err = 100.0 * (vol / (2.0 * exact[3] * np.sqrt(t)) - 1.0)
        level = 0.32   # front: where e_w rises through 0.32 (behind the front e_w < 0.29)
        print(f"{t / hd.DAY:8.2f} {np.trapz(np.abs(e_num - e_ex), x):12.4e} {front(x, e_num, level):10.3f} "
              f"{front(x_fine, e_fine, level):12.3f} {err:21.2f}")
        ax.plot(x_fine, e_fine, "k-", lw=1)
        ax.plot(x, e_num, "--", label=f"2P-TM, t = {t / hd.DAY:g} d")
    ax.set_xlabel("x [m]")
    ax.set_ylabel(r"$\epsilon^w$")
    ax.set_title("Kees & Miller (2002) Problem HD: 2P-TM (dashed) vs McWhorter-Sunada (solid)")
    ax.legend()


def imbibition():
    solutions, x, _ = load_solutions_from_hdf5(str(OUT), "2P_szymkiewicz_imbibition.h5")
    x = np.asarray(x).reshape(-1)
    s, _, dF, A = ms.solve(cc.diffusivity, cc.S_I, cc.S_0, cc.PHI)
    print(f"\nSzymkiewicz Example 1: A = {A:.4e} m/s^0.5")
    print(f"{'t [s]':>8} {'L1(e_w) [m]':>12} {'front num':>10} {'front exact':>12} {'infiltrated vol err [%]':>24}")
    fig, ax = plt.subplots(figsize=(8, 5))
    for t in sorted(solutions):
        e_num = solutions[t]["e_w"]
        S_ex = ms.saturation_profile(x, t, s, dF, A, cc.PHI)
        x_fine = np.linspace(0.0, cc.LENGTH, 20001)
        e_fine = cc.PHI * ms.saturation_profile(x_fine, t, s, dF, A, cc.PHI)
        err = 100.0 * (cc.infiltrated_volume(x, e_num / cc.PHI) / (2.0 * A * np.sqrt(t)) - 1.0)
        level = 0.03   # front: where e_w falls through 0.03 (S_w = 0.1)
        print(f"{t:8.0f} {np.trapz(np.abs(e_num - cc.PHI * S_ex), x):12.4e} {front(x, e_num, level):10.4f} "
              f"{front(x_fine, e_fine, level):12.4f} {err:24.2f}")
        ax.plot(x_fine, e_fine, "k-", lw=1)
        ax.plot(x, e_num, "--", label=f"2P-TM, t = {t:g} s")
    ax.set_xlabel("x [m]")
    ax.set_ylabel(r"$\epsilon^w$")
    ax.set_title("Szymkiewicz (2007) Example 1: 2P-TM (dashed) vs McWhorter-Sunada (solid)")
    ax.legend()


if __name__ == "__main__":
    drainage()
    imbibition()
    plt.show()
