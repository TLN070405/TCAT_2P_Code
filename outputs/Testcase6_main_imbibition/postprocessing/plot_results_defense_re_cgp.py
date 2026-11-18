"""
Defense slides 9-10 (Test Case 5) / 11-12 (Test Case 6): RE vs TCAT CGP.
Run re_*_defense.py and tcat_cgp_*_defense.py first.
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from TCAT2Phase.io.hdf5_io import load_solutions_from_hdf5

CASE = 6  # set to 5 in the Testcase5 copy of this script

outputs = Path(__file__).resolve().parent.parent / 'outputs'
solutions_re, mesh_points, meta_re = load_solutions_from_hdf5(str(outputs), f"RE_test_case_{CASE}_defense.h5")
solutions_cgp, _, meta_cgp = load_solutions_from_hdf5(str(outputs), f"tcat_cgp_test_case_{CASE}_defense.h5")
x = mesh_points[:, 0]


def plot_profiles(variable_name, ylabel, units, include_re):
    """Profiles of one variable at every output time."""
    plt.figure(figsize=(10, 6))
    for k, (time, variables) in enumerate(solutions_cgp.items()):
        line, = plt.plot(x, variables[variable_name], label=f"TCAT CGP, t={time:g}", linestyle='-')
        if include_re:
            plt.plot(x, solutions_re[time][variable_name], color=line.get_color(), linestyle='--',
                     label=f"RE, t={time:g}")
    plt.xlabel(r"$x\,[m]$", fontsize=10)
    plt.ylabel(f"{ylabel} [{units}]", fontsize=10)
    plt.legend(fontsize=8, ncol=2)
    plt.show()


def normalize(values):
    values = np.asarray(values)
    return (values - values[0]) / (values[-1] - values[0])


def plot_normalized():
    """Normalized time evolution at the monitor node (slides 10 and 12)."""
    plt.figure(figsize=(10, 6))
    t = meta_cgp["history_time"]
    for name, label in [("p_w", r"$p^w$"), ("k_wn", r"$K^{wn}$"), ("e_w", r"$\epsilon^w$"),
                        ("e_wn", r"$\epsilon^{wn}$"), ("j_wn", r"$J^{wn}$")]:
        plt.semilogx(t, normalize(meta_cgp[f"history_{name}"]), label=f"TCAT CGP {label}")
    t_re = meta_re["history_time"]
    plt.semilogx(t_re, normalize(meta_re["history_p_w"]), "k--", label=r"RE $p^w$")
    plt.semilogx(t_re, normalize(meta_re["history_e_w"]), "k:", label=r"RE $\epsilon^w$")
    plt.xlabel(r"$t\,[s]$", fontsize=10)
    plt.ylabel("normalized value [-]", fontsize=10)
    plt.title(f"x = {x[int(meta_cgp['monitor_node'])]:g} m")
    plt.legend(fontsize=10)
    plt.show()


plot_profiles("e_w", r"$\epsilon^w$", "-", include_re=True)
plot_profiles("p_w", r"$p^w$", "Pa", include_re=True)
plot_profiles("e_wn", r"$\epsilon^{wn}$", "1/m", include_re=False)
plot_profiles("j_wn", r"$J^{wn}$", "1/m", include_re=False)
plot_profiles("k_wn", r"$K^{wn}$", "1/m^2", include_re=False)
plot_normalized()
