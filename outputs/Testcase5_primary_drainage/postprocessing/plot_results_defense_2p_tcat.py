"""
Defense slides 16-17 (Test Case 5) / 18-19 and 21 (Test Case 6): 2P-TM vs full 2P-TCAT.
Run 2p_*_defense.py and tcat_full_*_defense.py first.
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from TCAT2Phase.io.hdf5_io import load_solutions_from_hdf5

CASE = 5  # set to 6 in the Testcase6 copy of this script

outputs = Path(__file__).resolve().parent.parent / 'outputs'
solutions_2p, mesh_points, meta_2p = load_solutions_from_hdf5(str(outputs), f"2P_test_case_{CASE}_defense.h5")
solutions_tcat, _, meta_tcat = load_solutions_from_hdf5(str(outputs), f"tcat_full_test_case_{CASE}_defense.h5")
x = mesh_points[:, 0]


def plot_profiles(variable_name, ylabel, units, include_2p):
    """Profiles of one variable at every output time."""
    plt.figure(figsize=(10, 6))
    for time, variables in solutions_tcat.items():
        line, = plt.plot(x, variables[variable_name], label=f"2P-TCAT, t={time:g}", linestyle='-')
        if include_2p:
            plt.plot(x, solutions_2p[time][variable_name], color=line.get_color(), linestyle='--',
                     label=f"2P-TM, t={time:g}")
    plt.xlabel(r"$x\,[m]$", fontsize=10)
    plt.ylabel(f"{ylabel} [{units}]", fontsize=10)
    plt.legend(fontsize=8, ncol=2)
    plt.show()


def normalize(values):
    values = np.asarray(values)
    return (values - values[0]) / (values[-1] - values[0])


def plot_normalized():
    """Normalized time evolution (slides 17, 19 and 21). The two models were monitored at
    different nodes in the defense code; see MONITOR_NODE in the drivers."""
    plt.figure(figsize=(10, 6))
    t = meta_tcat["history_time"]
    for name, label in [("p_w", r"$p^w$"), ("k_wn", r"$K^{wn}$"), ("e_w", r"$\epsilon^w$"),
                        ("e_wn", r"$\epsilon^{wn}$"), ("j_wn", r"$J^{wn}$"), ("g_wn", r"$G^{wn}_{xx}$")]:
        plt.semilogx(t, normalize(meta_tcat[f"history_{name}"]), label=f"2P-TCAT {label}")
    t_2p = meta_2p["history_time"]
    plt.semilogx(t_2p, normalize(meta_2p["history_p_w"]), "k--", label=r"2P-TM $p^w$")
    plt.semilogx(t_2p, normalize(meta_2p["history_e_w"]), "k:", label=r"2P-TM $\epsilon^w$")
    plt.xlabel(r"$t\,[s]$", fontsize=10)
    plt.ylabel("normalized value [-]", fontsize=10)
    plt.title(f"2P-TCAT at x = {x[int(meta_tcat['monitor_node'])]:g} m, "
              f"2P-TM at x = {x[int(meta_2p['monitor_node'])]:g} m")
    plt.legend(fontsize=10)
    plt.show()


plot_profiles("e_w", r"$\epsilon^w$", "-", include_2p=True)
plot_profiles("p_w", r"$p^w$", "Pa", include_2p=True)
plot_profiles("p_n", r"$p^n$", "Pa", include_2p=True)
plot_profiles("e_wn", r"$\epsilon^{wn}$", "1/m", include_2p=False)
plot_profiles("j_wn", r"$J^{wn}$", "1/m", include_2p=False)
plot_profiles("g_wn", r"$G^{wn}_{xx}$", "-", include_2p=False)
plot_profiles("k_wn", r"$K^{wn}$", "1/m^2", include_2p=False)
plot_normalized()
