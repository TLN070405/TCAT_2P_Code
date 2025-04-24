""" test case 8 """

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from TCAT2Phase.io.hdf5_io import load_solutions_from_hdf5


nodes_array=[65,101]


# Get the directory where the current script (re_test_case_1.py) is located
script_dir = Path(__file__).resolve().parent

# Navigate up one level and then get the "outputs" folder
script_dir = script_dir.parent / 'outputs'



# --------------- Pressure ------------------#
plt.figure(figsize=(10, 6))  # Standardize chart dimensions
for nodes in (nodes_array):
    # filenames
    filename = f"RE_test_case_8_{nodes}.h5"

    # Load the data from the file
    solutions, mesh_points,metadata = load_solutions_from_hdf5(str(script_dir), filename)
    # --------------- plot ------------------#
    for time, variables in solutions.items():
        plt.plot(mesh_points, variables["p_w"],
                label=f"nodes={nodes} ",
                linestyle="dashed")

# Adding axis labels with units and larger font sizes
plt.xlabel(r"$x\,[m]$", fontsize=10)
plt.ylabel(r"$p^w\,[Pa]$", fontsize=10)
plt.legend(fontsize=12)
plt.show()


# --------------- Pc-ew  ------------------#

plt.figure(figsize=(10, 6))  # Standardize chart dimensions
for nodes in (nodes_array):
    # filenames
    filename = f"RE_test_case_8_{nodes}.h5"

    # Load the data from the file
    solutions, mesh_points,metadata = load_solutions_from_hdf5(str(script_dir), filename)
    counter_hysteresis = metadata["counter"]
    # --------------- plot ------------------#
    for time, variables in solutions.items():
        
        plt.plot(variables['e_w_hysteresis'][0,1:counter_hysteresis],
            variables['p_c_hysteresis'] [0,1:counter_hysteresis],
            label='scanning curves',marker='o')

        plt.plot(variables['e_w_mpc_imb'],variables['pc_range'],label="MIC VG",linestyle="dashed")
        plt.plot(variables['e_w_mpc_dr'],variables['pc_range'],label="MDC VG",linestyle="dashed")
# Adding axis labels with units and larger font sizes
plt.xlabel(r"$x\,[m]$", fontsize=10)
plt.ylabel(r"$p^w\,[Pa]$", fontsize=10)
plt.legend(fontsize=12)
plt.show()
