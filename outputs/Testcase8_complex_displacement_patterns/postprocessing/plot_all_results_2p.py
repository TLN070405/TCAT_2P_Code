""" test case 8 """

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from TCAT2Phase.io.hdf5_io import load_solutions_from_hdf5
from matplotlib.lines import Line2D

nodes_array=[101]

# Get the directory where the current script (re_test_case_1.py) is located
script_dir = Path(__file__).resolve().parent

# Navigate up one level and then get the "outputs" folder
script_dir = script_dir.parent / 'outputs'

""" # --------------- Pressure ------------------#
plt.figure(figsize=(10, 6))  # Standardize chart dimensions
for nodes in (nodes_array):
    # filenames
    filename = f"2P_test_case_8_{nodes}.h5"

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
 """

# --------------- Pc-ew  ------------------#

# Plot settings
curves_thickness = 2.5
axis_titles_fs = 25
axis_numbers_fs = 20
numbered_points_fs = 25
legend_fs = 25



plt.figure(figsize=(10, 6))

for nodes in nodes_array:
    filename = f"2P_test_case_8_{nodes}.h5"
    solutions, mesh_points, metadata = load_solutions_from_hdf5(str(script_dir), filename)
    counter_hysteresis = metadata["counter"]
    # Define reversal indices
    reversal_indices = [1, 600, 3900, 8650, 11700, counter_hysteresis-1]
    for time, variables in solutions.items():
        # Primary drainage and main imbibition curves
        plt.plot(variables['e_w_mpc_dr'], variables['pc_range'], label="Primary drainage curve", color='r', linewidth=curves_thickness)
        plt.plot(variables['e_w_mpc_imb'], variables['pc_range'], label="Main imbibition curve", color='b', linewidth=curves_thickness)

        # Scanning curves (piecewise between reversals)
        revs =  reversal_indices
        colors = ['k', 'g'] * ((len(revs)-1)//2 + 1)  # alternate colors
        for i in range(len(revs) - 1):
            start = revs[i]  # avoid negative index
            end = revs[i + 1]
            plt.plot(variables['e_w_hysteresis'][0, start:end],
                     variables['p_c_hysteresis'][0, start:end],
                     label='Drainage scanning curves' if i == 0 else 'Imbibition scanning curves',
                     color=colors[i],
                     linewidth=curves_thickness)

        # Annotate reversal points (numbered 1, 2, 3, ...)
        for i, rev in enumerate(reversal_indices,1):
            offsets = [(0, 30), (0, 30), (0.006, -20), (0.004, 5), (-0.002, 0), (0, 0)]
            dx, dy = offsets[i-1]
            x = variables['e_w_hysteresis'][0, rev]
            y = variables['p_c_hysteresis'][0, rev]
            plt.text(x +dx , y + dy, f"{i}", fontsize=numbered_points_fs, color='black', ha='right', va='bottom')

# Remove duplicate labels from legend
legend_elements = [
    Line2D([0], [0], color='r', lw=curves_thickness, label='Primary drainage curve'),
    Line2D([0], [0], color='b', lw=curves_thickness, label='Main imbibition curve'),
    Line2D([0], [0], color='k', lw=curves_thickness, label='Drainage scanning curves'),
    Line2D([0], [0], color='g', lw=curves_thickness, label='Imbibition scanning curves'),
]
plt.legend(handles=legend_elements, fontsize=legend_fs)

# Axis labels and tick settings
plt.xlabel(r"$x\,[m]$", fontsize=axis_titles_fs)
plt.ylabel(r"$p^c\,[Pa]$", fontsize=axis_titles_fs)
plt.tick_params(axis='both', labelsize=axis_numbers_fs)
plt.show()
