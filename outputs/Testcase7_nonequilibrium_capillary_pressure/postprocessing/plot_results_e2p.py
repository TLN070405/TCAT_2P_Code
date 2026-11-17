""" 2 phases equilibrium test case """

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from TCAT2Phase.io.hdf5_io import load_solutions_from_hdf5

# Get the directory where the current script (re_test_case_1.py) is located
script_dir = Path(__file__).resolve().parent

# Navigate up one level and then get the "outputs" folder
script_dir = script_dir.parent / 'outputs'

tau_array=[0,1e7,1e8]

# --------------- Volume Fraction ------------------#
plt.figure(figsize=(10, 6))  # Standardize chart dimensions
for tau in (tau_array):
    # filenames
    filename = f"E2P_test_case_7_{tau}.h5"

    # Load the data from the file
    solutions, mesh_points,metadata = load_solutions_from_hdf5(str(script_dir), filename)
    # --------------- plot ------------------#
    for time, variables in solutions.items():
        plt.plot(mesh_points, variables["e_w"],
                label=f"tau={tau:g}",
                linestyle="dashed")

# Adding axis labels with units and larger font sizes
plt.xlabel(r"$x\,[m]$", fontsize=10)
plt.ylabel(r"$\epsilon^w\,[-]$", fontsize=10)
plt.legend(fontsize=12)
plt.show()


# --------------- Volume Fraction ------------------#
plt.figure(figsize=(10, 6))  # Standardize chart dimensions
for tau in (tau_array):
    # filenames
    filename = f"E2P_test_case_7_{tau}.h5"

    # Load the data from the file
    solutions, mesh_points,metadata = load_solutions_from_hdf5(str(script_dir), filename)
    # --------------- plot ------------------#
    for time, variables in solutions.items():
        plt.plot(mesh_points, variables["p_w"],
                label=f"tau={tau:g} ",
                linestyle="dashed")

# Adding axis labels with units and larger font sizes
plt.xlabel(r"$x\,[m]$", fontsize=20)
plt.ylabel(r"$p^w\,[Pa]$", fontsize=20)
plt.legend(fontsize=12)
plt.show()

# --------------- Volume Fraction ------------------#
plt.figure(figsize=(10, 6))  # Standardize chart dimensions
for tau in (tau_array):
    # filenames
    filename = f"E2P_test_case_7_{tau}.h5"

    # Load the data from the file
    solutions, mesh_points,metadata = load_solutions_from_hdf5(str(script_dir), filename)
    # --------------- plot ------------------#
    for time, variables in solutions.items():
        plt.plot(mesh_points, variables["p_n"],
                label=f"tau={tau:g} ",
                linestyle="dashed")

# Adding axis labels with units and larger font sizes
plt.xlabel(r"$x\,[m]$", fontsize=10)
plt.ylabel(r"$p^n\,[Pa]$", fontsize=10)
plt.legend(fontsize=12)
plt.show()
