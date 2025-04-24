""" 2 phases equilibrium test case """

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from TCAT2Phase.io.hdf5_io import load_solutions_from_hdf5

nodes_array=[33,65,101]


# Get the directory where the current script (re_test_case_1.py) is located
script_dir = Path(__file__).resolve().parent

# Navigate up one level and then get the "outputs" folder
script_dir = script_dir.parent / 'outputs'



# --------------- Volume Fraction ------------------#
plt.figure(figsize=(10, 6))  # Standardize chart dimensions
for nodes in (nodes_array):
    # filenames
    filename_re = f"RE_test_case_4_{nodes}.h5"
    filename_cgp = f"tcat_cgp_test_case_4_{nodes}.h5"

    # Load the data from the file
    solutions_re, mesh_points, metadata = load_solutions_from_hdf5(str(script_dir), filename_re)
    solutions_cgp, mesh_points , metadata= load_solutions_from_hdf5(str(script_dir), filename_cgp)

    # Plot for RE
    for time, variables in solutions_re.items():
        plt.plot(mesh_points, variables["e_w"],label=f"RE, t={time}, nodes={nodes}",linestyle="dashed")

    # Plot for TCAT CGP
    for time, variables in solutions_cgp.items():
        plt.plot(mesh_points, variables["e_w"],label=f"TCAT CGP, t={time}, nodes={nodes}",linestyle='-', marker='o')


# Adding axis labels with units and larger font sizes
plt.xlabel(r"$x\,[m]$", fontsize=10)
plt.ylabel(r"$\epsilon^w\,[-]$", fontsize=10)
plt.ylim([0, 0.4])
plt.legend(fontsize=12)
plt.show()

# --------------- Pressure ------------------#
plt.figure(figsize=(10, 6))  # Standardize chart dimensions
for nodes in (nodes_array):
    # filenames
    filename_re = f"RE_test_case_4_{nodes}.h5"
    filename_cgp = f"tcat_cgp_test_case_4_{nodes}.h5"

    # Load the data from the file
    solutions_re, mesh_points, metadata = load_solutions_from_hdf5(str(script_dir), filename_re)
    solutions_cgp, mesh_points , metadata= load_solutions_from_hdf5(str(script_dir), filename_cgp)

    
    # Plot for RE
    for time, variables in solutions_re.items():
        plt.plot(mesh_points, variables["p_w"],label=f"RE, t={time}, nodes={nodes}",linestyle="dashed")

    # Plot for TCAT CGP
    for time, variables in solutions_cgp.items():
        plt.plot(mesh_points, variables["p_w"],label=f"TCAT CGP, t={time}, nodes={nodes}",linestyle='-', marker='o')


# Adding axis labels with units and larger font sizes; note the label is now p^w with units in []
plt.xlabel(r"$x\,[m]$", fontsize=10)
plt.ylabel(r"$p^w\,[Pa]$", fontsize=10)
plt.legend(fontsize=12)
plt.show()

# --------------- j_wn ------------------#
plt.figure(figsize=(10, 6))  # Standardize chart dimensions
for nodes in (nodes_array):
    # filenames
    filename_cgp = f"tcat_cgp_test_case_4_{nodes}.h5"

    # Load the data from the file
    solutions_cgp, mesh_points , metadata= load_solutions_from_hdf5(str(script_dir), filename_cgp)

    # Plot for TCAT CGP
    for time, variables in solutions_cgp.items():
        plt.plot(mesh_points, variables["j_wn"],label=f"TCAT CGP, t={time}, nodes={nodes}",linestyle='-', marker='o')


# Adding axis labels with units and larger font sizes
plt.xlabel(r"$x\,[m]$", fontsize=10)
plt.ylabel(r"$J^{wn}_n\,[1/m]$", fontsize=10)
plt.legend(fontsize=12)
plt.show()




# ---------------e_wn------------------#
plt.figure(figsize=(10, 6))  # Standardize chart dimensions
for nodes in (nodes_array):
    # filenames
    filename_cgp = f"tcat_cgp_test_case_4_{nodes}.h5"

    # Load the data from the file
    solutions_cgp, mesh_points , metadata= load_solutions_from_hdf5(str(script_dir), filename_cgp)

    # Plot for TCAT CGP
    for time, variables in solutions_cgp.items():
        plt.plot(mesh_points, variables["e_wn"],label=f"TCAT CGP, t={time}, nodes={nodes}",linestyle='-', marker='o')


# Adding axis labels with units and larger font sizes
plt.xlabel(r"$x\,[m]$", fontsize=10)
plt.ylabel(r"$\epsilon^{wn}\,[1/m]$", fontsize=10)
plt.legend(fontsize=12)
plt.show()

# ---------------w_wn------------------#
plt.figure(figsize=(10, 6))  # Standardize chart dimensions
for nodes in (nodes_array):
    # filenames
    filename_cgp = f"tcat_cgp_test_case_4_{nodes}.h5"

    # Load the data from the file
    solutions_cgp, mesh_points , metadata= load_solutions_from_hdf5(str(script_dir), filename_cgp)

    # Plot for TCAT CGP
    for time, variables in solutions_cgp.items():
        plt.plot(mesh_points, variables["w_wn"],label=f"TCAT CGP, t={time}, nodes={nodes}",linestyle='-', marker='o')


# Adding axis labels with units and larger font sizes
plt.xlabel(r"$x\,[m]$", fontsize=10)
plt.ylabel(r"$w^{wn}\,[1/m]$", fontsize=10)
plt.ylim([-1, 1])
plt.legend(fontsize=12)
plt.show()


# ---------------k_wn------------------#
plt.figure(figsize=(10, 6))  # Standardize chart dimensions
for nodes in (nodes_array):
    # filenames
    filename_cgp = f"tcat_cgp_test_case_4_{nodes}.h5"

    # Load the data from the file
    solutions_cgp, mesh_points , metadata= load_solutions_from_hdf5(str(script_dir), filename_cgp)

    # Plot for TCAT CGP
    for time, variables in solutions_cgp.items():
        plt.plot(mesh_points, variables["k_wn"],label=f"TCAT CGP, t={time}, nodes={nodes}",linestyle='-', marker='o')


# Adding axis labels with units and larger font sizes
plt.xlabel(r"$x\,[m]$", fontsize=10)
plt.ylabel(r"$K^{wn}\,[1/m]$", fontsize=10)
plt.legend(fontsize=12)
plt.show()
