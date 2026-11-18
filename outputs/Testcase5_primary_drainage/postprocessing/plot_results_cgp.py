import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from TCAT2Phase.io.hdf5_io import load_solutions_from_hdf5


def plot_variable(variable_name, ylabel, units, mesh_points, solutions_cgp):
    """
    Generalized plotting function for different variables.
    
    Args:
    variable_name (str): The key for the variable to plot from the solutions.
    ylabel (str): Label for the y-axis.
    units (str): Units for the variable.
    mesh_points (array): The x-axis data (mesh points).
    solutions_cgp (dict): The dictionary containing the solution data.
    """
    plt.figure(figsize=(10, 6))  # Standardize chart dimensions

    # Plot for TCAT CGP
    for time, variables in solutions_cgp.items():
        plt.plot(mesh_points, variables[variable_name], label=f"TCAT CGP, t={time}", linestyle='-')

    # Adding axis labels with units and larger font sizes
    plt.xlabel(r"$x\,[m]$", fontsize=10)
    plt.ylabel(f"{ylabel} [{units}]", fontsize=10)
    plt.legend(fontsize=12)
    plt.show()

# Get the directory where the current script (re_test_case_1.py) is located
script_dir = Path(__file__).resolve().parent

# Navigate up one level and then get the "outputs" folder
script_dir = script_dir.parent / 'outputs'

# filename
filename_cgp = f"tcat_cgp_test_case_5.h5"
# Load the data from the file
solutions_cgp, mesh_points , metadata = load_solutions_from_hdf5(str(script_dir), filename_cgp)

# Plot different variables
plot_variable("e_w", r"$\epsilon^w$", "-", mesh_points, solutions_cgp)
plot_variable("p_w", r"$p^w$", "Pa", mesh_points, solutions_cgp)
plot_variable("j_wn", r"$J^{wn}_n$", "1/m", mesh_points, solutions_cgp)
plot_variable("e_wn", r"$\epsilon^{wn}$", "1/m", mesh_points, solutions_cgp)
plot_variable("w_wn", r"$w^{wn}$", "1/m", mesh_points, solutions_cgp)
plot_variable("k_wn", r"$K^{wn}$", "1/m", mesh_points, solutions_cgp)
