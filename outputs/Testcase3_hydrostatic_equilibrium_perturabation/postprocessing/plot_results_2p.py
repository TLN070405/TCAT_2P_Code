import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from TCAT2Phase.io.hdf5_io import load_solutions_from_hdf5


def plot_variable(variable_name, ylabel, units, mesh_points, solutions):
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
    for time, variables in solutions.items():
        plt.plot(mesh_points, variables[variable_name], label=f"Two-Phase, t={time}", linestyle='-')

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
filename = f"2P_test_case_3.h5"
# Load the data from the file
solutions, mesh_points , metadata = load_solutions_from_hdf5(str(script_dir), filename)

# Plot different variables
plot_variable("e_w", r"$\epsilon^w$", "-", mesh_points, solutions)
plot_variable("p_w", r"$p^w$", "Pa", mesh_points, solutions)
plot_variable("p_n", r"$p^n$", "Pa", mesh_points, solutions)