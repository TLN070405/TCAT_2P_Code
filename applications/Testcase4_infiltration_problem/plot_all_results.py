""" 2 phases equilibrium test case """

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from TCAT2Phase.io.hdf5_io import load_solutions_from_hdf5


# Get the directory where the current script (re_test_case_1.py) is located
script_dir = Path(__file__).resolve().parent

# filenames
filename_re = "RE_test_case_4.h5"
filename_cgp = "tcat_cgp_test_case_4.h5"
# Load the data from the file
solutions_re, mesh_points = load_solutions_from_hdf5(str(script_dir), filename_re)
solutions_cgp, mesh_points = load_solutions_from_hdf5(str(script_dir), filename_cgp)


# ---------------volume fraction------------------#
#Plot for RE
for time, variables in solutions_re.items():
    plt.plot(mesh_points, variables["e_w"], label=f"RE, t={time}",linestyle="dashed")

# Plot for CGP
for time, variables in solutions_cgp.items():
    plt.plot(mesh_points, variables["e_w"], label=f"TCAT CGP, t={time}",linestyle='-', marker='o')


# Adding labels and title
plt.xlabel(r"$x\, (m)$")
plt.ylabel(r'$\epsilon^w$')
plt.title("Test case 3")
plt.legend()

# Show the plot
plt.show()

# ---------------pressure------------------#
#Plot for RE
for time, variables in solutions_re.items():
    plt.plot(mesh_points, variables["p_w"], label=f"RE, t={time}",linestyle="dashed")

# Plot for CGP
for time, variables in solutions_cgp.items():
    plt.plot(mesh_points, variables["p_w"], label=f"TCAT CGP, t={time}",linestyle='-', marker='o')


# Adding labels and title
plt.xlabel(r"$x\, (m)$")
plt.ylabel(r'$p_w$')
plt.title("Test case 3")
plt.legend()

# Show the plot
plt.show()

# ---------------j_wn------------------#
#Plot for RE
for time, variables in solutions_re.items():
    plt.plot(mesh_points, variables["j_wn"], label=f"RE, t={time}",linestyle="dashed")

# Plot for CGP
for time, variables in solutions_cgp.items():
    plt.plot(mesh_points, variables["j_wn"], label=f"TCAT CGP, t={time}",linestyle='-', marker='o')


# Adding labels and title
plt.xlabel(r"$x\, (m)$")
plt.ylabel(r'$J^{wn}$')
plt.title("Test case 3")
plt.legend()

# Show the plot
plt.show()
