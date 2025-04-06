""" 2 phases equilibrium test case """

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from TCAT2Phase.io.hdf5_io import load_solutions_from_hdf5

def volume_fraction_exact(p_w):
    """ volume fraction exact solution"""
    n                = 3.4361
    m                = 1.0 - (1.0 / n)
    alpha            = 0.2001*1e-3
    porosity         = 0.369
    p_c = -p_w
    if p_c > 0.0:
        s_e =  (1.0 + (alpha*p_c)**n)**(-m)
        e_w = porosity*  s_e
    else:
        e_w = porosity
    return e_w

def pressure_w_exact(x):
    """wetting phase pressure exact solution"""
    return -1000-1000*9.81*x

def mean_curvature__exact(x):
    """j_wn exact solution """
    return -pressure_w_exact(x)/7.20e-2

# Get the directory where the current script (re_test_case_1.py) is located
script_dir = Path(__file__).resolve().parent

# filenames
filename_re = "RE_test_case_3.h5"
filename_cgp = "tcat_cgp_test_case_3.h5"
# Load the data from the file
solutions_re, mesh_points = load_solutions_from_hdf5(str(script_dir), filename_re)
solutions_cgp, mesh_points = load_solutions_from_hdf5(str(script_dir), filename_cgp)


e_w_exact= np.zeros([len(mesh_points)] )
p_w_exact= np.zeros([len(mesh_points)] )
j_wn_exact= np.zeros([len(mesh_points)] )


for idx, value in enumerate(mesh_points):
    e_w_exact[idx]  = volume_fraction_exact(pressure_w_exact(value[0]))
    p_w_exact[idx]  = pressure_w_exact(value[0])
    j_wn_exact[idx] = mean_curvature__exact(value[0])

# ---------------volume fraction------------------#
#Plot for RE
for time, variables in solutions_re.items():
    plt.plot(mesh_points, variables["e_w"], label=f"RE, t={time}",linestyle="dashed")

# Plot for CGP
for time, variables in solutions_cgp.items():
    plt.plot(mesh_points, variables["e_w"], label=f"TCAT CGP, t={time}",linestyle='-', marker='o')

# Plot exact
plt.plot(mesh_points, e_w_exact, label="Exact solution")
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

# Plot exact
plt.plot(mesh_points, p_w_exact, label="Exact solution")
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

# Plot exact
plt.plot(mesh_points, j_wn_exact, label="Exact solution")
# Adding labels and title
plt.xlabel(r"$x\, (m)$")
plt.ylabel(r'$J^{wn}$')
plt.title("Test case 3")
plt.legend()

# Show the plot
plt.show()
