""" 2 phases equilibrium test case """

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from TCAT2Phase.io.hdf5_io import load_solutions_from_hdf5

nodes_array=[101]

def volume_fraction_exact(p_w):
    """ volume fraction exact solution"""
    n =  37.8
    m = 1.0 - (1.0 / n)
    alpha = 2.28e-3
    porosity = 0.369
    e_wr = 0.0687*porosity
    e_nr = 0.

    p_c = -p_w
    if p_c > 0.0:
        s_e =  (1.0 + (alpha*p_c)**n)**(-m)
        e_w = s_e * (porosity - e_wr - e_nr) + e_wr
    else:
        e_w = porosity - e_nr
    return e_w

def pressure_w_exact(x):
    """wetting phase pressure exact solution"""
    return -1-1000*9.81*x

def interfacial_area_exact(x):
    """exact solution interfacial area"""
    coeff =800
    return coeff*volume_fraction_exact(pressure_w_exact(x))

def mean_curvature_exact(x):
    """j_wn exact solution """
    return -pressure_w_exact(x)/7.20e-2

def velocity_exact(x):
    """exact solution velocity of wn interface"""
    return 0.0

def gaussian_curvature_exact(e_w,e_wn,j_wn):
    """exact solution Gaussian curvature """
    b1 = 1.47e-1
    b2 = -2.08e-3
    b3 = 5.33e-2/(4.0 * 3.141592653589793)
    e_ss = 1.89e3
    kn_ns = 1.00e6
    js_ns = 2.00e3
    porosity =0.369
    D = 6.0*porosity/e_ss
    b1D = b1*D
    b2D2 = b2*D*D
    b3D3 = b3*D*D*D
                             
    e_ns =  e_ss*( 1- e_w/porosity )
    ewn_term = b1D  * (e_wn + e_ns )
    jwn_term = b2D2 * ( -e_wn * j_wn + e_ns *js_ns )

    k_wn = (((porosity - e_w - ewn_term - jwn_term)/b3D3) - e_ns * kn_ns)/e_wn
    return k_wn


# Get the directory where the current script (re_test_case_1.py) is located
script_dir = Path(__file__).resolve().parent

# Navigate up one level and then get the "outputs" folder
script_dir = script_dir.parent / 'outputs'



# --------------- Volume Fraction ------------------#
plt.figure(figsize=(10, 6))  # Standardize chart dimensions
for nodes in (nodes_array):
    # filenames
    filename_re = f"RE_test_case_3_{nodes}.h5"
    filename_cgp = f"tcat_cgp_test_case_3_{nodes}.h5"

    # Load the data from the file
    solutions_re, mesh_points, metadata = load_solutions_from_hdf5(str(script_dir), filename_re)
    solutions_cgp, mesh_points , metadata= load_solutions_from_hdf5(str(script_dir), filename_cgp)

    # Plot for RE
    for time, variables in solutions_re.items():
        plt.plot(mesh_points, variables["e_w"],label=f"RE, t={time}, nodes={nodes}",linestyle="dashed")

    # Plot for TCAT CGP
    for time, variables in solutions_cgp.items():
        plt.plot(mesh_points, variables["e_w"],label=f"TCAT CGP, t={time}, nodes={nodes}",linestyle='-', marker='o')

e_w_exact= np.zeros([len(mesh_points)] )
p_w_exact= np.zeros([len(mesh_points)] )
e_wn_exact= np.zeros([len(mesh_points)] )
j_wn_exact= np.zeros([len(mesh_points)] )
w_wn_exact= np.zeros([len(mesh_points)] )
k_wn_exact= np.zeros([len(mesh_points)] )

for idx, value in enumerate(mesh_points):
    e_w_exact[idx]  = volume_fraction_exact(pressure_w_exact(value[0]))
    p_w_exact[idx]  = pressure_w_exact(value[0])
    e_wn_exact[idx] = interfacial_area_exact(value[0])
    j_wn_exact[idx] = mean_curvature_exact(value[0])
    w_wn_exact[idx] = velocity_exact(value[0])
    k_wn_exact[idx] = gaussian_curvature_exact(e_w_exact[idx],e_wn_exact[idx],j_wn_exact[idx])

# Plot Analytic Solution with final time
plt.plot(mesh_points, e_w_exact,label="Analytic solution at equilibrium")

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
    filename_re = f"RE_test_case_3_{nodes}.h5"
    filename_cgp = f"tcat_cgp_test_case_3_{nodes}.h5"

    # Load the data from the file
    solutions_re, mesh_points, metadata = load_solutions_from_hdf5(str(script_dir), filename_re)
    solutions_cgp, mesh_points , metadata= load_solutions_from_hdf5(str(script_dir), filename_cgp)

    
    # Plot for RE
    for time, variables in solutions_re.items():
        plt.plot(mesh_points, variables["p_w"],label=f"RE, t={time}, nodes={nodes}",linestyle="dashed")

    # Plot for TCAT CGP
    for time, variables in solutions_cgp.items():
        plt.plot(mesh_points, variables["p_w"],label=f"TCAT CGP, t={time}, nodes={nodes}",linestyle='-', marker='o')

# Plot Analytic Solution with final time
plt.plot(mesh_points, p_w_exact,label="Analytic solution at equilibrium")

# Adding axis labels with units and larger font sizes; note the label is now p^w with units in []
plt.xlabel(r"$x\,[m]$", fontsize=10)
plt.ylabel(r"$p^w\,[Pa]$", fontsize=10)
plt.legend(fontsize=12)
plt.show()

# --------------- j_wn ------------------#
plt.figure(figsize=(10, 6))  # Standardize chart dimensions
for nodes in (nodes_array):
    # filenames
    filename_cgp = f"tcat_cgp_test_case_3_{nodes}.h5"

    # Load the data from the file
    solutions_cgp, mesh_points , metadata= load_solutions_from_hdf5(str(script_dir), filename_cgp)

    # Plot for TCAT CGP
    for time, variables in solutions_cgp.items():
        plt.plot(mesh_points, variables["j_wn"],label=f"TCAT CGP, t={time}, nodes={nodes}",linestyle='-', marker='o')

# Plot Analytic Solution with final time
plt.plot(mesh_points, j_wn_exact, label="Analytic solution at equilibrium")

# Adding axis labels with units and larger font sizes
plt.xlabel(r"$x\,[m]$", fontsize=10)
plt.ylabel(r"$J^{wn}_n\,[1/m]$", fontsize=10)
plt.legend(fontsize=12)
plt.show()




# ---------------e_wn------------------#
plt.figure(figsize=(10, 6))  # Standardize chart dimensions
for nodes in (nodes_array):
    # filenames
    filename_cgp = f"tcat_cgp_test_case_3_{nodes}.h5"

    # Load the data from the file
    solutions_cgp, mesh_points , metadata= load_solutions_from_hdf5(str(script_dir), filename_cgp)

    # Plot for TCAT CGP
    for time, variables in solutions_cgp.items():
        plt.plot(mesh_points, variables["e_wn"],label=f"TCAT CGP, t={time}, nodes={nodes}",linestyle='-', marker='o')

# Plot Analytic Solution with final time
plt.plot(mesh_points, e_wn_exact, label="Analytic solution at equilibrium")

# Adding axis labels with units and larger font sizes
plt.xlabel(r"$x\,[m]$", fontsize=10)
plt.ylabel(r"$\epsilon^{wn}\,[1/m]$", fontsize=10)
plt.legend(fontsize=12)
plt.show()

# ---------------w_wn------------------#
plt.figure(figsize=(10, 6))  # Standardize chart dimensions
for nodes in (nodes_array):
    # filenames
    filename_cgp = f"tcat_cgp_test_case_3_{nodes}.h5"

    # Load the data from the file
    solutions_cgp, mesh_points , metadata= load_solutions_from_hdf5(str(script_dir), filename_cgp)

    # Plot for TCAT CGP
    for time, variables in solutions_cgp.items():
        plt.plot(mesh_points, variables["w_wn"],label=f"TCAT CGP, t={time}, nodes={nodes}",linestyle='-', marker='o')

# Plot Analytic Solution with final time
plt.plot(mesh_points, w_wn_exact,label="Analytic solution at equilibrium")

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
    filename_cgp = f"tcat_cgp_test_case_3_{nodes}.h5"

    # Load the data from the file
    solutions_cgp, mesh_points , metadata= load_solutions_from_hdf5(str(script_dir), filename_cgp)

    # Plot for TCAT CGP
    for time, variables in solutions_cgp.items():
        plt.plot(mesh_points, variables["k_wn"],label=f"TCAT CGP, t={time}, nodes={nodes}",linestyle='-', marker='o')

# Plot Analytic Solution with final time
plt.plot(mesh_points, k_wn_exact,label="Analytic solution at equilibrium")

# Adding axis labels with units and larger font sizes
plt.xlabel(r"$x\,[m]$", fontsize=10)
plt.ylabel(r"$K^{wn}\,[1/m]$", fontsize=10)
plt.legend(fontsize=12)
plt.show()
