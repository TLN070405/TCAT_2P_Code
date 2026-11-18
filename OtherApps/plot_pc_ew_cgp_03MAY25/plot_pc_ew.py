import numpy as np
import matplotlib.pyplot as plt

# Load both CSV files (skip the header row)
drainage_data = np.loadtxt(
    "/Users/tomnelson/Documents/Research/TCAT_2P/TCAT_2P_Code/OtherApps/"
    "plot_pc_ew_cgp_03MAY25/drainage_pc_ew.csv",
    delimiter=",",
    skiprows=1
)
imbibition_data = np.loadtxt(
    "/Users/tomnelson/Documents/Research/TCAT_2P/TCAT_2P_Code/OtherApps/"
    "plot_pc_ew_cgp_03MAY25/imbibition_pc_ew.csv",
    delimiter=",",
    skiprows=1
)

# Extract columns
ew_drainage = drainage_data[:, 0]
pc_drainage = drainage_data[:, 1]

ew_imbibition = imbibition_data[:, 0]
pc_imbibition = imbibition_data[:, 1]

# Plot
#mpl.rcParams['text.usetex'] = True
plt.rcParams.update({ "font.family": "monospace", "font.monospace": "Computer Modern Typewriter"})

fs=20
fs_legend=20
fs_axes=16
plt.xticks(fontsize=fs_axes)  # Increase font size of X-axis values
plt.yticks(fontsize=fs_axes)  # Increase font size of Y-axis values

plt.plot(ew_drainage, pc_drainage, label="Drainage", color='red')
plt.plot(ew_imbibition, pc_imbibition, label="Imbibition", color='blue')
plt.legend(fontsize=fs_legend)
plt.xlabel(r'$\epsilon^{\overline{\overline{w}}} $'r' $[-]$',fontsize=fs)
plt.ylabel(r'$\gamma^{wn} J^{wn}_n$'r' $[Pa]$',fontsize=fs)
plt.show()
