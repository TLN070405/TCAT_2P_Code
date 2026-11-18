import sys
import math
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset



#######################################
def calculate_grid_resolution_error(nodes_array, time_interval, data_all, output_filename):
    # Initialize the L2 error array
    L2_error = np.zeros([len(nodes_array)-1, len(time_interval)])

    # Calculate the L2 error norm for each node and time
    for i in range(len(nodes_array)-1):
        for t in range(len(time_interval)):
            for j in range(nodes_array[i]):
                L2_error[i][t] += ((data_all[i+1][t][2*j] - data_all[i][t][j]) / data_all[i+1][t][2*j]) ** 2
        
        L2_error[i][t] = math.sqrt(L2_error[i][t] / nodes_array[i])

    # Create row and column headers
    row_headers = [f'{nodes_array[i]}--{nodes_array[i+1]}' for i in range(len(nodes_array)-1)]
    col_headers = [f'{time_interval[i]}' for i in range(len(time_interval))]

    # Create DataFrame and save to CSV with scientific notation
    df = pd.DataFrame(L2_error, index=row_headers, columns=col_headers)
    df.to_csv(f'grid_resolution_error/{output_filename}.csv', float_format='%.6e', index_label='time/nodes')

#######################################
def calculate_symmetry_error(time_interval, nodes_array, data, output_filename,sign):
    half_nodes = int((nodes_array[-1] - 1) / 2)
    err_symmetry = np.zeros([len(time_interval)])

    for t in range(len(time_interval)):
        for i in range(half_nodes):
            err_symmetry[t] += ((data[t][i] - sign*data[t][nodes_array[-1] - 1 - i]) / data[t][i]) ** 2
        err_symmetry[t] = math.sqrt(err_symmetry[t] / half_nodes)

    # Save to CSV with scientific notation and headers
    df = pd.DataFrame(err_symmetry)
    df.to_csv(f'symmetry_error/{output_filename}.csv', float_format='%.6e', index_label='time/error')

def main():
    Omega_0 = 0.
    Omega_L = 1.
    nodes_array= [201]
    #### read results
    # Read the data as strings first
    Ve_w_Time_TCAT_all = []
    Vp_w_Time_TCAT_all = []
    Ve_wn_Time_TCAT_all = []
    Vj_wn_Time_TCAT_all = []
    Vw_wn_Time_TCAT_all = []
    Vk_wn_Time_TCAT_all = []

    Ve_w_Time_RE_all = []
    Vp_w_Time_RE_all = []

    for node in nodes_array:
        Ve_w_Time_TCAT = np.genfromtxt(f'results_tcat_cgp/e_w_{node}.csv', delimiter=',', dtype=str)
        Vp_w_Time_TCAT  = np.genfromtxt(f'results_tcat_cgp/p_w_{node}.csv', delimiter=',', dtype=str)
        Ve_wn_Time_TCAT = np.genfromtxt(f'results_tcat_cgp/e_wn_{node}.csv', delimiter=',', dtype=str)
        Vj_wn_Time_TCAT = np.genfromtxt(f'results_tcat_cgp/j_wn_{node}.csv', delimiter=',', dtype=str)
        Vw_wn_Time_TCAT = np.genfromtxt(f'results_tcat_cgp/w_wn_{node}.csv', delimiter=',', dtype=str)
        Vk_wn_Time_TCAT = np.genfromtxt(f'results_tcat_cgp/k_wn_{node}.csv', delimiter=',', dtype=str)

        Vew_normalized_TCAT  = np.genfromtxt(f'results_tcat_cgp/e_w_normalized_{node}.csv', delimiter=',', dtype=str)
        Vpw_normalized_TCAT  = np.genfromtxt(f'results_tcat_cgp/p_w_normalized_{node}.csv', delimiter=',', dtype=str)
        Vewn_normalized_TCAT = np.genfromtxt(f'results_tcat_cgp/e_wn_normalized_{node}.csv', delimiter=',', dtype=str)
        Vjwn_normalized_TCAT = np.genfromtxt(f'results_tcat_cgp/j_wn_normalized_{node}.csv', delimiter=',', dtype=str)
        Vkwn_normalized_TCAT = np.genfromtxt(f'results_tcat_cgp/k_wn_normalized_{node}.csv', delimiter=',', dtype=str)
        Vtime_TCAT           = np.genfromtxt(f'results_tcat_cgp/time_steps_array_{node}.csv', delimiter=',', dtype=str)

        time_interval =np.genfromtxt(f'results_tcat_cgp/time_{node}.csv', delimiter=',', dtype=str)

        Ve_w_Time_TCAT  = Ve_w_Time_TCAT.astype(float)
        Vp_w_Time_TCAT  = Vp_w_Time_TCAT.astype(float)
        Ve_wn_Time_TCAT = Ve_wn_Time_TCAT.astype(float)
        Vj_wn_Time_TCAT = Vj_wn_Time_TCAT.astype(float)
        Vw_wn_Time_TCAT = Vw_wn_Time_TCAT.astype(float)
        Vk_wn_Time_TCAT = Vk_wn_Time_TCAT.astype(float)

        Vew_normalized_TCAT  = Vew_normalized_TCAT.astype(float)
        Vpw_normalized_TCAT  = Vpw_normalized_TCAT.astype(float)
        Vewn_normalized_TCAT = Vewn_normalized_TCAT.astype(float)
        Vjwn_normalized_TCAT = Vjwn_normalized_TCAT.astype(float)
        Vkwn_normalized_TCAT = Vkwn_normalized_TCAT.astype(float)
        Vtime_TCAT           = Vtime_TCAT.astype(float)


        time_interval   = time_interval.astype(float)
        

        Ve_w_Time_TCAT_all.append(Ve_w_Time_TCAT)
        Vp_w_Time_TCAT_all.append(Vp_w_Time_TCAT)
        Ve_wn_Time_TCAT_all.append(Ve_wn_Time_TCAT)
        Vj_wn_Time_TCAT_all.append(Vj_wn_Time_TCAT)
        Vw_wn_Time_TCAT_all.append(Vw_wn_Time_TCAT)
        Vk_wn_Time_TCAT_all.append(Vk_wn_Time_TCAT)
    
        # RE results
        Ve_w_Time_RE = np.genfromtxt(f'results_re/e_w_re_{node}.csv', delimiter=',', dtype=str)
        Vp_w_Time_RE  = np.genfromtxt(f'results_re/p_w_re_{node}.csv', delimiter=',', dtype=str)

        Vew_normalized_RE  = np.genfromtxt(f'results_re/e_w_normalized_re_{node}.csv', delimiter=',', dtype=str)
        Vpw_normalized_RE  = np.genfromtxt(f'results_re/p_w_normalized_re_{node}.csv', delimiter=',', dtype=str)

        Vtime_RE           = np.genfromtxt(f'results_re/time_steps_array_re_{node}.csv', delimiter=',', dtype=str)
        time_interval_RE   = np.genfromtxt(f'results_re/time_re_{node}.csv', delimiter=',', dtype=str)


        Ve_w_Time_RE  = Ve_w_Time_RE.astype(float)
        Vp_w_Time_RE  = Vp_w_Time_RE.astype(float)

        Vew_normalized_RE  = Vew_normalized_RE.astype(float)
        Vpw_normalized_RE  = Vpw_normalized_RE.astype(float)

        Vtime_RE           = Vtime_RE.astype(float)
        time_interval_RE   = time_interval_RE.astype(float)


        Ve_w_Time_RE_all.append(Ve_w_Time_RE)
        Vp_w_Time_RE_all.append(Vp_w_Time_RE)


    ########################################### Grid resolution error
    # tcat cgp error
    calculate_grid_resolution_error(nodes_array, time_interval, Ve_w_Time_TCAT_all, 'L2e_ew_tcat')
    calculate_grid_resolution_error(nodes_array, time_interval, Vp_w_Time_TCAT_all, 'L2e_pw_tcat')
    calculate_grid_resolution_error(nodes_array, time_interval, Ve_wn_Time_TCAT_all, 'L2e_ewn_tcat')
    calculate_grid_resolution_error(nodes_array, time_interval, Vj_wn_Time_TCAT_all, 'L2e_jwn_tcat')
    calculate_grid_resolution_error(nodes_array, time_interval, Vw_wn_Time_TCAT_all, 'L2e_wwn_tcat')
    calculate_grid_resolution_error(nodes_array, time_interval, Vk_wn_Time_TCAT_all, 'L2e_kwn_tcat')

    # Calculate L2 error norms for RE data
    calculate_grid_resolution_error(nodes_array, time_interval_RE, Ve_w_Time_RE_all, 'L2e_ew_re')
    calculate_grid_resolution_error(nodes_array, time_interval_RE, Vp_w_Time_RE_all, 'L2e_pw_re')

    ########################################### Symmetry error

    # Call the function for each symmetry quantification
    calculate_symmetry_error(time_interval, nodes_array, Ve_w_Time_TCAT, 'err_symmetry_ew',1)
    calculate_symmetry_error(time_interval, nodes_array, Vp_w_Time_TCAT, 'err_symmetry_pw',1)
    calculate_symmetry_error(time_interval, nodes_array, Ve_wn_Time_TCAT, 'err_symmetry_ewn',1)
    calculate_symmetry_error(time_interval, nodes_array, Vj_wn_Time_TCAT, 'err_symmetry_jwn',1)
    calculate_symmetry_error(time_interval, nodes_array, Vk_wn_Time_TCAT, 'err_symmetry_kwn',1)
    calculate_symmetry_error(time_interval, nodes_array, Vw_wn_Time_TCAT, 'err_anti_symmetry_wwn',-1)



    #####################################################################################################################
    ###############################################Visualization###########################################################
    #####################################################################################################################
    
    mesh = np.linspace(Omega_0, Omega_L, nodes_array[-1]) # mesh just for plots

    ######################PlotsPlotsPlotsPlots#######################
    ######################PlotsPlotsPlotsPlots#######################
    ######################PlotsPlotsPlotsPlots#######################
 


    # Font settings
    plt.rcParams.update({
        "font.family": "monospace",
        "font.monospace": "Computer Modern Typewriter"
    })

    # Font size variables
    fs = 20           # General font size (titles)
    fs_legend = 20    # Legend font size (if used)
    fs_axes = 20      # Reserved (not used directly)
    factor_size = 20     # Tick label size
    fs_scaling = 25   # Scaling factor (×10⁻³) font size

    ##### Visualization w_wn subplots #####

    n_plots = len(time_interval)
    colors = plt.cm.plasma(np.linspace(0.3, 1.0, n_plots))

    # Create subplots: 4 rows, 2 columns
    fig, axes = plt.subplots(nrows=4, ncols=2, figsize=(14, 12), sharex=True)

    # Flatten axes array to simplify access
    axes = axes.flatten()

    for i in range(n_plots):
        ax = axes[i]
        ax.plot(mesh, Vw_wn_Time_TCAT[i], color=colors[i], linestyle='-')
        
        # Title and label
        ax.set_title(f't = {time_interval[i]:g} sec', fontsize=fs)
        ax.set_ylabel(r'$w^{wn}_x\ [m/s]$', fontsize=25)
        
        # Tick label size
        ax.tick_params(axis='both', which='major', labelsize=fs_scaling)

        # ✅ Change scaling factor (offset text) size
        ax.xaxis.get_offset_text().set_fontsize(factor_size)
        ax.yaxis.get_offset_text().set_fontsize(factor_size)

    # Set shared x-axis label for the bottom row subplots
    for ax in axes[-2:]:
        ax.set_xlabel(r'$x\ [m]$', fontsize=25)
        ax.tick_params(axis='both', which='major', labelsize=fs_scaling)
        
        # ✅ Also apply scaling factor size to bottom row axes
        ax.xaxis.get_offset_text().set_fontsize(factor_size)
        ax.yaxis.get_offset_text().set_fontsize(factor_size)

    # Optional: Remove unused axes if n_plots < 8
    for j in range(n_plots, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()



 
   #######################################
if __name__ == "__main__":
    main()
