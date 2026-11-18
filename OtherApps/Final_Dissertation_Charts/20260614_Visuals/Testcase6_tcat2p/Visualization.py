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
    nodes_array= [51 ,101]
    #### read results
    # Read the data as strings first
    Ve_w_Time_TCAT_all = []
    Vp_w_Time_TCAT_all = []
    Vp_n_Time_TCAT_all = []
    Ve_wn_Time_TCAT_all = []
    Vj_wn_Time_TCAT_all = []
    Vg_wn_Time_TCAT_all = []
    Vw_wn_Time_TCAT_all = []
    Vk_wn_Time_TCAT_all = []

    Ve_w_Time_2P_all = []
    Vp_w_Time_2P_all = []
    Vp_n_Time_2P_all = []

    for node in nodes_array:
        Ve_w_Time_TCAT = np.genfromtxt(f'results_full_tcat/e_w_{node}.csv', delimiter=',', dtype=str)
        Vp_w_Time_TCAT  = np.genfromtxt(f'results_full_tcat/p_w_{node}.csv', delimiter=',', dtype=str)
        Vp_n_Time_TCAT  = np.genfromtxt(f'results_full_tcat/p_n_{node}.csv', delimiter=',', dtype=str)
        Ve_wn_Time_TCAT = np.genfromtxt(f'results_full_tcat/e_wn_{node}.csv', delimiter=',', dtype=str)
        Vj_wn_Time_TCAT = np.genfromtxt(f'results_full_tcat/j_wn_{node}.csv', delimiter=',', dtype=str)
        Vg_wn_Time_TCAT = np.genfromtxt(f'results_full_tcat/g_wn_{node}.csv', delimiter=',', dtype=str)
        Vw_wn_Time_TCAT = np.genfromtxt(f'results_full_tcat/w_wn_{node}.csv', delimiter=',', dtype=str)
        Vk_wn_Time_TCAT = np.genfromtxt(f'results_full_tcat/k_wn_{node}.csv', delimiter=',', dtype=str)

        Vew_normalized_TCAT  = np.genfromtxt(f'results_full_tcat/e_w_normalized_{node}.csv', delimiter=',', dtype=str)
        Vpw_normalized_TCAT  = np.genfromtxt(f'results_full_tcat/p_w_normalized_{node}.csv', delimiter=',', dtype=str)
        Vpn_normalized_TCAT  = np.genfromtxt(f'results_full_tcat/p_n_normalized_{node}.csv', delimiter=',', dtype=str)
        Vewn_normalized_TCAT = np.genfromtxt(f'results_full_tcat/e_wn_normalized_{node}.csv', delimiter=',', dtype=str)
        Vjwn_normalized_TCAT = np.genfromtxt(f'results_full_tcat/j_wn_normalized_{node}.csv', delimiter=',', dtype=str)
        Vgwn_normalized_TCAT = np.genfromtxt(f'results_full_tcat/g_wn_normalized_{node}.csv', delimiter=',', dtype=str)
        Vkwn_normalized_TCAT = np.genfromtxt(f'results_full_tcat/k_wn_normalized_{node}.csv', delimiter=',', dtype=str)
        
        Vtime_TCAT           = np.genfromtxt(f'results_full_tcat/time_steps_array_{node}.csv', delimiter=',', dtype=str)
        time_interval  = np.genfromtxt(f'results_full_tcat/time_{node}.csv', delimiter=',', dtype=str)


        Ve_w_Time_TCAT  = Ve_w_Time_TCAT.astype(float)
        Vp_w_Time_TCAT  = Vp_w_Time_TCAT.astype(float)
        Vp_n_Time_TCAT  = Vp_n_Time_TCAT.astype(float)
        Ve_wn_Time_TCAT = Ve_wn_Time_TCAT.astype(float)
        Vj_wn_Time_TCAT = Vj_wn_Time_TCAT.astype(float)
        Vg_wn_Time_TCAT = Vg_wn_Time_TCAT.astype(float)
        Vw_wn_Time_TCAT = Vw_wn_Time_TCAT.astype(float)
        Vk_wn_Time_TCAT = Vk_wn_Time_TCAT.astype(float)

        Vew_normalized_TCAT  = Vew_normalized_TCAT.astype(float)
        Vpw_normalized_TCAT  = Vpw_normalized_TCAT.astype(float)
        Vpn_normalized_TCAT  = Vpn_normalized_TCAT.astype(float)
        Vewn_normalized_TCAT = Vewn_normalized_TCAT.astype(float)
        Vjwn_normalized_TCAT = Vjwn_normalized_TCAT.astype(float)
        Vgwn_normalized_TCAT = Vgwn_normalized_TCAT.astype(float)
        Vkwn_normalized_TCAT = Vkwn_normalized_TCAT.astype(float)
        Vtime_TCAT           = Vtime_TCAT.astype(float)
        time_interval   = time_interval.astype(float)
        

        Ve_w_Time_TCAT_all.append(Ve_w_Time_TCAT)
        Vp_w_Time_TCAT_all.append(Vp_w_Time_TCAT)
        Vp_n_Time_TCAT_all.append(Vp_n_Time_TCAT)
        Ve_wn_Time_TCAT_all.append(Ve_wn_Time_TCAT)
        Vj_wn_Time_TCAT_all.append(Vj_wn_Time_TCAT)
        Vg_wn_Time_TCAT_all.append(Vg_wn_Time_TCAT)
        Vw_wn_Time_TCAT_all.append(Vw_wn_Time_TCAT)
        Vk_wn_Time_TCAT_all.append(Vk_wn_Time_TCAT)
    
        # 2P results

        Ve_w_Time_2P    = np.genfromtxt(f'results_2p/e_w_2p_{node}.csv', delimiter=',', dtype=str)
        Vp_w_Time_2P    = np.genfromtxt(f'results_2p/p_w_2p_{node}.csv', delimiter=',', dtype=str)
        Vp_n_Time_2P    = np.genfromtxt(f'results_2p/p_n_2p_{node}.csv', delimiter=',', dtype=str)

        Vew_normalized_2P  = np.genfromtxt(f'results_2p/e_w_normalized_2p_{node}.csv', delimiter=',', dtype=str)
        Vpw_normalized_2P  = np.genfromtxt(f'results_2p/p_w_normalized_2p_{node}.csv', delimiter=',', dtype=str)
        Vpn_normalized_2P  = np.genfromtxt(f'results_2p/p_n_normalized_2p_{node}.csv', delimiter=',', dtype=str)
        Vtime_2P           = np.genfromtxt(f'results_2p/time_steps_array_2p_{node}.csv', delimiter=',', dtype=str)
        time_interval_2P   = np.genfromtxt(f'results_2p/time_2p_{node}.csv', delimiter=',', dtype=str)



        Ve_w_Time_2P  = Ve_w_Time_2P.astype(float)
        Vp_w_Time_2P  = Vp_w_Time_2P.astype(float)
        Vp_n_Time_2P  = Vp_n_Time_2P.astype(float)

        Vew_normalized_2P  = Vew_normalized_2P.astype(float)
        Vpw_normalized_2P  = Vpw_normalized_2P.astype(float)
        Vpn_normalized_2P  = Vpn_normalized_2P.astype(float)

        Vtime_2P           = Vtime_2P.astype(float)
        time_interval_2P   = time_interval_2P.astype(float)


        Ve_w_Time_2P_all.append(Ve_w_Time_2P)
        Vp_w_Time_2P_all.append(Vp_w_Time_2P)
        Vp_n_Time_2P_all.append(Vp_n_Time_2P)


    ########################################### Grid resolution error
    # full tcat error
    calculate_grid_resolution_error(nodes_array, time_interval, Ve_w_Time_TCAT_all, 'L2e_ew_tcat')
    calculate_grid_resolution_error(nodes_array, time_interval, Vp_w_Time_TCAT_all, 'L2e_pw_tcat')
    calculate_grid_resolution_error(nodes_array, time_interval, Vp_n_Time_TCAT_all, 'L2e_pn_tcat')
    calculate_grid_resolution_error(nodes_array, time_interval, Ve_wn_Time_TCAT_all, 'L2e_ewn_tcat')
    calculate_grid_resolution_error(nodes_array, time_interval, Vj_wn_Time_TCAT_all, 'L2e_jwn_tcat')
    calculate_grid_resolution_error(nodes_array, time_interval, Vg_wn_Time_TCAT_all, 'L2e_gwn_tcat')
    calculate_grid_resolution_error(nodes_array, time_interval, Vw_wn_Time_TCAT_all, 'L2e_wwn_tcat')
    calculate_grid_resolution_error(nodes_array, time_interval, Vk_wn_Time_TCAT_all, 'L2e_kwn_tcat')

    # Calculate L2 error norms for 2P data
    calculate_grid_resolution_error(nodes_array, time_interval_2P, Ve_w_Time_2P_all, 'L2e_ew_2p')
    calculate_grid_resolution_error(nodes_array, time_interval_2P, Vp_w_Time_2P_all, 'L2e_pw_2p')
    calculate_grid_resolution_error(nodes_array, time_interval_2P, Vp_n_Time_2P_all, 'L2e_pn_2p')

    ########################################### Symmetry error

    # Call the function for each symmetry quantification
    calculate_symmetry_error(time_interval, nodes_array, Ve_w_Time_TCAT, 'err_symmetry_ew',1)
    calculate_symmetry_error(time_interval, nodes_array, Vp_w_Time_TCAT, 'err_symmetry_pw',1)
    calculate_symmetry_error(time_interval, nodes_array, Vp_n_Time_TCAT, 'err_symmetry_pn',1)
    calculate_symmetry_error(time_interval, nodes_array, Ve_wn_Time_TCAT, 'err_symmetry_ewn',1)
    calculate_symmetry_error(time_interval, nodes_array, Vj_wn_Time_TCAT, 'err_symmetry_jwn',1)
    calculate_symmetry_error(time_interval, nodes_array, Vg_wn_Time_TCAT, 'err_symmetry_gwn',1)
    calculate_symmetry_error(time_interval, nodes_array, Vk_wn_Time_TCAT, 'err_symmetry_kwn',1)
    calculate_symmetry_error(time_interval, nodes_array, Vw_wn_Time_TCAT, 'err_anti_symmetry_wwn',-1)



    #####################################################################################################################
    ###############################################Visualization###########################################################
    #####################################################################################################################
    
    mesh = np.linspace(Omega_0, Omega_L, nodes_array[-1]) # mesh just for plots

    ######################PlotsPlotsPlotsPlots#######################
    ######################PlotsPlotsPlotsPlots#######################
    ######################PlotsPlotsPlotsPlots#######################

    #mpl.rcParams['text.usetex'] = True
    plt.rcParams.update({ "font.family": "monospace", "font.monospace": "Computer Modern Typewriter"})
    fs=20
    fs_legend=20
    fs_axes=20
    fs_ticks =20
    plt.xticks(fontsize=fs_axes)  # Increase font size of X-axis values
    plt.yticks(fontsize=fs_axes)  # Increase font size of Y-axis values


    # Visualization: Volume fraction ew
    colors = plt.cm.Blues(np.linspace(0.3, 1.0, len(time_interval)))

    fig, ax = plt.subplots(figsize=(12, 6))

    for i in range(len(time_interval)):
        ax.plot(mesh, Ve_w_Time_TCAT[i], label=f'Full TCAT, t={time_interval[i]:g} sec', linestyle='-', color=colors[i])

    # Axis labels
    ax.set_xlabel(r'$x\ [m]$', fontsize=fs)
    ax.set_ylabel(r'$\epsilon^{\overline{\overline{w}}}\ [-]$', fontsize=fs)

    # Increase tick label font size
    ax.tick_params(axis='both', labelsize=fs_ticks)

    # Shrink the plot to the left to make space for the legend
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.75, box.height])  # 75% width for plot

    # Put legend to the right, centered vertically
    ax.legend(fontsize=fs_legend, loc='center left', bbox_to_anchor=(1.02, 0.5), handlelength=2, borderaxespad=0.5)

    plt.show()
    #####visualization  pressure pw

    colors = plt.cm.Greens(np.linspace(0.3, 1.0, len(time_interval)))

    fig, ax = plt.subplots(figsize=(12, 6))

    for i in range(len(time_interval)):
        ax.plot(mesh, Vp_w_Time_TCAT[i], label=f'Full TCAT, t={time_interval[i]:g} sec', linestyle='-', color=colors[i])

    # Axis labels
    ax.set_xlabel(r'$x\ [m]$', fontsize=fs)
    ax.set_ylabel(r' $p^w$'r' $[Pa]$',fontsize=fs)

    # Increase tick label font size
    ax.tick_params(axis='both', labelsize=fs_ticks)

    # Shrink the plot to the left to make space for the legend
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.75, box.height])  # 75% width for plot

    # Put legend to the right, centered vertically
    ax.legend(fontsize=fs_legend, loc='center left', bbox_to_anchor=(1.02, 0.5), handlelength=2, borderaxespad=0.5)

    plt.show()

    #####visualization  pressure pn

    colors = plt.cm.Greens(np.linspace(0.3, 1.0, len(time_interval)))

    fig, ax = plt.subplots(figsize=(12, 6))

    for i in range(len(time_interval)):
        ax.plot(mesh, Vp_n_Time_TCAT[i], label=f'Full TCAT, t={time_interval[i]:g} sec', linestyle='-', color=colors[i])

    # Axis labels
    ax.set_xlabel(r'$x\ [m]$', fontsize=fs)
    ax.set_ylabel(r' $p^n$'r' $[Pa]$',fontsize=fs)

    # Increase tick label font size
    ax.tick_params(axis='both', labelsize=fs_ticks)

    # Shrink the plot to the left to make space for the legend
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.75, box.height])  # 75% width for plot

    # Put legend to the right, centered vertically
    ax.legend(fontsize=fs_legend, loc='center left', bbox_to_anchor=(1.02, 0.5), handlelength=2, borderaxespad=0.5)

    plt.show()

    #####visualization   Ewn


    colors = plt.cm.plasma(np.linspace(0.3, 1.0, len(time_interval)))

    fig, ax = plt.subplots(figsize=(12, 6))

    for i in range(len(time_interval)):
        ax.plot(mesh, Ve_wn_Time_TCAT[i], label=f'Full TCAT, t={time_interval[i]:g} sec', linestyle='-', color=colors[i])

    # Axis labels
    ax.set_xlabel(r'$x\ [m]$', fontsize=fs)
    ax.set_ylabel(r'$\epsilon^{\overline{\overline{wn}}} $'r' $[1/m]$',fontsize=fs)

    # Increase tick label font size
    ax.tick_params(axis='both', labelsize=fs_ticks)

    # Shrink the plot to the left to make space for the legend
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.75, box.height])  # 75% width for plot

    # Put legend to the right, centered vertically
    ax.legend(fontsize=fs_legend, loc='center left', bbox_to_anchor=(1.02, 0.5), handlelength=2, borderaxespad=0.5)

    plt.show()

    #####visualization   Jwn
    colors = plt.cm.plasma(np.linspace(0.3, 1.0, len(time_interval)))

    fig, ax = plt.subplots(figsize=(12, 6))

    for i in range(len(time_interval)):
        ax.plot(mesh, Vj_wn_Time_TCAT[i], label=f'Full TCAT, t={time_interval[i]:g} sec', linestyle='-', color=colors[i])

    # Axis labels
    ax.set_xlabel(r'$x\ [m]$', fontsize=fs)
    ax.set_ylabel(r'$J^{wn}_n$'r' $[1/m]$',fontsize=fs)

    # Increase tick label font size
    ax.tick_params(axis='both', labelsize=fs_ticks)

    # Shrink the plot to the left to make space for the legend
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.75, box.height])  # 75% width for plot

    # Put legend to the right, centered vertically
    ax.legend(fontsize=fs_legend, loc='center left', bbox_to_anchor=(1.02, 0.5), handlelength=2, borderaxespad=0.5)

    plt.show()


    #####visualization   Gwn
    colors = plt.cm.plasma(np.linspace(0.3, 1.0, len(time_interval)))

    fig, ax = plt.subplots(figsize=(12, 6))

    for i in range(len(time_interval)):
        ax.plot(mesh, Vg_wn_Time_TCAT[i], label=f'Full TCAT, t={time_interval[i]:g} sec', linestyle='-', color=colors[i])

    # Axis labels
    ax.set_xlabel(r'$x\ [m]$', fontsize=fs)
    ax.set_ylabel(r'$G^{wn}_{xx}$'r' $[1/m]$',fontsize=fs)

    # Increase tick label font size
    ax.tick_params(axis='both', labelsize=fs_ticks)

    # Shrink the plot to the left to make space for the legend
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.75, box.height])  # 75% width for plot

    # Put legend to the right, centered vertically
    ax.legend(fontsize=fs_legend, loc='center left', bbox_to_anchor=(1.02, 0.5), handlelength=2, borderaxespad=0.5)

    plt.show()


    #####visualization   Wwn
    colors = plt.cm.plasma(np.linspace(0.3, 1.0, len(time_interval)))

    fig, ax = plt.subplots(figsize=(12, 6))

    for i in range(len(time_interval)):
        ax.plot(mesh, Vw_wn_Time_TCAT[i], label=f'Full TCAT, t={time_interval[i]:g} sec', linestyle='-', color=colors[i])

    # Axis labels
    ax.set_xlabel(r'$x\ [m]$', fontsize=fs)
    ax.set_ylabel(r'$w^{wn}_x$'r' $[m/s]$', fontsize=fs)

    # Increase tick label font size
    ax.tick_params(axis='both', labelsize=fs_ticks)

    # Legend outside plot
    ax.legend(fontsize=fs_legend, loc='center left', bbox_to_anchor=(1.02, 0.5), handlelength=2, borderaxespad=0.5)

    # Zoomed inset axes
    axins = inset_axes(ax, width="35%", height="35%", loc='upper right', borderpad=2)

    # Define zoom region
    x1, x2 = 0.1, 0.9 # example x-range to zoom into
    y1, y2 = -8e-10, 8e-10  # example y-range to zoom into

    for i in range(len(time_interval)):
        axins.plot(mesh, Vw_wn_Time_TCAT[i], linestyle='-', color=colors[i])

    axins.set_xlim(x1, x2)
    axins.set_ylim(y1, y2)
    axins.set_xticks([])
    axins.set_yticks([])

    # Add a box on the main plot to show where zoom is
    mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5")

    # Shrink the plot to the left to make space for the legend
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.75, box.height])  # 75% width for plot

    # Put legend to the right, centered vertically
    ax.legend(fontsize=fs_legend, loc='center left', bbox_to_anchor=(1.02, 0.5), handlelength=2, borderaxespad=0.5)

    plt.show()

    #####visualization   w_wn subplots

    n_plots = len(time_interval)
    colors = plt.cm.plasma(np.linspace(0.3, 1.0, n_plots))

    # Create subplots: 4 rows, 2 columns
    fig, axes = plt.subplots(nrows=4, ncols=2, figsize=(14, 12), sharex=True)

    # Flatten axes array to make indexing easier
    axes = axes.flatten()

    for i in range(n_plots):
        ax = axes[i]
        ax.plot(mesh, Vw_wn_Time_TCAT[i], color=colors[i], linestyle='-')
        ax.set_title(f't = {time_interval[i]:g} sec', fontsize=16)
        ax.set_ylabel(r'$w^{wn}_x$'r' $[m/s]$', fontsize=fs)
        ax.tick_params(axis='both', labelsize=fs_ticks)

    # Set shared x-axis label for the bottom row
    for ax in axes[-2:]:
        ax.set_xlabel(r'$x\ [m]$', fontsize=fs)

    # Remove unused subplots if any (not needed in your case since 8 plots fit exactly in 4x2)
    # But if n_plots < 8, you'd do:
    # for j in range(n_plots, len(axes)):
    #     fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()


    #####visualization   Kwn
    colors = plt.cm.plasma(np.linspace(0.3, 1.0, len(time_interval)))

    fig, ax = plt.subplots(figsize=(12, 6))

    for i in range(len(time_interval)):
        ax.plot(mesh, Vk_wn_Time_TCAT[i], label=f'Full TCAT, t={time_interval[i]:g} sec', linestyle='-', color=colors[i])

    # Axis labels
    ax.set_xlabel(r'$x\ [m]$', fontsize=fs)
    ax.set_ylabel(r'$K^{wn}_n$'r' $[1/m^2]$',fontsize=fs)

    # Increase tick label font size
    ax.tick_params(axis='both', labelsize=fs_ticks)

    # Shrink the plot to the left to make space for the legend
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.75, box.height])  # 75% width for plot

    # Put legend to the right, centered vertically
    ax.legend(fontsize=fs_legend, loc='center left', bbox_to_anchor=(1.02, 0.5), handlelength=2, borderaxespad=0.5)

    plt.show()






    # convergence of variables
    plt.plot(Vtime_TCAT, Vew_normalized_TCAT,label=r'TCAT CGP $\epsilon^{\overline{\overline{w}}}$')
    plt.plot(Vtime_TCAT, Vpw_normalized_TCAT,label=r'TCAT CGP $p^w$')
    #plt.plot(Vtime_TCAT, Vpn_normalized_TCAT,label=r'TCAT CGP $p^n$')
    plt.plot(Vtime_TCAT, Vewn_normalized_TCAT ,label=r'TCAT CGP $\epsilon^{\overline{\overline{wn}}}$')
    plt.plot(Vtime_TCAT, Vjwn_normalized_TCAT,label=r'TCAT CGP $J^{wn}_n$')
    plt.plot(Vtime_TCAT, Vgwn_normalized_TCAT,label=r'TCAT CGP $G^{wn}_{xx}$')
    plt.plot(Vtime_TCAT, Vkwn_normalized_TCAT,label=r'TCAT CGP $K^{wn}_n$')
    plt.legend(fontsize=fs_legend)
    plt.xlabel('time'r' $[s]$',fontsize=fs)
    plt.ylabel('normalized quantity'r' $[-]$',fontsize=fs)
    plt.show()



    # 2P visualization
    # Visualization: Volume fraction ew
    colors = plt.cm.Blues(np.linspace(0.3, 1.0, len(time_interval)))

    fig, ax = plt.subplots(figsize=(12, 6))

    for i in range(len(time_interval)):
        ax.plot(mesh, Ve_w_Time_2P[i], label=f'2P, t={time_interval[i]:g} sec', linestyle='-', color=colors[i])

    # Axis labels
    ax.set_xlabel(r'$x\ [m]$', fontsize=fs)
    ax.set_ylabel(r'$\epsilon^{\overline{\overline{w}}}\ [-]$', fontsize=fs)

    # Increase tick label font size
    ax.tick_params(axis='both', labelsize=fs_ticks)

    # Shrink the plot to the left to make space for the legend
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.75, box.height])  # 75% width for plot

    # Put legend to the right, centered vertically
    ax.legend(fontsize=fs_legend, loc='center left', bbox_to_anchor=(1.02, 0.5), handlelength=2, borderaxespad=0.5)

    plt.show()
    #####visualization  pressure pw

    colors = plt.cm.Greens(np.linspace(0.3, 1.0, len(time_interval)))

    fig, ax = plt.subplots(figsize=(12, 6))

    for i in range(len(time_interval)):
        ax.plot(mesh, Vp_w_Time_2P[i], label=f'2P, t={time_interval[i]:g} sec', linestyle='-', color=colors[i])

    # Axis labels
    ax.set_xlabel(r'$x\ [m]$', fontsize=fs)
    ax.set_ylabel(r' $p^w$'r' $[Pa]$',fontsize=fs)

    # Increase tick label font size
    ax.tick_params(axis='both', labelsize=fs_ticks)

    # Shrink the plot to the left to make space for the legend
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.75, box.height])  # 75% width for plot

    # Put legend to the right, centered vertically
    ax.legend(fontsize=fs_legend, loc='center left', bbox_to_anchor=(1.02, 0.5), handlelength=2, borderaxespad=0.5)

    plt.show()

    #####visualization  pressure pn

    colors = plt.cm.Greens(np.linspace(0.3, 1.0, len(time_interval)))

    fig, ax = plt.subplots(figsize=(12, 6))

    for i in range(len(time_interval)):
        ax.plot(mesh, Vp_n_Time_2P[i], label=f'2P, t={time_interval[i]:g} sec', linestyle='-', color=colors[i])

    # Axis labels
    ax.set_xlabel(r'$x\ [m]$', fontsize=fs)
    ax.set_ylabel(r' $p^w$'r' $[Pa]$',fontsize=fs)

    # Increase tick label font size
    ax.tick_params(axis='both', labelsize=fs_ticks)

    # Shrink the plot to the left to make space for the legend
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.75, box.height])  # 75% width for plot

    # Put legend to the right, centered vertically
    ax.legend(fontsize=fs_legend, loc='center left', bbox_to_anchor=(1.02, 0.5), handlelength=2, borderaxespad=0.5)

    plt.show()

    # convergence of variables
    plt.plot(Vtime_TCAT, Vew_normalized_TCAT,label=r'TCAT CGP $\epsilon^{\overline{\overline{w}}}$')
    plt.plot(Vtime_TCAT, Vpw_normalized_TCAT,label=r'TCAT CGP $p^w$')
    #plt.plot(Vtime_TCAT, Vpn_normalized_TCAT,label=r'TCAT CGP $p^n$')
    plt.plot(Vtime_TCAT, Vewn_normalized_TCAT ,label=r'TCAT CGP $\epsilon^{\overline{\overline{wn}}}$')
    plt.plot(Vtime_TCAT, Vjwn_normalized_TCAT,label=r'TCAT CGP $J^{wn}_n$')
    plt.plot(Vtime_TCAT, Vgwn_normalized_TCAT,label=r'TCAT CGP $G^{wn}_{xx}$')
    plt.plot(Vtime_TCAT, Vkwn_normalized_TCAT,label=r'TCAT CGP $K^{wn}_n$')
    plt.plot(Vtime_2P, Vew_normalized_2P,label=r'2P $\epsilon^{\overline{\overline{w}}}$', linestyle='--')
    plt.plot(Vtime_2P, Vpw_normalized_2P,label=r'2P $p^w$', linestyle='--')
    #plt.plot(Vtime_2P, Vpn_normalized_2P,label=r'2P $p^n$', linestyle='--')
    plt.legend(fontsize=fs_legend)
    plt.xlabel('time'r' $[s]$',fontsize=fs)
    plt.ylabel('normalized quantity'r' $[-]$',fontsize=fs)
    plt.show()
   #######################################
if __name__ == "__main__":
    main()
