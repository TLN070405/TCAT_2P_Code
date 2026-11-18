import sys
import math
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
#######################################
def main():
    Omega_0 = 0.
    Omega_L = 1.
    nodes_array= [51, 101, 201]
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
        Ve_w_Time_TCAT = np.genfromtxt(f'e_w_{node}.csv', delimiter=',', dtype=str)
        Vp_w_Time_TCAT  = np.genfromtxt(f'p_w_{node}.csv', delimiter=',', dtype=str)
        Ve_wn_Time_TCAT = np.genfromtxt(f'e_wn_{node}.csv', delimiter=',', dtype=str)
        Vj_wn_Time_TCAT = np.genfromtxt(f'j_wn_{node}.csv', delimiter=',', dtype=str)
        Vw_wn_Time_TCAT = np.genfromtxt(f'w_wn_{node}.csv', delimiter=',', dtype=str)
        Vk_wn_Time_TCAT = np.genfromtxt(f'k_wn_{node}.csv', delimiter=',', dtype=str)

        Vew_normalized_TCAT  = np.genfromtxt(f'e_w_normalized_{node}.csv', delimiter=',', dtype=str)
        Vpw_normalized_TCAT  = np.genfromtxt(f'p_w_normalized_{node}.csv', delimiter=',', dtype=str)
        Vewn_normalized_TCAT = np.genfromtxt(f'e_wn_normalized_{node}.csv', delimiter=',', dtype=str)
        Vjwn_normalized_TCAT = np.genfromtxt(f'j_wn_normalized_{node}.csv', delimiter=',', dtype=str)
        Vkwn_normalized_TCAT = np.genfromtxt(f'k_wn_normalized_{node}.csv', delimiter=',', dtype=str)
        Vtime_TCAT           = np.genfromtxt(f'time_steps_array_{node}.csv', delimiter=',', dtype=str)

        time_interval =np.genfromtxt(f'time_{node}.csv', delimiter=',', dtype=str)

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
        Ve_w_Time_RE = np.genfromtxt(f'e_w_re_{node}.csv', delimiter=',', dtype=str)
        Vp_w_Time_RE  = np.genfromtxt(f'p_w_re_{node}.csv', delimiter=',', dtype=str)

        Vew_normalized_RE  = np.genfromtxt(f'e_w_normalized_re_{node}.csv', delimiter=',', dtype=str)
        Vpw_normalized_RE  = np.genfromtxt(f'p_w_normalized_re_{node}.csv', delimiter=',', dtype=str)

        Vtime_RE           = np.genfromtxt(f'time_steps_array_re_{node}.csv', delimiter=',', dtype=str)
        time_interval_RE   = np.genfromtxt(f'time_re_{node}.csv', delimiter=',', dtype=str)


        Ve_w_Time_RE  = Ve_w_Time_RE.astype(float)
        Vp_w_Time_RE  = Vp_w_Time_RE.astype(float)

        Vew_normalized_RE  = Vew_normalized_RE.astype(float)
        Vpw_normalized_RE  = Vpw_normalized_RE.astype(float)

        Vtime_RE           = Vtime_RE.astype(float)
        time_interval_RE   = time_interval_RE.astype(float)


        Ve_w_Time_RE_all.append(Ve_w_Time_RE)
        Vp_w_Time_RE_all.append(Vp_w_Time_RE)

    # calculate L2 error norm: (e_w[time interval][node])
    # Ve_w_Time_TCAT_all[all_nodes][time][node]
    
    L2e_ew_tcat =  np.zeros([len(nodes_array)-1,len(time_interval)] )
    for i in range(0,len(nodes_array)-1):
        exact_index = int((nodes_array[-1]-1)/(nodes_array[i]-1))
        for t in range(0,len(time_interval)):
            for j in range(0,nodes_array[i]):
                L2e_ew_tcat[i][t]= L2e_ew_tcat[i][t] + ((Ve_w_Time_TCAT_all[i+1][t][2*j]- Ve_w_Time_TCAT_all[i][t][j])/Ve_w_Time_TCAT_all[-1][t][exact_index*j])**2
       
        L2e_ew_tcat[i][t] = math.sqrt(L2e_ew_tcat[i][t]/nodes_array[i])

    # Save to CSV
    # Define row and column headers
    row_headers = [f'{nodes_array[i]}--{nodes_array[i+1]}' for i in range(len(nodes_array)-1)]
    col_headers = [f'{time_interval[i]}' for i in range(len(time_interval)) ]

    # Create DataFrame
    df = pd.DataFrame(L2e_ew_tcat, index=row_headers, columns=col_headers)

    # Save to CSV with scientific notation and headers
    df.to_csv('L2e_ew_tcat.csv', float_format='%.6e', index_label='time/nodes')





    # calculate L2 error norm RE: (e_w[time interval][node])
    # Ve_w_Time_RE_all[all_nodes][time][node]

    L2e_ew_re =  np.zeros([len(nodes_array)-1,len(time_interval_RE)] )
    for i in range(0,len(nodes_array)-1):
        exact_index = int((nodes_array[-1]-1)/(nodes_array[i]-1))
        for t in range(0,len(time_interval_RE)):
            for j in range(0,nodes_array[i]):
                L2e_ew_re[i][t]= L2e_ew_re[i][t] + ((Ve_w_Time_RE_all[i+1][t][2*j]- Ve_w_Time_RE_all[i][t][j])/Ve_w_Time_RE_all[-1][t][exact_index*j])**2
       
        L2e_ew_re[i][t] = math.sqrt(L2e_ew_re[i][t]/nodes_array[i])

    # Save to CSV
    # Define row and column headers
    row_headers = [f'{nodes_array[i]}--{nodes_array[i+1]}' for i in range(len(nodes_array)-1)]
    col_headers = [f'{time_interval_RE[i]}' for i in range(len(time_interval_RE)) ]

    # Create DataFrame
    df = pd.DataFrame(L2e_ew_re, index=row_headers, columns=col_headers)

    # Save to CSV with scientific notation and headers
    df.to_csv('L2e_ew_re.csv', float_format='%.6e', index_label='time/nodes')
    #####################################################################################################################
    ###############################################Symetry###########################################################
    #####################################################################################################################
    
    
    
    
    
    ###### e_w symmetry quantification
    half_nodes=  int((nodes_array[-1]-1)/2)
    err_symmetry_ew = np.zeros([len(time_interval)] )

    for t in range(0,len(time_interval)):
        for i in range(0,half_nodes):
            err_symmetry_ew[t] = err_symmetry_ew[t] + (  Ve_w_Time_TCAT[t][i]-Ve_w_Time_TCAT[t][nodes_array[-1]-1-i]   )**2

        err_symmetry_ew[t] = math.sqrt(err_symmetry_ew[t]/half_nodes)

    # Save to CSV with scientific notation and headers
    # Create DataFrame
    df = pd.DataFrame(err_symmetry_ew)
    df.to_csv('err_symmetry_ew.csv', float_format='%.6e', index_label='time/error')

    ###### p_w symmetry quantification
    half_nodes=  int((nodes_array[-1]-1)/2)
    err_symmetry_pw = np.zeros([len(time_interval)] )

    for t in range(0,len(time_interval)):
        for i in range(0,half_nodes):
            err_symmetry_pw[t] = err_symmetry_pw[t] + (  Vp_w_Time_TCAT[t][i]-Vp_w_Time_TCAT[t][nodes_array[-1]-1-i]   )**2

        err_symmetry_pw[t] = math.sqrt(err_symmetry_pw[t]/half_nodes)

    # Save to CSV with scientific notation and headers
    # Create DataFrame
    df = pd.DataFrame(err_symmetry_pw)
    df.to_csv('err_symmetry_pw.csv', float_format='%.6e', index_label='time/error')

    ###### e_wn symmetry quantification
    half_nodes=  int((nodes_array[-1]-1)/2)
    err_symmetry_ewn = np.zeros([len(time_interval)] )

    for t in range(0,len(time_interval)):
        for i in range(0,half_nodes):
            err_symmetry_ewn[t] = err_symmetry_ewn[t] + (  Ve_wn_Time_TCAT[t][i]-Ve_wn_Time_TCAT[t][nodes_array[-1]-1-i]   )**2

        err_symmetry_ewn[t] = math.sqrt(err_symmetry_ewn[t]/half_nodes)

    # Save to CSV with scientific notation and headers
    # Create DataFrame
    df = pd.DataFrame(err_symmetry_ewn)
    df.to_csv('err_symmetry_ewn.csv', float_format='%.6e', index_label='time/error')

    ###### j_wn symmetry quantification
    half_nodes=  int((nodes_array[-1]-1)/2)
    err_symmetry_jwn = np.zeros([len(time_interval)] )

    for t in range(0,len(time_interval)):
        for i in range(0,half_nodes):
            err_symmetry_jwn[t] = err_symmetry_jwn[t] + (  Vj_wn_Time_TCAT[t][i]-Vj_wn_Time_TCAT[t][nodes_array[-1]-1-i]   )**2

        err_symmetry_jwn[t] = math.sqrt(err_symmetry_jwn[t]/half_nodes)

    # Save to CSV with scientific notation and headers
    # Create DataFrame
    df = pd.DataFrame(err_symmetry_jwn)
    df.to_csv('err_symmetry_jwn.csv', float_format='%.6e', index_label='time/error')

    ###### k_wn symmetry quantification
    half_nodes=  int((nodes_array[-1]-1)/2)
    err_symmetry_kwn = np.zeros([len(time_interval)] )

    for t in range(0,len(time_interval)):
        for i in range(0,half_nodes):
            err_symmetry_kwn[t] = err_symmetry_kwn[t] + (  Vk_wn_Time_TCAT[t][i]-Vk_wn_Time_TCAT[t][nodes_array[-1]-1-i]   )**2

        err_symmetry_kwn[t] = math.sqrt(err_symmetry_kwn[t]/half_nodes)

    # Save to CSV with scientific notation and headers
    # Create DataFrame
    df = pd.DataFrame(err_symmetry_kwn)
    df.to_csv('err_symmetry_kwn.csv', float_format='%.6e', index_label='time/error')



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
        ax.plot(mesh, Ve_w_Time_TCAT[i], label=f'TCAT CGP; t(s)={time_interval[i]:g}', linestyle='-', color=colors[i])

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
        ax.plot(mesh, Vp_w_Time_TCAT[i], label=f'TCAT CGP; t(s)={time_interval[i]:g}', linestyle='-', color=colors[i])

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
    #####visualization   Ewn


    colors = plt.cm.plasma(np.linspace(0.3, 1.0, len(time_interval)))

    fig, ax = plt.subplots(figsize=(12, 6))

    for i in range(len(time_interval)):
        ax.plot(mesh, Ve_wn_Time_TCAT[i], label=f'TCAT CGP; t(s)={time_interval[i]:g}', linestyle='-', color=colors[i])

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
        ax.plot(mesh, Vj_wn_Time_TCAT[i], label=f'TCAT CGP; t(s)={time_interval[i]:g}', linestyle='-', color=colors[i])

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
    
    #####visualization   Wwn
    colors = plt.cm.plasma(np.linspace(0.3, 1.0, len(time_interval)))

    fig, ax = plt.subplots(figsize=(12, 6))

    for i in range(len(time_interval)):
        ax.plot(mesh, Vw_wn_Time_TCAT[i], label=f'TCAT CGP; t(s)={time_interval[i]:g}', linestyle='-', color=colors[i])

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
        ax.set_title(f't = {time_interval[i]:g} s', fontsize=16)
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
        ax.plot(mesh, Vk_wn_Time_TCAT[i], label=f'TCAT CGP; t(s)={time_interval[i]:g}', linestyle='-', color=colors[i])

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
    plt.plot(Vtime_TCAT, Vew_normalized_TCAT,label='TCAT e_w')
    plt.plot(Vtime_TCAT, Vpw_normalized_TCAT,label='TCAT p_w')
    plt.plot(Vtime_TCAT, Vewn_normalized_TCAT ,label='TCAT e_wn')
    plt.plot(Vtime_TCAT, Vjwn_normalized_TCAT,label='TCAT j_wn')
    plt.plot(Vtime_TCAT, Vkwn_normalized_TCAT,label='TCAT k_wn')
    #plt.plot(Vtime_RE, Vew_normalized_RE,label='RE e_w')
    #plt.plot(Vtime_RE, Vpw_normalized_RE,label='RE p_w')
    plt.legend(fontsize=fs_legend)
    plt.xlabel('time'r' $[s]$',fontsize=fs)
    plt.ylabel('normalized quantity'r' $[-]$',fontsize=fs)
    plt.show()



    # RE visualization
    # Visualization: Volume fraction ew
    colors = plt.cm.Blues(np.linspace(0.3, 1.0, len(time_interval)))

    fig, ax = plt.subplots(figsize=(12, 6))

    for i in range(len(time_interval)):
        ax.plot(mesh, Ve_w_Time_RE[i], label=f'RE; t(s)={time_interval[i]:g}', linestyle='-', color=colors[i])

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
        ax.plot(mesh, Vp_w_Time_RE[i], label=f'RE; t(s)={time_interval[i]:g}', linestyle='-', color=colors[i])

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
   #######################################
if __name__ == "__main__":
    main()
