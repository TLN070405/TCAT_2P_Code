import sys
import math
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd

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

        Nx = len(Ve_w_Time_TCAT[0])

        node_i=int((Nx-1)/2)

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


 
        #####################################################################################################################
        ###############################################Visualization###########################################################
        #####################################################################################################################
    
    mesh = np.linspace(Omega_0, Omega_L, Nx) # mesh just for plots

    ######################PlotsPlotsPlotsPlots#######################
    ######################PlotsPlotsPlotsPlots#######################
    ######################PlotsPlotsPlotsPlots#######################

    #mpl.rcParams['text.usetex'] = True
    plt.rcParams.update({ "font.family": "monospace", "font.monospace": "Computer Modern Typewriter"})
    fs=20
    fs_legend=20
    fs_axes=16
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

    # Legend outside plot
    ax.legend(fontsize=10, loc='center left', bbox_to_anchor=(1.02, 0.5), handlelength=2, borderaxespad=0.5)

    # Layout adjustment for legend
    plt.tight_layout(rect=[0, 0, 0.85, 1])

    plt.show()
    #####visualization  pressure pw

    colors = plt.cm.Greens(np.linspace(0.3, 1.0, len(time_interval)))

    fig, ax = plt.subplots(figsize=(12, 6))

    for i in range(len(time_interval)):
        ax.plot(mesh, Vp_w_Time_TCAT[i], label=f'TCAT CGP; t(s)={time_interval[i]:g}', linestyle='-', color=colors[i])

    # Axis labels
    ax.set_xlabel(r'$x\ [m]$', fontsize=fs)
    ax.set_ylabel(r' $p^w$'r' $[Pa]$',fontsize=fs)

    # Legend outside plot
    ax.legend(fontsize=10, loc='center left', bbox_to_anchor=(1.02, 0.5), handlelength=2, borderaxespad=0.5)

    # Layout adjustment for legend
    plt.tight_layout(rect=[0, 0, 0.85, 1])

    plt.show()
    #####visualization   Ewn


    colors = plt.cm.plasma(np.linspace(0.3, 1.0, len(time_interval)))

    fig, ax = plt.subplots(figsize=(12, 6))

    for i in range(len(time_interval)):
        ax.plot(mesh, Ve_wn_Time_TCAT[i], label=f'TCAT CGP; t(s)={time_interval[i]:g}', linestyle='-', color=colors[i])

    # Axis labels
    ax.set_xlabel(r'$x\ [m]$', fontsize=fs)
    ax.set_ylabel(r'$\epsilon^{\overline{\overline{wn}}} $'r' $[1/m]$',fontsize=fs)

    # Legend outside plot
    ax.legend(fontsize=10, loc='center left', bbox_to_anchor=(1.02, 0.5), handlelength=2, borderaxespad=0.5)

    # Layout adjustment for legend
    plt.tight_layout(rect=[0, 0, 0.85, 1])

    plt.show()

    #####visualization   Jwn
    colors = plt.cm.plasma(np.linspace(0.3, 1.0, len(time_interval)))

    fig, ax = plt.subplots(figsize=(12, 6))

    for i in range(len(time_interval)):
        ax.plot(mesh, Vj_wn_Time_TCAT[i], label=f'TCAT CGP; t(s)={time_interval[i]:g}', linestyle='-', color=colors[i])

    # Axis labels
    ax.set_xlabel(r'$x\ [m]$', fontsize=fs)
    ax.set_ylabel(r'$J^{wn}_n$'r' $[1/m]$',fontsize=fs)

    # Legend outside plot
    ax.legend(fontsize=10, loc='center left', bbox_to_anchor=(1.02, 0.5), handlelength=2, borderaxespad=0.5)

    # Layout adjustment for legend
    plt.tight_layout(rect=[0, 0, 0.85, 1])

    plt.show()
    #####visualization   Wwn
    colors = plt.cm.plasma(np.linspace(0.3, 1.0, len(time_interval)))

    fig, ax = plt.subplots(figsize=(12, 6))

    for i in range(len(time_interval)):
        ax.plot(mesh, Vw_wn_Time_TCAT[i], label=f'TCAT CGP; t(s)={time_interval[i]:g}', linestyle='-', color=colors[i])

    # Axis labels
    ax.set_xlabel(r'$x\ [m]$', fontsize=fs)
    ax.set_ylabel(r'$w^{wn}_x$'r' $[m/s]$',fontsize=fs)

    # Legend outside plot
    ax.legend(fontsize=10, loc='center left', bbox_to_anchor=(1.02, 0.5), handlelength=2, borderaxespad=0.5)

    # Layout adjustment for legend
    plt.tight_layout(rect=[0, 0, 0.85, 1])

    plt.show()
    #####visualization   Kwn
    colors = plt.cm.plasma(np.linspace(0.3, 1.0, len(time_interval)))

    fig, ax = plt.subplots(figsize=(12, 6))

    for i in range(len(time_interval)):
        ax.plot(mesh, Vk_wn_Time_TCAT[i], label=f'TCAT CGP; t(s)={time_interval[i]:g}', linestyle='-', color=colors[i])

    # Axis labels
    ax.set_xlabel(r'$x\ [m]$', fontsize=fs)
    ax.set_ylabel(r'$K^{wn}_n$'r' $[1/m^2]$',fontsize=fs)

    # Legend outside plot
    ax.legend(fontsize=10, loc='center left', bbox_to_anchor=(1.02, 0.5), handlelength=2, borderaxespad=0.5)

    # Layout adjustment for legend
    plt.tight_layout(rect=[0, 0, 0.85, 1])

    plt.show()

    # convergence of variables
    plt.plot(Vtime_TCAT, Vew_normalized_TCAT,label=r'TCAT $\varepsilon^{\overline{\overline{w}}}$')
    plt.plot(Vtime_TCAT, Vpw_normalized_TCAT,label=r'TCAT $p^w$')
    plt.plot(Vtime_TCAT, Vewn_normalized_TCAT ,label=r'TCAT $\varepsilon^{\overline{\overline{wn}}}$')
    plt.plot(Vtime_TCAT, Vjwn_normalized_TCAT,label=r'TCAT $J^{wn}_{n}$')
    plt.plot(Vtime_TCAT, Vkwn_normalized_TCAT,label=r'TCAT $K^{wn}_{n}$')
    plt.legend(fontsize=fs_legend)
    plt.xlabel('time'r' $[s]$',fontsize=fs)
    plt.ylabel('normalized quantity'r' $[-]$',fontsize=fs)
    plt.xticks(fontsize=20)  # Set x-axis tick label font size to 12
    plt.yticks(fontsize=20)  # Set y-axis tick label font size to 12
    plt.show()


   #######################################
if __name__ == "__main__":
    main()
