import sys
import math
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


#######################################
def main():
    Omega_0 = 0.
    Omega_L = 1.

    #### read results
    # Read the data as strings first
    Ve_w_Time_TCAT = np.genfromtxt('e_w.csv', delimiter=',', dtype=str)
    Vp_w_Time_TCAT  = np.genfromtxt('p_w.csv', delimiter=',', dtype=str)
    Vp_n_Time_TCAT  = np.genfromtxt('p_n.csv', delimiter=',', dtype=str)
    Ve_wn_Time_TCAT = np.genfromtxt('e_wn.csv', delimiter=',', dtype=str)
    Vj_wn_Time_TCAT = np.genfromtxt('j_wn.csv', delimiter=',', dtype=str)
    Vg_wn_Time_TCAT = np.genfromtxt('g_wn.csv', delimiter=',', dtype=str)
    Vw_wn_Time_TCAT = np.genfromtxt('w_wn.csv', delimiter=',', dtype=str)
    Vk_wn_Time_TCAT = np.genfromtxt('k_wn.csv', delimiter=',', dtype=str)

    Vew_normalized_TCAT  = np.genfromtxt('e_w_normalized.csv', delimiter=',', dtype=str)
    Vpw_normalized_TCAT  = np.genfromtxt('p_w_normalized.csv', delimiter=',', dtype=str)
    Vpn_normalized_TCAT  = np.genfromtxt('p_n_normalized.csv', delimiter=',', dtype=str)
    Vewn_normalized_TCAT = np.genfromtxt('e_wn_normalized.csv', delimiter=',', dtype=str)
    Vjwn_normalized_TCAT = np.genfromtxt('j_wn_normalized.csv', delimiter=',', dtype=str)
    Vgwn_normalized_TCAT = np.genfromtxt('g_wn_normalized.csv', delimiter=',', dtype=str)
    Vkwn_normalized_TCAT = np.genfromtxt('k_wn_normalized.csv', delimiter=',', dtype=str)
    Vtime_TCAT           = np.genfromtxt('time_steps_array.csv', delimiter=',', dtype=str)


    Ve_w_Time_2P    = np.genfromtxt('e_w_2p.csv', delimiter=',', dtype=str)
    Vp_w_Time_2P    = np.genfromtxt('p_w_2p.csv', delimiter=',', dtype=str)
    Vp_n_Time_2P    = np.genfromtxt('p_n_2p.csv', delimiter=',', dtype=str)

    Vew_normalized_2P  = np.genfromtxt('e_w_normalized_2p.csv', delimiter=',', dtype=str)
    Vpw_normalized_2P  = np.genfromtxt('p_w_normalized_2p.csv', delimiter=',', dtype=str)
    Vpn_normalized_2P  = np.genfromtxt('p_n_normalized_2p.csv', delimiter=',', dtype=str)
    Vtime_2P           = np.genfromtxt('time_steps_array_2p.csv', delimiter=',', dtype=str)


    time_interval =np.genfromtxt('time.csv', delimiter=',', dtype=str)

    Nx = len(Ve_w_Time_TCAT[0])

    node_i=int((Nx-1)/2)

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


    Ve_w_Time_2P    = Ve_w_Time_2P.astype(float)
    Vp_w_Time_2P    = Vp_w_Time_2P.astype(float)
    Vp_n_Time_2P    = Vp_n_Time_2P.astype(float)

    Vew_normalized_2P  = Vew_normalized_2P.astype(float)
    Vpw_normalized_2P  = Vpw_normalized_2P.astype(float)
    Vpn_normalized_2P  = Vpn_normalized_2P.astype(float)
    Vtime_2P           = Vtime_2P.astype(float)

    time_interval   = time_interval.astype(float)


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
        ax.plot(mesh, Ve_w_Time_TCAT[i], label=f'TCAT 2P; t(s)={time_interval[i]:g}', linestyle='-', color=colors[i])
        ax.plot(mesh, Ve_w_Time_2P[i], label=f'2P ; t(s)={time_interval[i]:g}', linestyle=':', color=colors[i])

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
        ax.plot(mesh, Vp_w_Time_TCAT[i], label=f'TCAT 2P; t(s)={time_interval[i]:g}', linestyle='-', color=colors[i])
        ax.plot(mesh, Vp_w_Time_2P[i], label=f'2P ; t(s)={time_interval[i]:g}', linestyle=':', color=colors[i])

    # Axis labels
    ax.set_xlabel(r'$x\ [m]$', fontsize=fs)
    ax.set_ylabel(r' $p^w$'r' $[Pa]$',fontsize=fs)

    # Legend outside plot
    ax.legend(fontsize=10, loc='center left', bbox_to_anchor=(1.02, 0.5), handlelength=2, borderaxespad=0.5)

    # Layout adjustment for legend
    plt.tight_layout(rect=[0, 0, 0.85, 1])

    plt.show()

    #####visualization  pressure pn

    colors = plt.cm.Greens(np.linspace(0.3, 1.0, len(time_interval)))

    fig, ax = plt.subplots(figsize=(12, 6))

    for i in range(len(time_interval)):
        ax.plot(mesh, Vp_n_Time_TCAT[i], label=f'TCAT 2P; t(s)={time_interval[i]:g}', linestyle='-', color=colors[i])
        ax.plot(mesh, Vp_n_Time_2P[i], label=f'2P ; t(s)={time_interval[i]:g}', linestyle=':', color=colors[i])
    # Axis labels
    ax.set_xlabel(r'$x\ [m]$', fontsize=fs)
    ax.set_ylabel(r' $p^n$'r' $[Pa]$',fontsize=fs)

    # Legend outside plot
    ax.legend(fontsize=10, loc='center left', bbox_to_anchor=(1.02, 0.5), handlelength=2, borderaxespad=0.5)

    # Layout adjustment for legend
    plt.tight_layout(rect=[0, 0, 0.85, 1])

    plt.show()


    #####visualization   Ewn


    colors = plt.cm.inferno(np.linspace(0.3, 1.0, len(time_interval)))

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

    #####visualization   Gwn
    colors = plt.cm.plasma(np.linspace(0.3, 1.0, len(time_interval)))

    fig, ax = plt.subplots(figsize=(12, 6))

    for i in range(len(time_interval)):
        ax.plot(mesh, Vg_wn_Time_TCAT[i], label=f'TCAT 2P; t(s)={time_interval[i]:g}', linestyle='-', color=colors[i])

    # Axis labels
    ax.set_xlabel(r'$x\ [m]$', fontsize=fs)
    ax.set_ylabel(r'$G^{wn}_{xx}$'r' $[-]$',fontsize=fs)

    # Legend outside plot
    ax.legend(fontsize=10, loc='center left', bbox_to_anchor=(1.02, 0.5), handlelength=2, borderaxespad=0.5)

    # Layout adjustment for legend
    plt.tight_layout(rect=[0, 0, 0.85, 1])

    plt.show()

    #####visualization   Wwn
    colors = plt.cm.coolwarm(np.linspace(0.3, 1.0, len(time_interval)))

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
    colors = plt.cm.cividis(np.linspace(0.3, 1.0, len(time_interval)))

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
    plt.plot(Vtime_TCAT, Vew_normalized_TCAT, label=r'2P-TCAT $\epsilon^{\overline{\overline{w}}}$')
    plt.plot(Vtime_TCAT, Vpw_normalized_TCAT, label=r'2P-TCAT $p^w$')
    plt.plot(Vtime_TCAT, Vewn_normalized_TCAT, label=r'2P-TCAT $\epsilon^{\overline{\overline{wn}}}$')
    plt.plot(Vtime_TCAT, Vjwn_normalized_TCAT, label=r'2P-TCAT $J_n^{wn}$')
    plt.plot(Vtime_TCAT, Vgwn_normalized_TCAT, label=r'2P-TCAT $G^{wn}_{xx}$')
    plt.plot(Vtime_TCAT, Vkwn_normalized_TCAT, label=r'2P-TCAT $K_n^{wn}$')
    plt.plot(Vtime_2P, Vew_normalized_2P, label=r'2P-TM $\epsilon^{\overline{\overline{w}}}$', linestyle=':')
    plt.plot(Vtime_2P, Vpw_normalized_2P, label=r'2P-TM $p^w$', linestyle=':')

    plt.legend(fontsize=fs_legend)
    plt.xlabel('time'r' $[s]$',fontsize=fs)
    plt.ylabel('normalized quantity'r' $[-]$',fontsize=fs)
    plt.show()



   #######################################
if __name__ == "__main__":
    main()
