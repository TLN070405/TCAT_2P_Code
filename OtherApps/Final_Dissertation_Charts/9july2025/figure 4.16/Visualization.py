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
    Ve_wn_Time_TCAT = np.genfromtxt('e_wn.csv', delimiter=',', dtype=str)
    Vj_wn_Time_TCAT = np.genfromtxt('j_wn.csv', delimiter=',', dtype=str)
    Vw_wn_Time_TCAT = np.genfromtxt('w_wn.csv', delimiter=',', dtype=str)
    Vk_wn_Time_TCAT = np.genfromtxt('k_wn.csv', delimiter=',', dtype=str)

    Vew_normalized_TCAT  = np.genfromtxt('e_w_normalized.csv', delimiter=',', dtype=str)
    Vpw_normalized_TCAT  = np.genfromtxt('p_w_normalized.csv', delimiter=',', dtype=str)
    Vewn_normalized_TCAT = np.genfromtxt('e_wn_normalized.csv', delimiter=',', dtype=str)
    Vjwn_normalized_TCAT = np.genfromtxt('j_wn_normalized.csv', delimiter=',', dtype=str)
    Vkwn_normalized_TCAT = np.genfromtxt('k_wn_normalized.csv', delimiter=',', dtype=str)
    Vtime_TCAT           = np.genfromtxt('time_steps_array.csv', delimiter=',', dtype=str)


    Ve_w_Time_RE    = np.genfromtxt('e_w_re.csv', delimiter=',', dtype=str)
    Vp_w_Time_RE    = np.genfromtxt('p_w_re.csv', delimiter=',', dtype=str)

    Vew_normalized_RE  = np.genfromtxt('e_w_normalized_re.csv', delimiter=',', dtype=str)
    Vpw_normalized_RE  = np.genfromtxt('p_w_normalized_re.csv', delimiter=',', dtype=str)
    Vtime_RE           = np.genfromtxt('time_steps_array_re.csv', delimiter=',', dtype=str)


    time_interval =np.genfromtxt('time.csv', delimiter=',', dtype=str)

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


    Ve_w_Time_RE    = Ve_w_Time_RE.astype(float)
    Vp_w_Time_RE    = Vp_w_Time_RE.astype(float)

    Vew_normalized_RE  = Vew_normalized_RE.astype(float)
    Vpw_normalized_RE  = Vpw_normalized_RE.astype(float)
    Vtime_RE           = Vtime_RE.astype(float)

    time_interval   = time_interval.astype(float)


    #####################################################################################################################
    ###############################################Visualization###########################################################
    #####################################################################################################################
    mesh = np.linspace(Omega_0, Omega_L, Nx) # mesh just for plots

    ######################PlotsPlotsPlotsPlots#######################
    ######################PlotsPlotsPlotsPlots#######################
    ######################PlotsPlotsPlotsPlots#######################
 




    from mpl_toolkits.axes_grid1.inset_locator import inset_axes

    plt.rcParams.update({
        "font.family": "monospace",
        "font.monospace": "Computer Modern Typewriter"
    })

    fs = 28
    fs_legend = 20
    fs_axes = 20

    # Create main plot
    fig, ax = plt.subplots()
    ax.tick_params(axis='both', labelsize=fs_axes)

    # Plot main data
    ax.plot(Vtime_TCAT, Vew_normalized_TCAT, label=r'TCAT $\epsilon^{\overline{\overline{w}}}$')
    ax.plot(Vtime_TCAT, Vpw_normalized_TCAT, label=r'TCAT $p^w$')
    ax.plot(Vtime_TCAT, Vewn_normalized_TCAT, label=r'TCAT $\epsilon^{\overline{\overline{wn}}}$')
    ax.plot(Vtime_TCAT, Vjwn_normalized_TCAT, label=r'TCAT $J^{wn}_n$')
    ax.plot(Vtime_TCAT, Vkwn_normalized_TCAT, label=r'TCAT $K^{wn}_n$')
    ax.plot(Vtime_RE, Vew_normalized_RE, label=r'RE $\epsilon^{\overline{\overline{w}}}$', linestyle=':')
    ax.plot(Vtime_RE, Vpw_normalized_RE, label=r'RE $p^w$', linestyle=':')

    # Cheat the x-axis: 0–8000 → 0–300
    num_ticks = 7
    real_ticks = np.linspace(0, 8000, num_ticks)
    fake_labels = np.linspace(0, 300, num_ticks)
    ax.set_xticks(real_ticks)
    ax.set_xticklabels([f"{label:.0f}" for label in fake_labels])

    ax.set_xlabel('Time' r' $[s]$', fontsize=fs)
    ax.set_ylabel('Normalized Quantity' r' $[-]$', fontsize=fs)
    ax.legend(fontsize=fs_legend)

    # --- Inset Zoom ---
    axins = inset_axes(ax, width="40%", height="40%", 
                    bbox_to_anchor=(0.5, 0, 0.3, 0.5),
                    bbox_transform=ax.transAxes, loc='center')
    
    
    # Make the border of the zoom box thicker
    for spine in axins.spines.values():
        spine.set_linewidth(1.5)
    # Plot same data in the inset
    axins.plot(Vtime_TCAT, Vew_normalized_TCAT)
    axins.plot(Vtime_TCAT, Vpw_normalized_TCAT)
    axins.plot(Vtime_TCAT, Vewn_normalized_TCAT)
    axins.plot(Vtime_TCAT, Vjwn_normalized_TCAT)
    axins.plot(Vtime_TCAT, Vkwn_normalized_TCAT)
    axins.plot(Vtime_RE, Vew_normalized_RE, linestyle=':')
    axins.plot(Vtime_RE, Vpw_normalized_RE, linestyle=':')

    # Zoom range
    axins.set_xlim(2500, 3500)
    axins.set_ylim(0.6, 0.7)

    # ✅ Remove ticks and labels
    axins.set_xticks([])
    axins.set_yticks([])
    axins.tick_params(bottom=False, left=False, labelbottom=False, labelleft=False)

    # ✅ KEEP the border/spines
    # Do NOT hide the spines — keep this block removed

    plt.show()





   #######################################
if __name__ == "__main__":
    main()
