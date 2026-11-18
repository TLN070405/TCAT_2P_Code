import sys
import math
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
import matplotlib.ticker as ticker


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
    return L2_error


def main():
    Omega_0 = 0.
    Omega_L = 1.
    nodes_array= [51 ,101,201]
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
    L2_cgp_ew  = calculate_grid_resolution_error(nodes_array, time_interval, Ve_w_Time_TCAT_all, 'L2e_ew_tcat')
    L2_cgp_pw = calculate_grid_resolution_error(nodes_array, time_interval, Vp_w_Time_TCAT_all, 'L2e_pw_tcat')
    L2_cgp_ewn = calculate_grid_resolution_error(nodes_array, time_interval, Ve_wn_Time_TCAT_all, 'L2e_ewn_tcat')
    L2_cgp_jwn = calculate_grid_resolution_error(nodes_array, time_interval, Vj_wn_Time_TCAT_all, 'L2e_jwn_tcat')
    L2_cgp_wwn = calculate_grid_resolution_error(nodes_array, time_interval, Vw_wn_Time_TCAT_all, 'L2e_wwn_tcat')
    L2_cgp_kwn = calculate_grid_resolution_error(nodes_array, time_interval, Vk_wn_Time_TCAT_all, 'L2e_kwn_tcat')

    # Calculate L2 error norms for RE data
    L2_re_ew = calculate_grid_resolution_error(nodes_array, time_interval_RE, Ve_w_Time_RE_all, 'L2e_ew_re')
    L2_re_pw = calculate_grid_resolution_error(nodes_array, time_interval_RE, Vp_w_Time_RE_all, 'L2e_pw_re')





    #####################################################################################################################
    ###############################################Visualization###########################################################
    #####################################################################################################################
    

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
    colors_ew = plt.cm.Blues(np.linspace(0.8, 0.4, len(nodes_array)-1))
    colors_pw = plt.cm.Greens(np.linspace(0.8, 0.4, len(nodes_array)-1))
    colors_ewn = plt.cm.Reds(np.linspace(0.8, 0.4, len(nodes_array)-1))
    colors_jwn = plt.cm.pink(np.linspace(0.8, 0.4, len(nodes_array)-1))
    colors_wwn = plt.cm.Purples(np.linspace(0.8, 0.4, len(nodes_array)-1))
    colors_kwn = plt.cm.Greys(np.linspace(0.8, 0.4, len(nodes_array)-1))

    fig, ax = plt.subplots(figsize=(12, 6))
    start =3
    for i in range(len(nodes_array)-1):
        ax.plot(time_interval[start:], L2_cgp_ew[i][start:], label = fr'TCAT CGP $\epsilon^{{\overline{{\overline{{w}}}}}}$ error {nodes_array[i]}-{nodes_array[i+1]}', linestyle='-', color=colors_ew[i])
    for i in range(len(nodes_array)-1):
        ax.plot(time_interval[start:], L2_cgp_pw[i][start:], label=fr'TCAT CGP $p^w$ error {nodes_array[i]}-{nodes_array[i+1]}', linestyle='-', color=colors_pw[i])
    for i in range(len(nodes_array)-1):
        ax.plot(time_interval[start:], L2_cgp_ewn[i][start:], label=fr'TCAT CGP $\epsilon^{{\overline{{\overline{{wn}}}}}}$ error {nodes_array[i]}-{nodes_array[i+1]}', linestyle='-', color=colors_ewn[i])
    for i in range(len(nodes_array)-1):
        ax.plot(time_interval[start:], L2_cgp_jwn[i][start:], label=fr'TCAT CGP $J^{{wn}}_n$ error {nodes_array[i]}-{nodes_array[i+1]}', linestyle='-', color=colors_jwn[i])
    #for i in range(len(nodes_array)-1):
    #    ax.plot(time_interval, L2_cgp_wwn[i][:], label=fr'TCAT CGP $w^{{wn}}_x$ error {nodes_array[i]}-{nodes_array[i+1]}', linestyle='-', color=colors_wwn[i])
    for i in range(len(nodes_array)-1):
        ax.plot(time_interval[start:], L2_cgp_kwn[i][start:], label=fr'TCAT CGP $K^{{wn}}_n$ error {nodes_array[i]}-{nodes_array[i+1]}', linestyle='-', color=colors_kwn[i])

    for i in range(len(nodes_array)-1):
        ax.plot(time_interval[start:], L2_re_ew[i][start:], label=fr'RE $\epsilon^{{\overline{{\overline{{w}}}}}}$ error {nodes_array[i]}-{nodes_array[i+1]}', linestyle='--', color=colors_ew[i])
    for i in range(len(nodes_array)-1):
        ax.plot(time_interval[start:], L2_re_pw[i][start:], label=fr'RE $p^w$ error {nodes_array[i]}-{nodes_array[i+1]}', linestyle='--', color=colors_pw[i])
    # Axis labels
    ax.set_xlabel('time [sec]', fontsize=fs)
    ax.set_ylabel('L2 error', fontsize=fs)

    # Increase tick label font size
    ax.tick_params(axis='both', labelsize=fs_ticks)

    # Shrink the plot to the left to make space for the legend
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.75, box.height])  # 75% width for plot

    # Put legend to the right, centered vertically
    ax.legend(fontsize=fs_legend, loc='center left', bbox_to_anchor=(1.02, 0.5), handlelength=2, borderaxespad=0.5)


    # Force scientific notation on both axes
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
    ax.xaxis.get_major_formatter().set_scientific(True)
    ax.xaxis.get_major_formatter().set_powerlimits((-2, 2))  # Controls when sci notation kicks in

    ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
    ax.yaxis.get_major_formatter().set_scientific(True)
    ax.yaxis.get_major_formatter().set_powerlimits((-2, 2))

    # Increase font size of scientific notation scaling factor
    ax.ticklabel_format(style='sci', axis='both', scilimits=(-2, 2))  # Ensure sci notation is used

    # Set font size for the offset text (e.g., ×10⁻³)
    ax.xaxis.offsetText.set_fontsize(fs_ticks)  # Use your fs_ticks or any custom size
    ax.yaxis.offsetText.set_fontsize(fs_ticks)
    plt.show()
  
   #######################################
if __name__ == "__main__":
    main()
