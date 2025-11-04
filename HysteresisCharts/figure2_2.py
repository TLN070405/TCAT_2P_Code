"""
hysteresis ew and pc (multiple endpoints saving)
"""
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
#
porosity = 0.41

#imb
alpha_imb = 9e-4
n_imb     = 6.0
e_nr_imb  = 0.
e_wr_imb  = 0.

#dr
alpha_dr  = 6e-4
n_dr      = 9.0
e_nr_dr   = 0.
e_wr_dr   = 0.

#
m_imb         = 1.0 - (1.0 / n_imb)
n_inv_imb     = 1.0/n_imb
m_inv_imb     = 1.0/m_imb
alpha_inv_imb = 1.0/alpha_imb
m_dr          = 1.0 - (1.0 / n_dr)
n_inv_dr      = 1.0/n_dr
m_inv_dr      = 1.0/m_dr
alpha_inv_dr  = 1.0/alpha_dr

#
htol = 0

#
N = 1000
ve_w_imb = np.linspace(e_wr_imb, porosity-e_nr_imb, N)
ve_w_dr  = np.linspace(e_wr_dr, porosity-e_nr_dr, N)
vp_c_imb = np.empty([len(ve_w_imb)], dtype = np.double)
vp_c_dr  = np.empty([len(ve_w_dr)], dtype=np.double)


# functions
def capillary_pressure_dr( e_w):
    """
    Capillary pressure function
    """
    se = (e_w - e_wr_dr) / (porosity - e_nr_dr  - e_wr_dr)

    if se > 0.0 and se < 1.0:
        p_c = (  ( se**(-m_inv_dr)  - 1.0)**n_inv_dr )*alpha_inv_dr
    else:
        p_c = 0.0
    return p_c

def capillary_pressure_imb( e_w):
    """
    Capillary pressure function
    """
    se = (e_w - e_wr_imb) / (porosity - e_nr_imb - e_wr_imb)
    if se > 0.0 and se < 1.0:
        p_c = (  ( se**(-m_inv_imb)  - 1.0)**n_inv_imb )*alpha_inv_imb
    else:
        p_c = 0.0
    return p_c

def volume_fraction_dr(p_c):
    """
    Volume fraction w phase function
    """
    if p_c > 0.0:
        se =  (1.0 + (alpha_dr*p_c)**n_dr)**(-m_dr)
        e_w =(porosity - e_nr_dr  - e_wr_dr)*  se + e_wr_dr
    else:
        e_w = porosity-e_nr_dr
    return e_w

def volume_fraction_imb(p_c):
    """
    Volume fraction w phase function
    """
    if p_c > 0.0:
        se =  (1.0 + (alpha_imb*p_c)**n_imb)**(-m_imb)
        e_w =(porosity - e_nr_imb - e_wr_imb)*  se + e_wr_imb
    else:
        e_w = porosity-e_nr_imb
    return e_w

def volume_fraction_nonwetting_residual( e_w):
    """
    Capillary pressure function
    """
    se = (e_w - e_wr_dr) / (porosity - e_nr_dr  - e_wr_dr)

    if se > 0.0 and se < 1.0:
        p_c = (  ( se**(-m_inv_dr)  - 1.0)**n_inv_dr )*alpha_inv_dr
    else:
        p_c = 0.0
    return p_c



def  capillary_pressure_hysteresis( e_w, process, e_w_reversal, e_w_endpoint, p_c_reversal, p_c_endpoint):
    """
    hysteretic pc Parker Lenhard scaling model
    """
    if process=='imb':
        e_w_reversal_mpc = volume_fraction_imb(p_c_reversal)
        e_w_endpoint_mpc = volume_fraction_imb(p_c_endpoint)
        e_w_hysteretic   = ((e_w-e_w_endpoint)/(e_w_reversal-e_w_endpoint))*(e_w_reversal_mpc-e_w_endpoint_mpc) + e_w_endpoint_mpc
        p_c = capillary_pressure_imb(e_w_hysteretic)
    elif process=='dr':
        e_w_reversal_mpc = volume_fraction_dr(p_c_reversal)
        e_w_endpoint_mpc = volume_fraction_dr(p_c_endpoint)
        e_w_hysteretic   = ((e_w-e_w_endpoint)/(e_w_reversal-e_w_endpoint))*(e_w_reversal_mpc-e_w_endpoint_mpc) + e_w_endpoint_mpc
        p_c = capillary_pressure_dr(e_w_hysteretic)
    return p_c

def update_process(switch_condition,process_old,process_new):
    """
    function to update the current process 
    """
    if switch_condition > htol:
        process_old = process_new
        process_new = 'imb'
    elif switch_condition < htol:
        process_old = process_new
        process_new = 'dr'
    return process_old, process_new

def update_scaling_points(process_new, process_old, e_w, p_c, e_w_reversal, e_w_endpoint, p_c_reversal, p_c_endpoint):
    """
    function to update hysteresis  reversal and end points.
    """
    # Update reversal and endpoints (ew and pc)
    if process_new != process_old:
        p_c_endpoint[0]  = p_c_reversal[0]
        p_c_reversal[0]  = p_c
        e_w_endpoint[0]  = e_w_reversal[0]
        e_w_reversal[0]  = e_w


def ew_bc(t):
    """
    Boundary volume fraction
    """
    ew_t = [0.9*porosity, 0.1*porosity,0.7*porosity, 0.3*porosity, 0.7*porosity]
    time = [0. ,  100, 200,  300, 400]

    if time[0]<=t<=time[1]:
        a   = (ew_t[1] - ew_t[0])/(time[1] - time[0])
        b   = ew_t[0] - a * time[0]
        e_w =  a * t + b
    elif time[1]<t<=time[2]:
        a   = (ew_t[2] - ew_t[1])/(time[2] - time[1])
        b   = ew_t[1] - a * time[1]
        e_w = a * t + b
    elif time[2]<t<=time[3]:
        a   = (ew_t[3] - ew_t[2])/(time[3] - time[2])
        b   = ew_t[2] - a * time[2]
        e_w =  a * t + b
    elif time[3]<t<=time[4]:
        a   = (ew_t[4]-ew_t[3])/(time[4] - time[3])
        b   = ew_t[3] - a * time[3]
        e_w = a * t + b
    return  e_w


e_w_reversal = np.full((8),  porosity)
e_w_endpoint = np.full((8),  0.003 )
p_c_reversal = np.full((8), capillary_pressure_dr(porosity) )
p_c_endpoint = np.full((8), capillary_pressure_dr(0.003 ) )


# initialize the scaling points
process_new  = 'dr'
process_old  = 'dr'

e_w_old      = ew_bc(0)
p_c_old      = capillary_pressure_dr(e_w_old)

ve_w_hysteresis = np.zeros([401], dtype = np.double)
vp_c_hysteresis = np.zeros([401], dtype = np.double)

ve_w_hysteresis[0] = e_w_old
vp_c_hysteresis[0] = p_c_old

for i in range(1,401):
    t=i
    e_w = ew_bc(t)
    process_old, process_new = update_process(e_w-e_w_old, process_old, process_new)
    update_scaling_points(process_new,process_old,e_w_old,p_c_old,e_w_reversal, e_w_endpoint, p_c_reversal, p_c_endpoint)
    p_c = capillary_pressure_hysteresis(e_w, process_new, e_w_reversal[0], e_w_endpoint[0], p_c_reversal[0], p_c_endpoint[0])
    e_w_old = e_w
    p_c_old = p_c
    vp_c_hysteresis[i] = p_c
    ve_w_hysteresis[i] = e_w









def capillary_pressure_imb_adjusted( e_w):
    """
    Capillary pressure function
    """
    #imb
    alpha_imb = 10e-4
    n_imb     = 6.0
    e_nr_imb  = 0.
    e_wr_imb  = 0.

    #
    m_imb         = 1.0 - (1.0 / n_imb)
    n_inv_imb     = 1.0/n_imb
    m_inv_imb     = 1.0/m_imb
    alpha_inv_imb = 1.0/alpha_imb

    se = (e_w - e_wr_imb) / (porosity - e_nr_imb - e_wr_imb)
    if se > 0.0 and se < 1.0:
        p_c = (  ( se**(-m_inv_imb)  - 1.0)**n_inv_imb )*alpha_inv_imb
    else:
        p_c = 0.0
    return p_c

# Visualization
for i in range(0,N):
    vp_c_dr[i]  = capillary_pressure_dr(ve_w_dr[i])
    vp_c_imb[i] = capillary_pressure_imb_adjusted(ve_w_imb[i])

























# Plot settings
curves_thickness = 2.5
axis_titles_fs = 30
axis_numbers_fs = 25
numbered_points_fs = 30
legend_fs = 30

fig, ax = plt.subplots()

# Main curves
ax.plot(ve_w_dr[2:] / porosity, vp_c_dr[2:], label="Primary drainage curve", color='r', linewidth=curves_thickness)
ax.plot(ve_w_imb[2:] / porosity, vp_c_imb[2:], label="Main imbibition curve", color='b', linewidth=curves_thickness)

# Scanning curve segments (start index, end index, color, label)
segments = [
    (0, 101, 'k', "Drainage scanning curves"),
    (100, 201, 'g', "Imbibition scanning curves"),
    (200, 301, 'k', None),  # second black curve
    (300, 401, 'g', None)   # second green curve
]

# Loop over scanning curves
for idx, (start, end, color, label) in enumerate(segments):
    x = ve_w_hysteresis[start:end] / porosity
    y = vp_c_hysteresis[start:end]

    # Plot the scanning curve
    ax.plot(x, y, color=color, linewidth=curves_thickness, label=label)

    # Add solid black square point at the beginning
    ax.plot(x[0], y[0], marker='s', markersize=8, color='black')

    # Offsets for text placement
    offsets = [(0.004, 50), (0.004, 50), (0.004, 0.01), (-0.02, -0.015)]
    dx, dy = offsets[idx]
    ax.text(x[0] + dx, y[0] + dy, str(idx + 1), fontsize=numbered_points_fs, verticalalignment='center')

    # Add arrow in the middle
    mid_idx = len(x) // 2
    arrow = FancyArrowPatch(
        (x[mid_idx], y[mid_idx]),
        (x[mid_idx + 1], y[mid_idx + 1]),
        arrowstyle='->',
        mutation_scale=20,
        color=color,
        linewidth=2
    )
    ax.add_patch(arrow)

    # Draw dashed line and square marker for point 3 and 4
    if idx in [2, 3]:  # Point 3 and 4
        x_point, y_point = x[0], y[0]

        # Get imbibition curve data
        imb_x = ve_w_imb[2:] / porosity
        imb_y = vp_c_imb[2:]

        # Find closest y match on the blue curve
        closest_idx = (np.abs(imb_y - y_point)).argmin()
        x_blue = imb_x[closest_idx]
        y_blue = imb_y[closest_idx]

        # Draw dashed black horizontal line
        ax.hlines(y=y_point, xmin=x_point, xmax=x_blue, linestyles='dashed', colors='black', linewidth=2)

        # Add black square at intersection point
        ax.plot(x_blue, y_blue, marker='s', markersize=8, color='black')

        # Add number next to intersection with blue curve
        if idx == 2:
            ax.text(x_blue + 0.005, y_blue + 50, "6", fontsize=numbered_points_fs, verticalalignment='center')
        elif idx == 3:
            ax.text(x_blue + 0.005, y_blue + 50, "5", fontsize=numbered_points_fs, verticalalignment='center')


# New horizontal dashed line at y=1500 from second black curve to blue curve
y_line = 1500

# Extract second black curve data (segment 200:301)
x_black2 = ve_w_hysteresis[200:301] / porosity
y_black2 = vp_c_hysteresis[200:301]

# Find x where black curve crosses y=1500 (approximate by closest point)
idx_black2 = (np.abs(y_black2 - y_line)).argmin()
x_black2_at_y = x_black2[idx_black2]

# Blue curve data
imb_x = ve_w_imb[2:] / porosity
imb_y = vp_c_imb[2:]

# Find closest y=1500 point on blue curve
idx_blue = (np.abs(imb_y - y_line)).argmin()
x_blue_at_y = imb_x[idx_blue]

# Draw black dashed horizontal line from second black curve to blue curve at y=1500
ax.hlines(y=y_line, xmin=x_black2_at_y, xmax=x_blue_at_y, colors='black', linestyles='dashed', linewidth=2)

# Add black square at intersection with blue curve
ax.plot(x_blue_at_y, y_line, marker='s', markersize=8, color='black')

# Add number next to intersection with blue curve at y = 1500
ax.text(x_blue_at_y + 0.005, y_line + 50, "7", fontsize=numbered_points_fs, verticalalignment='center')

# Add black square at intersection with second black curve (already found x_black2_at_y)
ax.plot(x_black2_at_y, y_line, marker='s', markersize=8, color='black')

# Label "ksi" near the intersection with second black curve
ax.text(x_black2_at_y + 0.005, y_line + 50, r"$\xi$", fontsize=numbered_points_fs, verticalalignment='center', color='black')

# Add black square at intersection with second green curve (segment 300:401) at y=1500
x_green2 = ve_w_hysteresis[300:401] / porosity
y_green2 = vp_c_hysteresis[300:401]
idx_green2 = (np.abs(y_green2 - y_line)).argmin()
x_green2_at_y = x_green2[idx_green2]
ax.plot(x_green2_at_y, y_line, marker='s', markersize=8, color='black')

# Label "psi" near the intersection with second green curve
ax.text(x_green2_at_y + 0.005, y_line + 50, r"$\zeta$", fontsize=numbered_points_fs, verticalalignment='center', color='black')

# Remove duplicate legend entries
handles, labels = ax.get_legend_handles_labels()
by_label = dict(zip(labels, handles))
ax.legend(by_label.values(), by_label.keys(), fontsize=legend_fs)

# Axis labels and styling
ax.set_xlabel(r"$\overline{S}^{wa}\,[-]$", fontsize=axis_titles_fs)
ax.set_ylabel(r"$p^c\,[Pa]$", fontsize=axis_titles_fs)
ax.tick_params(axis='both', labelsize=axis_numbers_fs)

plt.show()
