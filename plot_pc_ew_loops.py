"""
hysteresis ew and pc (multiple endpoints saving)
"""
import math
import numpy as np
import matplotlib.pyplot as plt

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

def  capillary_pressure_hysteresis( e_w, process_new, e_w_reversal, e_w_endpoint, p_c_reversal, p_c_endpoint):
    """
    hysteretic pc Parker Lenhard scaling model
    """
    if process_new=='imb':
        e_w_reversal_mpc = volume_fraction_imb(p_c_reversal)
        e_w_endpoint_mpc = volume_fraction_imb(p_c_endpoint)
        e_w_hysteretic   = ((e_w-e_w_endpoint)/(e_w_reversal-e_w_endpoint))*(e_w_reversal_mpc-e_w_endpoint_mpc) + e_w_endpoint_mpc
        p_c = capillary_pressure_imb(e_w_hysteretic)
    elif process_new=='dr':
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

def update_scaling_points(process_new,process_old,e_w,p_c,e_w_reversal, e_w_endpoint, p_c_reversal, p_c_endpoint):
    """
    function to update hysteresis  reversal and end points.
    """
    # Update reversal and endpoints (ew and pc)
    if process_new != process_old:
        for i in range(15, 0, -1):
            p_c_endpoint[i]  = p_c_endpoint[i-1]
            p_c_reversal[i]  = p_c_reversal[i-1]
            e_w_endpoint[i]  = e_w_endpoint[i-1]
            e_w_reversal[i]  = e_w_reversal[i-1]
        p_c_endpoint[0]  = p_c_reversal[0]
        p_c_reversal[0]  = p_c
        e_w_endpoint[0]  = e_w_reversal[0]
        e_w_reversal[0]  = e_w
        #
    #if (process_new=='dr' and e_w < e_w_endpoint[0]) or (process_new=='imb' and e_w > e_w_endpoint[0]):
    if (process_new=='dr' and e_w < e_w_endpoint[0]) or (process_new=='imb' and e_w > e_w_endpoint[0]):
        for i in range(0, 14):
            p_c_endpoint[i]  = p_c_endpoint[i+2]
            p_c_reversal[i]  = p_c_reversal[i+2]
            e_w_endpoint[i]  = e_w_endpoint[i+2]
            e_w_reversal[i]  = e_w_reversal[i+2]



    return e_w_reversal[0], e_w_endpoint[0], p_c_reversal[0], p_c_endpoint[0]

def ew_bc(t):
    """
    Boundary volume fraction
    """
    ew_t = [0.3, 0.12, 0.3, 0.15, 0.26, 0.18,   0.23, 0.20,  0.215, 0.12]
    time = [0. ,  100, 200,  300,  400,  500,    600,  700,   800, 900]

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
    elif time[4]<t<=time[5]:
        a   = (ew_t[5]-ew_t[4])/(time[5] - time[4])
        b   = ew_t[4] - a * time[4]
        e_w = a * t + b
    elif time[5]<t<=time[6]:
        a   = (ew_t[6]-ew_t[5])/(time[6] - time[5])
        b   = ew_t[5] - a * time[5]
        e_w = a * t + b
    elif time[6]<t<=time[7]:
        a   = (ew_t[7]-ew_t[6])/(time[7] - time[6])
        b   = ew_t[6] - a * time[6]
        e_w = a * t + b
    elif time[7]<t<=time[8]:
        a   = (ew_t[8]-ew_t[7])/(time[8] - time[7])
        b   = ew_t[7] - a * time[7]
        e_w = a * t + b
    elif time[8]<t<=time[9]:
        a   = (ew_t[9]-ew_t[8])/(time[9] - time[8])
        b   = ew_t[8] - a * time[8]
        e_w = a * t + b
    return  e_w
 

e_w_reversal = np.full((16),  porosity)
e_w_endpoint = np.full((16),  0.003 )
p_c_reversal = np.full((16), capillary_pressure_dr(porosity) )
p_c_endpoint = np.full((16), capillary_pressure_dr(0.003 ) )


# initialize the scaling points
process_new  = 'dr'
process_old  = 'dr'

e_w_old      = ew_bc(0)
p_c_old      = capillary_pressure_dr(e_w_old)

ve_w_hysteresis = np.zeros([900], dtype = np.double)
vp_c_hysteresis = np.zeros([900], dtype = np.double)

ve_w_hysteresis[0] = e_w_old
vp_c_hysteresis[0] = p_c_old

for i in range(1,900):
    e_w = ew_bc(i)
    process_old, process_new = update_process(e_w-e_w_old, process_old, process_new)
    e_w_reversal1, e_w_endpoint1, p_c_reversal1, p_c_endpoint1 = update_scaling_points(process_new,process_old,e_w_old,p_c_old,e_w_reversal, e_w_endpoint, p_c_reversal, p_c_endpoint)
    p_c = capillary_pressure_hysteresis(e_w, process_new, e_w_reversal1, e_w_endpoint1, p_c_reversal1, p_c_endpoint1)
    e_w_old = e_w
    p_c_old = p_c
    vp_c_hysteresis[i] = p_c
    ve_w_hysteresis[i] = e_w

# Visualization
for i in range(0,N):
    vp_c_dr[i]  = capillary_pressure_dr(ve_w_dr[i])
    vp_c_imb[i] = capillary_pressure_imb(ve_w_imb[i])
print(e_w_reversal)
print(p_c_reversal)   
print(e_w_endpoint)
print(p_c_endpoint)
         



VEw_new_dr=np.empty([N], dtype=np.double)
VPc_new_dr=np.empty([N], dtype=np.double)
VEw_reversal_dr = 0.215
VPc_reversal_dr = 1390.90882345
VEw_endpoint_dr =  0.12
VPc_endpoint_dr = 1881.92253228


for i in range(0,N):
    VEw_new_dr[i]=VEw_reversal_dr+i*((VEw_endpoint_dr-VEw_reversal_dr)/(N-1))
for i in range(0,N):
    VPc_new_dr[i]=capillary_pressure_hysteresis( VEw_new_dr[i], 'dr', VEw_reversal_dr, VEw_endpoint_dr, VPc_reversal_dr, VPc_endpoint_dr)


plt.plot(ve_w_hysteresis[0:100], vp_c_hysteresis[0:100], label='PDC')
plt.plot(ve_w_hysteresis[100:200], vp_c_hysteresis[100:200], label='scanning curve1')
plt.plot(ve_w_hysteresis[200:300], vp_c_hysteresis[200:300], label='scanning curve2')
plt.plot(ve_w_hysteresis[300:399], vp_c_hysteresis[300:399], label='scanning curve3')
plt.plot(ve_w_hysteresis[400:499], vp_c_hysteresis[400:499], label='scanning curve4')
plt.plot(ve_w_hysteresis[500:599], vp_c_hysteresis[500:599], label='scanning curve5')
plt.plot(ve_w_hysteresis[600:699], vp_c_hysteresis[600:699], label='scanning curve6')
plt.plot(ve_w_hysteresis[700:799], vp_c_hysteresis[700:799], label='scanning curve7')
plt.plot(ve_w_hysteresis[800:899], vp_c_hysteresis[800:899], label='scanning curve8')
plt.plot(VEw_new_dr, VPc_new_dr, label='scaled curve')
plt.plot(ve_w_imb[200:1000], vp_c_imb[200:1000], label="MIC VG",linestyle="dashed",color='k')
plt.plot(ve_w_dr[200:1000] , vp_c_dr[200:1000] , label="PDC VG",linestyle="dashed",color='k')
plt.xlabel('e_w')
plt.ylabel('p_c')
plt.title('Hysteresis pc-ew curves')
plt.legend()
plt.show()
