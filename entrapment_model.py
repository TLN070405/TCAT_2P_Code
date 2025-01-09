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
e_nr_imb  = 0.1
e_wr = 0.0

#dr
alpha_dr  = 6e-4
n_dr      = 9.0
e_nr_dr   = 0.

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
ve_w_imb = np.linspace(e_wr, porosity-e_nr_imb, N)
ve_w_dr  = np.linspace(e_wr, porosity-e_nr_dr, N)
vp_c_imb = np.empty([len(ve_w_imb)], dtype = np.double)
vp_c_dr  = np.empty([len(ve_w_dr)], dtype=np.double)


# functions
def capillary_pressure_dr( e_w):
    """
    Capillary pressure function
    """
    se = (e_w - e_wr) / (porosity - e_nr_dr  - e_wr)

    if se > 0.0 and se < 1.0:
        p_c = (  ( se**(-m_inv_dr)  - 1.0)**n_inv_dr )*alpha_inv_dr
    else:
        p_c = 0.0
    return p_c

def capillary_pressure_imb( e_w):
    """
    Capillary pressure function
    """
    se = (e_w - e_wr) / (porosity - e_nr_imb - e_wr)
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
        e_w =(porosity - e_nr_dr  - e_wr)*  se + e_wr
    else:
        e_w = porosity-e_nr_dr
    return e_w

def volume_fraction_imb(p_c):
    """
    Volume fraction w phase function
    """
    if p_c > 0.0:
        se =  (1.0 + (alpha_imb*p_c)**n_imb)**(-m_imb)
        e_w =(porosity - e_nr_imb - e_wr)*  se + e_wr
    else:
        e_w = porosity-e_nr_imb
    return e_w

def residual_volume_fraction_nw(e_w_reversal):
    """
    residual nw volume fraction from Lands formula
    """
    e_nr = (porosity-e_wr)*(porosity-e_w_reversal)*(e_nr_imb)/(   (porosity-e_wr)*e_nr_imb -  e_nr_imb*(porosity-e_w_reversal)  +   (porosity-e_wr)*(porosity-e_w_reversal)  )

    return e_nr

def entrapped_volume_fraction_nw(e_w,e_nr,e_w_reversal):
    """
    entrapped nw volume fraction from Parker Lenhard formula
    """
    if e_nr == 0.0:
        e_nt= 0.0
    else:
        e_nt = e_nr*(e_w-e_w_reversal)/(porosity-e_w_reversal-e_nr)

    return e_nt

def apparent_volume_fraction(e_w,e_nt):
    """
    apparent volume fraction from Parker Lenhard formula
    """
    e_a = e_w + e_nt

    return e_a


def capillary_pressure_hysteresis( e_a, process, e_a_reversal, e_a_endpoint, p_c_reversal, p_c_endpoint):
    """
    hysteretic pc Parker Lenhard scaling model
    """
    if process=='imb':
        e_a_reversal_mpc = volume_fraction_imb(p_c_reversal)
        e_a_endpoint_mpc = volume_fraction_imb(p_c_endpoint)
        e_a_hysteretic   = ((e_a-e_a_endpoint)/(e_a_reversal-e_a_endpoint))*(e_a_reversal_mpc-e_a_endpoint_mpc) + e_a_endpoint_mpc
        p_c = capillary_pressure_imb(e_a_hysteretic)
    elif process=='dr':
        e_a_reversal_mpc = volume_fraction_dr(p_c_reversal)
        e_a_endpoint_mpc = volume_fraction_dr(p_c_endpoint)
        e_a_hysteretic   = ((e_a-e_a_endpoint)/(e_a_reversal-e_a_endpoint))*(e_a_reversal_mpc-e_a_endpoint_mpc) + e_a_endpoint_mpc
        p_c = capillary_pressure_dr(e_a_hysteretic)
    return p_c

def update_process(switch_condition, process_old, process_new):
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
    ew_t = [0.3, 0.15, porosity-e_nr_imb + 0.01, 0.14, 0.25, 0.23,   0.2, 0.35, 0.33]
    time = [0. ,  100, 200,  300, 400,  500,   600,  700,  800]

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
    return  e_w


e_w_reversal = np.full((8),  porosity)
e_w_endpoint = np.full((8),  e_wr+0.003 )
p_c_reversal = np.full((8), capillary_pressure_dr(porosity) )
p_c_endpoint = np.full((8), capillary_pressure_dr(e_wr+0.003 ) )


# initialize the scaling points
process_new  = 'dr'
process_old  = 'dr'

e_w_old      = ew_bc(0)
p_c_old      = capillary_pressure_dr(e_w_old)

ve_w_hysteresis = np.zeros([800], dtype = np.double)
vp_c_hysteresis = np.zeros([800], dtype = np.double)

ve_w_hysteresis[0] = e_w_old
vp_c_hysteresis[0] = p_c_old

for i in range(1,300):
    t=i
    e_w = ew_bc(t)


    process_old, process_new = update_process(e_w-e_w_old, process_old, process_new)
    update_scaling_points(process_new,process_old,e_w_old,p_c_old,e_w_reversal, e_w_endpoint, p_c_reversal, p_c_endpoint)

    e_nr = residual_volume_fraction_nw(e_w_reversal[0])
    e_nt = entrapped_volume_fraction_nw(e_w,e_nr,e_w_reversal[0])
    e_a  = e_w + e_nt
    p_c = capillary_pressure_hysteresis(e_a, process_new, e_w_reversal[0], e_w_endpoint[0], p_c_reversal[0], p_c_endpoint[0])
    e_w_old = e_w
    p_c_old = p_c
    vp_c_hysteresis[i] = p_c
    ve_w_hysteresis[i] = e_w

# Visualization
for i in range(0,N):
    vp_c_dr[i]  = capillary_pressure_dr(ve_w_dr[i])
    vp_c_imb[i] = capillary_pressure_imb(ve_w_imb[i])



#plt.plot(ve_w_hysteresis, vp_c_hysteresis, label='scanning curves',color='r')
plt.plot(ve_w_imb, vp_c_imb, label="MIC VG",linestyle="dashed",color='k')
plt.plot(ve_w_dr, vp_c_dr , label="MDC VG",linestyle="dashed",color='k')
plt.plot(ve_w_hysteresis[0:300], vp_c_hysteresis[0:300] , label="Scanning curve",color='r')
plt.xlabel('e_w')
plt.ylabel('p_c')
plt.title('Hysteresis pc-ew curves')
plt.legend()
plt.show()





N =2000
ve_a_dr = np.zeros([N], dtype = np.double)
ve_a_imb = np.zeros([N], dtype = np.double)
vpc = np.zeros([N], dtype = np.double)
for i in range(0,N):
    vpc[i] = 2*i
    e_w  = volume_fraction_imb(vpc[i])
    e_nt = entrapped_volume_fraction_nw(e_w,e_nr_imb,e_wr)
    ve_a_imb[i] = apparent_volume_fraction(e_w,e_nt)
    ve_a_dr[i]  = volume_fraction_dr(vpc[i])


plt.plot(ve_a_imb,vpc, label="Scanning curve",color='r')
plt.plot(ve_a_dr,vpc, label="Scanning curve",color='g')
plt.xlabel('e_a')
plt.ylabel('pc')
plt.title('Hysteresis pc-ew curves')
plt.show()
