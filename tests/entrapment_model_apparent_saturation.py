"""
hysteresis sw and pc (multiple endpoints saving)
"""
import math
import numpy as np
import matplotlib.pyplot as plt


#imb
alpha_imb = 9e-4
n_imb     = 6.0
s_nr_max  = 0.4
s_wr = 0.25

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



# functions
def capillary_pressure_dr( s_w,s_nr):
    """
    Capillary pressure function
    """
    se = (s_w - s_wr) / (1 - s_nr  - s_wr)

    if se > 0.0 and se < 1.0:
        p_c = (  ( se**(-m_inv_dr)  - 1.0)**n_inv_dr )*alpha_inv_dr
    else:
        p_c = 0.0
    return p_c

def capillary_pressure_imb( s_w,s_nr):
    """
    Capillary pressure function
    """
    se = (s_w - s_wr) / (1 - s_nr  - s_wr)
    if se > 0.0 and se < 1.0:
        p_c = (  ( se**(-m_inv_imb)  - 1.0)**n_inv_imb )*alpha_inv_imb
    else:
        p_c = 0.0
    return p_c

def saturation_w_dr(p_c, s_nr):
    """
    saturation w phase function
    """
    if p_c > 0.0:
        se =  (1.0 + (alpha_dr*p_c)**n_dr)**(-m_dr)
        s_w =(1 - s_nr  - s_wr)*  se + s_wr
    else:
        s_w = 1-s_nr
    return s_w

def saturation_w_imb(p_c, s_nr):
    """
    saturation w phase function
    """
    if p_c > 0.0:
        se =  (1.0 + (alpha_imb*p_c)**n_imb)**(-m_imb)
        s_w =(1 - s_nr  - s_wr)*  se + s_wr
    else:
        s_w = 1-s_nr
    return s_w

def residual_saturation_nw(s_w_reversal):
    """
    residual nw saturation  from Lands formula
    """
    if s_w_reversal==1.0:
        s_nr = 0.0
    else:
        s_nr = 1/(  1/(1-s_w_reversal) +   1/(s_nr_max) - 1/(1-s_wr) )

    return s_nr

def effective_entrapped_saturation_nw(s_w,s_nr,s_w_reversal):
    """
    effective entrapped nw saturation  from Parker Lenhard formula
    """
    if s_nr == 0:
        s_nt_e = 0
    else:
        s_nt_e =  s_nr*(s_w-s_w_reversal)/( (1-s_wr)*(1-s_w_reversal-s_nr)    )

    return s_nt_e

def apparent_saturation(s_w_e,s_nt_e):
    """
    apparent saturation from Parker Lenhard formula
    """
    s_a = s_w_e + s_nt_e

    return s_a

# functions
def main_apparent_saturation_dr( p_c):
    """
    apparent saturation function
    """
    if p_c > 0.0:
        s_a =  (1.0 + (alpha_dr*p_c)**n_dr)**(-m_dr)
    else:
        s_a = 1.0
    return s_a

def main_apparent_saturation_imb(p_c):
    """
    apparent saturation function
    """
    if p_c > 0.0:
        s_a =  (1.0 + (alpha_imb*p_c)**n_imb)**(-m_imb)
    else:
        s_a = 1.0
    return s_a

def main_capillary_pressure_dr( s_a):
    """
    Capillary pressure function
    """

    if s_a > 0.0 and s_a < 1.0:
        p_c = (  ( s_a**(-m_inv_dr)  - 1.0)**n_inv_dr )*alpha_inv_dr
    else:
        p_c = 0.0
    return p_c

def main_capillary_pressure_imb( s_a):
    """
    Capillary pressure function
    """
    if s_a > 0.0 and s_a < 1.0:
        p_c = (  ( s_a**(-m_inv_imb)  - 1.0)**n_inv_imb )*alpha_inv_imb
    else:
        p_c = 0.0
    return p_c



def capillary_pressure_hysteresis( s_a, process, s_a_reversal, s_a_endpoint, p_c_reversal, p_c_endpoint):
    """
    hysteretic pc Parker Lenhard scaling model
    """
    if process=='imb':
        s_a_reversal_mpc = main_apparent_saturation_imb(p_c_reversal)
        s_a_endpoint_mpc = main_apparent_saturation_imb(p_c_endpoint)
        s_a_hysteretic   = ((s_a-s_a_endpoint)/(s_a_reversal-s_a_endpoint))*(s_a_reversal_mpc-s_a_endpoint_mpc) + s_a_endpoint_mpc
        p_c = main_capillary_pressure_imb(s_a_hysteretic)
    elif process=='dr':
        s_a_reversal_mpc = main_apparent_saturation_dr(p_c_reversal)
        s_a_endpoint_mpc = main_apparent_saturation_dr(p_c_endpoint)
        s_a_hysteretic   = ((s_a-s_a_endpoint)/(s_a_reversal-s_a_endpoint))*(s_a_reversal_mpc-s_a_endpoint_mpc) + s_a_endpoint_mpc
        p_c = main_capillary_pressure_dr(s_a_hysteretic)
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

def update_scaling_points(process_new, process_old, s_a, p_c, s_a_reversal, s_a_endpoint, p_c_reversal, p_c_endpoint):
    """
    function to update hysteresis  reversal and end points.
    """
    # Update reversal and endpoints (ew and pc)
    if process_new != process_old:
        p_c_endpoint  = p_c_reversal
        p_c_reversal  = p_c
        s_a_endpoint  = s_a_reversal
        s_a_reversal  = s_a


def sa_bc(t):
    """
    Boundary volume fraction
    """
    sa_t = [0.9, 0.2, 0.8, 0.2]
    time = [0. ,  100, 200,  300]

    if time[0]<=t<=time[1]:
        a   = (sa_t[1] - sa_t[0])/(time[1] - time[0])
        b   = sa_t[0] - a * time[0]
        sa =  a * t + b
    elif time[1]<t<=time[2]:
        a   = (sa_t[2] - sa_t[1])/(time[2] - time[1])
        b   = sa_t[1] - a * time[1]
        sa = a * t + b
    elif time[2]<t<=time[3]:
        a   = (sa_t[3] - sa_t[2])/(time[3] - time[2])
        b   = sa_t[2] - a * time[2]
        sa =  a * t + b
    return  sa


s_a_reversal = 1.0
s_a_endpoint = s_wr+0.0003
p_c_reversal = main_capillary_pressure_dr(s_a_reversal)
p_c_endpoint = main_capillary_pressure_dr(s_a_endpoint)


# initialize the scaling points
process_new  = 'dr'
process_old  = 'dr'

s_a_old      = sa_bc(0)
p_c_old      = main_capillary_pressure_dr(s_a_old)

vs_a_hysteresis = np.zeros([300], dtype = np.double)
vs_w_hysteresis = np.zeros([300], dtype = np.double)
vp_c_hysteresis = np.zeros([300], dtype = np.double)

vs_a_hysteresis[0] = s_a_old
vs_w_hysteresis[0] = s_a_old
vp_c_hysteresis[0] = p_c_old

for i in range(1,300):
    t=i
    s_a = sa_bc(t)
    # Switch process

    if s_a-s_a_old > htol:
        process_old = process_new
        process_new = 'imb'
    elif s_a-s_a_old < htol:
        process_old = process_new
        process_new = 'dr'

    # Update reversal and endpoints (s_a and pc)
    if process_new != process_old:
        p_c_endpoint  = p_c_reversal
        p_c_reversal  = p_c_old
        s_a_endpoint  = s_a_reversal
        s_a_reversal  = s_a_old

    p_c = capillary_pressure_hysteresis(s_a, process_new, s_a_reversal, s_a_endpoint, p_c_reversal, p_c_endpoint)
    s_a_old = s_a
    p_c_old = p_c
    vp_c_hysteresis[i] = p_c
    vs_a_hysteresis[i] = s_a




N=300
vp_c_dr=np.zeros([N], dtype = np.double)
vp_c_imb=np.zeros([N], dtype = np.double)
vs_a_dr = np.linspace(0.0, 1.0, N)
vs_a_imb = np.linspace(0.0, 1.0, N)

vs_w_dr = np.linspace(0.0, 1.0, N)
vs_w_imb = np.linspace(0.0, 1.0-s_nr_max, N)

# Visualization
for i in range(0,N):
    vp_c_dr[i]  = main_capillary_pressure_dr(vs_a_dr[i])
    vp_c_imb[i] = main_capillary_pressure_imb(vs_a_imb[i])


plt.plot(vs_a_imb, vp_c_imb, label="MIC VG",linestyle="dashed",color='b')
plt.plot(vs_a_dr, vp_c_dr , label="PDC VG",linestyle="dashed",color='green')
plt.plot(vs_a_hysteresis[0:300], vp_c_hysteresis[0:300] , label="Scanning curve",color='r')
plt.xlabel('s_apparent')
plt.ylabel('p_c')
plt.title('Hysteresis pc-ew curves')
plt.legend()
plt.show()



