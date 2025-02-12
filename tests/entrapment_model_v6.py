"""
hysteresis sw and pc (multiple endpoints saving)
"""
import math
import numpy as np
import matplotlib.pyplot as plt


#imb
alpha_imb = 9e-4
n_imb     = 6.0
s_nr_max  = 0.2
s_wr = 0.1
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
htol = 1e-8


# VAn Genuchten functions
def capillary_pressure_dr( s_w):
    """
    Capillary pressure function
    """
    se = (s_w - s_wr) / (1  - s_wr)

    if se > 0.0 and se < 1.0:
        p_c = (  ( se**(-m_inv_dr)  - 1.0)**n_inv_dr )*alpha_inv_dr
    else:
        p_c = 0.0
    return p_c

def capillary_pressure_imb( s_w):
    """
    Capillary pressure function
    """
    se = (s_w - s_wr) / (1 - s_nr_max  - s_wr)
    if se > 0.0 and se < 1.0:
        p_c = (  ( se**(-m_inv_imb)  - 1.0)**n_inv_imb )*alpha_inv_imb
    else:
        p_c = 0.0
    return p_c

def saturation_w_dr(p_c):
    """
    saturation w phase function
    """
    if p_c > 0.0:
        se =  (1.0 + (alpha_dr*p_c)**n_dr)**(-m_dr)
        s_w =(1  - s_wr)*  se + s_wr
    else:
        s_w = 1
    return s_w

def saturation_w_imb(p_c):
    """
    saturation w phase function
    """
    if p_c > 0.0:
        se =  (1.0 + (alpha_imb*p_c)**n_imb)**(-m_imb)
        s_w =(1 - s_nr_max  - s_wr)*  se + s_wr
    else:
        s_w = 1-s_nr_max
    return s_w

def effective_saturation_w(s_w):
    """
    effective saturation w phase function
    """

    se= (s_w-s_wr)/(1  - s_wr)

    return se


# entrapment functions
def Lands_residual_saturation_nw(s_w_reversal):
    """
    residual nw saturation  from Lands formula
    """
    if s_w_reversal==1 or s_nr_max==0:
        s_nr = 0
    else:
        s_nr = 1/(  1/(1-s_w_reversal) +   1/(s_nr_max) - 1/(1-s_wr) )

    return s_nr


def linear_effective_entrapped_saturation_nw(s_w,s_w_reversal,s_w_endpoint,s_nt_e_reversal,s_nt_e_endpoint):
    """
    Linear effective entrapped nw saturation from Parker Lenhard formula  Snt = a * S_w + b
    """
    a = (s_nt_e_endpoint-s_nt_e_reversal)/(s_w_endpoint-s_w_reversal)
    b = s_nt_e_endpoint - a * s_w_endpoint
    s_nt_e = a * s_w + b

    return s_nt_e


# scaling functions
def main_apparent_saturation_dr(p_c):
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

def main_capillary_pressure_dr(s_a):
    """
    Capillary pressure function
    """

    if s_a > 0.0 and s_a < 1.0:
        p_c = (  ( s_a**(-m_inv_dr)  - 1.0)**n_inv_dr )*alpha_inv_dr
    else:
        p_c = 0.0
    return p_c

def main_capillary_pressure_imb(s_a):
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




def sw_bc(t):
    """
    Boundary volume fraction
    """
    tr1 = Lands_residual_saturation_nw(0.5)
    sw_t = [0.3, 0.10001, 0.8, 0.65, 0.8]
    time = [0 ,  2000, 4000,  6000 , 8000]

    if time[0]<=t<=time[1]:
        a   = (sw_t[1] - sw_t[0])/(time[1] - time[0])
        b   = sw_t[0] - a * time[0]
        sw =  a * t + b
    elif time[1]<t<=time[2]:
        a   = (sw_t[2] - sw_t[1])/(time[2] - time[1])
        b   = sw_t[1] - a * time[1]
        sw = a * t + b
    elif time[2]<t<=time[3]:
        a   = (sw_t[3] - sw_t[2])/(time[3] - time[2])
        b   = sw_t[2] - a * time[2]
        sw =  a * t + b
    elif time[3]<t<=time[4]:
        a   = (sw_t[4] - sw_t[3])/(time[4] - time[3])
        b   = sw_t[3] - a * time[3]
        sw =  a * t + b
    return  sw


N=8001
vs_w_hysteresis = np.zeros([N], dtype = np.double)
vp_c_hysteresis = np.zeros([N], dtype = np.double)
vs_a_hysteresis = np.zeros([N], dtype = np.double)

# initialize the scaling points
process_new  = 'dr'
process_old  = 'dr'
path_type    = 0

s_w_reversal = 1.0
s_w_endpoint = s_wr + 1e-6
p_c_reversal = capillary_pressure_dr(s_w_reversal)
p_c_endpoint = capillary_pressure_dr(s_w_endpoint)

s_a_reversal = (s_w_reversal-s_wr)/(1-s_wr)
s_a_endpoint = (s_w_endpoint-s_wr)/(1-s_wr)
s_nt_e_endpoint = 0.0
s_nt_e_reversal = 0.0


s_w_old      = sw_bc(0)
p_c_old      = capillary_pressure_dr(s_w_old)
s_a_old      = (s_w_old-s_wr)/(1-s_wr)
s_nt_e_old   = 0.0
vs_w_hysteresis[0] = s_w_old
vp_c_hysteresis[0] = p_c_old
vs_a_hysteresis[0] = s_a_old



for i in range(1,N):
    s_w = sw_bc(i)

    # Determine the current process
    if s_w-s_w_old > htol:
        process_old = process_new
        process_new = 'imb'
    elif s_w-s_w_old < -htol:
        process_old = process_new
        process_new = 'dr'
    else:
        process_old = process_new

    # Update reversal and endpoints (s_a and pc)
    if process_new != process_old:
        path_type = path_type + 1
        s_a_endpoint  = s_a_reversal
        s_a_reversal  = s_a_old
        if path_type == 1:
            s_w_reversal  = s_w_old
            s_w_endpoint  = 1-Lands_residual_saturation_nw(s_w_reversal)
            p_c_endpoint  = 0.0
            s_nt_e_endpoint = Lands_residual_saturation_nw(s_w_reversal)/(1-s_wr)
            s_nt_e_reversal = 0.0
        else:
            s_w_endpoint  = s_w_reversal
            s_w_reversal  = s_w_old
            p_c_endpoint  = p_c_reversal
            s_nt_e_endpoint = s_nt_e_reversal
            s_nt_e_reversal = s_nt_e_old

        p_c_reversal  = p_c_old

    # Update apparent saturation
    s_nt_e_old = linear_effective_entrapped_saturation_nw(s_w,s_w_reversal,s_w_endpoint,s_nt_e_reversal,s_nt_e_endpoint)
    s_a = effective_saturation_w(s_w) +  s_nt_e_old

    # Calculate p_c
    p_c    = capillary_pressure_hysteresis(s_a, process_new, s_a_reversal, s_a_endpoint, p_c_reversal, p_c_endpoint)

    s_w_old = s_w
    s_a_old = s_a
    p_c_old = p_c

    vs_w_hysteresis[i] = s_w
    vp_c_hysteresis[i] = p_c
    vs_a_hysteresis[i] = s_a


N2=1000
vp_c_dr=np.zeros([N2], dtype = np.double)
vp_c_imb=np.zeros([N2], dtype = np.double)
vs_a_dr = np.linspace(0.0, 1.0, N2)
vs_a_imb = np.linspace(0.0, 1.0, N2)

vs_w_dr = np.linspace(s_wr, 1.0, N2)
vs_w_imb = np.linspace(s_wr, 1.0-s_nr_max, N2)

# Visualization
for i in range(0,N2):
    vp_c_dr[i]  = capillary_pressure_dr(vs_w_dr[i])
    vp_c_imb[i] = capillary_pressure_imb(vs_w_imb[i])


plt.plot(vs_w_imb[1:], vp_c_imb[1:], label="MIC VG",linestyle="dashed",color='b')
plt.plot(vs_w_dr[1:], vp_c_dr[1:] , label="MDC VG",linestyle="dashed",color='r')
plt.plot(vs_w_hysteresis[0:N], vp_c_hysteresis[0:N] , label="Second drainge process (scanning curve)",color='k')
plt.xlabel('s_w')
plt.ylabel('p_c')
plt.title('Hysteresis pc-sw curves')
plt.legend()
plt.show()

plt.plot(vs_w_imb[1:], vp_c_imb[1:], label="MIC VG",linestyle="dashed",color='b')
plt.plot(vs_w_dr[1:], vp_c_dr[1:] , label="MDC VG",linestyle="dashed",color='r')
plt.plot(vs_w_hysteresis[0:N], vp_c_hysteresis[0:N] , label="Second drainge process (scanning curve)",color='k')
plt.xlabel('s_w')
plt.ylabel('p_c')
plt.title('Hysteresis pc-sw curves points')
plt.legend()
plt.show()


# Visualization
for i in range(0,N2):
    vp_c_dr[i]  = main_capillary_pressure_dr(vs_a_dr[i])
    vp_c_imb[i] = main_capillary_pressure_imb(vs_a_imb[i])

plt.plot(vs_a_imb[1:], vp_c_imb[1:], label="MIC VG",linestyle="dashed",color='b')
plt.plot(vs_a_dr[1:], vp_c_dr[1:] , label="MDC VG",linestyle="dashed",color='r')
plt.plot(vs_a_hysteresis[0:N], vp_c_hysteresis[0:N] , label="Second drainge process (scanning curve)",color='k')
plt.xlabel('s_a')
plt.ylabel('p_c')
plt.title('Hysteresis pc-sa curves')
plt.legend()
plt.show()
























