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
        se  =  (1.0 + (alpha_imb*p_c)**n_imb)**(-m_imb)
        s_w = (1 - s_nr  - s_wr)*  se + s_wr
    else:
        s_w = 1-s_nr
    return s_w

def effective_saturation_w(s_w, s_nr):
    """
    effective saturation w phase function
    """

    se= (s_w-s_wr)/(1 - s_nr  - s_wr)

    return se


def Lands_residual_saturation_nw(s_w_reversal,process):
    """
    residual nw saturation  from Lands formula
    """
    if process =='dr':
        s_nr = 0.0
    else:
        s_nr = 1/(  1/(1-s_w_reversal) +   1/(s_nr_max) - 1/(1-s_wr) )

    return s_nr

def effective_entrapped_saturation_nw(s_w, s_nr, s_w_reversal):
    """
    effective entrapped nw saturation  from Parker Lenhard formula
    """
    if s_nr == 0:
        s_nt_e = 0
    else:
        s_nt_e =  s_nr*(s_w-s_w_reversal)/( (1-s_wr)*(1-s_w_reversal-s_nr)    )

    return s_nt_e

def apparent_saturation(s_w_e, s_nt_e):
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





N=5000

vs_a_imb_func = np.zeros([N], dtype = np.double)
vs_a_dr_func  = np.zeros([N], dtype = np.double)

vs_a_imb      = np.zeros([N], dtype = np.double)
vs_a_dr       = np.zeros([N], dtype = np.double)

vs_w_imb      = np.zeros([N], dtype = np.double)
vs_w_dr       = np.zeros([N], dtype = np.double)

vs_e_imb      = np.zeros([N], dtype = np.double)
vs_e_dr       = np.zeros([N], dtype = np.double)

vp_c = np.linspace(0.0, N, N)

# Visualization
for i in range(0,N):
    vs_a_imb_func[i]  = main_apparent_saturation_imb(vp_c[i])
    vs_a_dr_func[i]   = main_apparent_saturation_dr(vp_c[i])

    vs_w_imb[i]  = saturation_w_imb(vp_c[i], s_nr_max)
    vs_w_dr[i]   = saturation_w_dr(vp_c[i], 0)

    vs_e_imb[i]  = (vs_w_imb[i] -  s_wr) /(1-s_wr)
    vs_e_dr[i]   = (vs_w_dr[i] -  s_wr) /(1-s_wr)

    vs_a_imb[i]  = vs_e_imb[i] + effective_entrapped_saturation_nw(vs_w_imb[i], s_nr_max, s_wr +0.0003)
    vs_a_dr[i]   = vs_e_dr[i]

    


plt.plot(vs_a_imb_func, vp_c, label="apparent func imb"                     ,color='b')
plt.plot(vs_a_dr_func , vp_c, label="apparent func dr "  ,linestyle="dashed",color='r')
plt.plot(vs_a_imb, vp_c, label="apparent imb "                              ,color='g')
plt.plot(vs_a_dr, vp_c, label="apparent dr "             ,linestyle="dashed",color='k')
plt.xlabel('s_a')
plt.ylabel('p_c')
plt.title('Hysteresis pc-sa curves')
plt.legend()
plt.show()

plt.plot(vs_w_imb, vp_c, label="actual imb "                              ,color='b')
plt.plot(vs_w_dr, vp_c, label="actual dr "             ,linestyle="dashed",color='r')

plt.xlabel('s_w')
plt.ylabel('p_c')
plt.title('Hysteresis pc-sa curves check')
plt.legend()
plt.show()

plt.plot(vs_e_imb, vp_c, label="effective imb "                              ,color='b')
plt.plot(vs_e_dr, vp_c, label="effective dr "             ,linestyle="dashed",color='r')
plt.xlabel('s_e')
plt.ylabel('p_c')
plt.title('Hysteresis pc-sa curves check')
plt.legend()
plt.show()



plt.plot(vs_a_imb, vp_c, label="apparent imb "                              ,color='g')
plt.plot(vs_a_dr, vp_c, label="apparent dr "             ,linestyle="dashed",color='k')
plt.xlabel('s_a')
plt.ylabel('p_c')
plt.title('Hysteresis pc-sa curves check')
plt.legend()
plt.show()





















