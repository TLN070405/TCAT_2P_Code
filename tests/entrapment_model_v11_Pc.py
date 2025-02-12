"""
hysteresis sw and pc (multiple endpoints saving)
"""
import math
import numpy as np
import matplotlib.pyplot as plt

reversals_number= 0
exceeds_number = 0
#
porosity = 0.4

#imb
alpha_imb = 9e-4
n_imb     = 6.0
e_nr_imb  = 0.2*porosity
e_wr_imb  = 0.1*porosity

#dr
alpha_dr  = 6e-4
n_dr      = 9.0
e_nr_dr   = 0.
e_wr_dr   =  e_wr_imb

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


# entrapment functions
def lands_residual_effective_saturation_nw(s_a_reversal):
    """
    residual nw saturation  from Lands formula
    """
    if s_a_reversal == 1.0 or e_nr_imb == 0:
        s_nr_e = 0.0
    else:
        r_nw = ((porosity-e_wr_imb)/e_nr_imb)-1
        s_nr_e = (1 - s_a_reversal)/(1 + r_nw*(1 - s_a_reversal))

    return s_nr_e


def linear_effective_entrapped_saturation_nw(s, s_reversal, s_endpoint, s_nt_e_reversal, s_nt_e_endpoint):
    """
    Linear effective entrapped nw saturation from Parker Lenhard formula  snt_eff = a * s + b
    e : is a saturation could be actual or apparent
    """
    power=1
    a = (s_nt_e_endpoint-s_nt_e_reversal)/(s_endpoint**power-s_reversal**power)
    b = s_nt_e_endpoint - a * s_endpoint**power
    s_nt_e = a * (s**power)+ b

    return s_nt_e

def hysteretic_apparent_saturation_w( p_c, drainage_process, s_a_reversal, s_a_endpoint, p_c_reversal, p_c_endpoint):
    """
    hysteretic pc Parker Lenhard scaling model
    """
    if drainage_process == False:
        s_a_reversal_mpc = (volume_fraction_imb(p_c_reversal)-e_wr_imb)/(porosity - e_nr_imb - e_wr_imb)
        s_a_endpoint_mpc = (volume_fraction_imb(p_c_endpoint)-e_wr_imb)/(porosity - e_nr_imb - e_wr_imb)
        s_a              = (volume_fraction_imb(p_c)-e_wr_imb)/(porosity - e_nr_imb - e_wr_imb)
    elif drainage_process == True:
        s_a_reversal_mpc = (volume_fraction_dr(p_c_reversal)-e_wr_dr)/(porosity - e_nr_dr - e_wr_dr)
        s_a_endpoint_mpc = (volume_fraction_dr(p_c_endpoint)-e_wr_dr)/(porosity - e_nr_dr - e_wr_dr)
        s_a              = (volume_fraction_dr(p_c)-e_wr_dr)/(porosity - e_nr_dr - e_wr_dr)
    else:
        print("process should be either dr or imb")
    s_a_hyst = ((s_a-s_a_endpoint_mpc)/(s_a_reversal_mpc-s_a_endpoint_mpc))*(s_a_reversal-s_a_endpoint) + s_a_endpoint
    return s_a_hyst


def update_process(switch_condition):
    """
    function to update the current process 
    """
    drainage_process_old[0] = drainage_process_new[0]
    if switch_condition > htol:
        drainage_process_new[0] = False  #imb
    elif switch_condition < -htol:
        drainage_process_new[0] = True   #dr

def update_scaling_points(s_a,s_nt_e,p_c):
    """
    Update reversal and endpoints (s_a and pc)
    """
    global reversals_number,exceeds_number

    if drainage_process_new[0] != drainage_process_old[0]:
        reversals_number = reversals_number+1
        for i in range(max_reversal_points-1, 0, -1):
            p_c_endpoint[i]     = p_c_endpoint[i-1]
            p_c_reversal[i]     = p_c_reversal[i-1]
            s_a_endpoint[i]     = s_a_endpoint[i-1]
            s_a_reversal[i]     = s_a_reversal[i-1]
            s_nt_e_endpoint[i]  = s_nt_e_endpoint[i-1]
            s_nt_e_reversal[i]  = s_nt_e_reversal[i-1]

        #verify if we are on primary dr curve
        if reversals_number - 2*exceeds_number ==1:
            path_type[0] = 1
        else:
            path_type[0] = 0
        #update scaling points
        s_a_endpoint[0]  = s_a_reversal[0]
        s_a_reversal[0]  = s_a
        if path_type[0] == 1:
            p_c_endpoint[0]    = 0.0
            s_nt_e_endpoint[0] = lands_residual_effective_saturation_nw(s_a_reversal[0])
            s_nt_e_reversal[0] = 0.0
        else:
            p_c_endpoint[0]    = p_c_reversal[0]
            s_nt_e_endpoint[0] = s_nt_e_reversal[0]
            s_nt_e_reversal[0] = s_nt_e

        p_c_reversal[0]  = p_c

    if (drainage_process_new[0]== True and s_a <= s_a_endpoint[0]) or (drainage_process_new[0]== False and s_a >= s_a_endpoint[0]):

        exceeds_number = exceeds_number+1
        for i in range( 0, max_reversal_points-2):
            p_c_endpoint[i]     = p_c_endpoint[i+2]
            p_c_reversal[i]     = p_c_reversal[i+2]
            s_a_endpoint[i]     = s_a_endpoint[i+2]
            s_a_reversal[i]     = s_a_reversal[i+2]
            s_nt_e_endpoint[i]  = s_nt_e_endpoint[i+2]
            s_nt_e_reversal[i]  = s_nt_e_reversal[i+2]

## # # # # # # # #  Drainage process# # # # # # # # #
def pc_bc(t):
    pc_t = [100, 1500, 0.0, 1800, 0.0,2000,0.0]
    time = [0 ,   200, 400,  600, 800, 1000,1200]

    if time[0]<=t<=time[1]:
        a   = (pc_t[1] - pc_t[0])/(time[1] - time[0])
        b   = pc_t[0] - a * time[0]
        pc =  a * t + b
    elif time[1]<t<=time[2]:
        a   = (pc_t[2] - pc_t[1])/(time[2] - time[1])
        b   = pc_t[1] - a * time[1]
        pc = a * t + b
    elif time[2]<t<=time[3]:
        a   = (pc_t[3] - pc_t[2])/(time[3] - time[2])
        b   = pc_t[2] - a * time[2]
        pc =  a * t + b
    elif time[3]<t<=time[4]:
        a   = (pc_t[4] - pc_t[3])/(time[4] - time[3])
        b   = pc_t[3] - a * time[3]
        pc =  a * t + b
    elif time[4]<t<=time[5]:
        a   = (pc_t[5] - pc_t[4])/(time[5] - time[4])
        b   = pc_t[4] - a * time[4]
        pc =  a * t + b
    elif time[5]<t<=time[6]:
        a   = (pc_t[6] - pc_t[5])/(time[6] - time[5])
        b   = pc_t[5] - a * time[5]
        pc =  a * t + b
    return  pc


# initialize the scaling points
max_reversal_points=7

e_w_reversal = porosity
e_w_endpoint = e_wr_imb +1e-6

drainage_process_new  = np.full((1), True)
drainage_process_old  = np.full((1), True)
s_a_reversal    = np.full((max_reversal_points), (e_w_reversal-e_wr_imb)/(porosity-e_wr_imb))
s_a_endpoint    = np.full((max_reversal_points), (e_w_endpoint-e_wr_imb)/(porosity-e_wr_imb))
path_type       = np.full((1), 0)
p_c_reversal    = np.full((max_reversal_points), capillary_pressure_dr(e_w_reversal))
p_c_endpoint    = np.full((max_reversal_points), capillary_pressure_dr(e_w_endpoint))
s_nt_e_reversal = np.full((max_reversal_points), 0.0)
s_nt_e_endpoint = np.full((max_reversal_points), 0.0)


Nt = 1201
e_w_hysteresis = np.zeros([Nt], dtype = np.double)
p_c_hysteresis = np.zeros([Nt], dtype = np.double)
s_a_hysteresis = np.zeros([Nt], dtype = np.double)
s_nt_e_hysteresis = np.zeros([Nt], dtype = np.double)

e_w_hysteresis[0]    = volume_fraction_dr(pc_bc(0))
p_c_hysteresis[0]    = pc_bc(0)
s_nt_e_hysteresis[0] = 0.0
s_a_hysteresis[0]    = (e_w_hysteresis[0]-e_wr_imb)/(porosity-e_wr_imb) + s_nt_e_hysteresis[0]





# solve the problem by looping over time
for i in range(1,Nt):
    p_c = pc_bc(i)
    switch_condition = p_c_hysteresis[i-1]-p_c
    update_process(switch_condition)
    update_scaling_points(s_a_hysteresis[i-1],s_nt_e_hysteresis[i-1],p_c_hysteresis[i-1])

    # Update apparent saturation
    s_a = hysteretic_apparent_saturation_w(p_c,
                                            drainage_process_new[0],
                                            s_a_reversal[0],
                                            s_a_endpoint[0],
                                            p_c_reversal[0],
                                            p_c_endpoint[0])
    s_nt_e = linear_effective_entrapped_saturation_nw(s_a,
                                                      s_a_reversal[0],
                                                      s_a_endpoint[0],
                                                      s_nt_e_reversal[0],
                                                      s_nt_e_endpoint[0])

    e_w_hysteresis[i] = porosity* ((s_a - s_nt_e) * (1-e_wr_imb/porosity) + e_wr_imb/porosity)
    p_c_hysteresis[i] = p_c
    s_a_hysteresis[i] = s_a
    s_nt_e_hysteresis[i] = s_nt_e





# Visualization # Visualization# Visualization# Visualization# Visualization# Visualization


N2=1000
vp_c_dr=np.zeros([N2], dtype = np.double)
vp_c_imb=np.zeros([N2], dtype = np.double)
ve_w_dr = np.linspace(e_wr_imb, porosity, N2)
ve_w_imb = np.linspace(e_wr_imb, porosity-e_nr_imb, N2)

# Visualization ew
for i in range(0,N2):
    vp_c_dr[i]  = capillary_pressure_dr(ve_w_dr[i])
    vp_c_imb[i] = capillary_pressure_imb(ve_w_imb[i])


plt.plot(ve_w_imb[1:], vp_c_imb[1:], label="MIC VG",linestyle="dashed",color='b')
plt.plot(ve_w_dr[1:], vp_c_dr[1:] , label="MDC VG",linestyle="dashed",color='r')
plt.plot(e_w_hysteresis[0:Nt], p_c_hysteresis[0:Nt] , label="Second drainge process (scanning curve)",color='k')
plt.xlabel('s_w')
plt.ylabel('p_c')
plt.title('Hysteresis pc-sw curves')
plt.legend()
plt.show()


vs_a_dr = np.linspace(0.0, 1.0, N2)
vs_a_imb = np.linspace(0.0, 1.0, N2)
# Visualization sa
for i in range(0,N2):
    vp_c_dr[i]  = capillary_pressure_dr(vs_a_dr[i]*(porosity - e_nr_dr - e_wr_dr) + e_wr_dr)
    vp_c_imb[i] = capillary_pressure_imb(vs_a_imb[i]*(porosity - e_nr_imb - e_wr_imb) + e_wr_imb)


plt.plot(vs_a_imb[1:], vp_c_imb[1:], label="MIC VG",linestyle="dashed",color='b')
plt.plot(vs_a_dr[1:], vp_c_dr[1:] , label="MDC VG",linestyle="dashed",color='r')
plt.plot(s_a_hysteresis[0:Nt], p_c_hysteresis[0:Nt] , label="Second drainge process (scanning curve)",color='k')
plt.xlabel('s_a')
plt.ylabel('p_c')
plt.title('Hysteresis pc-sa curves')
plt.legend()
plt.show()
