# the parameters are for coarse sand

import time
import sys
import math
import numpy as np
import cython
import matplotlib.pyplot as plt
from libc.stdlib cimport malloc

############
phi=1.0
############

alpha_imb= 9e-4
n_imb= 6.0
alpha_dr=6e-4
n_dr=  9.0
############
Epsilon_nr= 0.0
Epsilon_wr= 0.0
Const_Epsilon=phi-Epsilon_nr - Epsilon_wr


###
m_imb = 1.0 - (1.0 / n_imb)
n_inv_imb=1.0/n_imb
n_inv1_imb=n_inv_imb-1.0
n1_imb=n_imb-1.0
m_inv_imb=1.0/m_imb
m1_imb=-m_imb-1.0
m2_imb=2.0*m_imb
alpha_inv_imb=1.0/alpha_imb
alpha_m_n_imb=alpha_imb*m_imb*n_imb
###
m_dr = 1.0 - (1.0 / n_dr)
n_inv_dr=1.0/n_dr
n_inv1_dr=n_inv_dr-1.0
n1_dr=n_dr-1.0
m_inv_dr=1.0/m_dr
m1_dr=-m_dr-1.0
m2_dr=2.0*m_dr
alpha_inv_dr=1.0/alpha_dr
alpha_m_n_dr=alpha_dr*m_dr*n_dr
######################



tol=1e-9
#######################################
#######################################
cdef double Pc_imb(double Epsilon_w):
    Se = (Epsilon_w - Epsilon_wr) / Const_Epsilon
    if (Se > 0.0 and Se < 1.0):
       P_c = (  ( Se**(-m_inv_imb)  - 1.0)**n_inv_imb )*alpha_inv_imb
    else:
       P_c = 0.0
    return P_c
#######################################
cdef double  Pc_dr(double Epsilon_w):
    Se = (Epsilon_w - Epsilon_wr) / Const_Epsilon
    if (Se > 0.0 and Se < 1.0):
       P_c = (  ( Se**(-m_inv_dr)  - 1.0)**n_inv_dr )*alpha_inv_dr
    else:
       P_c = 0.0
    return P_c

#######################################
cdef double  Epsilon_w_of_Pc_imb(double  P_c):
    if (P_c > 0.0):
       Se =  (1.0 + (alpha_imb*P_c)**n_imb)**(-m_imb)
       Epsilon_w =Const_Epsilon*  Se + Epsilon_wr
    else:
       Epsilon_w = phi-Epsilon_nr
    return Epsilon_w
#######################################
cdef double  Epsilon_w_of_Pc_dr(double  P_c):
    if (P_c > 0.0):
       Se =  (1.0 + (alpha_dr*P_c)**n_dr)**(-m_dr)
       Epsilon_w =Const_Epsilon*  Se + Epsilon_wr
    else:
       Epsilon_w = phi-Epsilon_nr
    return Epsilon_w


#######################################
cdef double Pc_Hyst_Parker(double Ew, Ih_new,double Ew_reversal,double Ew_endpoint,double Pc_reversal,double Pc_endpoint):
    cdef double Pc_tn
    if(Ih_new==-1):
      Ew_reversal_MPC=Epsilon_w_of_Pc_imb(Pc_reversal)
      Ew_endpoint_MPC=Epsilon_w_of_Pc_imb(Pc_endpoint)
      Pc_tn=Pc_imb( ((Ew-Ew_endpoint)/(Ew_reversal-Ew_endpoint))*(Ew_reversal_MPC-Ew_endpoint_MPC) + Ew_endpoint_MPC)
    elif(Ih_new==1):
      Ew_reversal_MPC=Epsilon_w_of_Pc_dr(Pc_reversal)
      Ew_endpoint_MPC=Epsilon_w_of_Pc_dr(Pc_endpoint)
      Pc_tn=Pc_dr( ((Ew-Ew_endpoint)/(Ew_reversal-Ew_endpoint))*(Ew_reversal_MPC-Ew_endpoint_MPC) + Ew_endpoint_MPC)
    return Pc_tn

#######################################
cdef  double Epsilon_w_of_Pc_Hyst_Parker(double Pc_tn, Ih_new,double Ew_reversal,double Ew_endpoint,double Pc_reversal,double Pc_endpoint):
    cdef double Epsilon_w
    #######
    if(Ih_new==-1):
      Ew_reversal_MPC=Epsilon_w_of_Pc_imb(Pc_reversal)
      Ew_endpoint_MPC=Epsilon_w_of_Pc_imb(Pc_endpoint)
      Epsilon_w=(((Epsilon_w_of_Pc_imb(Pc_tn)-Ew_endpoint_MPC)/(Ew_reversal_MPC-Ew_endpoint_MPC))*(Ew_reversal-Ew_endpoint))+Ew_endpoint
    elif(Ih_new==1):
      Ew_reversal_MPC=Epsilon_w_of_Pc_dr(Pc_reversal)
      Ew_endpoint_MPC=Epsilon_w_of_Pc_dr(Pc_endpoint)
      Epsilon_w=(((Epsilon_w_of_Pc_dr(Pc_tn)-Ew_endpoint_MPC)/(Ew_reversal_MPC-Ew_endpoint_MPC))*(Ew_reversal-Ew_endpoint))+Ew_endpoint
    #######
    return Epsilon_w

#######################################




#######################################
cdef double Ew_func_time(double t):
    if(t>0 and t<=100):
      a=(phi-0.01)/(100-0)
      b=0.01-a*0
      Ew=a*t+b
    elif(t>100 and t<=200):
      a=(0.1-phi)/(200-100)
      b=phi-a*100
      Ew=a*t+b
    elif(t>200 and t<=300):
      a=(0.5-0.1)/(300-200)
      b=0.1-a*200
      Ew=a*t+b
    elif(t>300 and t<=400):
      a=(0.2-0.5)/(400-300)
      b=0.5-a*300
      Ew=a*t+b
    elif(t>400 and t<=500):
      a=(0.3-0.5)/(500-400)
      b=0.5-a*400
      Ew=a*t+b
    return Ew

#######################################
cdef void main():



    VTime = np.linspace(1, 500,1000 )
    VEw_hyst=np.zeros([len(VTime)], dtype=np.double)
    VPc_hyst=np.zeros([len(VTime)], dtype=np.double)
    ####initialization
    VEw_hyst[0]= 0.01
    VPc_hyst[0]= Pc_imb(VEw_hyst[0])
    VEw_reversal= 0.00003
    VEw_endpoint= phi
    VPc_reversal=Pc_imb(VEw_reversal)
    VPc_endpoint=Pc_imb(VEw_endpoint)
    VPc_endpoint_old=VPc_endpoint

    Ew_Old = VEw_hyst[0]
    VIh_new=-1
    VIh_old=-1
    for i in range(1,len(VTime)):
        VEw_hyst[i]=Ew_func_time(VTime[i])
        if(VEw_hyst[i]-VEw_hyst[i-1]>=tol):
          VIh_old= VIh_new
          VIh_new= -1
        else:
          VIh_old=VIh_new
          VIh_new= 1
        if(VIh_new != VIh_old):
          VPc_endpoint= VPc_reversal
          VPc_reversal= Pc_Hyst_Parker(Ew_Old,VIh_old,VEw_reversal,VEw_endpoint,VPc_reversal, VPc_endpoint_old)
          VPc_endpoint_old= VPc_endpoint
          VEw_endpoint=VEw_reversal
          VEw_reversal=Ew_Old #VEw_hyst[i]
        VPc_hyst[i]=Pc_Hyst_Parker(VEw_hyst[i],VIh_new,VEw_reversal,VEw_endpoint,VPc_reversal, VPc_endpoint)
        Ew_Old =VEw_hyst[i]
    VEpsilon_w = np.linspace(0.003, phi, 1000)
    VPc_imb=np.zeros([len(VEpsilon_w)], dtype=np.double)
    VPc_dr=np.zeros([len(VEpsilon_w)], dtype=np.double)
    for i in range(len(VEpsilon_w)):
          VPc_dr[i] =Pc_dr(VEpsilon_w[i])
          VPc_imb[i]=Pc_imb(VEpsilon_w[i])

    #### Visualization
    plt.rcParams.update({
    "text.usetex": True,
    "font.family": "monospace",
    "font.monospace": 'Computer Modern Typewriter'
    })

    plt.plot(VEpsilon_w,VPc_imb,label="MIC VG",linestyle="dashed")
    plt.plot(VEpsilon_w,VPc_dr,label="PDC VG",linestyle="dashed")
    plt.plot(VEw_hyst,VPc_hyst,color='k',label="Scanning curves")
    plt.legend()
    plt.xlabel(r'\LARGE $\bar S^{wa}$')
    plt.ylabel(r'\LARGE $p^c $')
    plt.show()








main()
