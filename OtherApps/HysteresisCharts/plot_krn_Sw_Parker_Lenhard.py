#just a visualization of P-S realtions with imb/dr coefficients
import math
import numpy as np
import matplotlib.pyplot as plt

############
phi=0.41
n= 6.0


Epsilon_nr_imb=phi*0.2
Epsilon_wr_imb=phi*0.13
####dr
Epsilon_nr_dr=phi*0.0
Epsilon_wr_dr=phi*0.13
Const_Epsilon_dr   =phi-Epsilon_nr_dr - Epsilon_wr_dr
Const_Epsilon_imb  =phi-Epsilon_nr_imb - Epsilon_wr_imb
Sm=0.13
S_nr=0.2

###
m = 1.0 - (1.0 / n)
m_inv=1.0/m
m2=2.0*m
 

##############################################################################
##############################################################################
##############################################################################
##############################################################################
 
 


#######################################
def  Krn_of_Ew(Snf_e,Sw_a):
    Krn = (Snf_e**0.5)*( 1 - (Sw_a)**m_inv  )**m2
    return Krn
####################################### 
def  Krn_of_Ew_dr(  Epsilon_w):
    Se = (Epsilon_w - Epsilon_wr_dr) / Const_Epsilon_dr
    if (Se >= 0.0 and Se < 1.0):
       Krn = (  (1.0 - Se)**0.5 )* (1.0 - Se**m_inv)**m2 
    else:
       Krn = 0.0
    return Krn
####################################### 
def  Krn_of_Ew_imb(  Epsilon_w):
    Se = (Epsilon_w - Epsilon_wr_imb) / Const_Epsilon_imb
    if (Se >= 0.0 and Se < 1.0):
       Krn = (  (1.0 - Se)**0.5 )* (1.0 - Se**m_inv)**m2 
    else:
       Krn = 0.0
    return Krn
#######################################
def  Krw_of_Ew(Sw_e,Snr_e,Sw_a,Sw_reversal_e):
    Ex1=( 1 - (Snr_e/(1-Sw_reversal_e)) )*((1-Sw_a**m_inv)**m)
    Ex2= (Snr_e/(1-Sw_reversal_e))*((1-Sw_reversal_e**m_inv)**m)
    Krw = (Sw_e**0.5)*(( 1 -Ex1- Ex2 )**2)
    return Krw
#######################################
def  Krw_of_Ew_dr( Epsilon_w):
    Se = (Epsilon_w - Epsilon_wr_dr) / Const_Epsilon_dr
    if (Se >= 0.0 and Se < 1.0):
       Krw = (Se**0.5)  *(  (1.0 - (  (1.0 - Se**m_inv)**m  )   )**2.0)
    else:
       Krw = 1.0
    return Krw
def  Krw_of_Ew_imb( Epsilon_w):
    Se = (Epsilon_w - Epsilon_wr_imb) / Const_Epsilon_imb
    if (Se >= 0.0 and Se < 1.0):
       Krw = (Se**0.5)  *(  (1.0 - (  (1.0 - Se**m_inv)**m )   )**2.0)
    else:
       Krw = 1.0
    return Krw
####################################### 
#######################################
def Sw_a_func(Sw_reversal,Sw): 
    ############
    Sw_e=(Sw - Sm) / (1-Sm)
    Sw_e_reversal=(Sw_reversal - Sm) / (1-Sm)
    ############
    Snri_e=S_nr/(1-Sm)
    Rnw=(1/Snri_e)-1
    ############
    Snr_e=(1-Sw_e_reversal)/(1+Rnw*(1-Sw_e_reversal))
    ############
    Snt_e=Snr_e*(Sw_e - Sw_e_reversal)/(1-Sw_e_reversal-Snr_e)
    ############
    Sw_a=Sw_e+Snt_e
    return Sw_a
#######################################
####################################### 
def Snt_e_func(Sw_reversal,Sw): 
    ############
    Sw_e=(Sw - Sm) / (1-Sm)
    Sw_e_reversal=(Sw_reversal - Sm) / (1-Sm)
    ############
    Snri_e=S_nr/(1-Sm)
    Rnw=(1/Snri_e)-1
    ############
    Snr_e=(1-Sw_e_reversal)/(1+Rnw*(1-Sw_e_reversal))
    ############
    Snt_e=Snr_e*(Sw_e - Sw_e_reversal)/(1-Sw_e_reversal-Snr_e)
    return Snt_e
 
 
def Snr_e_func(Sw_reversal): 
    ############
    Sw_e_reversal=(Sw_reversal - Sm) / (1-Sm)
    ############
    Snri_e=S_nr/(1-Sm)
    Rnw=(1/Snri_e)-1
    ############
    Snr_e=(1-Sw_e_reversal)/(1+Rnw*(1-Sw_e_reversal))
    ############
    return Snr_e
 
    
########################################################################
########################################################################
########################################################################
########################################################################
N=1001
Sw_reversal_imb1       = 0.13
Sw_reversal_imb2       = 0.2
Sw_reversal_imb3       = 0.5

 

 
VSw_new_imb1           = np.linspace(Sw_reversal_imb1, 1, N)
VSw_new_imb2           = np.linspace(Sw_reversal_imb2, 1, N)
VSw_new_imb3           = np.linspace(Sw_reversal_imb3, 1, N)

VSw_new_dr             = np.linspace(Epsilon_wr_dr/phi, 1, N)

 
Vkrn_new_imb1     = np.empty([N], dtype=np.double) 
Vkrn_new_imb2     = np.empty([N], dtype=np.double) 
Vkrn_new_imb3     = np.empty([N], dtype=np.double) 
Vkrn_new_dr_VG    = np.empty([N], dtype=np.double) 
 

Vkrw_new_imb1     = np.empty([N], dtype=np.double) 
Vkrw_new_imb2     = np.empty([N], dtype=np.double) 
Vkrw_new_imb3     = np.empty([N], dtype=np.double) 
Vkrw_new_dr_VG    = np.empty([N], dtype=np.double) 
 
###

###
 


########################################################################
########################################################################
########################################################################
########################################################################
for i in range(0,N):
    ############ calculate Sw_a_new ############
    Sw_a=Sw_a_func(Sw_reversal_imb1,VSw_new_imb1[i])
    Snt_e=Snt_e_func(Sw_reversal_imb1,VSw_new_imb1[i])
    Sn=1-VSw_new_imb1[i]
    Sn_e=Sn/(1-Sm)
    Snf_e=Sn_e-Snt_e
    Vkrn_new_imb1[i]=Krn_of_Ew(Snf_e,Sw_a)
    
    Sw_a=Sw_a_func(Sw_reversal_imb2,VSw_new_imb2[i])
    Snt_e=Snt_e_func(Sw_reversal_imb2,VSw_new_imb2[i])
    Sn=1-VSw_new_imb2[i]
    Sn_e=Sn/(1-Sm)
    Snf_e=Sn_e-Snt_e
    Vkrn_new_imb2[i]=Krn_of_Ew(Snf_e,Sw_a)
    
    Sw_a=Sw_a_func(Sw_reversal_imb3,VSw_new_imb3[i])
    Snt_e=Snt_e_func(Sw_reversal_imb3,VSw_new_imb3[i])
    Sn=1-VSw_new_imb3[i]
    Sn_e=Sn/(1-Sm)
    Snf_e=Sn_e-Snt_e
    Vkrn_new_imb3[i]=Krn_of_Ew(Snf_e,Sw_a)
    
    
    Epsilon_w=VSw_new_dr[i]*phi
    Vkrn_new_dr_VG[i]  =  Krn_of_Ew_dr(  Epsilon_w)
    Sw_a             = Sw_a_func(Sw_reversal_imb1,VSw_new_imb1[i])
    Sw_e             = (VSw_new_imb1[i]-Sm)/(1-Sm)
    Snr_e            = Snr_e_func(Sw_reversal_imb1)
    Sw_reversal_e    = (Sw_reversal_imb1 - Sm) / (1-Sm)
    Vkrw_new_imb1[i] = Krw_of_Ew(Sw_e,Snr_e,Sw_a,Sw_reversal_e)
    
    Sw_a             = Sw_a_func(Sw_reversal_imb2,VSw_new_imb2[i])
    Sw_e             = (VSw_new_imb2[i]-Sm)/(1-Sm)
    Snr_e            = Snr_e_func(Sw_reversal_imb2)
    Sw_reversal_e    = (Sw_reversal_imb2 - Sm) / (1-Sm)
    Vkrw_new_imb2[i] = Krw_of_Ew(Sw_e,Snr_e,Sw_a,Sw_reversal_e)
    
    Sw_a             = Sw_a_func(Sw_reversal_imb3,VSw_new_imb3[i])
    Sw_e             = (VSw_new_imb3[i]-Sm)/(1-Sm)
    Snr_e            = Snr_e_func(Sw_reversal_imb3)
    Sw_reversal_e    = (Sw_reversal_imb3 - Sm) / (1-Sm)
    Vkrw_new_imb3[i] = Krw_of_Ew(Sw_e,Snr_e,Sw_a,Sw_reversal_e)
    
    Epsilon_w=VSw_new_dr[i]*phi
    Vkrw_new_dr_VG[i]  = Krw_of_Ew_dr( Epsilon_w)

    
plt.plot(VSw_new_imb1, Vkrn_new_imb1,label=r'$k^n_{r} $' 'Main imb')
plt.plot(VSw_new_imb2, Vkrn_new_imb2,label=r'$k^n_{r} $' 'scanning imb1')
plt.plot(VSw_new_imb3, Vkrn_new_imb3,label=r'$k^n_{r} $' 'scanning imb2')
plt.plot(VSw_new_dr, Vkrn_new_dr_VG,label=r'$k^n_{r} $' ' Main dr')
plt.plot(VSw_new_imb1, Vkrw_new_imb1,label=r'$k^w_{r} $' 'Main imb')
plt.plot(VSw_new_imb2, Vkrw_new_imb2,label=r'$k^w_{r} $' 'scanning imb1')
plt.plot(VSw_new_imb3, Vkrw_new_imb3,label=r'$k^w_{r} $' 'scanning imb2')
plt.plot(VSw_new_dr, Vkrw_new_dr_VG,label=r'$k^w_{r} $' ' Main dr')

plt.legend()
plt.xlabel('Sw')
plt.ylabel('krn-krw') 
plt.show()

 
