#just a visualization of P-S realtions with imb/dr coefficients
import math
import numpy as np
import matplotlib.pyplot as plt

############
 




phi=0.41

####imb
alpha_imb= 9e-4
n_imb= 6.0
Epsilon_nr_imb=phi*0.2
Epsilon_wr_imb=phi*0.13
####dr
alpha_dr=6e-4
n_dr=  9.0
Epsilon_nr_dr=phi*0.0
Epsilon_wr_dr=phi*0.13

 
############




####
Const_Epsilon_dr=phi-Epsilon_nr_dr - Epsilon_wr_dr
Const_Epsilon_imb=phi-Epsilon_nr_imb - Epsilon_wr_imb
 
 
 
 

m_imb = 1.0 - (1.0 / n_imb)
n_inv_imb=1.0/n_imb
m_inv_imb=1.0/m_imb
alpha_inv_imb=1.0/alpha_imb
 
m_dr = 1.0 - (1.0 / n_dr)
n_inv_dr=1.0/n_dr
m_inv_dr=1.0/m_dr
alpha_inv_dr=1.0/alpha_dr
 
tol=1e-6
 
############
VEpsilon_w_imb = np.linspace(Epsilon_wr_imb, phi-Epsilon_nr_imb, 1000)
VSw_imb=VEpsilon_w_imb/phi
VPc_imb=np.empty([len(VEpsilon_w_imb)], dtype=np.double) 
VSw_a_imb=np.empty([len(VEpsilon_w_imb)], dtype=np.double) 
############
VEpsilon_w_dr = np.linspace(Epsilon_wr_dr, phi-Epsilon_nr_dr, 1000)
VSw_dr=VEpsilon_w_dr/phi
VPc_dr=np.empty([len(VEpsilon_w_dr)], dtype=np.double) 
VSw_a_dr=np.empty([len(VEpsilon_w_dr)], dtype=np.double) 
VSw_e_dr=(VEpsilon_w_dr - Epsilon_wr_dr) / Const_Epsilon_dr

 
 
############
def Pc_dr( Epsilon_w):
    Se = (Epsilon_w - Epsilon_wr_dr) / Const_Epsilon_dr
 
    if (Se > 0.0 and Se < 1.0):
       P_c = (  ( Se**(-m_inv_dr)  - 1.0)**n_inv_dr )*alpha_inv_dr
    else:
       P_c = 0.0
    return P_c
############
def Pc_imb( Epsilon_w):
    Se = (Epsilon_w - Epsilon_wr_imb) / Const_Epsilon_imb
 
    if (Se > 0.0 and Se < 1.0):
       P_c = (  ( Se**(-m_inv_imb)  - 1.0)**n_inv_imb )*alpha_inv_imb
    else:
       P_c = 0.0
    return P_c
############
def Epsilon_w_of_Pc_dr(P_c):
    if (P_c > 0.0):
       Se =  (1.0 + (alpha_dr*P_c)**n_dr)**(-m_dr)
       Epsilon_w =Const_Epsilon_dr*  Se + Epsilon_wr_dr
    else:
       Epsilon_w = phi-Epsilon_nr_dr
    return Epsilon_w
############
def Epsilon_w_of_Pc_imb(P_c):
    if (P_c > 0.0):
       Se =  (1.0 + (alpha_imb*P_c)**n_imb)**(-m_imb)
       Epsilon_w =Const_Epsilon_imb*  Se + Epsilon_wr_imb
    else:
       Epsilon_w = phi-Epsilon_nr_imb
    return Epsilon_w
############ 
def Sw_a_func(Ew_reversal,Ew): 
    ############
    Sw_e=(Ew - Epsilon_wr_dr) / Const_Epsilon_dr
    Sw_e_reversal=(Ew_reversal - Epsilon_wr_dr) / Const_Epsilon_dr
    ############
    Snri_e=Epsilon_nr_imb/Const_Epsilon_dr
    Rnw=(1/Snri_e)-1
    ############
    Snr_e=(1-Sw_e_reversal)/(1+Rnw*(1-Sw_e_reversal))
    ############
    Snt_e=Snr_e*(Sw_e - Sw_e_reversal)/(1-Sw_e_reversal-Snr_e)
    ############
    Sw_a=Sw_e+Snt_e
    return Sw_a
############ 
def Snt_e_func(Ew_reversal,Ew): 
    ############
    Sw_e=(Ew - Epsilon_wr_dr) / Const_Epsilon_dr
    Sw_e_reversal=(Ew_reversal - Epsilon_wr_dr) / Const_Epsilon_dr
    ############
    Snri_e=Epsilon_nr_imb/Const_Epsilon_dr
    Rnw=(1/Snri_e)-1
    ############
    Snr_e=(1-Sw_e_reversal)/(1+Rnw*(1-Sw_e_reversal))
    ############
    Snt_e=Snr_e*(Sw_e - Sw_e_reversal)/(1-Sw_e_reversal-Snr_e)
    return Snt_e
############

def Pc_Hyst_Parker_Land(Sw_a, Ih_new, Sw_a_reversal, Sw_a_endpoint, Pc_reversal, Pc_endpoint):
    if(Ih_new==-1):
      Sw_a_reversal_MPC= (Epsilon_w_of_Pc_imb(Pc_reversal)-Epsilon_wr_imb)/Const_Epsilon_imb
      Sw_a_endpoint_MPC= (Epsilon_w_of_Pc_imb(Pc_endpoint)-Epsilon_wr_imb)/Const_Epsilon_imb
      x=((Sw_a-Sw_a_endpoint)/(Sw_a_reversal-Sw_a_endpoint))*(Sw_a_reversal_MPC-Sw_a_endpoint_MPC) + Sw_a_endpoint_MPC
      Pc_tn=Pc_imb(x*Const_Epsilon_imb+ Epsilon_wr_imb)
    elif(Ih_new==1):
      Sw_a_reversal_MPC= (Epsilon_w_of_Pc_dr(Pc_reversal)-Epsilon_wr_dr)/Const_Epsilon_dr
      Sw_a_endpoint_MPC= (Epsilon_w_of_Pc_dr(Pc_endpoint)-Epsilon_wr_dr)/Const_Epsilon_dr
      x=((Sw_a-Sw_a_endpoint)/(Sw_a_reversal-Sw_a_endpoint))*(Sw_a_reversal_MPC-Sw_a_endpoint_MPC) + Sw_a_endpoint_MPC
      Pc_tn=Pc_dr(x*Const_Epsilon_dr+ Epsilon_wr_dr)
    return Pc_tn
     
 
############ 
def func_E_w0_bc( t): 
    Ew_t0= phi
    Ew_t1= 0.25
    Ew_t2= 0.3468
    Ew_t3= 0.3
    if(t>=0 and t<=500):
      aa=(Ew_t1-Ew_t0)/(500-0)
      bb=Ew_t0-aa*0
      E_w0=aa*t+bb
    elif(t>500 and t<=1000): 
      aa=(Ew_t2-Ew_t1)/(1000-500)
      bb=Ew_t1-aa*500
      E_w0=aa*t+bb
    elif(t>1000 and t<=1500): 
      aa=(Ew_t3-Ew_t2)/(1500-1000)
      bb=Ew_t2-aa*1000
      E_w0=aa*t+bb
    return E_w0
############



for i in range(len(VPc_dr)):
    VPc_dr[i] =Pc_dr(VEpsilon_w_dr[i])
    VPc_imb[i]=Pc_imb(VEpsilon_w_imb[i] )
    VSw_a_imb[i]=Sw_a_func(Epsilon_wr_imb,VEpsilon_w_imb[i])
    VSw_a_dr[i]=VSw_e_dr[i] 




############
N=1501
VTime = np.linspace(0, 1500, N)
VEw_new=np.empty([N], dtype=np.double) 
VPc_new=np.empty([N], dtype=np.double) 
VSw_a=np.empty([N], dtype=np.double) 
VSnt_e=np.empty([N], dtype=np.double)

VPc_new_1=np.empty([500], dtype=np.double) 
VPc_new_2=np.empty([500], dtype=np.double) 
VPc_new_3=np.empty([500], dtype=np.double) 
VEw_new_1=np.empty([500], dtype=np.double)
VEw_new_2=np.empty([500], dtype=np.double)
VEw_new_3=np.empty([500], dtype=np.double)
############Initialization of hyst 
Ih_new=1
Ih_old=1
Ew_old= func_E_w0_bc(0)
###
Ew_reversal=phi
Sw_a_reversal=(Ew_reversal-Epsilon_wr_dr)/Const_Epsilon_dr
###
Ew_endpoint= Epsilon_wr_imb+1e-5
Sw_a_endpoint=(Ew_endpoint-Epsilon_wr_dr)/Const_Epsilon_dr
###
Pc_reversal=Pc_dr(Ew_reversal)
Pc_endpoint=Pc_dr(Ew_endpoint)
Pc_endpoint_old=Pc_endpoint

Sw_a_old=(Ew_old - Epsilon_wr_dr) / Const_Epsilon_dr

 
############
for i in range(0,N):
    ############ calculate Sw_a_new ############
    VEw_new[i]=func_E_w0_bc(VTime[i])
    ############
    if(VEw_new[i]-Ew_old>tol):
      Ih_old=Ih_new
      Ih_new=-1 #imb  
      if(Ih_new != Ih_old):
        Ew_endpoint = Ew_reversal
        Ew_reversal = Ew_old
      Sw_a_new= Sw_a_func(Ew_reversal,VEw_new[i]) 
      VSnt_e[i]=Snt_e_func(Ew_reversal,VEw_new[i])  
    else:
      Ih_old=Ih_new
      Ih_new=1  #dr
      if(Ih_new != Ih_old):
        Ew_endpoint = Ew_reversal
        Ew_reversal = Ew_old 
      if(i>500):  
        Sw_a_new=(VEw_new[i] - Epsilon_wr_dr) /Const_Epsilon_dr + Snt_e_func(Ew_endpoint,VEw_new[i]) 
      else:
        Sw_a_new=(VEw_new[i] - Epsilon_wr_dr) /Const_Epsilon_dr
     
    ############################################
    if(Ih_new != Ih_old):
       Pc_endpoint = Pc_reversal
       Pc_reversal = Pc_Hyst_Parker_Land(Sw_a_old, Ih_old, Sw_a_reversal, Sw_a_endpoint, Pc_reversal, Pc_endpoint_old)
       Pc_endpoint_old = Pc_endpoint
       Sw_a_endpoint = Sw_a_reversal
       Sw_a_reversal = Sw_a_old    

    ############
    VPc_new[i]=Pc_Hyst_Parker_Land(Sw_a_new, Ih_new, Sw_a_reversal, Sw_a_endpoint, Pc_reversal, Pc_endpoint)
    ############
    Ew_old=VEw_new[i]
    Sw_a_old=Sw_a_new
    VSw_a[i]=Sw_a_new
    

VPc_new_1= VPc_new[0:500]
VPc_new_2= VPc_new[500:1000]
VPc_new_3= VPc_new[1000:1501]
VEw_new_1= VEw_new[0:500]
VEw_new_2= VEw_new[500:1000]
VEw_new_3= VEw_new[1000:1501]


plt.plot(VEpsilon_w_dr[1:1000], VPc_dr[1:1000],label="main drainage VG")
plt.plot(VEpsilon_w_imb[1:1000], VPc_imb[1:1000],label="main imbibition VG")
plt.plot(VEw_new_1, VPc_new_1,label="First scanning dr")
plt.plot(VEw_new_2, VPc_new_2,label="First scanning imb")
plt.plot(VEw_new_3, VPc_new_3,label="Secondary scanning dr")
plt.legend()
plt.xlabel("E_w")
plt.ylabel("Pc") 
plt.title("Pc-Ew Parker-Land")
plt.show()



"""
plt.plot(VSw_a_dr[1:1000], VPc_dr[1:1000],label="main drainage VG")
plt.plot(VSw_a_imb[1:1000], VPc_imb[1:1000],label="main imbibition VG")
plt.plot(VSw_a, VPc_new,label="Parker-Land")
plt.legend()
plt.xlabel("S_w")
plt.ylabel("Pc") 
plt.title("Pc-Sw apparent VG")
plt.show()
"""


