#just a visualization of P-S realtions with imb/dr coefficients
import math
import numpy as np
import matplotlib.pyplot as plt

############
 




############
phi=0.4
 
 
####
alpha_imb= 1e-3
n_imb= 2.494
alpha_dr=5e-4 
n_dr= 3
####
K= 1e-10
############
Epsilon_nr= 0.0
Epsilon_wr= 0.0
 
 
####
Const_Epsilon=phi-Epsilon_nr - Epsilon_wr
 
####
 
        


 

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
 
############

 

 
####################################### 
def  Krw_of_Pc_dr( P_c):
    if (P_c > 0.0):
       Se =  (1.0 + ((alpha_dr*P_c)**n_dr))**(-m_dr)
       Krw = (Se**0.5)  *(  (1.0 - (  (1.0 - Se**m_inv_dr)**m_dr  )   )**2.0)
    else:
       Krw = 1.0
    return Krw
####################################### 
def   Krw_of_Pc_imb( P_c):
    if (P_c > 0.0):
       Se =  (1.0 + ((alpha_imb*P_c)**n_imb))**(-m_imb)
       Krw = (Se**0.5)  *(  (1.0 - (  (1.0 - Se**m_inv_imb)**m_imb  )   )**2.0)
    else:
       Krw = 1.0
    return Krw
####################################### 
def   Krn_of_Pc_dr(  P_c):
    if (P_c > 0.0):
       Se =  (1.0 + (alpha_dr*P_c)**n_dr)**(-m_dr)
       Krn = (  (1.0 - Se)**(0.5)  )*(   (1.0 - (Se**(m_inv_dr)))**(m2_dr)   )
    else:
       Krn = 0.0
    return Krn
####################################### 
def   Krn_of_Pc_imb(  P_c):
    if (P_c > 0.0):
       Se =  (1.0 + (alpha_imb*P_c)**n_imb)**(-m_imb)
       Krn = (  (1.0 - Se)**(0.5)  )*(   (1.0 - (Se**(m_inv_imb)))**(m2_imb)   )
    else:
       Krn = 0.0
    return Krn
#######################################
def  Pc_imb( Epsilon_w):
    Se = (Epsilon_w - Epsilon_wr) / Const_Epsilon
    if (Se > 0.0 and Se < 1.0):
       P_c = (  ( Se**(-m_inv_imb)  - 1.0)**n_inv_imb )*alpha_inv_imb
    else:
       P_c = 0.0
    return P_c
#######################################
def   Pc_dr( Epsilon_w):
    Se = (Epsilon_w - Epsilon_wr) / Const_Epsilon
    if (Se > 0.0 and Se < 1.0):
       P_c = (  ( Se**(-m_inv_dr)  - 1.0)**n_inv_dr )*alpha_inv_dr
    else:
       P_c = 0.0
    return P_c
 
####################################### 
def   Epsilon_w_of_Pc_imb(  P_c):
    if (P_c > 0.0):
       Se =  (1.0 + (alpha_imb*P_c)**n_imb)**(-m_imb)
       Epsilon_w =Const_Epsilon*  Se + Epsilon_wr
    else:
       Epsilon_w = phi-Epsilon_nr
    return Epsilon_w
####################################### 
def   Epsilon_w_of_Pc_dr(  P_c):
    if (P_c > 0.0):
       Se =  (1.0 + (alpha_dr*P_c)**n_dr)**(-m_dr)
       Epsilon_w =Const_Epsilon*  Se + Epsilon_wr
    else:
       Epsilon_w = phi-Epsilon_nr
    return Epsilon_w
 
 
####################################### 
def  Krn_of_Ew_dr(  Epsilon_w):
    Se = (Epsilon_w - Epsilon_wr) / Const_Epsilon
    if (Se >= 0.0 and Se < 1.0):
       Krn = (  (1.0 - Se)**0.5 )* (1.0 - Se**m_inv_dr)**m2_dr  
    else:
       Krn = 0.0
    return Krn
####################################### 
def  Krn_of_Ew_imb(  Epsilon_w):
    Se = (Epsilon_w - Epsilon_wr) / Const_Epsilon
    if (Se >= 0.0 and Se < 1.0):
       Krn = (  (1.0 - Se)**0.5 )* (1.0 - Se**m_inv_imb)**m2_imb   
    else:
       Krn = 0.0
    return Krn
####################################### 
def  Krw_of_Ew_dr( Epsilon_w):
    Se = (Epsilon_w - Epsilon_wr) / Const_Epsilon
    if (Se >= 0.0 and Se < 1.0):
       Krw = (Se**0.5)  *(  (1.0 - (  (1.0 - Se**m_inv_dr)**m_dr  )   )**2.0)
    else:
       Krw = 1.0
    return Krw
####################################### 
def  Krw_of_Ew_imb( Epsilon_w):
    Se = (Epsilon_w - Epsilon_wr) / Const_Epsilon
    if (Se >= 0.0 and Se < 1.0):
       Krw = (Se**0.5)  *(  (1.0 - (  (1.0 - Se**m_inv_imb)**m_imb  )   )**2.0)
    else:
       Krw = 1.0
    return Krw
    

#######################################
def  F_Krn_of_Ew_dr(krn,Epsilon_w):
    Se = (Epsilon_w - Epsilon_wr) / Const_Epsilon
    F=krn-((  (1.0 - Se)**0.5 )* (1.0 - Se**m_inv_dr)**m2_dr ) 
    return F  
####################################### 
def  F_Krn_of_Ew_imb(krn,Epsilon_w):
    Se = (Epsilon_w - Epsilon_wr) / Const_Epsilon
    F=   krn-((  (1.0 - Se)**0.5 )* (1.0 - Se**m_inv_imb)**m2_imb   )
    return F
#######################################
def  F_Krw_of_Ew_dr(krw,Epsilon_w):
    Se = (Epsilon_w - Epsilon_wr) / Const_Epsilon
    F= krw-( (Se**0.5)  *(  (1.0 - (  (1.0 - Se**m_inv_dr)**m_dr  )   )**2.0))
    return F  
####################################### 
def  F_Krw_of_Ew_imb(krw,Epsilon_w):
    Se = (Epsilon_w - Epsilon_wr) / Const_Epsilon
    F= krw- ((Se**0.5)  *(  (1.0 - (  (1.0 - Se**m_inv_imb)**m_imb  )   )**2.0))
    return F
#######################################
def  Epsilon_w_of_krn_dr(krn):
    e=1e-6
    a=0
    b=phi
    delta = 1
    while delta > e:
        m = (a + b) / 2
        delta = abs(b - a)
        if F_Krn_of_Ew_dr(krn,m) == 0:
            return m
        elif F_Krn_of_Ew_dr(krn,a) * F_Krn_of_Ew_dr(krn,m)  > 0:
            a = m
        else:
            b = m
    return m
#######################################
def  Epsilon_w_of_krn_imb(krn):
    e=1e-6
    a=0
    b=phi
    delta = 1
    while delta > e:
        m = (a + b) / 2
        delta = abs(b - a)
        if F_Krn_of_Ew_imb(krn,m) == 0:
            return m
        elif F_Krn_of_Ew_imb(krn,a) * F_Krn_of_Ew_imb(krn,m)  > 0:
            a = m
        else:
            b = m
    return m

#######################################
def  Epsilon_w_of_krw_dr(krw):
    e=1e-6
    a=0
    b=phi
    delta = 1
    while delta > e:
        m = (a + b) / 2
        delta = abs(b - a)
        if F_Krw_of_Ew_dr(krw,m) == 0:
            return m
        elif F_Krw_of_Ew_dr(krw,a) * F_Krw_of_Ew_dr(krw,m)  > 0:
            a = m
        else:
            b = m
    return m
#######################################
def  Epsilon_w_of_krw_imb(krw):
    e=1e-6
    a=0
    b=phi
    delta = 1
    while delta > e:
        m = (a + b) / 2
        delta = abs(b - a)
        if F_Krw_of_Ew_imb(krw,m) == 0:
            return m
        elif F_Krw_of_Ew_imb(krw,a) * F_Krw_of_Ew_imb(krw,m)  > 0:
            a = m
        else:
            b = m
    return m
#######################################
def  Pc_Hyst_Parker( Ew, Ih_new, Ew_reversal, Ew_endpoint, Pc_reversal, Pc_endpoint):
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
def   Krn_Hyst_Parker( Ew, Ih_new, Ew_reversal, Ew_endpoint, krn_reversal, krn_endpoint):
    if(Ih_new==-1):
      Ew_reversal_MPC=Epsilon_w_of_krn_imb(krn_reversal)
      Ew_endpoint_MPC=Epsilon_w_of_krn_imb(krn_endpoint)
      Krn_tn=Krn_of_Pc_imb(Pc_imb( ((Ew-Ew_endpoint)/(Ew_reversal-Ew_endpoint))*(Ew_reversal_MPC-Ew_endpoint_MPC) + Ew_endpoint_MPC))
    elif(Ih_new==1):
      Ew_reversal_MPC=Epsilon_w_of_krn_dr(krn_reversal)
      Ew_endpoint_MPC=Epsilon_w_of_krn_dr(krn_endpoint)
      Krn_tn=Krn_of_Pc_dr(Pc_dr( ((Ew-Ew_endpoint)/(Ew_reversal-Ew_endpoint))*(Ew_reversal_MPC-Ew_endpoint_MPC) + Ew_endpoint_MPC))
    return Krn_tn
 
####################################### 
def  Krw_Hyst_Parker( Ew, Ih_new, Ew_reversal, Ew_endpoint, krw_reversal, krw_endpoint):
    if(Ih_new==-1):
      Ew_reversal_MPC=Epsilon_w_of_krw_imb(krw_reversal)
      Ew_endpoint_MPC=Epsilon_w_of_krw_imb(krw_endpoint)
      Krw_tn=Krw_of_Pc_imb(Pc_imb( ((Ew-Ew_endpoint)/(Ew_reversal-Ew_endpoint))*(Ew_reversal_MPC-Ew_endpoint_MPC) + Ew_endpoint_MPC))
    elif(Ih_new==1):
      Ew_reversal_MPC=Epsilon_w_of_krw_dr(krw_reversal)
      Ew_endpoint_MPC=Epsilon_w_of_krw_dr(krw_endpoint)
      Krw_tn=Krw_of_Pc_dr(Pc_dr( ((Ew-Ew_endpoint)/(Ew_reversal-Ew_endpoint))*(Ew_reversal_MPC-Ew_endpoint_MPC) + Ew_endpoint_MPC))
    return Krw_tn
 
############ 
def func_E_w0_bc( t): 
    Ew_t0= phi
    Ew_t1= phi*0.4
    Ew_t2= phi*0.8
    Ew_t3= phi*0.5
    Ew_t4= phi*0.7
    Ew_t5= phi*0.6
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
    elif(t>1500 and t<=2000): 
      aa=(Ew_t4-Ew_t3)/(2000-1500)
      bb=Ew_t3-aa*1500
      E_w0=aa*t+bb
    elif(t>2000 and t<=2500): 
      aa=(Ew_t5-Ew_t4)/(2500-2000)
      bb=Ew_t4-aa*2000
      E_w0=aa*t+bb
    return E_w0
############








############
N=2501
VTime = np.linspace(0, 2500, N)
VEw_new=np.empty([N], dtype=np.double) 
VPc_new=np.empty([N], dtype=np.double) 
Vkrw_new=np.empty([N], dtype=np.double) 
Vkrn_new=np.empty([N], dtype=np.double) 



VEpsilon_w = np.linspace(0.003, phi, 1000)
VPc_imb=np.empty([len(VEpsilon_w)], dtype=np.double) 
VPc_dr=np.empty([len(VEpsilon_w)], dtype=np.double) 
Vkrw_dr=np.empty([len(VEpsilon_w)], dtype=np.double) 
Vkrw_imb=np.empty([len(VEpsilon_w)], dtype=np.double) 
Vkrn_dr=np.empty([len(VEpsilon_w)], dtype=np.double) 
Vkrn_imb=np.empty([len(VEpsilon_w)], dtype=np.double) 
############
for i in range(len(VPc_dr)):
    VPc_dr[i]    = Pc_dr(VEpsilon_w[i])
    VPc_imb[i]   = Pc_imb(VEpsilon_w[i] )
    Vkrn_dr[i]   = Krn_of_Ew_dr( VEpsilon_w[i])
    Vkrn_imb[i]  = Krn_of_Ew_imb( VEpsilon_w[i])
    Vkrw_dr[i]   = Krw_of_Ew_dr( VEpsilon_w[i])
    Vkrw_imb[i]  = Krw_of_Ew_imb( VEpsilon_w[i])
############Initialization 
Ih_new=1
Ih_old=1
Ew_old= func_E_w0_bc(0)
###
Ew_reversal=phi
Ew_endpoint= Epsilon_wr+1e-5
###
Pc_reversal=Pc_dr(Ew_reversal)
Pc_endpoint=Pc_dr(Ew_endpoint)
Pc_endpoint_old=Pc_endpoint
###
krn_reversal=Krn_of_Ew_dr(Ew_reversal)
krn_endpoint=Krn_of_Ew_dr(Ew_endpoint)
krn_endpoint_old=krn_endpoint
###
krw_reversal=Krw_of_Ew_dr(Ew_reversal)
krw_endpoint=Krw_of_Ew_dr(Ew_endpoint)
krw_endpoint_old=krw_endpoint
###
 
tol=1e-5
############
for i in range(0,N):
    ############ calculate Sw_a_new ############
    VEw_new[i]=func_E_w0_bc(VTime[i])
    ############
    if(VEw_new[i]-Ew_old>tol):
      Ih_old=Ih_new
      Ih_new=-1 #imb  
    else:
      Ih_old=Ih_new
      Ih_new=1  #dr
 

    ############################################
    if(Ih_new != Ih_old):
       Pc_endpoint     = Pc_reversal
       Pc_reversal     = Pc_Hyst_Parker(Ew_old, Ih_old, Ew_reversal, Ew_endpoint, Pc_reversal, Pc_endpoint_old)
       Pc_endpoint_old = Pc_endpoint
       ###
       krn_endpoint    = krn_reversal
       krn_reversal    = Krn_Hyst_Parker(Ew_old, Ih_old, Ew_reversal, Ew_endpoint, krn_reversal, krn_endpoint_old)
       krn_endpoint_old= krn_endpoint
       ###
       krw_endpoint    = krw_reversal
       krw_reversal    = Krw_Hyst_Parker(Ew_old, Ih_old, Ew_reversal, Ew_endpoint, krw_reversal, krw_endpoint_old)
       krw_endpoint_old= krw_endpoint
       
       ###
       Ew_endpoint     = Ew_reversal
       Ew_reversal     = Ew_old    
    ############
    VPc_new[i]         = Pc_Hyst_Parker(VEw_new[i], Ih_new, Ew_reversal, Ew_endpoint, Pc_reversal, Pc_endpoint)
    Vkrn_new[i]        = Krn_Hyst_Parker(VEw_new[i], Ih_new, Ew_reversal, Ew_endpoint, krn_reversal, krn_endpoint)
    Vkrw_new[i]        = Krw_Hyst_Parker(VEw_new[i], Ih_new, Ew_reversal, Ew_endpoint, krw_reversal, krw_endpoint)
    ############
    Ew_old=VEw_new[i]
 

 
############ 
plt.plot(VEpsilon_w, VPc_dr,label="main drainage VG")
plt.plot(VEpsilon_w, VPc_imb,label="main imbibition VG")
plt.plot(VEw_new, VPc_new,label="Hyst Pc-Ew")
plt.legend()
plt.xlabel("E_w")
plt.ylabel("Pc") 
plt.title("Pc-Ew (volume fraction) VG")
plt.show()
############ Krn visualtization
plt.plot(VEpsilon_w, Vkrn_dr   ,label="main drainage VG")
plt.plot(VEpsilon_w, Vkrn_imb  ,label="main imbibition VG")
plt.plot(VEw_new   , Vkrn_new  ,label="hyst krn-Ew")
plt.legend()
plt.xlabel("E_w")
plt.ylabel("krn") 
plt.title("krn-Ew (volume fraction) VG")
plt.show()
############ Krw visualtization
plt.plot(VEpsilon_w, Vkrw_dr   ,label="main drainage VG")
plt.plot(VEpsilon_w, Vkrw_imb  ,label="main imbibition VG")
plt.plot(VEw_new   , Vkrw_new  ,label="hyst krn-Ew")
plt.legend()
plt.xlabel("E_w")
plt.ylabel("krw") 
plt.title("krw-Ew (volume fraction) VG")
plt.show()
 
