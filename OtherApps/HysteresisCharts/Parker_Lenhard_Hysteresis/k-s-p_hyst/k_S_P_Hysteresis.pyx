# the parameters are for coarse sand

import petsc4py
import time
import sys
import math
import numpy as np
import cython
import matplotlib.pyplot as plt
petsc4py.init(sys.argv)
from petsc4py import PETSc
from libc.stdlib cimport malloc


OptDB = PETSc.Options()
###############
cdef unsigned int MaxStepRejections,MaxSNESFailures,MaxTimeSteps
cdef unsigned int Nz
cdef int d
cdef double t_out,t0,Omega_0,Omega_L,dz,Cz,phi,mu_w,mu_n,g,alpha,n,K,S0,SL,Epsilon_nr,Epsilon_wr,rho_w0,rho_n0,a,b
cdef double Const_Epsilon,k_mun_half,k_muw_half,m,n_inv,n_inv1,n1,m_inv,m1,m2,alpha_inv,alpha_m_n
cdef double atol,rtol
cdef double Ew_bc_0,Ew_bc_L,Pw_bc_0,Pw_bc_L,Pn_bc_0,Pn_bc_L,Ew_IC,Pw_IC,Pn_IC
############
Counter=1
Node_hyst=0
############
time_interval=np.array([1.0,10.0,20.0,30.0,100.0,200.0,500.0,600,700,800,900,1000])

t0=0.0
############
Nz=101
Omega_0= 0.0
Omega_L= 1.0

############
phi=0.4
mu_w=1e-3
mu_n=1.78e-5
############
g=9.8066
d=0
####
alpha_imb= 1e-3
n_imb= 2.494
alpha_dr=5e-4 
n_dr=  2.6
####
K= 1e-10
############
Epsilon_nr= 0.0
Epsilon_wr= 0.0
rho_w0 =1e3
rho_n0 = 1.2
a= 4.9e-10
b= 1.189e-5
####
Const_Epsilon=phi-Epsilon_nr - Epsilon_wr
k_mun_half=0.5*(K/mu_n)
k_muw_half=0.5*(K/mu_w)
####

############
rtol=1e-8
atol=1e-8
tol_hyst=1e-9
MaxStepRejections=500
MaxSNESFailures=5000
MaxTimeSteps=100000

############
Ew_bc_0= phi-0.003                         
Pw_bc_0= -346.73 
Ew_bc_L= phi-0.003  
Pw_bc_L= -346.73   
Ew_IC= phi-0.003                      
Pw_IC=-346.73         


############
dz =(Omega_L - Omega_0) / (Nz - 1.0)
Cz =  1.0 / dz

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
######################dynamic arrays #################
cdef double *Vp_c = <double*>malloc(Nz * sizeof(double))
cdef double *Vp_w = <double*>malloc(Nz * sizeof(double))
cdef double *Vp_n = <double*>malloc(Nz * sizeof(double))
cdef double *Vrho_w_of_P_w = <double*>malloc(Nz * sizeof(double))
cdef double *Vrho_n_of_P_n = <double*>malloc(Nz * sizeof(double))
cdef double *VKrw_of_P_c = <double*>malloc(Nz * sizeof(double))
cdef double *VKrn_of_P_c = <double*>malloc(Nz * sizeof(double))
cdef double *VKrw_rho_w = <double*>malloc(Nz * sizeof(double))
cdef double *VKrn_rho_n = <double*>malloc(Nz * sizeof(double))
cdef double *VDrvt_Pc_by_Epsilon_w = <double*>malloc(Nz * sizeof(double))
Clock_Array=np.zeros([len(time_interval)], dtype=float) 
ftime=np.zeros([len(time_interval)], dtype=float) 
######################dynamic arrays related to hysteresis model #################
cdef double *VPc_tn1 = <double*>malloc(Nz * sizeof(double))
cdef double *VEw_tn1 = <double*>malloc(Nz * sizeof(double))
cdef double *Vkrw_tn1 = <double*>malloc(Nz * sizeof(double))
cdef double *Vkrn_tn1 = <double*>malloc(Nz * sizeof(double))


cdef double *VEw_reversal = <double*>malloc(Nz * sizeof(double))
cdef double *VEw_endpoint = <double*>malloc(Nz * sizeof(double))
cdef double *VPc_reversal = <double*>malloc(Nz * sizeof(double))
cdef double *VPc_endpoint = <double*>malloc(Nz * sizeof(double))
cdef double *VPc_endpoint_old = <double*>malloc(Nz * sizeof(double))
cdef double *Vkrw_reversal = <double*>malloc(Nz * sizeof(double))
cdef double *Vkrw_endpoint = <double*>malloc(Nz * sizeof(double))
cdef double *Vkrw_endpoint_old = <double*>malloc(Nz * sizeof(double))

cdef double *Vkrn_reversal = <double*>malloc(Nz * sizeof(double))
cdef double *Vkrn_endpoint = <double*>malloc(Nz * sizeof(double))
cdef double *Vkrn_endpoint_old = <double*>malloc(Nz * sizeof(double))

VEw_hyst=np.zeros([MaxTimeSteps], dtype=np.double)  
VPc_hyst=np.zeros([MaxTimeSteps], dtype=np.double) 
Vkrw_hyst=np.zeros([MaxTimeSteps], dtype=np.double)
Vkrn_hyst=np.zeros([MaxTimeSteps], dtype=np.double)
VEw_endpoint_hyst=np.zeros([MaxTimeSteps], dtype=np.double) 
VTime_hyst=np.zeros([MaxTimeSteps], dtype=float) 
VIh_old=np.ones([Nz], dtype=int) 
VIh_new=np.ones([Nz], dtype=int) 

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
cdef double  Drvt_Pc_by_Epsilon_w_imb(double Epsilon_w):
    Se = (Epsilon_w - Epsilon_wr) / Const_Epsilon
    if (Se > 0.0 and Se < 1.0):
       Drvt_P_c = -( ( Se**(-m_inv_imb) - 1.0)**n_inv1_imb )*(1.0/(alpha_m_n_imb*(Epsilon_w-Epsilon_wr)*(Se**m_inv_imb)))
    else:
       Drvt_P_c = 0.0
    return Drvt_P_c
#######################################
cdef double  Drvt_Pc_by_Epsilon_w_dr(double Epsilon_w):
    Se = (Epsilon_w - Epsilon_wr) / Const_Epsilon
    if (Se > 0.0 and Se < 1.0):
       Drvt_P_c = -( ( Se**(-m_inv_dr) - 1.0)**n_inv1_dr )*(1.0/(alpha_m_n_dr*(Epsilon_w-Epsilon_wr)*(Se**m_inv_dr)))
    else:
       Drvt_P_c = 0.0
    return Drvt_P_c
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
cdef double  Krw_of_Pc_dr(double P_c):
    if (P_c > 0.0):
       Se =  (1.0 + ((alpha_dr*P_c)**n_dr))**(-m_dr)
       Krw = (Se**0.5)  *(  (1.0 - (  (1.0 - Se**m_inv_dr)**m_dr  )   )**2.0)
    else:
       Krw = 1.0
    return Krw
####################################### 
cdef double  Krw_of_Pc_imb(double P_c):
    if (P_c > 0.0):
       Se =  (1.0 + ((alpha_imb*P_c)**n_imb))**(-m_imb)
       Krw = (Se**0.5)  *(  (1.0 - (  (1.0 - Se**m_inv_imb)**m_imb  )   )**2.0)
    else:
       Krw = 1.0
    return Krw
####################################### 
cdef double  Krn_of_Pc_dr(double  P_c):
    if (P_c > 0.0):
       Se =  (1.0 + (alpha_dr*P_c)**n_dr)**(-m_dr)
       Krn = (  (1.0 - Se)**(0.5)  )*(   (1.0 - (Se**(m_inv_dr)))**(m2_dr)   )
    else:
       Krn = 0.0
    return Krn
####################################### 
cdef double  Krn_of_Pc_imb(double  P_c):
    if (P_c > 0.0):
       Se =  (1.0 + (alpha_imb*P_c)**n_imb)**(-m_imb)
       Krn = (  (1.0 - Se)**(0.5)  )*(   (1.0 - (Se**(m_inv_imb)))**(m2_imb)   )
    else:
       Krn = 0.0
    return Krn
####################################### 
cdef double  Krn_of_Ew_dr(  Epsilon_w):
    Se = (Epsilon_w - Epsilon_wr) / Const_Epsilon
    if (Se >= 0.0 and Se < 1.0):
       Krn = (  (1.0 - Se)**0.5 )* (1.0 - Se**m_inv_dr)**m2_dr  
    else:
       Krn = 0.0
    return Krn
####################################### 
cdef double  Krn_of_Ew_imb(  Epsilon_w):
    Se = (Epsilon_w - Epsilon_wr) / Const_Epsilon
    if (Se >= 0.0 and Se < 1.0):
       Krn = (  (1.0 - Se)**0.5 )* (1.0 - Se**m_inv_imb)**m2_imb   
    else:
       Krn = 0.0
    return Krn
####################################### 
cdef double  Krw_of_Ew_dr( Epsilon_w):
    Se = (Epsilon_w - Epsilon_wr) / Const_Epsilon
    if (Se >= 0.0 and Se < 1.0):
       Krw = (Se**0.5)  *(  (1.0 - (  (1.0 - Se**m_inv_dr)**m_dr  )   )**2.0)
    else:
       Krw = 1.0
    return Krw
####################################### 
cdef double  Krw_of_Ew_imb( Epsilon_w):
    Se = (Epsilon_w - Epsilon_wr) / Const_Epsilon
    if (Se >= 0.0 and Se < 1.0):
       Krw = (Se**0.5)  *(  (1.0 - (  (1.0 - Se**m_inv_imb)**m_imb  )   )**2.0)
    else:
       Krw = 1.0
    return Krw
#######################################
cdef double  F_Krn_of_Ew_dr(krn,Epsilon_w):
    Se = (Epsilon_w - Epsilon_wr) / Const_Epsilon
    F=krn-((  (1.0 - Se)**0.5 )* (1.0 - Se**m_inv_dr)**m2_dr ) 
    return F  
####################################### 
cdef double F_Krn_of_Ew_imb(krn,Epsilon_w):
    Se = (Epsilon_w - Epsilon_wr) / Const_Epsilon
    F=   krn-((  (1.0 - Se)**0.5 )* (1.0 - Se**m_inv_imb)**m2_imb   )
    return F
#######################################
cdef double F_Krw_of_Ew_dr(krw,Epsilon_w):
    Se = (Epsilon_w - Epsilon_wr) / Const_Epsilon
    F= krw-( (Se**0.5)  *(  (1.0 - (  (1.0 - Se**m_inv_dr)**m_dr  )   )**2.0))
    return F  
####################################### 
cdef double F_Krw_of_Ew_imb(krw,Epsilon_w):
    Se = (Epsilon_w - Epsilon_wr) / Const_Epsilon
    F= krw- ((Se**0.5)  *(  (1.0 - (  (1.0 - Se**m_inv_imb)**m_imb  )   )**2.0))
    return F
#######################################
cdef double Epsilon_w_of_krn_dr(krn):
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
cdef double Epsilon_w_of_krn_imb(krn):
    # this is solved using dichotomie method
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
cdef double Epsilon_w_of_krw_dr(krw):
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
cdef double Epsilon_w_of_krw_imb(krw):
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
cdef double  Epsilon_n_of_Pc(double Epsilon_w):
    Epsilon_n = phi - Epsilon_w
    return Epsilon_n
####################################### 
cdef double  rhow_of_Pw(double P_w):
    rho_w =rho_w0*math.exp(a*P_w)
    return rho_w
#######################################  
cdef double  Drvt_rhow_of_Pw(double P_w):
    Drvt_rho_w =a*rho_w0*math.exp(a*P_w)
    return Drvt_rho_w
#######################################
cdef double  rhon_of_Pn(double P_n):
    rho_n =b* P_n+rho_n0
    return rho_n
#######################################
cdef double  Drvt_rhon_of_Pn(double P_n):
    Drvt_rho_n =b
    return Drvt_rho_n   








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
cdef  double Drvt_Pc_by_Epsilon_w_Hyst_Parker(double Ew, Ih_new,double Ew_reversal,double Ew_endpoint,double Pc_reversal,double Pc_endpoint):
    cdef double Drvt_P_c
    if(Ih_new==-1):
      Ew_reversal_MPC=Epsilon_w_of_Pc_imb(Pc_reversal)
      Ew_endpoint_MPC=Epsilon_w_of_Pc_imb(Pc_endpoint)
      Drvt_P_c=((Ew_reversal_MPC-Ew_endpoint_MPC)/(Ew_reversal-Ew_endpoint))* Drvt_Pc_by_Epsilon_w_imb( ((Ew-Ew_endpoint)/(Ew_reversal-Ew_endpoint))*(Ew_reversal_MPC-Ew_endpoint_MPC) + Ew_endpoint_MPC)
    elif(Ih_new==1):
      Ew_reversal_MPC=Epsilon_w_of_Pc_dr(Pc_reversal)
      Ew_endpoint_MPC=Epsilon_w_of_Pc_dr(Pc_endpoint)
      Drvt_P_c=((Ew_reversal_MPC-Ew_endpoint_MPC)/(Ew_reversal-Ew_endpoint))* Drvt_Pc_by_Epsilon_w_dr( ((Ew-Ew_endpoint)/(Ew_reversal-Ew_endpoint))*(Ew_reversal_MPC-Ew_endpoint_MPC) + Ew_endpoint_MPC)
    #######  
    return Drvt_P_c
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
cdef double Krn_Hyst_Parker( Ew, Ih_new, Ew_reversal, Ew_endpoint, krn_reversal, krn_endpoint):
    if(Ih_new==-1):
      Ew_reversal_MPC=Epsilon_w_of_krn_imb(krn_reversal)
      Ew_endpoint_MPC=Epsilon_w_of_krn_imb(krn_endpoint)
      Krn_tn=Krn_of_Pc_imb( Pc_imb(((Ew-Ew_endpoint)/(Ew_reversal-Ew_endpoint))*(Ew_reversal_MPC-Ew_endpoint_MPC) + Ew_endpoint_MPC))
    elif(Ih_new==1):
      Ew_reversal_MPC=Epsilon_w_of_krn_dr(krn_reversal)
      Ew_endpoint_MPC=Epsilon_w_of_krn_dr(krn_endpoint)
      Krn_tn=Krn_of_Pc_dr(Pc_dr( ((Ew-Ew_endpoint)/(Ew_reversal-Ew_endpoint))*(Ew_reversal_MPC-Ew_endpoint_MPC) + Ew_endpoint_MPC))
    return Krn_tn
 
####################################### 
cdef double Krw_Hyst_Parker( Ew, Ih_new, Ew_reversal, Ew_endpoint, krw_reversal, krw_endpoint):
    if(Ih_new==-1):
      Ew_reversal_MPC=Epsilon_w_of_krw_imb(krw_reversal)
      Ew_endpoint_MPC=Epsilon_w_of_krw_imb(krw_endpoint)
      Krw_tn=Krw_of_Pc_imb(Pc_imb( ((Ew-Ew_endpoint)/(Ew_reversal-Ew_endpoint))*(Ew_reversal_MPC-Ew_endpoint_MPC) + Ew_endpoint_MPC))
    elif(Ih_new==1):
      Ew_reversal_MPC=Epsilon_w_of_krw_dr(krw_reversal)
      Ew_endpoint_MPC=Epsilon_w_of_krw_dr(krw_endpoint)
      Krw_tn=Krw_of_Pc_dr(Pc_dr( ((Ew-Ew_endpoint)/(Ew_reversal-Ew_endpoint))*(Ew_reversal_MPC-Ew_endpoint_MPC) + Ew_endpoint_MPC))
    return Krw_tn
 
#######################################
cdef double func_P_w0_bc(double t): 
    Pw_t0=-346.73
    Pw_t1=-1000
    Pw_t2=-500
    Pw_t3=-1000
    Pw_t4=-300
    if(t>=0 and t<=500):
      aa=(Pw_t1-Pw_t0)/(500-0)
      bb=Pw_t0-aa*0
      P_w0=aa*t+bb
    elif(t>500 and t<=1000): 
      aa=(Pw_t2-Pw_t1)/(1000-500)
      bb=Pw_t1-aa*500
      P_w0=aa*t+bb
    elif(t>1000 and t<=1500):  
      aa=(Pw_t3-Pw_t2)/(1500-1000)
      bb=Pw_t2-aa*1000
      P_w0=aa*t+bb
    elif(t>1500 and t<=2000):  
      aa=(Pw_t4-Pw_t3)/(2000-1500)
      bb=Pw_t3-aa*1500
      P_w0=aa*t+bb  
    return P_w0
 
  
  
    
#######################################
cdef int Residual(ts,t,U,Udot,F):
# residual function is like sundials residual, the prototype is fixed but the name of the function can be changed
#u[i][0]=Epsilon_w[i]
#u[i][1]=Psi_w[i]
 
    cdef unsigned int i
    cdef double CEpsilon_n_of_Pc,CDrvt_rho_n_of_Pn
    cdef double A_SP_11,A_SP_12,A_SP_21,A_SP_22
    cdef double Kw_plus,Kw_minus,rho_w_plus,rho_w_minus,Kn_plus,Kn_minus,rho_n_plus,rho_n_minus,Od_w,Od_n
    global Counter
    da=ts.getDM()
    #######
    localU=da.getLocalVec()
    da.globalToLocal(U,localU)
    #######
    u=da.getVecArray(localU)
    udot=da.getVecArray(Udot)
    f=da.getVecArray(F)
    Ndt=ts.getStepNumber()
    for i in range(0,Nz):
        Vp_c[i] =  Pc_Hyst_Parker(u[i][0],VIh_new[i],VEw_reversal[i],VEw_endpoint[i],VPc_reversal[i], VPc_endpoint[i])   
        VDrvt_Pc_by_Epsilon_w[i]=Drvt_Pc_by_Epsilon_w_Hyst_Parker(u[i][0],VIh_new[i],VEw_reversal[i],VEw_endpoint[i],VPc_reversal[i], VPc_endpoint[i]) 
        Vp_w[i]           = u[i][1]
        Vp_n[i]           = Vp_c[i] + Vp_w[i]
        Vrho_w_of_P_w[i]  = rhow_of_Pw(Vp_w[i])
        VKrw_of_P_c[i]    = Krw_Hyst_Parker( u[i][0], VIh_new[i],VEw_reversal[i], VEw_endpoint[i], Vkrw_reversal[i], Vkrw_endpoint[i]) 
        Vrho_n_of_P_n[i]  = rhon_of_Pn(Vp_n[i])
        VKrn_of_P_c[i]    = Krn_Hyst_Parker( u[i][0], VIh_new[i],VEw_reversal[i], VEw_endpoint[i], Vkrn_reversal[i], Vkrn_endpoint[i])  
        VKrw_rho_w[i]     = Vrho_w_of_P_w[i]* VKrw_of_P_c[i]
        VKrn_rho_n[i]     = Vrho_n_of_P_n[i]* VKrn_of_P_c[i]
          
    for i in range(1,Nz-1):
        #########
        CEpsilon_n_of_P_c=Epsilon_n_of_Pc(u[i][0])
        CDrvt_rho_n_of_P_n=Drvt_rhon_of_Pn(Vp_n[i])
        #########
        A_SP_11 = Vrho_w_of_P_w[i]
        A_SP_12 = u[i][0] *Drvt_rhow_of_Pw(Vp_w[i])
        A_SP_21 = CEpsilon_n_of_P_c*CDrvt_rho_n_of_P_n*VDrvt_Pc_by_Epsilon_w[i]-Vrho_n_of_P_n[i]
        A_SP_22 = CEpsilon_n_of_P_c*CDrvt_rho_n_of_P_n
        #########
        Kw_plus= k_muw_half*(VKrw_rho_w[i+1] + VKrw_rho_w[i])
        Kw_minus= k_muw_half*(VKrw_rho_w[i-1] + VKrw_rho_w[i])
        rho_w_plus = 0.5*(Vrho_w_of_P_w[i+1] + Vrho_w_of_P_w[i])
        rho_w_minus =0.5*(Vrho_w_of_P_w[i-1] + Vrho_w_of_P_w[i])
        Kn_plus = k_mun_half*(VKrn_rho_n[i+1] + VKrn_rho_n[i])
        Kn_minus = k_mun_half*(VKrn_rho_n[i-1] + VKrn_rho_n[i])
        rho_n_plus = 0.5*(Vrho_n_of_P_n[i+1] + Vrho_n_of_P_n[i])
        rho_n_minus =0.5*(Vrho_n_of_P_n[i-1] + Vrho_n_of_P_n[i])
        Od_w = Cz * (Kw_plus*(Cz*(Vp_w[i+1] - Vp_w[i]) + d*g*rho_w_plus) - Kw_minus * (Cz*(Vp_w[i] - Vp_w[i-1]) + d*g*rho_w_minus))
        Od_n = Cz * (Kn_plus*(Cz*(Vp_n[i+1] - Vp_n[i]) + d*g*rho_n_plus) - Kn_minus * (Cz*(Vp_n[i] - Vp_n[i-1]) + d*g*rho_n_minus))
        f[i][0] =  A_SP_11 * udot[i][0] + A_SP_12 * udot[i][1] - Od_w
        f[i][1] =  A_SP_21 * udot[i][0] + A_SP_22 * udot[i][1] - Od_n
    #########
    f[0][0] = u[0][0] - Epsilon_w_of_Pc_Hyst_Parker( -func_P_w0_bc(t), VIh_new[0], VEw_reversal[0], VEw_endpoint[0], VPc_reversal[0], VPc_endpoint[0])  
    f[0][1] = u[0][1] - func_P_w0_bc(t)
    f[Nz-1][0] = u[Nz-1][0] - Ew_bc_L
    f[Nz-1][1] = u[Nz-1][1] - Pw_bc_L
    #########    
    if(Ndt==Counter):
      Solution=ts.getSolution()
      dt=ts.getTimeStep()
      for i in range(0,Nz):  
          VPc_tn1[i]=Pc_Hyst_Parker(Solution[2*i],VIh_new[i],VEw_reversal[i],VEw_endpoint[i],VPc_reversal[i],VPc_endpoint[i])  
          Vkrw_tn1[i]=Krw_Hyst_Parker(Solution[2*i],VIh_new[i],VEw_reversal[i],VEw_endpoint[i],Vkrw_reversal[i],Vkrw_endpoint[i]) 
          Vkrn_tn1[i]=Krn_Hyst_Parker(Solution[2*i],VIh_new[i],VEw_reversal[i],VEw_endpoint[i],Vkrn_reversal[i],Vkrn_endpoint[i])  
          #####                          
          if(Solution[2*i]-VEw_tn1[i]>tol_hyst):
            VIh_old[i]=VIh_new[i]
            VIh_new[i]=-1
          else:      
            VIh_old[i]=VIh_new[i]
            VIh_new[i]=1
          #####  
          VEw_tn1[i]=Solution[2*i] 
          #####
          if(VIh_new[i] != VIh_old[i]): 
            VPc_endpoint[i] = VPc_reversal[i] 
            VPc_reversal[i] = Pc_Hyst_Parker(Solution[2*i], VIh_old[i], VEw_reversal[i], VEw_endpoint[i], VPc_reversal[i], VPc_endpoint_old[i])
            VPc_endpoint_old[i]= VPc_endpoint[i]
            #####
            Vkrw_endpoint[i] = Vkrw_reversal[i] 
            Vkrw_reversal[i] = Krw_Hyst_Parker(Solution[2*i], VIh_old[i], VEw_reversal[i], VEw_endpoint[i], Vkrw_reversal[i], Vkrw_endpoint_old[i])
            Vkrw_endpoint_old[i]= Vkrw_endpoint[i]
            #####
            Vkrn_endpoint[i] = Vkrn_reversal[i] 
            Vkrn_reversal[i] = Krn_Hyst_Parker(Solution[2*i], VIh_old[i], VEw_reversal[i], VEw_endpoint[i], Vkrn_reversal[i], Vkrn_endpoint_old[i])
            Vkrn_endpoint_old[i]= Vkrn_endpoint[i]
            #####
            VEw_endpoint[i] = VEw_reversal[i]
            VEw_reversal[i] = Solution[2*i]
          #####
      VEw_hyst[Counter]  =VEw_tn1[Node_hyst]  #here we store Sw(t) at a chosen node called Node_Hyst
      VPc_hyst[Counter]  =VPc_tn1[Node_hyst]  #here we store Pc(t) at Node_Hyst to plot Pc vs Sw
      Vkrw_hyst[Counter] =Vkrw_tn1[Node_hyst]  #here we store krw(t) at Node_Hyst to plot krw vs Sw
      Vkrn_hyst[Counter] =Vkrn_tn1[Node_hyst]  #here we store krn(t) at Node_Hyst to plot krn vs Sw
      
      
      VEw_endpoint_hyst[Counter]=VEw_endpoint[Node_hyst]
      VTime_hyst[Counter]=VTime_hyst[Counter-1]+dt
      Counter=Counter+1

 
        
#######################################
def Initialization(ts,U):
# a private function that we created to initialize the solution vector at t=t0
    cdef unsigned int i
    da=ts.getDM()
    u=da.getVecArray(U)
    #####
    u[0][0] = Epsilon_w_of_Pc_dr(-func_P_w0_bc(0))
    u[0][1] = func_P_w0_bc(0) 
    u[Nz-1][0] = Ew_bc_L
    u[Nz-1][1] = Pw_bc_L 
    for i in range(1,Nz-1):
        u[i][0] = Ew_IC
        u[i][1] = Pw_IC
        VEw_tn1[i] = u[i][0]
        VPc_tn1[i] = Pc_dr(u[i][0])
        Vkrw_tn1[i]= Krw_of_Pc_dr(VPc_tn1[i])
        Vkrn_tn1[i]= Krn_of_Pc_dr(VPc_tn1[i])
        
        
        VEw_reversal[i] = phi
        VEw_endpoint[i] = 0.003
        VPc_reversal[i] = Pc_dr(VEw_reversal[i])
        VPc_endpoint[i] = Pc_dr(VEw_endpoint[i])
        VPc_endpoint_old[i]=VPc_endpoint[i]
        
        Vkrw_reversal[i] = Krw_of_Pc_dr(VPc_reversal[i])
        Vkrw_endpoint[i] = Krw_of_Pc_dr(VPc_endpoint[i])
        Vkrw_endpoint_old[i]=Vkrw_endpoint[i]
        
        Vkrn_reversal[i] = Krn_of_Pc_dr(VPc_reversal[i])
        Vkrn_endpoint[i] = Krn_of_Pc_dr(VPc_endpoint[i])
        Vkrn_endpoint_old[i]=Vkrn_endpoint[i]
    #####    x=0
    VEw_tn1[0] = u[0][0]  
    VPc_tn1[0] = -u[0][1] 
    Vkrw_tn1[0]= Krw_of_Pc_dr(VPc_tn1[0])
    Vkrn_tn1[0]= Krn_of_Pc_dr(VPc_tn1[0])
    
    VEw_reversal[0] = phi 
    VEw_endpoint[0] = 0.003
    
    VPc_reversal[0] = Pc_dr(VEw_reversal[0])
    VPc_endpoint[0] = Pc_dr(VEw_endpoint[0])
    VPc_endpoint_old[0] = VPc_endpoint[0]
    
    Vkrw_reversal[0] = Krw_of_Pc_dr(VPc_reversal[0])
    Vkrw_endpoint[0] = Krw_of_Pc_dr(VPc_endpoint[0])
    Vkrw_endpoint_old[0]=Vkrw_endpoint[0]    
    
    Vkrn_reversal[0] = Krn_of_Pc_dr(VPc_reversal[0])
    Vkrn_endpoint[0] = Krn_of_Pc_dr(VPc_endpoint[0])
    Vkrn_endpoint_old[0]=Vkrn_endpoint[0]
    #####    x=L
    VEw_tn1[Nz-1] = u[Nz-1][0]  
    VPc_tn1[Nz-1] = -u[Nz-1][1] 
    Vkrw_tn1[Nz-1]= Krw_of_Pc_dr(VPc_tn1[Nz-1])
    Vkrn_tn1[Nz-1]= Krn_of_Pc_dr(VPc_tn1[Nz-1])
    
    VEw_reversal[Nz-1] = phi 
    VEw_endpoint[Nz-1] = 0.003
    
    VPc_reversal[Nz-1] = Pc_dr(VEw_reversal[Nz-1])
    VPc_endpoint[Nz-1] = Pc_dr(VEw_endpoint[Nz-1])
    VPc_endpoint_old[Nz-1] = VPc_endpoint[Nz-1]
    
    Vkrw_reversal[Nz-1] = Krw_of_Pc_dr(VPc_reversal[Nz-1])
    Vkrw_endpoint[Nz-1] = Krw_of_Pc_dr(VPc_endpoint[Nz-1])
    Vkrw_endpoint_old[Nz-1]=Vkrw_endpoint[Nz-1]    
    
    Vkrn_reversal[Nz-1] = Krn_of_Pc_dr(VPc_reversal[Nz-1])
    Vkrn_endpoint[Nz-1] = Krn_of_Pc_dr(VPc_endpoint[Nz-1])
    Vkrn_endpoint_old[Nz-1]=Vkrn_endpoint[Nz-1]
    #####
    VEw_hyst[0] = VEw_tn1[Node_hyst]
    VPc_hyst[0] = VPc_tn1[Node_hyst]
    Vkrw_hyst[0]= Vkrw_tn1[Node_hyst]
    Vkrn_hyst[0]= Vkrn_tn1[Node_hyst]
    VEw_endpoint_hyst[0]=VEw_endpoint[Node_hyst]
    VTime_hyst[0]= t0
#######################################
cdef void main():
#######################################    
# here we created a 1D grid, to let the solver store the jacobian as a sparse matrix (this is one of the ways to do that) 
    cdef unsigned int k
    comm = PETSc.COMM_WORLD 
    bz= PETSc.DMDA.BoundaryType.NONE
    da=PETSc.DMDA().create(dof= 2,sizes=(Nz, ),boundary_type=bz,stencil_type=None,stencil_width=1, comm=comm ) 
    da.setUp()
    u= da.createGlobalVec()   
#######################################    
    da.setMatType(PETSc.Mat.Type.AIJ)           # create a sparse matrix related to the grid created before (actually this is the type)
    da.createMatrix()                           # create the matrix
#######################################
    ts = PETSc.TS().create(PETSc.COMM_WORLD)    # create a time-stepping solver object 
    ts.setProblemType(ts.ProblemType.NONLINEAR) # set type of problem 
    ts.setType(ts.Type.BDF)                     # set type to BDF method	
    ts.setIFunction(Residual)                   # pass the residual function to "ts" so it can use it 
#######################################    
    ts.setDM(da)                                # attach the ts solver to the grid created
#######################################      
    snes=ts.getSNES()                           # get the snes solver
    snes.setFromOptions()                       # give the user the possibility to personalize the "snes" at running time
#######################################        
    ksp=snes.getKSP()                           # get the ksp solver
    ksp.setFromOptions()                        # give the user the possibility to personalize the "ksp" at running time
#######################################
    pc=ksp.getPC()                              # get the pc object from ksp
    pc.setFromOptions()                         # give the user the possibility to personalize the "pc" at running time
####################################### 
    ts.setMaxSteps(MaxTimeSteps)                    # maximum steps allowed to take to reach final time
    ts.setMaxStepRejections(MaxStepRejections)  # maximum time steps rejections allowed ,       
    ts.setMaxSNESFailures(MaxSNESFailures)
    ts.setTolerances(rtol, atol)                # set tolerances of the the chosen ts (here bdf )
    ts.setTime(t0)                              # set initial time t0 
    ts.setExactFinalTime(ts.ExactFinalTime.MATCHSTEP)  # this option is to match the output time of ts with the output time defined by user.
    #ts.setTimeStep(First_Time_Step)
    ts.setFromOptions()                         # give the user the possibility to personalize the ts solver at running time
#######################################
    Initialization(ts,u)
#############Calling the solver and start solving the DAE system ###################### 
    ThetaW=PETSc.Vec().createSeq(Nz) 
    Ew_Node_hyst=np.zeros([len(time_interval)], dtype=np.double) 
    mesh = np.linspace(Omega_0, Omega_L, Nz)
    for i in range(len(time_interval)):
       ts.setMaxTime(time_interval[i])
       CPU_start = time.process_time()
       ts.solve(u)
       CPU_end = time.process_time()
       ftime[i]=ts.getTime()
       Clock_Array[i]=CPU_end-CPU_start
       #####visualization
       U=da.getVecArray(u)
       Ew_Node_hyst[i]=U[Node_hyst][0]
       for k in range(0,Nz):
          ThetaW[k]=U[k][0]
       plt.plot(mesh,ThetaW,label='t(sec)=%5.4f'%ftime[i])
       plt.legend()
       NTimeSteps=ts.getStepNumber()
       if(NTimeSteps==MaxTimeSteps): 
         break
    ######  
       
    plt.xlabel("Position (m)")
    plt.ylabel("Epsilon_w") 
    plt.title("Solution 2Phase-SP 1D Szymkiewicz Paper ")
    plt.show()
    ######
    plt.plot(time_interval,Ew_Node_hyst)
    plt.xlabel("time(s)")
    plt.ylabel("E_w") 
    plt.title("Saturation in function of time at Node %d"%Node_hyst)
    plt.show()
    ######
    VEpsilon_w = np.linspace(0.5*phi, phi, 1000)
    VPc_imb=np.zeros([len(VEpsilon_w)], dtype=np.double) 
    VPc_dr=np.zeros([len(VEpsilon_w)], dtype=np.double) 
    Vkrw_imb=np.zeros([len(VEpsilon_w)], dtype=np.double)
    Vkrw_dr=np.zeros([len(VEpsilon_w)], dtype=np.double)
    Vkrn_imb=np.zeros([len(VEpsilon_w)], dtype=np.double)
    Vkrn_dr=np.zeros([len(VEpsilon_w)], dtype=np.double)
    for i in range(len(VEpsilon_w)):
          VPc_dr[i] =Pc_dr(VEpsilon_w[i])
          VPc_imb[i]=Pc_imb(VEpsilon_w[i]) 
          Vkrw_imb[i] =Krw_of_Ew_imb(VEpsilon_w[i])
          Vkrw_dr[i]  =Krw_of_Ew_dr(VEpsilon_w[i])
          Vkrn_imb[i] =Krn_of_Ew_imb(VEpsilon_w[i])
          Vkrn_dr[i]  =Krn_of_Ew_dr(VEpsilon_w[i])
          
    plt.plot(VEpsilon_w,VPc_imb,label="MIC VG",linestyle="dashed")
    plt.plot(VEpsilon_w,VPc_dr,label="MDC VG",linestyle="dashed")
    plt.plot(VEw_hyst[0:Counter],VPc_hyst[0:Counter],label="Parker-Lenhard Hysteresis model")
    plt.legend()
    plt.xlabel("Ew")
    plt.ylabel("Pc") 
    plt.title("Saturation vs Pc at Node %d"%Node_hyst)
    plt.show()
    ######krw
    plt.plot(VEpsilon_w,Vkrw_imb,label="MIC VG",linestyle="dashed")
    plt.plot(VEpsilon_w,Vkrw_dr,label="MDC VG",linestyle="dashed")
    plt.plot(VEw_hyst[0:Counter],Vkrw_hyst[0:Counter],label="Parker-Lenhard")
    plt.legend()
    plt.xlabel("Ew")
    plt.ylabel("krw") 
    plt.title("Saturation vs krw at Node %d"%Node_hyst)
    plt.show()
    ######krn
    plt.plot(VEpsilon_w,Vkrn_imb,label="MIC VG",linestyle="dashed")
    plt.plot(VEpsilon_w,Vkrn_dr,label="MDC VG",linestyle="dashed")
    plt.plot(VEw_hyst[0:Counter],Vkrn_hyst[0:Counter],label="Parker-Lenhard")
    plt.legend()
    plt.xlabel("Ew")
    plt.ylabel("krn") 
    plt.title("Saturation vs krn at Node %d"%Node_hyst)
    plt.show()
    ######
    
    
    plt.plot(VTime_hyst[0:Counter],VPc_hyst[0:Counter])
    plt.xlabel("time(s)")
    plt.ylabel("Pc") 
    plt.title("Pc vs Time at Node %d"%Node_hyst)
    plt.show()
    ######
    plt.plot(VTime_hyst[0:Counter],VEw_endpoint_hyst[0:Counter])
    plt.xlabel("time(s)")
    plt.ylabel("Ew_endpoint") 
    plt.title("Ew_endpoint vs Time at Node %d"%Node_hyst)
    plt.show()
    ######
    
    
    ## print results about solver on screen
    ts.view()
 
 
      
    ############################
    
    ## print results about solver on screen
    ts.view()
    
    
main() 

