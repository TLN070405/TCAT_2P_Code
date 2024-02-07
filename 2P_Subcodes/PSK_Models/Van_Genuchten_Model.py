import sys
import math
import petsc4py
petsc4py.init(sys.argv)
from petsc4py import PETSc
#######################################
OptDB = PETSc.Options()
Paper = OptDB.getString('Paper', 'Helmig') # get paper name from terminal so we can decide which inputs and PSK model to use
if(Paper=='Malgorzata'):
   sys.path.insert(0, './inputs/inputs_Malgorzata')
   import inputs as inp # import the inupts.py file as a module so we can access directly the parameters defined inside it
elif( Paper=='Helmig'):
   sys.path.insert(0, './inputs/inputs_Helmig')  
   import inputs as inp # import the inupts.py file as a module so we can access directly the parameters defined inside it
#######################################


phi=inp.phi
Epsilon_nr= inp.Epsilon_nr
Epsilon_wr= inp.Epsilon_wr
n=inp.n
rho_w0= inp.rho_w0
rho_n0= inp.rho_n0
mu_w=inp.mu_w 
mu_n=inp.mu_n
g= inp.g
d=inp.d
alpha= inp.alpha
a= inp.a
b=  inp.b
K=inp.K
m = inp.m
Const_Epsilon = inp.Const_Epsilon
n_inv=inp.n_inv
n_inv1=inp.n_inv1
n1=inp.n1
m_inv=inp.m_inv
m1=inp.m1
m2=inp.m2
alpha_inv=inp.alpha_inv
alpha_m_n=inp.alpha_m_n
Cz=inp.Cz

#######################################
def  Pc( Epsilon_w):
    Se = (Epsilon_w - Epsilon_wr) / Const_Epsilon
    if (Se > 0.0 and Se < 1.0):
       P_c = (  ( Se**(-m_inv)  - 1.0)**n_inv )*alpha_inv
    else:
       P_c = 0.0
    return P_c
#######################################
def  Drvt_Pc_by_Epsilon_w( Epsilon_w):
    Se = (Epsilon_w - Epsilon_wr) / Const_Epsilon
    if (Se > 0.0 and Se < 1.0):
       Drvt_P_c = -( ( Se**(-m_inv) - 1.0)**n_inv1 )*(1.0/(alpha_m_n*(Epsilon_w-Epsilon_wr)*(Se**m_inv)))
    else:
       Drvt_P_c = 0.0
    return Drvt_P_c
####################################### 
def  Krw_of_Pc( P_c):
    if (P_c > 0.0):
       Se =  (1.0 + ((alpha*P_c)**n))**(-m)
       Krw = (Se**0.5)  *(  (1.0 - (  (1.0 - Se**m_inv)**m  )   )**2.0)
    else:
       Krw = 1.0
    return Krw
####################################### 
def  Krn_of_Pc(  P_c):
    if (P_c > 0.0):
       Se =  (1.0 + (alpha*P_c)**n)**(-m)
       Krn = ((1.0 - Se)**0.5)*((1.0 - (Se**(m_inv)))**(m2))
    else:
       Krn = 0.0
    return Krn
####################################### 

def  Epsilon_w_of_Pc(  P_c):
    if (P_c > 0.0):
       Se =  (1.0 + (alpha*P_c)**n)**(-m)
       Epsilon_w =Const_Epsilon*  Se + Epsilon_wr
    else:
       Epsilon_w = phi-Epsilon_nr
    return Epsilon_w
####################################### 
def  Drvt_Epsilon_w_of_Pc(  P_c):
    if (P_c > 0.0):
       Drvt_Se = -alpha_m_n*((alpha*P_c)**n1)*((1.0 + ((alpha*P_c)**n))**m1)
       Drvt_Epsilon_w = Const_Epsilon*Drvt_Se
    else:
       Drvt_Epsilon_w = 0.0
    return Drvt_Epsilon_w
####################################### 
def  Epsilon_n_of_Pc( Epsilon_w):
    Epsilon_n = phi - Epsilon_w
    return Epsilon_n
####################################### 
def  rhow_of_Pw( P_w):
    rho_w =rho_w0*math.exp(a*P_w)
    return rho_w
#######################################  
def  Drvt_rhow_of_Pw( P_w):
    Drvt_rho_w =a*rho_w0*math.exp(a*P_w)
    return Drvt_rho_w
#######################################
def  rhon_of_Pn( P_n):
    rho_n =b* P_n+rho_n0
    return rho_n
#######################################
def  Drvt_rhon_of_Pn( P_n):
    Drvt_rho_n =b
    return Drvt_rho_n
#######################################
def Flux_qw(Ew_i,Pw_i,Pw_i1,sign):
    qw_t=rhow_of_Pw(Pw_i)*((K/mu_w)*Krw_of_Pc(Pc(Ew_i))*( Cz*sign*(Pw_i1-Pw_i) + rhow_of_Pw(Pw_i)*g*d) )
    return qw_t
#######################################
