import sys
import math
sys.path.insert(0, '../inputs/inputs_Fucik')
import inputs as inp

phi= inp.phi
Epsilon_nr= inp.Epsilon_nr
Epsilon_wr= inp.Epsilon_wr
rho_w0 =inp.rho_w0
rho_n0 =inp.rho_n0
tau=inp.tau
lamda=inp.lamda
Pd=inp.Pd
K=inp.K
mu_w=inp.mu_w
mu_n=inp.mu_n
a=inp.a
b=inp.b
g=inp.g
Const_Epsilon=inp.Const_Epsilon
Cz=inp.Cz
d=inp.d
#######################################
def  Pc( Epsilon_w):
    Se = (Epsilon_w - Epsilon_wr) / Const_Epsilon
    if (Se > 0.0 and Se <= 1.0):
       P_c = Pd*((Se)**(-1.0/lamda))
    else:
       P_c = Pd
    return P_c
#######################################
def  Drvt_Pc_by_Epsilon_w( Epsilon_w):
    Se = (Epsilon_w - Epsilon_wr) / Const_Epsilon
    if (Se > 0.0 and Se < 1.0):
       Drvt_P_c =  (-Pd/(lamda*(phi-Epsilon_wr)))*(Se**((-1/lamda)-1))
    else:
       Drvt_P_c = 0.0
    return Drvt_P_c
####################################### 
def  Krw_of_Pc( P_c):
    if (P_c > Pd):
       Se =   (P_c/Pd)**(-lamda)
       Krw = Se**(3+(2/lamda)) 
    else:
       Krw = 1.0
    return Krw
####################################### 
def  Krn_of_Pc(  P_c):
    if (P_c > Pd):
       Se =   (P_c/Pd)**(-lamda) 
       Krn = ((1.0-Se)**2)*(1-(Se**(1.0+(2/lamda))))
    else:
       Krn = 0.0
    return Krn
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
    qw_t=rhow_of_Pw(Pw_i)*(-(K/mu_w)*Krw_of_Pc(Pc(Ew_i))*( Cz*sign*(Pw_i1-Pw_i) + rhow_of_Pw(Pw_i)*g*d) )
    return qw_t
    
 
