import math
#######################################
def Pc(inp,Epsilon_w):
    Se = (Epsilon_w - inp.Epsilon_wr) / inp.Const_Epsilon
    if (Se > 0.0 and Se < 1.0):
       P_c = (  ( Se**(-inp.m_inv)  - 1.0)**inp.n_inv )*inp.alpha_inv
    else:
       P_c = 0.0
    return P_c
####################################### 
def  Drvt_Epsilon_w_of_Pc(inp, P_c):
    if (P_c > 0.0):
       Drvt_Se = -inp.alpha_m_n*((inp.alpha*P_c)**inp.n1)*((1.0 + ((inp.alpha*P_c)**inp.n))**inp.m1)
       Drvt_Epsilon_w = inp.Const_Epsilon*Drvt_Se
    else:
       Drvt_Epsilon_w = 0.0
    return Drvt_Epsilon_w

####################################### 
def  Krw_of_Pc(inp,P_c):
    if (P_c > 0.0):
       Se =  (1.0 + ((inp.alpha*P_c)**inp.n))**(-inp.m)
       Krw =    (Se**0.5)  *(  (1.0 - (  (1.0 - Se**inp.m_inv)**inp.m  )   )**2.0)
    else:
       Krw = 1.0
    return Krw
    
    
####################################### 
def  Epsilon_w_of_Pc(inp,P_c):
    if (P_c > 0.0):
       Se =  (1.0 + (inp.alpha*P_c)**inp.n)**(-inp.m)
       Epsilon_w =inp.Const_Epsilon*  Se + inp.Epsilon_wr
    else:
       Epsilon_w = inp.epsilon-inp.Epsilon_nr
    return Epsilon_w
 
#######################################
def  rhow_of_Pw(inp,P_w):
    rho_w =inp.rho_w0*math.exp(inp.Beta_w*P_w)
    return rho_w
#######################################
def  Drvt_rhow_of_Pw(inp,P_w):
    Drvt_rho_w =inp.Beta_w*inp.rho_w0*math.exp(inp.Beta_w*P_w)
    return Drvt_rho_w


#######################################
