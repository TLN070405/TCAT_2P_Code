import math
#######################################
def  rhow_of_Pw(inp,P_w):
    Beta_w=inp.Beta_w
    rho_w0=inp.rho_w0
    rho_w =rho_w0*math.exp(Beta_w*P_w)
    return rho_w
#######################################  
def  Drvt_rhow_of_Pw(inp,P_w):
    Beta_w=inp.Beta_w
    rho_w0=inp.rho_w0
    Drvt_rho_w =Beta_w*rho_w0*math.exp(Beta_w*P_w)
    return Drvt_rho_w
#######################################
def  Linear_Geometric_Constants(inp,ew):
   
    N_Spheres        = 4021
    V                = 0.024*0.024*0.024
    r                = 0.000806994 #(meter) # (Total_volume-Spheres_Volume)/Total_volume=porosity ==>(V-4021*(4/3)*pi*r^3)/V=porosity
    e_s               = (4021*4*inp.pi*r*r)/V
    a_ens            = -e_s/inp.epsilon
    b_ens            = e_s
    ####
    e_ns             =  a_ens*ew+b_ens
    e_ws             =  e_s-e_ns
    js_ns            = 2/ r
    kn_ns            = 1/(r*r) 
    D_sauter         = inp.epsilon6/(e_ns+e_ws)
    #D_sauter         = 2*r               # 6*Vp/Ap
    b0D1             = inp.b0*D_sauter
    b1D2             = inp.b1*D_sauter*D_sauter
    b2D3             = inp.b2*D_sauter*D_sauter*D_sauter

    return e_ns,e_ws,js_ns,kn_ns,b0D1,b1D2,b2D3
#######################################
def  Kwn_Linear_Geometric( inp, ew, e_wn, j_wn, b0D1, b1D2, b2D3, e_ns, js_ns, kn_ns):
    c0=inp.epsilon-ew
    c1=b0D1*(e_wn+e_ns)
    c2=b1D2*(-e_wn*j_wn+e_ns*js_ns)
    c3=b2D3*e_ns*kn_ns
    c4=b2D3*e_wn
    kwn=(c0-c1-c2-c3)/c4
    return kwn 
#######################################
def   Wx_wn_IC_func(inp,dew_dx, Ewn_eq,  Pw,  Ewn,  Jwn): 
    Wx_wn=-(dew_dx/(Ewn**2))*(  inp.C_wn*(Pw+inp.Gamma_wn*Jwn) - (inp.Prod_Gamma_K_wn/(Pw+inp.Gamma_wn*Jwn+inp.Tol))*(Ewn_eq-Ewn)   )
    return Wx_wn 

#######################################
