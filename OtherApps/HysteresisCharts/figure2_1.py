#just a visualization of P-S realtions with imb/dr coefficients
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
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


Const_Epsilon_dr=phi-Epsilon_nr_dr - Epsilon_wr_dr
Const_Epsilon_imb=phi-Epsilon_nr_imb - Epsilon_wr_imb

####

 
 
 
 

m_imb = 1.0 - (1.0 / n_imb)
n_inv_imb=1.0/n_imb
m_inv_imb=1.0/m_imb
alpha_inv_imb=1.0/alpha_imb
 
m_dr = 1.0 - (1.0 / n_dr)
n_inv_dr=1.0/n_dr
m_inv_dr=1.0/m_dr
alpha_inv_dr=1.0/alpha_dr
 
 
############
VEpsilon_w_imb = np.linspace(Epsilon_wr_imb, phi-Epsilon_nr_imb, 1000)
VPc_imb=np.empty([len(VEpsilon_w_imb)], dtype=np.double) 
 
############
VEpsilon_w_dr = np.linspace(Epsilon_wr_dr, phi-Epsilon_nr_dr, 1000)
VPc_dr=np.empty([len(VEpsilon_w_dr)], dtype=np.double) 
 

 
 
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
############
def En_r_func(Ew_reversal): 
    ############
    Sw_e_reversal=(Ew_reversal - Epsilon_wr_dr) / Const_Epsilon_dr
    ############
    Snri_e=Epsilon_nr_imb/Const_Epsilon_dr
    Rnw=(1/Snri_e)-1
    ############
    Snr_e=(1-Sw_e_reversal)/(1+Rnw*(1-Sw_e_reversal))
    En_r=Snr_e*Const_Epsilon_dr
    return En_r





for i in range(len(VPc_dr)):
    VPc_dr[i] =Pc_dr(VEpsilon_w_dr[i])
    VPc_imb[i]=Pc_imb(VEpsilon_w_imb[i] )


####
N=1000
N_curves_imb=5
N_curves_dr=5



VEw_reversal_imb_range=np.linspace(Epsilon_wr_imb+phi*0.1, 0.37, N_curves_imb)
VEw_reversal_dr_range=np.linspace(0.14, 0.31, N_curves_dr)
####
VEw_reversal_imb=np.empty([N_curves_imb], dtype=np.double) 
VPc_reversal_imb=np.empty([N_curves_imb], dtype=np.double) 
VEw_endpoint_imb=np.empty([N_curves_imb], dtype=np.double) 
VPc_endpoint_imb=np.empty([N_curves_imb], dtype=np.double) 
VEw_new_imb=np.empty([N,N_curves_imb], dtype=np.double)
VPc_new_imb=np.empty([N,N_curves_imb], dtype=np.double)
####
VEw_reversal_dr=np.empty([N_curves_dr], dtype=np.double) 
VPc_reversal_dr=np.empty([N_curves_dr], dtype=np.double) 
VEw_endpoint_dr=np.empty([N_curves_dr], dtype=np.double) 
VPc_endpoint_dr=np.empty([N_curves_dr], dtype=np.double) 
VEw_new_dr=np.empty([N,N_curves_dr], dtype=np.double)
VPc_new_dr=np.empty([N,N_curves_dr], dtype=np.double)
####

####
for j in range(0,N_curves_imb): 
    VEw_reversal_imb[j]=VEw_reversal_imb_range[j]
    VPc_reversal_imb[j]=Pc_dr(VEw_reversal_imb[j])
    VEw_endpoint_imb[j]=phi-En_r_func(VEw_reversal_imb[j])
    VPc_endpoint_imb[j]=0
for j in range(0,N_curves_dr): 
    VEw_reversal_dr[j]= VEw_reversal_dr_range[j]
    VPc_reversal_dr[j]= Pc_imb(VEw_reversal_dr[j])
    VEw_endpoint_dr[j]= VEpsilon_w_dr[2]
    VPc_endpoint_dr[j]= Pc_dr(VEw_endpoint_dr[j])
    
    
####  Precise the range of volume fractions

for j in range(0,N_curves_imb):
    for i in range(0,N):
        VEw_new_imb[i][j]=VEw_reversal_imb[j]+i*((VEw_endpoint_imb[j]-VEw_reversal_imb[j])/(N-1))
for j in range(0,N_curves_dr):
    for i in range(0,N):
        VEw_new_dr[i][j]=VEw_reversal_dr[j]+i*((VEw_endpoint_dr[j]-VEw_reversal_dr[j])/(N-1))

 
############ Calculate the scaled graphs by calculating only Pc

for i in range(0,N):
    for j in range(0,N_curves_imb): 
        VPc_new_imb[i][j]=Pc_Hyst_Parker( VEw_new_imb[i][j], -1, VEw_reversal_imb[j], VEw_endpoint_imb[j], VPc_reversal_imb[j], VPc_endpoint_imb[j])
    for j in range(0,N_curves_dr):
        VPc_new_dr[i][j]=Pc_Hyst_Parker( VEw_new_dr[i][j], 1, VEw_reversal_dr[j], VEw_endpoint_dr[j], VPc_reversal_dr[j], VPc_endpoint_dr[j])






# plot hysteresis pw vs ew
# control the figure sizes using these params:
curves_thickness = 2.5
axis_titles_fs = 25
axis_numbers_fs = 20
legend_fs = 25
plt.figure(figsize=(10, 6))



# Plot primary drainage and main imbibition (1D arrays)
plt.plot(VEpsilon_w_dr[2:N], VPc_dr[2:N], color='r', label="Primary drainage curve", linewidth=curves_thickness)
plt.plot(VEpsilon_w_imb[2:N], VPc_imb[2:N], 'b', label="Main imbibition curve", linewidth=curves_thickness)

# Plot each imbibition scanning curve (green) and add arrows
for j in range(VEw_new_imb.shape[1]):
    x = VEw_new_imb[:, j]
    y = VPc_new_imb[:, j]
    label = "Imbibition scanning curves" if j == 0 else None
    plt.plot(x, y, color='g', label=label, linewidth=curves_thickness)
    
    # Arrow at midpoint
    i = len(x) // 2
    arrow = FancyArrowPatch(
        (x[i], y[i]), (x[i+1], y[i+1]),
        arrowstyle='->', color='g', mutation_scale=20, linewidth=2,
        transform=plt.gca().transData
    )
    plt.gca().add_patch(arrow)

# Plot each drainage scanning curve (black) and add arrows
for j in range(VEw_new_dr.shape[1]):
    x = VEw_new_dr[:, j]
    y = VPc_new_dr[:, j]
    label = "Drainage scanning curves" if j == 0 else None
    plt.plot(x, y, color='k', label=label, linewidth=curves_thickness)
    
    # Arrow at midpoint
    i = len(x) // 2
    arrow = FancyArrowPatch(
        (x[i], y[i]), (x[i+1], y[i+1]),
        arrowstyle='->', color='k', mutation_scale=20, linewidth=2,
        transform=plt.gca().transData
    )
    plt.gca().add_patch(arrow)

# Remove duplicate labels (for legend)
handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys(), fontsize=legend_fs)

plt.xlabel(r"$\epsilon^w\,[-]$", fontsize=axis_titles_fs)
plt.ylabel(r"$p^c\,[Pa]$", fontsize=axis_titles_fs)
plt.tick_params(axis='both', labelsize=axis_numbers_fs)  # Increase tick label size
plt.grid(False)
plt.show()




