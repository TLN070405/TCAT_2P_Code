import matplotlib.pyplot as plt
import numpy as np

##### Fixed Values #####
# Wetting-phase volume fraction
e_w = 0.300
# Wetting-phase pressure (Pa)
p_w = 375
# Interfacial tension (N/m)
Gamma_wn  = 7.20e-2
# Mean curvature (1/m)
J_wn = p_w / Gamma_wn  # TCAT relationship: J = pw / gamma

##### Domain porosity #####
epsilon = 0.369
epsilon6 = epsilon * 6

##### Linear geometric state equation constants #####
pi = np.pi
b0 = 1.47e-1
b1 = -2.08e-3
b2 = (5.33e-2) / (4 * pi)

##### Helper Functions #####
def Linear_Geometric_Constants(ew):
    e_ss = 1.89e3
    js_ns = 2.00e3
    kn_ns = 1.00e6

    Swi = ew / epsilon
    e_ns = (1 - Swi) * e_ss
    e_ws = Swi * e_ss
    D_sauter = epsilon6 / (e_ns + e_ws)

    b0D1 = b0 * D_sauter
    b1D2 = b1 * D_sauter**2
    b2D3 = b2 * D_sauter**3
    return e_ns, e_ws, js_ns, kn_ns, b0D1, b1D2, b2D3
"""
def Kwn_Linear_Geometric(e_wn, ew, j_wn, b0D1, b1D2, b2D3, e_ns, js_ns, kn_ns):
    c0 = epsilon - ew
    c1 = b0D1 * (e_wn + e_ns)
    c2 = b1D2 * (-e_wn * j_wn + e_ns * js_ns)
    c3 = b2D3 * e_ns * kn_ns
    c4 = b2D3 * e_wn
    kwn = (c0 - c1 - c2 - c3) / c4
    return kwn
"""
def Kwn_Linear_Geometric(e_wn, ew, j_wn, b0D1, b1D2, b2D3, e_ns, js_ns, kn_ns):
    c0 = epsilon - ew
    c1 = b0D1 * (e_wn + e_ns)
    c2 = b1D2 * (-e_wn * j_wn + e_ns * js_ns)
    c3 = e_ns * kn_ns / e_wn
    c4 = b2D3 * e_wn
    kwn = ((c0 - c1 - c2) / c4) - c3
    return kwn

##### Generate Data #####
e_ns, e_ws, js_ns, kn_ns, b0D1, b1D2, b2D3 = Linear_Geometric_Constants(e_w)
e_wn_values = np.linspace(7700, 62000, 300)
kwn_values = [Kwn_Linear_Geometric(e_wn, e_w, J_wn, b0D1, b1D2, b2D3, e_ns, js_ns, kn_ns) for e_wn in e_wn_values]

##### Plot #####
plt.figure(figsize=(10, 6))
plt.plot(e_wn_values, kwn_values, label=r'$K_n^{wn}$ [1/m$^2$]', color='blue')
plt.xlabel(r'Interfacial Area $\epsilon^{\overline{\overline{wn}}}$ [1/m]', fontsize=14)
plt.ylabel(r'Gaussian Curvature $K_n^{wn}$ [1/m$^2$]', fontsize=14)
plt.title('Gaussian Curvature vs. Interfacial Area', fontsize=16)
plt.grid(True)
plt.legend(fontsize=12)
plt.tight_layout()
plt.show()

##### Summary Output #####
print(f"Fixed parameters:\n  e_w = {e_w}\n  p_w = {p_w} Pa\n  Gamma_wn = {Gamma_wn} N/m\n  J_wn = {J_wn:.2f} 1/m")
