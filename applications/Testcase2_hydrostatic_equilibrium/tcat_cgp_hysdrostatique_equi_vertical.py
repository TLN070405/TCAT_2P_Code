import numpy as np
import TCAT2Phase
from TCAT2Phase.state_equations import state_equations
from TCAT2Phase.state_equations import state_equations_interface
import matplotlib.pyplot as plt
import math
import solver_options

# Domain
domain_size = np.array([[0.0,10.0]])
domain =  TCAT2Phase.properties.domain.initialize_domain(
                            dims = 1,
                            domain_size = domain_size,
                            )

# Discretization
nodes = 101

# Mesh
mesh = TCAT2Phase.mesh.grid.Mesh()
mesh.uniform_mesh_1d(domain,nodes)

# Boundaries
model = TCAT2Phase.numerics.model.Model(
                                model_type = 'cgp-TCAT',
                                form = 'spijvk',
                                domain = domain,
                                nodes = nodes,
                                mesh = mesh
                                    )

# Set boundary types: 0 - Dirchlet
#                     1 - Neumann 
boundary_type = {'e_w': [[0, 0]],
                 'p_w': [[0, 0]],
                 'e_wn': [[0, 0]],
                 'j_wn': [[0, 0]],
                 'w_wn': [[0, 0]],
                 'k_wn': [[0, 0]]}

model.set_boundary_type(boundary_type = boundary_type)

# Set solid phase properties
s_phase = TCAT2Phase.properties.phase.Solid('s',porosity = 0.369,
                                                permeability = None,
                                                gravity = -9.81,
                                                dynamic_capillary_pressure_parameter = None)

# Set wetting-phase properties
w_den_params = {'rho0':1.e3, 'beta':1e-8}
w_phase=TCAT2Phase.properties.phase.Fluid('w',viscosity = None,
                                              momentum_resistance = 1.5e6,
                                              residual_volume_fraction = 0.0,
                                              density_function = state_equations.density_exp_pressure,
                                              d_density_function = state_equations.d_density_exp_pressure,
                                              **w_den_params)

# Set w-n interphase properties
tcat = state_equations_interface.TCAT(b1 = 1.47e-1,
                                      b2 = -2.08e-3,
                                      b3 = 5.33e-2/(4.0 * 3.141592653589793),
                                      porosity = s_phase.porosity,
                                      e_ns = 1. ,
                                      e_ws = 1. ,
                                      js_ns = 1.,
                                      kn_ns = 1.167425480799784)

def ew_ic_func(x, pw0):
    c1= math.exp(1e-8*pw0)
    c2=-1e-8 * 1000 * 9.81
    c3= 0.36796438207402804/(math.exp(-0.5*c2*0)*(1-c1*math.exp(c2*0))**(-0.5))
    ew = c3*math.exp(-0.5*c2*x)*(1-c1*math.exp(c2*x))**(-0.5)
    return ew

def dew_ic_func(x, pw0):
    c1= math.exp(1e-8*pw0)
    c2=-1e-8 * 1000 * 9.81
    c3= 0.36796438207402804/(math.exp(-0.5*c2*0)*(1-c1*math.exp(c2*0))**(-0.5))
    dew = -0.5*c2*c3*math.exp(-0.5*c2*x)/((1-c1*math.exp(c2*x))**0.5)
    return dew

def interfacial_area_equilibrium(x): return 100*ew_ic_func(x,-1000)

wn_interface = TCAT2Phase.properties.interface.TCATInterface(tcat,
                                                             interfacial_dynamics = 1e-7,
                                                             interfacial_tension = 72e-3,
                                                             interfacial_area_relaxation = 1.41e-2,
                                                             interfacial_area_equilibrium = interfacial_area_equilibrium)

TCAT2Phase.numerics.model.Model.set_model_phases( model, s_phase, w_phase, None, wn_interface)


def kwn_ic_func(ew, e_wn, j_wn):
    b1 = 1.47e-1
    b2 = -2.08e-3
    b3 = 5.33e-2/(4.0 * 3.141592653589793)
    d_sauter         =  3*0.369
    b1d1             =  b1*d_sauter
    b2d2             =  b2*d_sauter*d_sauter
    b3d3             =  b3*d_sauter*d_sauter*d_sauter
    c0=0.369-ew
    c1=b1d1*(e_wn+1)
    c2=b2d2*(-e_wn*j_wn+1*1)
    c3=b3d3*1*1.167425480799784
    c4=b3d3*e_wn
    kwn=(c0-c1-c2-c3)/c4
    return kwn

def wx_wn_ic_func(dew_dx, e_wn_eq, p_w, e_wn, j_wn):
    wx_wn=-(dew_dx/e_wn**2)*(1e-7*(p_w+72e-3*j_wn)  +(72e-3*1.5e6/(p_w+72e-3*j_wn+1e-3))*(e_wn_eq-e_wn)   )
    return wx_wn

# Set initial conditions
def ew_ic(x): return ew_ic_func(x,-1000)
def pw_ic(x): return -1000-1000*9.81*x
def ewn_ic(x): return 100*ew_ic(x)
def jwn_ic(x): return -pw_ic(x)/72e-3
def wwn_ic(x): return wx_wn_ic_func(dew_ic_func(x, -1000), interfacial_area_equilibrium(x) ,  pw_ic(x), ewn_ic(x),  jwn_ic(x))
def kwn_ic(x): return kwn_ic_func(ew_ic_func(x,-1000), ewn_ic(x), jwn_ic(x))

initial_conditions = {'e_w': ew_ic,
                      'p_w': pw_ic,
                      'e_wn': ewn_ic,
                      'j_wn': jwn_ic,
                      'w_wn': wwn_ic,
                      'k_wn': kwn_ic}

model.set_initial_conditions(initial_conditions)

# Set boundary values
def ew0_bc(t): return ew_ic(0)
def pw0_bc(t): return  pw_ic(0)
def ewn0_bc(t): return ewn_ic(0)
def jwn0_bc(t): return jwn_ic(0)
def wwn0_bc(t): return wwn_ic(0)
def kwn0_bc(t): return kwn_ic(0)

def ewL_bc(t): return  ew_ic(10)
def pwL_bc(t): return pw_ic(10)
def ewnL_bc(t): return ewn_ic(10)
def jwnL_bc(t): return jwn_ic(10)
def wwnL_bc(t): return wwn_ic(10)
def kwnL_bc(t): return kwn_ic(10)


boundary_conditions = {'e_w': [[ ew0_bc,ewL_bc ]],
                  'p_w': [[ pw0_bc,pwL_bc ]],
                  'e_wn': [[ ewn0_bc,ewnL_bc ]],
                  'j_wn': [[ jwn0_bc,jwnL_bc ]],
                  'w_wn': [[ wwn0_bc,wwnL_bc ]],
                  'k_wn': [[ kwn0_bc,kwnL_bc ]]}

model.set_boundary_conditions(boundary_conditions)


# Time
time_interval = np.array([1e-5,1e-4,1e-3,1e-2,1,10,20,30,50])
model.set_simulation_time(time_interval)

# Set petsc solver options
solver_options.solver_options(max_time_steps=100000,bdf_order=1,pc_type='sor')


# Solve model and return model.solutions
# which is a dictionary where
# solutions[time][u_name] = np.array
u, ts = TCAT2Phase.numerics.solver.setup_solver(model)
solutions = TCAT2Phase.numerics.solver.solver_time(u,ts,model)


for time,variable in solutions.items():
    plt.plot(mesh.points,variable['e_w'],label=f't={time}')
    plt.xlabel(r'$x\, (m)$')
    plt.ylabel(r'$\epsilon^w$')
    plt.title('tcat cgp')
    plt.legend()
plt.show()


for time,variable in solutions.items():
    print(time,variable)
    plt.plot(mesh.points,variable['p_w'],label=f't={time}')
    plt.xlabel(r'$x\, (m)$')
    plt.ylabel(r'$p_w$')
    plt.title('tcat cgp')
    plt.legend()
plt.show()

for time,variable in solutions.items():
    print(time,variable)
    plt.plot(mesh.points,variable['e_wn'],label=f't={time}')
    plt.xlabel(r'$x\, (m)$')
    plt.ylabel(r'$\epsilon^{wn}$')
    plt.title('tcat cgp')
    plt.legend()
plt.show()

for time,variable in solutions.items():
    print(time,variable)
    plt.plot(mesh.points,variable['j_wn'],label=f't={time}')
    plt.xlabel(r'$x\, (m)$')
    plt.ylabel(r'$J^{wn}$')
    plt.title('tcat cgp')
    plt.legend()
plt.show()

for time,variable in solutions.items():
    print(time,variable)
    plt.plot(mesh.points,variable['w_wn'],label=f't={time}')
    plt.xlabel(r'$x\, (m)$')
    plt.ylabel(r'$w^{wn}$')
    plt.title('tcat cgp')
    plt.legend()
plt.show()

for time,variable in solutions.items():
    print(time,variable)
    plt.plot(mesh.points,variable['k_wn'],label=f't={time}')
    plt.xlabel(r'$x\, (m)$')
    plt.ylabel(r'$K^{wn}$')
    plt.title('tcat cgp')
    plt.legend()
plt.show()