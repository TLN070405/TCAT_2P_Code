import numpy as np
from numpy import linalg 
from petsc4py import PETSc
import TCAT2Phase
from TCAT2Phase.numerics import utils
from TCAT2Phase.state_equations import state_equations
from TCAT2Phase.state_equations import state_equations_interface
import matplotlib.pyplot as plt
import math
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
boundary_type = [0,0]
model = TCAT2Phase.numerics.model.initialize_model(
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

# Set boundary names
boundary_name = {'e_w': [['e_w', 'e_w']],
                 'p_w': [['p_w', 'p_w']],
                 'e_wn': [['e_wn', 'e_wn']],
                 'j_wn': [['j_wn', 'j_wn']],
                 'w_wn': [['w_wn', 'w_wn']],
                 'k_wn': [['k_wn', 'k_wn']]}

model.set_boundary_name(boundary_name = boundary_name)



# Set solid phase properties
s_phase = TCAT2Phase.properties.phase.Solid('s')
s_phase.set_porosity(0.369)
s_phase.set_gravity(-9.81)

# Set wetting-phase properties
w_phase = TCAT2Phase.properties.phase.Fluid('w')
w_phase.set_momentum_resistance(1.5e6)
w_den_params = {'rho0':1.e3,
                'beta':1e-8}
w_phase.set_density_function(state_equations.density_exp_pressure,**w_den_params)
w_phase.set_d_density_function(state_equations.d_density_exp_pressure,**w_den_params)

# Set w-n interphase properties
wn_interface = TCAT2Phase.properties.interface.Interface()

b1 = 1.47e-1
b2 = -2.08e-3
b3 = 5.33e-2/(4.0 * 3.141592653589793)

tcat = state_equations_interface.TCAT(
    b1 = b1,
    b2 = b2,
    b3 = b3,
    porosity = s_phase.porosity,
    e_ns = 1. ,
    e_ws = 1. ,
    js_ns = 1.,
    kn_ns = 1.167425480799784
    )
wn_interface.set_interfacial_dynamics(1e-7)
wn_interface.set_interfacial_tension(72e-3)
wn_interface.set_interfacial_area_relaxation(1.41e-2)
wn_interface.set_volume_fraction_function(tcat.volume_fraction_w)

w_phase.set_rel_perm_function(tcat.w_relative_permability)
w_phase.set_volume_fraction_function(tcat.volume_fraction_w)

model.solid = s_phase
model.w_phase = w_phase
model.interface = wn_interface


def ew_ic_func(x, pw0):
    c1= math.exp(1e-8*pw0)
    c2=-1e-8 * 1000 * 9.81
    c3= 0.36796438207402804/(math.exp(-0.5*c2*0)*(1-c1*math.exp(c2*0))**(-0.5))
    ew = c3*math.exp(-0.5*c2*x)*(1-c1*math.exp(c2*x))**(-0.5)
    return ew

def kwn_ic_func(ew, e_wn, j_wn):
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
ew_ic= np.zeros([nodes])
pw_ic= np.zeros([nodes])
ewn_ic= np.zeros([nodes])
jwn_ic= np.zeros([nodes])
wwn_ic= np.zeros([nodes])
kwn_ic= np.zeros([nodes])
ewn_eq = np.zeros([nodes])
x_loc = np.zeros([nodes])
d_ew_d_x  = np.zeros([nodes])


for i in range(0,nodes):
    x_loc[i] = i*10/(nodes-1)
    ew_ic[i] = ew_ic_func(x_loc[i],-1000)
    pw_ic[i] = -1000-1000*9.81*x_loc[i]+1e-4
    ewn_ic[i] = 100*ew_ic[i]
    jwn_ic[i] = -pw_ic[i]/wn_interface.interfacial_tension
    ewn_eq[i] = ewn_ic[i]

for i in range(0,nodes):
    if i==0:
        d_ew_d_x[i] = (ew_ic[i+1]-ew_ic[i])/(10/(nodes-1))
    else:
        d_ew_d_x[i] = (ew_ic[i]-ew_ic[i-1])/(10/(nodes-1))
    wwn_ic[i] = wx_wn_ic_func( d_ew_d_x[i], ewn_eq[i] ,  pw_ic[i],  ewn_ic[i],  jwn_ic[i])
    kwn_ic[i] = kwn_ic_func( ew_ic[i],  ewn_ic[i], jwn_ic[i])


wn_interface.set_interfacial_area_equilibrium(ewn_eq)

initial_conditions = {'e_w': ew_ic,
                      'p_w': pw_ic,
                      'e_wn': ewn_ic,
                      'j_wn': jwn_ic,
                      'w_wn': wwn_ic,
                      'k_wn': kwn_ic}

model.set_initial_conditions(initial_conditions)

# Set boundary values
def ew0_bc(t): return ew_ic[0]
def pw0_bc(t): return  pw_ic[0]
def ewn0_bc(t): return ewn_ic[0]
def jwn0_bc(t): return jwn_ic[0]
def wwn0_bc(t): return wwn_ic[0] 
def kwn0_bc(t): return kwn_ic[0]

def ewL_bc(t): return  ew_ic[nodes-1]
def pwL_bc(t): return pw_ic[nodes-1]
def ewnL_bc(t): return ewn_ic[nodes-1]
def jwnL_bc(t): return jwn_ic[nodes-1]
def wwnL_bc(t): return wwn_ic[nodes-1] 
def kwnL_bc(t): return kwn_ic[nodes-1]


boundary_value = {'e_w': [[ ew0_bc,ewL_bc ]],
                  'p_w': [[ pw0_bc,pwL_bc ]],
                  'e_wn': [[ ewn0_bc,ewnL_bc ]],
                  'j_wn': [[ jwn0_bc,jwnL_bc ]],
                  'w_wn': [[ wwn0_bc,wwnL_bc ]],
                  'k_wn': [[ kwn0_bc,kwnL_bc ]]}

model.set_boundary_value(boundary_value = boundary_value)


# Time
time_interval = np.array([1e-5,1e-4,1e-3,1e-2,1,10,20,30,50])
model.set_simulation_time(time_interval)

# Numerical Parameters
OptDB = PETSc.Options()

# Time-Stepper Options
OptDB.setValue('snes_fd_color', None)
OptDB.setValue('ts_type', 'bdf')
OptDB.setValue('ts_bdf_order', 1)
OptDB.setValue('ts_max_steps',100000)
OptDB.setValue('ts_max_reject',500)
OptDB.setValue('ts_monitor', None)

# Nonlinear Solver Options
OptDB.setValue('ts_max_snes_failures',10000)

# Linear Solver Options
OptDB.setValue('ts_rtol', 1.e-8)
OptDB.setValue('ts_atol', 1.e-8)
# Preconditioner Options
OptDB.setValue('pc_type', 'sor')


# Solve model and return model.solutions
# which is a dictionary where
# solutions[time][u_name] = np.array
u, ts = TCAT2Phase.numerics.solver.setup_solver(model)
solutions = TCAT2Phase.numerics.solver.solver_time(u,ts,model)

for time,variable in solutions.items():
    plt.plot(mesh.points,variable['e_w'],label=f't={time}')
    plt.xlabel('x(m)')
    plt.ylabel('e_w')
    plt.title('tcat cgp')
    plt.legend()
plt.show()


for time,variable in solutions.items():
    print(time,variable)
    plt.plot(mesh.points,variable['p_w'],label=f't={time}')
    plt.xlabel('x(m)')
    plt.ylabel('p_w')
    plt.title('tcat cgp')
    plt.legend()
plt.show()

for time,variable in solutions.items():
    print(time,variable)
    plt.plot(mesh.points,variable['e_wn'],label=f't={time}')
    plt.xlabel('x(m)')
    plt.ylabel('e_wn')
    plt.title('tcat cgp')
    plt.legend()
plt.show()

for time,variable in solutions.items():
    print(time,variable)
    plt.plot(mesh.points,variable['j_wn'],label=f't={time}')
    plt.xlabel('x(m)')
    plt.ylabel('j_wn')
    plt.title('tcat cgp')
    plt.legend()
plt.show()

for time,variable in solutions.items():
    print(time,variable)
    plt.plot(mesh.points,variable['w_wn'],label=f't={time}')
    plt.xlabel('x(m)')
    plt.ylabel('w_wn')
    plt.title('tcat cgp')
    plt.legend()
plt.show()

for time,variable in solutions.items():
    print(time,variable)
    plt.plot(mesh.points,variable['k_wn'],label=f't={time}')
    plt.xlabel('x(m)')
    plt.ylabel('k_wn')
    plt.title('tcat cgp')
    plt.legend()
plt.show()