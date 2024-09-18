import numpy as np
from numpy import linalg 
from petsc4py import PETSc
import TCAT2Phase
from TCAT2Phase.numerics import utils
from TCAT2Phase.state_equations import state_equations
from TCAT2Phase.state_equations import state_equations_interface
from TCAT2Phase.state_equations import state_equations_hysteresis
import matplotlib.pyplot as plt
# Domain

domain_size = np.array([[0.0,1.0]])
domain =  TCAT2Phase.properties.domain.initialize_domain(
                            dims = 1,
                            domain_size = domain_size,
                            )

# Discretization
nodes =2

# Mesh
mesh = TCAT2Phase.mesh.grid.Mesh()
mesh.uniform_mesh_1d(domain,nodes)

# Boundaries
boundary_type = [0,0]
model = TCAT2Phase.numerics.model.initialize_model(
                                model_type = 'two_phase',
                                form = 'sp',
                                domain = domain,
                                nodes = nodes,
                                mesh = mesh
                                    )

# Set boundary types: 0 - Dirchlet
#                     1 - Neumann (but not supported yet)
boundary_type = {'e_w': [[0, 0]],
                 'p_w': [[0, 0]]}

model.set_boundary_type(boundary_type = boundary_type)

# Set boundary names
boundary_name = {'e_w': [['e_w', 'e_w']],
                 'p_w': [['p_w', 'p_w']]}

model.set_boundary_name(boundary_name = boundary_name)




# Set solid phase properties
s_phase = TCAT2Phase.properties.phase.Solid('s')
s_phase.set_porosity(0.4)
s_phase.set_permeability(1e-10)
s_phase.set_gravity(0.0)

# Set wetting-phase properties
w_phase = TCAT2Phase.properties.phase.Fluid('w')
w_phase.set_residual_volume_fraction(0.0)
w_phase.set_viscosity(1.e-3)
w_den_params = {'rho0':1.e3,
                'beta':4.9e-10}
w_phase.set_density_function(state_equations.density_exp_pressure,**w_den_params)
w_phase.set_d_density_function(state_equations.d_density_exp_pressure,**w_den_params)

# Set nonwetting-phase properties
n_phase = TCAT2Phase.properties.phase.Fluid('n')
n_phase.set_residual_volume_fraction(0.0)
n_phase.set_viscosity(1.78e-5)
n_den_params = {'rho0':1.2,
                'alpha':1.189e-5
                }
n_phase.set_density_function(state_equations.density_linear_pressure,**n_den_params)
n_phase.set_d_density_function(state_equations.d_density_linear_pressure,**n_den_params)
n_en_params = {'porosity':s_phase.porosity
                }
n_phase.set_volume_fraction_function(state_equations.volume_fraction,**n_en_params)

# Set up the model phases.
model.solid = s_phase
model.w_phase = w_phase
model.n_phase = n_phase
# Set w-n interphase properties
wn_interface_imb = TCAT2Phase.properties.interface.Interface()

n = 2.494
m = 1.0 - (1.0 / n)
alpha = 1e-3

vg_imb = state_equations_interface.VanGenuchten(
    m = m,
    n = n,
    alpha = alpha,
    porosity = s_phase.porosity,
    e_wr = w_phase.residual_volume_fraction,
    e_nr = n_phase.residual_volume_fraction,
    )

wn_interface_imb.set_pc_function(vg_imb.capillary_pressure)
wn_interface_imb.set_d_pc_function(vg_imb.d_capillary_pressure)
wn_interface_imb.set_volume_fraction_function(vg_imb.volume_fraction_w)
wn_interface_imb.set_d_volume_fraction_function(vg_imb.d_volume_fraction_w)

# Set wn-interface properties drainage
wn_interface_dr = TCAT2Phase.properties.interface.Interface()

n = 2.494
m = 1.0 - (1.0 / n)
alpha = 5e-4

vg_dr = state_equations_interface.VanGenuchten(
    m = m,
    n = n,
    alpha = alpha,
    porosity = s_phase.porosity,
    e_wr = w_phase.residual_volume_fraction,
    e_nr = n_phase.residual_volume_fraction,
    )
wn_interface_dr.set_pc_function(vg_dr.capillary_pressure)
wn_interface_dr.set_d_pc_function(vg_dr.d_capillary_pressure)
wn_interface_dr.set_volume_fraction_function(vg_dr.volume_fraction_w)
wn_interface_dr.set_d_volume_fraction_function(vg_dr.d_volume_fraction_w)

w_phase.set_rel_perm_function(vg_dr.w_relative_permability)
n_phase.set_rel_perm_function(vg_dr.n_relative_permability)


# Set hysteresis data and model
model.hysteresis = True

drainage_new         = np.full((nodes), True)
p_c_reversal         = np.full((nodes), 0.0)
p_c_endpoint         = np.full((nodes), 0.0)
e_w_reversal         = np.full((nodes), 0.0)
e_w_endpoint         = np.full((nodes), 0.0)
ew_hysteresis        = np.full((nodes,100000), 0.0)
pc_hysteresis        = np.full((nodes,100000), 0.0)

hysteresis_tol = 1e-6

for n in range(0,nodes):
    ew_hysteresis[n][0]  = 0.397
    pc_hysteresis[n][0]  = 346.73
    e_w_reversal[n]      = 0.4
    e_w_endpoint[n]      = 0.003
    p_c_reversal[n]      = wn_interface_dr.pc_function(e_w_reversal[n])
    p_c_endpoint[n]      = wn_interface_dr.pc_function(e_w_endpoint[n])


model.interface = state_equations_hysteresis.ParkerLenhard(
    model,
    wn_interface_imb,
    wn_interface_dr,
    drainage_new,
    p_c_reversal,
    p_c_endpoint,
    e_w_reversal,
    e_w_endpoint,
    ew_hysteresis,
    pc_hysteresis,
    hysteresis_tol
    )

# Set initial conditions
ew_ic = np.full((nodes), 0.397)
pw_ic = np.full((nodes), -346.73)

initial_conditions = {'e_w': ew_ic,
                      'p_w': pw_ic}

model.set_initial_conditions(initial_conditions)

# Set boundary values

def pw0_bc(t):
    """
    Boundary wetting phase pressure
    """
    pw_t0 = -346.73
    pw_t1 = -7000
    pw_t2 = -2000
    pw_t3 = -4000
    if 0<=t<=500:
        aa=(pw_t1-pw_t0)/(500-0)
        bb=pw_t0-aa*0
        p_w0=aa*t+bb
    elif 500<=t<=1000:
        aa=(pw_t2-pw_t1)/(1000-500)
        bb=pw_t1-aa*500
        p_w0=aa*t+bb
    elif 1000<=t<=1500:
        aa=(pw_t3-pw_t2)/(1500-1000)
        bb=pw_t2-aa*1000
        p_w0=aa*t+bb
    return p_w0

def ew0_bc(t):
    """
    Boundary volume fraction
    """
    model.current_cell = 0
    e_w_bc = model.interface.volume_fraction_function(-pw0_bc(t))

    return  e_w_bc

def ewL_bc(t):
    """
    Boundary volume fraction
    """
    return  0.397

def pwL_bc(t):
    """
    Boundary wetting phase pressure
    """
    return  -346.73


boundary_value = {'e_w': [[ ew0_bc,ewL_bc ]],
                  'p_w': [[ pw0_bc,pwL_bc  ]]}

model.set_boundary_value(boundary_value = boundary_value)

# Time
time_interval = np.array([1500])

model.set_simulation_time(time_interval)

# Numerical Parameters
OptDB = PETSc.Options()

# Time-Stepper Options
OptDB.setValue('snes_fd_color', None)
OptDB.setValue('ts_type', 'bdf')
OptDB.setValue('ts_bdf_order', 5)
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
solutions = TCAT2Phase.numerics.solver.solver(model)

for time,variable in solutions.items():
    plt.plot(mesh.points,variable['e_w'],label=f't={time}')
    plt.xlabel("x (m)")
    plt.ylabel("e_w")
    plt.title("Malgorzata Solution")
    plt.legend()
plt.show()


for time,variable in solutions.items():
    plt.plot(mesh.points,variable['p_w'],label=f't={time}')
    plt.xlabel('x (m)')
    plt.ylabel('p_w')
    plt.title('Malgorzata Solution')
    plt.legend()
plt.show()

# Hysteresis visualization
e_w_range    = np.linspace(0.05, s_phase.porosity, 1000)
p_c_mpc_dr   = np.zeros([len(e_w_range)], dtype=np.double)
p_c_mpc_imb  = np.zeros([len(e_w_range)], dtype=np.double)

for n, value in enumerate(e_w_range):
    p_c_mpc_dr[n]   = wn_interface_dr.pc_function(value)
    p_c_mpc_imb[n]  = wn_interface_imb.pc_function(value)


counter_hysteresis=model.interface.data.counter_hysteresis

plt.plot(model.interface.data.ew_hysteresis[0,0:counter_hysteresis],
         model.interface.data.pc_hysteresis[0,0:counter_hysteresis],
         label='scanning curves')
plt.plot(e_w_range,p_c_mpc_imb,label="MIC VG",linestyle="dashed")
plt.plot(e_w_range,p_c_mpc_dr,label="MDC VG",linestyle="dashed")
plt.xlabel('e_w')
plt.ylabel('p_c')
plt.title('Hysteresis pc-ew curves')
plt.legend()
plt.show()

time_hysteresis    = np.linspace(0, counter_hysteresis, counter_hysteresis)

plt.plot(time_hysteresis,model.interface.data.ew_hysteresis[0,0:counter_hysteresis])
plt.xlabel('time step number')
plt.ylabel('ew ')
plt.title('ew vs time step number at node 0')
plt.show()

plt.plot(time_hysteresis,model.interface.data.pc_hysteresis[0,0:counter_hysteresis])
plt.xlabel('time step number')
plt.ylabel('pc ')
plt.title('pc vs time step number at node 0')
plt.show()

plt.plot(time_hysteresis,model.interface.data.ew_endpoint_hysteresis[0,0:counter_hysteresis])
plt.xlabel('time step number')
plt.ylabel('ew endpoint')
plt.title('ew endpoint vs time step number at node 0')
plt.show()
