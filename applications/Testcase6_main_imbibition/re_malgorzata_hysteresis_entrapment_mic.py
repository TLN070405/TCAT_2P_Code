import numpy as np
from numpy import linalg 
from petsc4py import PETSc
import TCAT2Phase
from TCAT2Phase.numerics import utils
from TCAT2Phase.state_equations import state_equations
from TCAT2Phase.state_equations import state_equations_interface
from TCAT2Phase.state_equations import state_equations_hysteresis
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
# Domain

domain_size = np.array([[0.0,1.0]])
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
                                        model_type="Richards", form="pr", domain=domain, nodes=nodes, mesh=mesh
                                        )

# Set boundary types: 0 - Dirchlet
#                     1 - Neumann qw
#                     2 - Neumann qn
boundary_type = {'p_w': [[0, 0]]}

model.set_boundary_type(boundary_type = boundary_type)

# Set solid phase properties
s_phase = TCAT2Phase.properties.phase.Solid(
    "s",
    porosity=0.4,
    permeability=1e-10,
    gravity=0.0,
    dynamic_capillary_pressure_parameter=None,
)

# Set wetting-phase properties
w_den_params = {"rho0": 1.0e3, "beta": 4.9e-10}
w_phase = TCAT2Phase.properties.phase.Fluid(
    "w",
    viscosity=1.0e-3,
    momentum_resistance=0.0,
    residual_volume_fraction=0.04,
    density_function=state_equations.density_exp_pressure,
    d_density_function=state_equations.d_density_exp_pressure,
    **w_den_params,
)

# Set nonwetting-phase properties
n_den_params = {"rho0": 1.2, "alpha": 1.189e-5}
n_phase = TCAT2Phase.properties.phase.Fluid(
    "n",
    viscosity=1.78e-5,
    momentum_resistance=0.0,
    residual_volume_fraction=0.2*0.4,
    density_function=state_equations.density_linear_pressure,
    d_density_function=state_equations.d_density_linear_pressure,
    **n_den_params,
)

# Set w-n interphase properties
vg_imb = state_equations_interface.VanGenuchten(
    m=1.0 - (1.0 /6.0),
    n=6.0,
    alpha=9e-4,
    porosity=s_phase.porosity,
    e_wr=w_phase.residual_volume_fraction,
    e_nr=n_phase.residual_volume_fraction,
)

wn_interface_imb = TCAT2Phase.properties.interface.TraditionalInterface(vg_imb)

vg_dr = state_equations_interface.VanGenuchten(
    m=1.0 - (1.0 /9.0),
    n=9.0,
    alpha=6e-4,
    porosity=s_phase.porosity,
    e_wr=w_phase.residual_volume_fraction,
    e_nr=n_phase.residual_volume_fraction,
)

wn_interface_dr = TCAT2Phase.properties.interface.TraditionalInterface(vg_dr)

# Set model phases and interfaces
TCAT2Phase.numerics.model.Model.set_model_phases(
    model, s_phase, w_phase, n_phase, wn_interface
)





# Set hysteresis data and model
model.hysteresis = True
max_reversals = 20
max_time_steps =100000
e_wr = w_phase.residual_volume_fraction
e_nr = n_phase.residual_volume_fraction
porosity=s_phase.porosity

drainage_process_new  = np.full((nodes), False)
e_w_reversal         = np.full((nodes,max_reversals), e_wr +1e-5 )
e_w_endpoint         = np.full((nodes,max_reversals), porosity- e_nr)
p_c_reversal         = np.full((nodes,max_reversals), wn_interface_imb.pc_function(e_w_reversal[0][0]) )
p_c_endpoint         = np.full((nodes,max_reversals), wn_interface_imb.pc_function(e_w_endpoint[0][0]) )
s_a_reversal         = np.full((nodes,max_reversals), (e_w_reversal[0][0]-e_wr)/(porosity-e_wr))
s_a_endpoint         = np.full((nodes,max_reversals), 1.0)
s_nte_reversal       = np.full((nodes,max_reversals), 0.0)
s_nte_endpoint       = np.full((nodes,max_reversals), e_nr/(porosity-e_wr))

path_is_pdc          = np.full((nodes),False )


hysteresis_tol = 1e-6
switch_condition ='p_c'

ParkerLenhard_parameters = {'drainage_process_new':drainage_process_new,
                            'pc_reversal':p_c_reversal,
                            'pc_endpoint':p_c_endpoint,
                            'ew_reversal':e_w_reversal,
                            'ew_endpoint':e_w_endpoint,
                            'sa_reversal':s_a_reversal,
                            'sa_endpoint':s_a_endpoint,
                            'snte_reversal':s_nte_reversal,
                            'snte_endpoint':s_nte_endpoint,
                            'path_is_pdc':path_is_pdc,
                            'switch_condition':switch_condition,
                            'hysteresis_tol':hysteresis_tol,
                            'max_reversals':max_reversals,
                            'max_time_steps':max_time_steps }
model.interface = state_equations_hysteresis.ParkerLenhard(model, wn_interface_imb, wn_interface_dr, **ParkerLenhard_parameters )



# Set initial conditions
pw_ic = np.full((nodes), -2000)

initial_conditions = {'p_w': pw_ic }

model.set_initial_conditions(initial_conditions)

# Set boundary values

def pw0_bc(t):
    """
    Boundary wetting phase pressure
    Optimized to reduce redundancy
    """
    pw_values = [-2000, 0.0, -1800, -1000, -4000]
    time_values = [0, 500, 1000, 1500, 2000, 2500, 3000]

    for i in range(1, len(time_values)):
        if time_values[i-1] < t <= time_values[i]:
            aa = (pw_values[i] - pw_values[i-1]) / (time_values[i] - time_values[i-1])
            bb = pw_values[i-1] - aa * time_values[i-1]
            p_w0 = aa * t + bb
            return p_w0

    return pw_values[-1]  # If t is out of the given time range


def pn0_bc(t): return  0.0
def pnL_bc(t): return  0.0
def pwL_bc(t): return  -2000


boundary_value = {'p_w': [[ pw0_bc,pwL_bc]],
                  'p_n': [[ pn0_bc,pnL_bc  ]]}

model.set_boundary_value(boundary_value = boundary_value)

# Time
time_interval = np.array([2000])
#time_interval = np.array([1.0,10.0,20.0,30.0,100.0,200.0,500.0,600.0,700.0])

model.set_simulation_time(time_interval)

# Numerical Parameters
OptDB = PETSc.Options()

# Time-Stepper Options
OptDB.setValue('snes_fd_color', None)
OptDB.setValue('ts_type', 'bdf')
OptDB.setValue('ts_bdf_order', 5)
OptDB.setValue('ts_max_steps',10000)
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
u,ts = TCAT2Phase.numerics.solver.setup_solver(model)
solutions = TCAT2Phase.numerics.solver.solver_time(u,ts,model)

for time,variable in solutions.items():
    plt.plot(mesh.points,variable['p_w'],label=f't={time}')
    plt.xlabel(r'$x\, (m)$')
    plt.ylabel(r'$p_w$')
    plt.title("Malgorzata Solution")
    plt.legend()
plt.show()


# Hysteresis visualization
pc_range   = np.linspace(0.0, 4000, 4000)
e_w_mpc_dr   = np.zeros([len(pc_range)], dtype=np.double)
e_w_mpc_imb  = np.zeros([len(pc_range)], dtype=np.double)

for n, value in enumerate(pc_range):
    e_w_mpc_dr[n]   = wn_interface_dr.volume_fraction_function(value)
    e_w_mpc_imb[n]  = wn_interface_imb.volume_fraction_function(value)


counter_hysteresis=model.interface.data.counter_hysteresis

plt.plot(model.interface.data.ew_hysteresis[0,1:counter_hysteresis],
         model.interface.data.pc_hysteresis[0,1:counter_hysteresis],
         label='scanning curves',marker='o')
plt.plot(e_w_mpc_imb,pc_range,label="MIC VG",linestyle="dashed")
plt.plot(e_w_mpc_dr,pc_range,label="MDC VG",linestyle="dashed")
plt.xlabel(r'$\epsilon^w$')
plt.ylabel(r'$p_c$')
plt.title('Hysteresis pc-ew curves')
plt.legend()
plt.show()

plt.plot(model.interface.data.ew_hysteresis[1,1:counter_hysteresis],
         model.interface.data.pc_hysteresis[1,1:counter_hysteresis],
         label='scanning curves',marker='o')
plt.plot(e_w_mpc_imb,pc_range,label="MIC VG",linestyle="dashed")
plt.plot(e_w_mpc_dr,pc_range,label="MDC VG",linestyle="dashed")
plt.xlabel(r'$\epsilon^w$')
plt.ylabel(r'$p_c$')
plt.title('Hysteresis pc-ew curves')
plt.legend()
plt.show()
