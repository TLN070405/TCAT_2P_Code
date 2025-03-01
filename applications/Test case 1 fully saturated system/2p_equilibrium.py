""" 2 phases equilibrium test case """

import TCAT2Phase
from TCAT2Phase.state_equations import state_equations_interface
import solver_options
import numpy as np
import matplotlib.pyplot as plt

# Domain
domain_size = np.array([[0.0,1.0]])
domain =  TCAT2Phase.properties.domain.initialize_domain(
                            dims = 1,
                            domain_size = domain_size,
                            )

# Mesh
nodes = 51
mesh = TCAT2Phase.mesh.grid.Mesh()
mesh.uniform_mesh_1d(domain,nodes)

# Boundaries
model = TCAT2Phase.numerics.model.initialize_model(
                                model_type = 'two_phase',
                                form = 'sp',
                                domain = domain,
                                nodes = nodes,
                                mesh = mesh)

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
s_phase = TCAT2Phase.properties.phase.initialize_solid_phase(porosity=0.4,
                                                            permeability=1e-10,
                                                            gravity=9.8066,
                                                            dynamic_capillary_pressure_parameter=0)

# Set wetting-phase properties
w_phase=TCAT2Phase.properties.phase.initialize_wetting_phase(viscosity=1.e-3,
                                                            momentum_resistance=0.0,
                                                            residual_volume_fraction=0.0,
                                                            w_den_params= {'rho0':1.e3,
                                                                           'beta':4.9e-10},
                                                            density_function_type='exp')

# Set nonwetting-phase properties
n_phase=TCAT2Phase.properties.phase.initialize_nonwetting_phase(viscosity=1.78e-5,
                                                                momentum_resistance=0.0,
                                                                residual_volume_fraction=0.0,
                                                                n_den_params={'rho0':1.2,
                                                                              'alpha':1.189e-5},
                                                                density_function_type='linear')

# Set w-n interphase properties
vg = state_equations_interface.VanGenuchten(m = 0.5990376904570971,
                                            n = 2.494,
                                            alpha = 1.e-3,
                                            porosity = s_phase.porosity,
                                            e_wr = w_phase.residual_volume_fraction,
                                            e_nr = n_phase.residual_volume_fraction)

wn_interface = TCAT2Phase.properties.interface.initialize_traditional_interface(vg)

# Set model phases and interfaces
TCAT2Phase.numerics.model.set_model_phases(model,s_phase,w_phase,n_phase,wn_interface)

# Set initial conditions
ew_ic= np.full((nodes), 0.10)
pw_ic= np.full((nodes), -model.interface.pc_function(0.10))
ew_ic[0] =  0.12
pw_ic[0] =  -model.interface.pc_function(0.12)
ew_ic[nodes-1] = 0.12
pw_ic[nodes-1] = -model.interface.pc_function(0.12)

initial_conditions = {'e_w': ew_ic,
                      'p_w': pw_ic }
model.set_initial_conditions(initial_conditions)

# Time
time_interval = np.array([1.])
model.set_simulation_time(time_interval)

# Set petsc solver options
solver_options.solver_options(max_time_steps=100000,bdf_order=5,pc_type='sor')

# Solve model and return model.solutions which is a dictionary where
# solutions[time][u_name] = np.array

for i in range (0,3):
    ew_values=[0.12, 0.13, 0.14]
    def ew0_bc(t): return ew_values[i]
    def pw0_bc(t): return -model.interface.pc_function(ew_values[i])
    def ewL_bc(t): return ew_values[i]
    def pwL_bc(t): return -model.interface.pc_function(ew_values[i])

    # Set boundary values
    boundary_value = {'e_w': [[ ew0_bc,ewL_bc ]],
                      'p_w': [[ pw0_bc,pwL_bc ]]}

    model.set_boundary_value(boundary_value = boundary_value)
    u,ts = TCAT2Phase.numerics.solver.setup_solver(model)
    solutions = TCAT2Phase.numerics.solver.solver_error(u,ts,model)

# visualization
plt.gca().set_ylim(0.02, 0.41)  # Y-axis limits

for time,variable in solutions.items():
    plt.plot(mesh.points,variable['e_w'],label=f't={time}')
    plt.xlabel(r'$x\, (m)$')
    plt.ylabel(r'$\epsilon^w$')
    plt.title("equilibrium Solution")
    plt.legend()
plt.show()

for time,variable in solutions.items():
    #print(time,variable)
    plt.plot(mesh.points,variable['p_w'],label=f't={time}')
    plt.xlabel(r'$x\, (m)$')
    plt.ylabel(r'$p_w$')
    plt.title('equilibrium Solution')
    plt.legend()
plt.show()
