import numpy as np
import TCAT2Phase
from TCAT2Phase.numerics import utils
from TCAT2Phase.state_equations import state_equations
from TCAT2Phase.state_equations import state_equations_interface
from TCAT2Phase.state_equations import state_equations_hysteresis
import matplotlib.pyplot as plt
# Set solid phase properties

domain_size = np.array([[0.0,1.0]])
domain =  TCAT2Phase.properties.domain.initialize_domain(
                            dims = 1,
                            domain_size = domain_size,
                            )

# Discretization
nodes = 3

# Mesh
mesh = TCAT2Phase.mesh.grid.Mesh()
mesh.uniform_mesh_1d(domain,nodes)

# Create iumbibition model
model_test = TCAT2Phase.numerics.model.initialize_model(
                                model_type = 'two_phase',
                                form = 'pp',
                                domain = domain,
                                nodes = nodes,
                                mesh = mesh
                                    )


s_phase = TCAT2Phase.properties.phase.Solid('s')
s_phase.set_porosity(0.4)

# Set wetting-phase properties
w_phase = TCAT2Phase.properties.phase.Fluid('w')
w_phase.set_residual_volume_fraction(0.0)

# Set non-wetting phase properties
n_phase = TCAT2Phase.properties.phase.Fluid('n')
n_phase.set_residual_volume_fraction(0.0)

# Set up the model phases.
model_test.solid = s_phase
model_test.w_phase = w_phase
model_test.n_phase = n_phase

# Set wn-interface properties imbibition
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
wn_interface_imb.set_volume_fraction_function(vg_imb.volume_fraction_w)

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
wn_interface_dr.set_volume_fraction_function(vg_dr.volume_fraction_w)

model_test.interface = state_equations_hysteresis.Hysteresis(model_test,wn_interface_imb,wn_interface_dr)

 
 

time    = np.linspace(0, 2000,4000 )


model_test.hysteresis = True

drainage_new         = np.full((1), True)
drainage_old         = np.full((1), True)

e_w_reversal         = np.full((1), 0.4) 
e_w_endpoint         = np.full((1), 0.003)
p_c_reversal         = np.full((1), wn_interface_dr.pc_function(e_w_reversal[0]) )
p_c_endpoint         = np.full((1), wn_interface_dr.pc_function(e_w_endpoint[0]))

e_w_reversal_old   =  np.full((1), 0.4) 
e_w_endpoint_old   = np.full((1), 0.003)
p_c_reversal_old   = np.full((1), wn_interface_dr.pc_function(e_w_reversal[0]) )
p_c_endpoint_old   = np.full((1), wn_interface_dr.pc_function(e_w_endpoint[0]))

e_w_old              = np.full((1), 0.397) 
p_c_old              = np.full((1), 346.73 ) 

e_w_hysteresis       = np.full(([len(time)]), 0.397) 
p_c_hysteresis       = np.full(([len(time)]), 346.73 )

hysteresis_tol    = 1e-6


data_hysteresis = {'drainage_old'         : drainage_old,
                   'drainage_new'         : drainage_new,
                   'p_c_old'              : p_c_old,
                   'e_w_old'              : e_w_old, 
                   'e_w_reversal'         : e_w_reversal, 
                   'e_w_endpoint'         : e_w_endpoint, 
                   'p_c_reversal'         : p_c_reversal,
                   'p_c_endpoint'         : p_c_endpoint , 
                   'p_c_endpoint_old'     : p_c_endpoint_old,
                   'p_c_reversal_old'     : p_c_reversal_old,
                   'e_w_reversal_old'     : e_w_reversal_old,
                   'e_w_endpoint_old'     : e_w_endpoint_old,
                   'e_w_hysteresis'       : e_w_hysteresis,
                   'p_c_hysteresis'       : p_c_hysteresis,
                   'state'                : "old",
                   'hysteresis_tol'       : hysteresis_tol}

model_test.set_data_hysteresis(data_hysteresis)



def pw0_bc(t): 
    Pw_t0 = -346.73
    Pw_t1 = -7000
    Pw_t2 = -2000
    Pw_t3 = -4000
    Pw_t4 = -3000
    if(t>=0 and t<=500):
      aa=(Pw_t1-Pw_t0)/(500-0)
      bb=Pw_t0-aa*0
      P_w0=aa*t+bb
    elif(t>500 and t<=1000): 
      aa=(Pw_t2-Pw_t1)/(1000-500)
      bb=Pw_t1-aa*500
      P_w0=aa*t+bb
    elif(t>1000 and t<=1500):  
      aa=(Pw_t3-Pw_t2)/(1500-1000)
      bb=Pw_t2-aa*1000
      P_w0=aa*t+bb
    elif(t>1500 and t<=2000):  
      aa=(Pw_t4-Pw_t3)/(2000-1500)
      bb=Pw_t3-aa*1500
      P_w0=aa*t+bb  
    return P_w0
 
 



drainage_old    = model_test.data_hysteresis['drainage_old']
drainage_new    = model_test.data_hysteresis['drainage_new']
pc_reversal     = model_test.data_hysteresis['p_c_reversal']
pc_endpoint     = model_test.data_hysteresis['p_c_endpoint' ]
ew_reversal     = model_test.data_hysteresis['e_w_reversal']
ew_endpoint     = model_test.data_hysteresis['e_w_endpoint']
pc_reversal_old = model_test.data_hysteresis['p_c_reversal_old']
pc_endpoint_old = model_test.data_hysteresis['p_c_endpoint_old']
ew_reversal_old = model_test.data_hysteresis['e_w_reversal_old']
ew_endpoint_old = model_test.data_hysteresis['e_w_endpoint_old']
pc_old          = model_test.data_hysteresis['p_c_old']
ew_old          = model_test.data_hysteresis['e_w_old']

ew_hysteresis   = model_test.data_hysteresis['e_w_hysteresis']
pc_hysteresis   = model_test.data_hysteresis['p_c_hysteresis']
hysteresis_tol  = model_test.data_hysteresis['hysteresis_tol']


for i in range(1,len(time)):
            solution_now = pw0_bc(time[i])
            model_test.data_hysteresis['state'] = "new"
            model_test.current_cell = 0
            ew_old[0]  = model_test.interface.volume_fraction_function(-solution_now)
            if(pc_old[0]-(-solution_now)>hysteresis_tol):
                drainage_old[0] = drainage_new[0]
                drainage_new[0] = False
            else:      
                drainage_old[0] = drainage_new[0]
                drainage_new[0] = True
            #  
            pc_old[0]        = -solution_now
            #  store hysteresis ew and pc just for visualization
            ew_hysteresis[i] = ew_old[0]
            pc_hysteresis[i] = pc_old[0]
            # update reversal and endpoints (ew and pc)
            if(drainage_new[0] != drainage_old[0]): 
                model_test.data_hysteresis['state'] = "old"
                model_test.current_cell = 0
                
                pc_reversal_old[0] = pc_reversal[0]
                pc_endpoint_old[0] = pc_endpoint[0]
                ew_endpoint_old[0] = ew_endpoint[0]
                ew_reversal_old[0] = ew_reversal[0]

                ew_endpoint[0]    = ew_reversal[0]
                ew_reversal[0]    = model_test.interface.volume_fraction_function(-solution_now) 
                pc_endpoint[0]    = pc_reversal[0]
                pc_reversal[0]    = -solution_now




# Plot results.
e_w_mpc = np.linspace(0.05, s_phase.porosity, 1000)
p_c_imb = np.zeros([len(e_w_mpc)], dtype=np.double) 
p_c_dr  = np.zeros([len(e_w_mpc)], dtype=np.double) 

for i in range(len(e_w_mpc)):
    p_c_dr[i] = wn_interface_dr.pc_function(e_w_mpc[i])
    p_c_imb[i]= wn_interface_imb.pc_function(e_w_mpc[i]) 

plt.plot(e_w_mpc,p_c_imb,label="MIC VG",linestyle="dashed")
plt.plot(e_w_mpc,p_c_dr,label="PDC VG",linestyle="dashed")
plt.plot(ew_hysteresis,pc_hysteresis,color='k',label="Scanning curves")
plt.legend()
plt.xlabel("e_w")
plt.ylabel("p_c") 
plt.show()
    

 