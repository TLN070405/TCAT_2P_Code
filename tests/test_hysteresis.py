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
model_imb = TCAT2Phase.numerics.model.initialize_model(
                                model_type = 'two_phase',
                                form = 'sp',
                                domain = domain,
                                nodes = nodes,
                                mesh = mesh
                                    )


s_phase = TCAT2Phase.properties.phase.Solid('s')
s_phase.set_porosity(0.4)

# Set wetting-phase properties
w_phase_imb = TCAT2Phase.properties.phase.Fluid('w')
w_phase_imb.set_residual_volume_fraction(0.0)

# Set non-wetting phase properties
n_phase = TCAT2Phase.properties.phase.Fluid('n')
n_phase.set_residual_volume_fraction(0.0)


# Set wn-interface properties
wn_interface_imb = TCAT2Phase.properties.interface.Interface()


n = 6.0
m = 1.0 - (1.0 / n)
alpha = 9e-4

vg_imb = state_equations_interface.VanGenuchten(
    m = m,
    n = n,
    alpha = alpha,
    porosity = s_phase.porosity,
    e_wr = w_phase_imb.residual_volume_fraction,
    e_nr = n_phase.residual_volume_fraction,
    )

w_phase_imb.set_volume_fraction_function(vg_imb.volume_fraction_w)



model_imb.solid = s_phase
model_imb.w_phase = w_phase_imb
model_imb.n_phase = n_phase
model_imb.interface = wn_interface_imb

model_imb.interface.set_pc_function(vg_imb.capillary_pressure)
model_imb.interface.set_pc_hysteretic_function(state_equations_hysteresis.pc_hysteretic)

 
# Create drainage model

model_dr = TCAT2Phase.numerics.model.initialize_model(
                                model_type = 'two_phase',
                                form = 'pp',
                                domain = domain,
                                nodes = nodes,
                                mesh = mesh
                                    )

w_phase_dr = TCAT2Phase.properties.phase.Fluid('w')
w_phase_dr.set_residual_volume_fraction(0.0)
# Set wn-interface properties
wn_interface_dr = TCAT2Phase.properties.interface.Interface()


n = 9.0
m = 1.0 - (1.0 / n)
alpha = 6e-4

vg_dr = state_equations_interface.VanGenuchten(
    m = m,
    n = n,
    alpha = alpha,
    porosity = s_phase.porosity,
    e_wr = w_phase_dr.residual_volume_fraction,
    e_nr = n_phase.residual_volume_fraction,
    )

w_phase_dr.set_volume_fraction_function(vg_dr.volume_fraction_w)


model_dr.solid = s_phase
model_dr.w_phase = w_phase_dr
model_dr.n_phase = n_phase
model_dr.interface = wn_interface_dr

model_dr.interface.set_pc_function(vg_dr.capillary_pressure)

# Set hysteresis data arrays

drainage_old= np.zeros([nodes] )
drainage_new= np.zeros([nodes] )
p_c_old= np.zeros([nodes] )
e_w_old= np.zeros([nodes] )
e_w_reversal= np.zeros([nodes] )
e_w_endpoint= np.zeros([nodes] )
p_c_reversal= np.zeros([nodes] )
p_c_endpoint= np.zeros([nodes] )
p_c_endpoint_old= np.zeros([nodes] )


drainage_new[0]     = False
drainage_old[0]     = False
e_w_reversal[0]     = 0.00003
e_w_endpoint[0]     = model_imb.solid.porosity
p_c_reversal[0]     = model_imb.interface.pc_function(e_w_reversal[0])
p_c_endpoint[0]     = model_imb.interface.pc_function(e_w_endpoint[0])
p_c_endpoint_old[0] = p_c_endpoint[0]
e_w_old[0]          = 0.01
p_c_old[0]          = model_imb.interface.pc_function(e_w_old[0] )


data_hysteresis = {'drainage_old'   : drainage_old,
                   'drainage_new'   : drainage_new,
                   'p_c_old'         : p_c_old,
                   'e_w_old'         : e_w_old, 
                   'e_w_reversal'    : e_w_reversal, 
                   'e_w_endpoint'    : e_w_endpoint, 
                   'p_c_reversal'    : p_c_reversal,
                   'p_c_endpoint'    : p_c_endpoint , 
                   'p_c_endpoint_old': p_c_endpoint_old }

model_imb.set_data_hysteresis(data_hysteresis)



def  ew_func_time( t):
    if(t>0 and t<=100): 
      a=(s_phase.porosity-0.01)/(100-0)
      b=0.01-a*0
      ew=a*t+b
    elif(t>100 and t<=200): 
      a=(0.1-s_phase.porosity)/(200-100)
      b=0.4-a*100
      ew=a*t+b
    elif(t>200 and t<=300):  
      a=(0.3-0.1)/(300-200)
      b=0.1-a*200
      ew=a*t+b
    elif(t>300 and t<=400): 
      a=(0.2-0.3)/(400-300)
      b=0.3-a*300
      ew=a*t+b
    elif(t>400 and t<=500): 
      a=(0.3-0.2)/(500-400)
      b=0.2-a*400
      ew=a*t+b
    return ew
 
 

time    = np.linspace(1, 500,1000 )
e_w = np.zeros([len(time)], dtype=np.double) 
p_c = np.zeros([len(time)], dtype=np.double) 

e_w[0] = 0.01
p_c[0] = model_imb.interface.pc_function(e_w[0] )

 
for i in range(1,len(time)):
        e_w[i]=ew_func_time(time[i])
        if(e_w[i]-e_w[i-1]>=1e-9): 
          drainage_old[0] = drainage_new[0]
          drainage_new[0]= False
        else:
          drainage_old[0] = drainage_new[0]
          drainage_new[0]= True
        if(drainage_new[0] != drainage_old[0]):
          p_c_endpoint[0]= p_c_reversal[0]
          p_c_reversal[0]= model_imb.interface.pc_hysteretic_function(model_imb,model_dr,e_w_old[0],drainage_old[0],e_w_reversal[0],e_w_endpoint[0],p_c_reversal[0], p_c_endpoint_old[0])
          p_c_endpoint_old[0]= p_c_endpoint[0]
          e_w_endpoint[0]=e_w_reversal[0]
          e_w_reversal[0]= e_w_old[0] 

        p_c[i]=model_imb.interface.pc_hysteretic_function(model_imb,model_dr,e_w[i],drainage_old[0],e_w_reversal[0],e_w_endpoint[0],p_c_reversal[0], p_c_endpoint[0]) 
        e_w_old[0] =e_w[i]


e_w_mpc = np.linspace(0.003, s_phase.porosity, 1000)
p_c_imb = np.zeros([len(e_w_mpc)], dtype=np.double) 
p_c_dr  = np.zeros([len(e_w_mpc)], dtype=np.double) 
for i in range(len(e_w_mpc)):
    p_c_dr[i] =model_dr.interface.pc_function(e_w_mpc[i])
    p_c_imb[i]=model_imb.interface.pc_function(e_w_mpc[i]) 

plt.plot(e_w_mpc,p_c_imb,label="MIC VG",linestyle="dashed")
plt.plot(e_w_mpc,p_c_dr,label="PDC VG",linestyle="dashed")
plt.plot(e_w,p_c,color='k',label="Scanning curves")
plt.legend()
plt.xlabel("e_w")
plt.ylabel("p_c") 
plt.show()
    