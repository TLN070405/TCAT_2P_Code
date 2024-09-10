import numpy as np
from TCAT2Phase.state_equations import state_equations_hysteresis
import matplotlib.pyplot as plt
import conftest as ct

def set_model():
    """
    Set up model and interfaces
    """
    model = ct.initialize_model(domain_size=np.array([[0.0,1.0]]),
                                dims = 1,
                                nodes = 2,
                                model_type = 'two_phase',
                                form = 'sp')

    s_phase = ct.set_solid_phase(porosity=0.4, permeability=0, gravity=0 )
    w_phase = ct.set_wet_phase(ewr=0, viscosity=0)
    n_phase = ct.set_nonwet_phase(enr=0, viscosity=0)
    ct.setup_phases(model, s_phase, w_phase, n_phase)
    vg_imb = ct.create_vg_model(s_phase, w_phase, n_phase, m=1-1/6, n=6, alpha=9e-4)
    vg_dr  = ct.create_vg_model(s_phase, w_phase, n_phase, m=1-1/9, n=9, alpha=6e-4)
    wn_interface_imb = ct.setup_interface(vg_imb)
    wn_interface_dr  = ct.setup_interface(vg_dr)

    return model,wn_interface_imb,wn_interface_dr

def setup_hysteresis(model,wn_interface_imb,wn_interface_dr):
    """
    Set up hysteresis data and model
    """

    model.hysteresis = True
    nodes = model.nodes
    drainage_new   = np.full((nodes), False)
    e_w_reversal   = np.full((nodes), 0.0003)
    e_w_endpoint   = np.full((nodes), model.solid.porosity)
    p_c_reversal   = np.full((nodes), wn_interface_imb.pc_function(e_w_reversal[0]))
    p_c_endpoint   = np.full((nodes), wn_interface_imb.pc_function(e_w_endpoint[0]))
    ew_hysteresis  = np.full((nodes,5000),  0.01)
    pc_hysteresis  = np.full((nodes,5000), wn_interface_imb.pc_function(ew_hysteresis[0][0] ) )
    hysteresis_tol = 0.

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

def ew_func_time(t):
    """
    Time dependant volume fraction to simulate hysteresis
    """
    if 0<=t<=100:
        a=(0.4-0.01)/(100-0)
        b=0.01-a*0
        ew=a*t+b
    elif 100<t<=200:
        a=(0.1-0.4)/(200-100)
        b=0.4-a*100
        ew=a*t+b
    elif 200<t<=300:
        a=(0.3-0.1)/(300-200)
        b=0.1-a*200
        ew=a*t+b
    return ew

def test_hysteresis_parkerlenhard():
    """
    Test function for Parker Lenhard hysteresis model
    """
    for n, value in enumerate(time):
        counter=model.interface.data.counter_hysteresis
        model.current_cell = 0
        ew = ew_func_time(value)
        pc = model.interface.pc_function(ew)
        switch_condition = ew-model.interface.data.ew_hysteresis[0,counter-1]
        model.interface.update_process(switch_condition)
        model.interface.update_scaling_points(ew, pc)
        model.interface.store_hysteresis_results(ew, pc)
        model.interface.data.counter_hysteresis += 1


# set the model
model,wn_interface_imb,wn_interface_dr = set_model()

# fill hysteresis data
time=np.linspace(1, 300,3000)
setup_hysteresis(model,wn_interface_imb,wn_interface_dr)

#call hysteresis test function 
test_hysteresis_parkerlenhard()

# Plot main imb and dr curves besides scanning curves.
e_w_mpc = np.linspace(0.003, 0.4, 1000)
p_c_imb = np.zeros([len(e_w_mpc)], dtype=np.double)
p_c_dr  = np.zeros([len(e_w_mpc)], dtype=np.double)

for n, value in enumerate(e_w_mpc):
    p_c_dr[n]   = wn_interface_dr.pc_function(value)
    p_c_imb[n]  = wn_interface_imb.pc_function(value)

counter_hysteresis = model.interface.data.counter_hysteresis
plt.plot(e_w_mpc,p_c_imb,label="MIC VG",linestyle="dashed")
plt.plot(e_w_mpc,p_c_dr,label="PDC VG",linestyle="dashed")
plt.plot(model.interface.data.ew_hysteresis[0,0:counter_hysteresis],
         model.interface.data.pc_hysteresis[0,0:counter_hysteresis],
         label='scanning curves')
plt.legend()
plt.xlabel("e_w")
plt.ylabel("p_c")
plt.title('Hysteresis pc-ew curves')
plt.show()

# plot ew endpoint versus number of time steps 
number_dt   = np.linspace(0, counter_hysteresis, counter_hysteresis)
plt.plot(number_dt,model.interface.data.ew_endpoint_hysteresis[0,0:counter_hysteresis])
plt.xlabel('time step number')
plt.ylabel('ew endpoint')
plt.title('ew endpoint vs time step number at node 0')
plt.show()