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
import solver_options
from pathlib import Path
from TCAT2Phase.io.hdf5_io import save_solutions_to_hdf5

nodes_array=[65,101]

for nodes in (nodes_array):
    # Domain

    domain_size = np.array([[0.0,1.0]])
    domain = TCAT2Phase.properties.domain.initialize_domain(
                                                            dims=1,
                                                            domain_size=domain_size,
                                                            )
    mesh = TCAT2Phase.mesh.grid.Mesh()
    mesh.uniform_mesh_1d(domain, nodes)

    # Boundaries
    model = TCAT2Phase.numerics.model.Model(
                                            model_type="Richards", form="pr", domain=domain, nodes=nodes, mesh=mesh
                                            )

    # Set boundary types: 0 - Dirchlet
    #                     1 - Neumann (but not supported yet)
    boundary_type = {'p_w': [[0, 0]]}

    model.set_boundary_type(boundary_type)

    # Set solid phase properties
    s_phase = TCAT2Phase.properties.phase.Solid(
        "s",
        porosity=0.4,
        permeability=1e-10,
        gravity= 0.0,
        dynamic_capillary_pressure_parameter=None,
    )

    # Set wetting-phase properties
    w_den_params = {"rho0": 1.0e3, "beta": 4.9e-10}
    w_phase = TCAT2Phase.properties.phase.Fluid(
        "w",
        viscosity=1.e-3,
        momentum_resistance=0.0,
        residual_volume_fraction=0.1*0.4,
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

    # Set w-n interfaces properties
    vg_imb = state_equations_interface.VanGenuchten(
        m=1.0 - (1.0 /6.),
        n=6.,
        alpha= 9.e-4,
        porosity=s_phase.porosity,
        e_wr=w_phase.residual_volume_fraction,
        e_nr=n_phase.residual_volume_fraction,
    )

    wn_interface_imb = TCAT2Phase.properties.interface.TraditionalInterface(vg_imb)

    vg_dr = state_equations_interface.VanGenuchten(
        m= 1.0 - (1.0 /9.0),
        n= 9.0,
        alpha= 6.e-4,
        porosity=s_phase.porosity,
        e_wr=w_phase.residual_volume_fraction,
        e_nr= 0.,
    )

    wn_interface_dr = TCAT2Phase.properties.interface.TraditionalInterface(vg_dr)

    #set relative permeabilities
    w_phase.set_relative_permeability_function(vg_dr.w_relative_permeability)
    n_phase.set_relative_permeability_function(vg_dr.n_relative_permeability)

    # Set hysteresis data and hysteretic interface
    model.hysteresis = True
    max_reversals = 20
    max_time_steps =100000
    e_wr = w_phase.residual_volume_fraction

    drainage_process_new  = np.full((nodes), True)
    e_w_reversal         = np.full((nodes,max_reversals), s_phase.porosity)
    e_w_endpoint         = np.full((nodes,max_reversals), w_phase.residual_volume_fraction +1e-5)
    s_a_reversal         = np.full((nodes,max_reversals), (e_w_reversal[0][0]-e_wr)/(s_phase.porosity-e_wr))
    s_a_endpoint         = np.full((nodes,max_reversals), (e_w_endpoint[0][0]-e_wr)/(s_phase.porosity-e_wr))
    p_c_reversal         = np.full((nodes,max_reversals), wn_interface_dr.pc_function(e_w_reversal[0][0]) )
    p_c_endpoint         = np.full((nodes,max_reversals), wn_interface_dr.pc_function(e_w_endpoint[0][0]) )
    s_nte_reversal       = np.full((nodes,max_reversals), 0.0)
    s_nte_endpoint       = np.full((nodes,max_reversals), 0.0)
    path_is_pdc          = np.full((nodes),True )

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


    wn_interface = state_equations_hysteresis.ParkerLenhard(model, wn_interface_imb, wn_interface_dr, **ParkerLenhard_parameters )

    # Set model phases and interfaces
    TCAT2Phase.numerics.model.Model.set_model_phases(model, s_phase, w_phase, n_phase, wn_interface)

    # Set initial conditions
    def pw_ic(x):
        """wetting phase pressure ic at t=0"""
        return -wn_interface_dr.pc_function(0.35)

    initial_conditions = {"p_w": pw_ic}
    model.set_initial_conditions(initial_conditions)

    # Set boundary values
    def pw0_bc(t):
        """
        Boundary wetting phase pressure using linear interpolation
        """
        # Boundary pressure values corresponding to the time intervals
        pw_values = [-wn_interface_dr.pc_function(0.35), -1500, 0.0, -1300, -900, -1800, 0.0]
        time_values = [0, 500, 1000, 1500, 2000, 2500, 3000]

        # Loop through each time interval to find the appropriate one
        for i in range(1, len(time_values)):
            if time_values[i-1] < t <= time_values[i]:
                # Linear interpolation
                aa = (pw_values[i] - pw_values[i-1]) / (time_values[i] - time_values[i-1])
                bb = pw_values[i-1] - aa * time_values[i-1]
                p_w0 = aa * t + bb
                return p_w0
        # If t is outside the time range, return None or a default value (optional)
        return pw_values[-1]

    def pwL_bc(t): return  -wn_interface_dr.pc_function(0.35)


    boundary_conditions = {"p_w": [[pw0_bc, pwL_bc]]}
    model.set_boundary_conditions(boundary_conditions)

    # Time
    time_interval = np.array([3000])

    model.set_simulation_time(time_interval)

    # Numerical Parameters
    OptDB = PETSc.Options()

    # Numerical Parameters
    solver_options.solver_options(max_time_steps=100000, bdf_order=5, pc_type="sor")

    # Solve model and return model.solutions
    # which is a dictionary where
    # solutions[time][u_name] = np.array
    u,ts = TCAT2Phase.numerics.solver.setup_solver(model)
    solutions = TCAT2Phase.numerics.solver.solver_time(u,ts,model)

    # Hysteresis visualization
    pc_range   = np.linspace(0.0, 4000, 4000)
    e_w_mpc_dr   = np.zeros([len(pc_range)], dtype=np.double)
    e_w_mpc_imb  = np.zeros([len(pc_range)], dtype=np.double)

    for n, value in enumerate(pc_range):
        e_w_mpc_dr[n]   = wn_interface_dr.volume_fraction_function(value)
        e_w_mpc_imb[n]  = wn_interface_imb.volume_fraction_function(value)


    for time,variable in solutions.items():
        variable['e_w_hysteresis'] = model.interface.data.ew_hysteresis
        variable['p_c_hysteresis'] = model.interface.data.pc_hysteresis
        variable['e_w_mpc_dr']     = e_w_mpc_dr
        variable['e_w_mpc_imb']    = e_w_mpc_imb
        variable['pc_range']       = pc_range
        plt.plot(mesh.points,variable['p_w'],label=f't={time}')
        plt.xlabel("x (m)")
        plt.ylabel("p_w")
        plt.title("Malgorzata Solution")
        plt.legend()
    #plt.show()


    counter_hysteresis=model.interface.data.counter_hysteresis
    for time,variable in solutions.items():
        plt.plot(variable['e_w_hysteresis'][0,1:counter_hysteresis],
            variable['p_c_hysteresis'] [0,1:counter_hysteresis],
            label='scanning curves',marker='o')
    plt.plot(e_w_mpc_imb,pc_range,label="MIC VG",linestyle="dashed")
    plt.plot(e_w_mpc_dr,pc_range,label="MDC VG",linestyle="dashed")
    plt.xlabel(r'$\epsilon^w$')
    plt.ylabel(r'$p_c$')
    plt.title('Hysteresis pc-ew curves')
    plt.legend()
    #plt.show()

    plt.plot(model.interface.data.ew_hysteresis[0,1:counter_hysteresis],
            model.interface.data.pc_hysteresis[0,1:counter_hysteresis],
            label='scanning curves',marker='o')
    plt.plot(e_w_mpc_imb,pc_range,label="MIC VG",linestyle="dashed")
    plt.plot(e_w_mpc_dr,pc_range,label="MDC VG",linestyle="dashed")
    plt.xlabel(r'$\epsilon^w$')
    plt.ylabel(r'$p_c$')
    plt.title('Hysteresis pc-ew curves')
    plt.legend()
    #plt.show()

    plt.plot(model.interface.data.ew_hysteresis[1,1:counter_hysteresis],
            model.interface.data.pc_hysteresis[1,1:counter_hysteresis],
            label='scanning curves',marker='o')
    plt.plot(e_w_mpc_imb,pc_range,label="MIC VG",linestyle="dashed")
    plt.plot(e_w_mpc_dr,pc_range,label="MDC VG",linestyle="dashed")
    plt.xlabel(r'$\epsilon^w$')
    plt.ylabel(r'$p_c$')
    plt.title('Hysteresis pc-ew curves')
    plt.legend()
    #plt.show()

    # Export results
    filename = f"RE_test_case_8_{nodes}.h5"
    file_path = file_path =  Path(__file__).resolve().parent.parent.parent / 'outputs'/ 'Testcase8_complex_displacement_patterns'/'outputs'

    # Save the data to the correct file path
    metadata = {"counter":counter_hysteresis, "nodes":nodes}
    save_solutions_to_hdf5(str(file_path), filename,mesh, solutions,metadata)
