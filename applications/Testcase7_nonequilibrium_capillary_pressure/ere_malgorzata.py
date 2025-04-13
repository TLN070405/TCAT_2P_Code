import numpy as np
import TCAT2Phase
from TCAT2Phase.state_equations import state_equations
from TCAT2Phase.state_equations import state_equations_interface
import matplotlib.pyplot as plt
import solver_options
from pathlib import Path
from TCAT2Phase.io.hdf5_io import save_solutions_to_hdf5


nodes_array=[33,65,101]
tau_array=[0,1e7,1e8]
for nodes in (nodes_array):
    for tau in (tau_array):
        # Domain
        domain_size = np.array([[0.0,1.0]])
        domain =  TCAT2Phase.properties.domain.initialize_domain(
                                    dims = 1,
                                    domain_size = domain_size,
                                    )

        # Discretization
        #nodes = 33

        # Mesh
        mesh = TCAT2Phase.mesh.grid.Mesh()
        mesh.uniform_mesh_1d(domain,nodes)

        # Model initialization
        model = TCAT2Phase.numerics.model.Model(model_type="extended_Richards", form="sp", domain=domain, nodes=nodes, mesh=mesh)

        # Set boundary types: 0 - Dirchlet
        #                     1 - Neumann (but not supported yet)
        boundary_type = {'e_w': [[0, 0]],
                        'p_w': [[0, 0]]}

        model.set_boundary_type(boundary_type)

        # Set solid phase properties
        s_phase = TCAT2Phase.properties.phase.Solid(
            "s",
            porosity=0.4,
            permeability=1e-10 ,
            gravity = -9.8066,
            dynamic_capillary_pressure_parameter = tau,
        )

        # Set wetting-phase properties
        w_den_params = {"rho0": 1.e3, "beta": 4.9e-10}
        w_phase = TCAT2Phase.properties.phase.Fluid(
            "w",
            viscosity=1.0e-3,
            momentum_resistance=0.0,
            residual_volume_fraction= 0.,
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
            residual_volume_fraction=0.0,
            density_function=state_equations.density_linear_pressure,
            d_density_function=state_equations.d_density_linear_pressure,
            **n_den_params,
        )

        # Set w-n interphase properties
        vg = state_equations_interface.VanGenuchten(
            m = 0.5990376904570971,
            n = 2.494,
            alpha = 1.e-3,
            porosity=s_phase.porosity,
            e_wr=w_phase.residual_volume_fraction,
            e_nr=n_phase.residual_volume_fraction,
        )

        wn_interface = TCAT2Phase.properties.interface.TraditionalInterface(vg)

        #set relative permeabilities
        w_phase.set_relative_permeability_function(vg.w_relative_permeability)
        n_phase.set_relative_permeability_function(vg.n_relative_permeability)
        
        # Set model phases and interfaces
        TCAT2Phase.numerics.model.Model.set_model_phases(
            model, s_phase, w_phase, n_phase, wn_interface
        )

        # Set boundary values
        ew0_bc = 0.12
        ewL_bc = 0.008
        pw0_bc = -2113.152788450121
        pwL_bc = -13706.965861446095

        boundary_conditions = {'e_w': [[ ew0_bc,ewL_bc ]], 'p_w': [[ pw0_bc,pwL_bc  ]]}

        model.set_boundary_conditions(boundary_conditions)

        # Set initial conditions
        def ew_ic(x):
            """volume fraction ic at t=0""" 
            return 0.008

        def pw_ic(x):
            """wetting phase pressure ic at t=0"""
            return -13706.965861446095

        initial_conditions = {'e_w': ew_ic, 'p_w': pw_ic }

        model.set_initial_conditions(initial_conditions)


        # Time
        time_interval = np.array([20000.])
        model.set_simulation_time(time_interval)

        # Set petsc solver options
        solver_options.solver_options(max_time_steps=100000, bdf_order=5, pc_type="sor")

        # Solve model and return model.solutions
        # which is a dictionary where
        # solutions[time][u_name] = np.array
        u, ts = TCAT2Phase.numerics.solver.setup_solver(model)
        solutions = TCAT2Phase.numerics.solver.solver_time(u,ts,model)

        # Export results
        filename = f"ERE_test_case_7_{nodes}_{tau}.h5"
        file_path = Path(__file__).resolve().parent/ 'outputs'

        # Save the data to the correct file path
        metadata = metadata = {"tau": tau, "nodes":nodes}
        save_solutions_to_hdf5(str(file_path), filename,mesh, solutions,metadata)
