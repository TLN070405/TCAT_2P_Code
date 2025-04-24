""" 2 phases equilibrium test case """

import TCAT2Phase
from TCAT2Phase.state_equations import state_equations_interface
from TCAT2Phase.state_equations import state_equations
import solver_options
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from TCAT2Phase.io.hdf5_io import save_solutions_to_hdf5


nodes_array=[33,65,101]

for nodes in (nodes_array):
    # Domain
    domain_size = np.array([[0.0, 1.0]])
    domain = TCAT2Phase.properties.domain.initialize_domain(
                                                            dims=1,
                                                            domain_size=domain_size,
                                                            )

    # Mesh

    mesh = TCAT2Phase.mesh.grid.Mesh()
    mesh.uniform_mesh_1d(domain, nodes)

    # Boundaries
    model = TCAT2Phase.numerics.model.Model(
                                            model_type="Richards", form="pr", domain=domain, nodes=nodes, mesh=mesh
                                            )

    # Set boundary types: 0 - Dirchlet
    #                     1 - Neumann qw
    #                     2 - Neumann qn
    boundary_type = {"p_w": [[0, 0]]}

    model.set_boundary_type(boundary_type)

    # Set solid phase properties
    s_phase = TCAT2Phase.properties.phase.Solid(
        "s",
        porosity=0.369,
        permeability=2.67e-9,
        gravity= 9.81,
        dynamic_capillary_pressure_parameter=None,
    )

    # Set wetting-phase properties
    w_den_params = {"rho0": 1.0e3, "beta": 1e-8}
    w_phase = TCAT2Phase.properties.phase.Fluid(
        "w",
        viscosity=1.0e-3,
        momentum_resistance=0.0,
        residual_volume_fraction=0.0687*0.369,
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
        m=1.0 - (1.0 /37.8),
        n=37.8,
        alpha=2.28e-3,
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

    # Set initial conditions
    def pw_ic(x):
        """wetting phase pressure ic at t=0"""
        return ((2000-2500.)/(1.-0.))*x + 2500.

    initial_conditions = {"p_w": pw_ic}
    model.set_initial_conditions(initial_conditions)

    # Time
    time_interval = np.array([1,5,10])
    model.set_simulation_time(time_interval)

    # Set petsc solver options
    solver_options.solver_options(max_time_steps=100000, bdf_order=5, pc_type="sor")

    # Solve model and return model.solutions which is a dictionary where
    # solutions[time][u_name] = np.array

    pw0_bc = 2500.
    pwL_bc = 2000.
    # Set boundary values
    boundary_conditions = {"p_w": [[pw0_bc, pwL_bc]]}

    model.set_boundary_conditions(boundary_conditions)
    u, ts = TCAT2Phase.numerics.solver.setup_solver(model)
    solutions = TCAT2Phase.numerics.solver.solver_time(u, ts, model)

    # visualization
    #plt.gca().set_ylim(0.02, 0.41)  # Y-axis limits

    for time, variable in solutions.items():
        plt.plot(mesh.points, variable["p_w"], label=f"t={time}")
        plt.xlabel(r"$x\, (m)$")
        plt.ylabel(r"$p_w$")
        plt.title("equilibrium Solution")
        plt.legend()
    #plt.show()


    for time,variable in solutions.items():
        p_w = variable['p_w']
        e_w=np.zeros([nodes] )
        j_wn=np.zeros([nodes] )
        for n in range(0,nodes):
            e_w[n]= model.interface.volume_fraction_function(-p_w[n])
            j_wn[n]= -p_w[n]/ 7.20e-2
        variable["e_w"] = e_w
        variable["j_wn"] = j_wn
        plt.plot(mesh.points,e_w,label=f't={time}')
        plt.xlabel(r'$x\, (m)$')
        plt.ylabel(r'$\epsilon^w$')
        plt.title('Volume fraction vs space')
        plt.legend()
    #plt.show()

    # Export results
    filename = f"RE_test_case_1_{nodes}.h5"
    file_path =  Path(__file__).resolve().parent.parent.parent / 'outputs'/ 'Testcase1_fully_saturated_system'/'outputs'

    # Save the data to the correct file path
    metadata  = { "nodes":nodes}
    save_solutions_to_hdf5(str(file_path), filename,mesh, solutions,metadata)
