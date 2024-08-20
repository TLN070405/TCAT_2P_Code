from petsc4py import PETSc
import TCAT2Phase
from TCAT2Phase.state_equations import state_equations_interface
from TCAT2Phase.state_equations import state_equations
from TCAT2Phase.state_equations import state_equations_hysteresis


def setup_model(domain_size,dims,nodes,model_type,form):
    """
    initialize model
    """
    domain =  TCAT2Phase.properties.domain.initialize_domain(
                                dims = dims,
                                domain_size = domain_size,
                                )

    # Mesh
    mesh = TCAT2Phase.mesh.grid.Mesh()
    mesh.uniform_mesh_1d(domain,nodes)

    # Create model
    model = TCAT2Phase.numerics.model.initialize_model(
                                    model_type = model_type,
                                    form = form,
                                    domain = domain,
                                    nodes = nodes,
                                    mesh = mesh
                                        )

    return model

def setup_boundary_initial_conditions(model,
                                      boundary_type,
                                      boundary_name,
                                      boundary_value,
                                      initial_conditions):
    """
    Set boundary and initial conditions
    """
    model.set_boundary_type(boundary_type = boundary_type)
    model.set_boundary_name(boundary_name = boundary_name)
    model.set_boundary_value(boundary_value = boundary_value)
    model.set_initial_conditions(initial_conditions)

def setup_solid_phase(porosity,permeability,gravity ):
    """
    Set solid phase properties
    """
    s_phase = TCAT2Phase.properties.phase.Solid('s')
    s_phase.set_porosity(porosity)
    s_phase.set_permeability(permeability)
    s_phase.set_gravity(gravity)

    return s_phase

def setup_wet_phase(ewr,viscosity,w_den_params):
    """
    Set wetting phase properties
    """
    w_phase = TCAT2Phase.properties.phase.Fluid('w')
    w_phase.set_residual_volume_fraction(ewr)
    w_phase.set_viscosity(viscosity)
    w_phase.set_density_function(state_equations.density_exp_pressure,**w_den_params)
    w_phase.set_d_density_function(state_equations.d_density_exp_pressure,**w_den_params)

    return w_phase

def setup_nonwet_phase(enr,viscosity,n_den_params,n_en_params):
    """
    Set non-wetting phase properties
    """
    n_phase = TCAT2Phase.properties.phase.Fluid('n')
    n_phase.set_residual_volume_fraction(enr)
    n_phase.set_viscosity(viscosity)
    n_phase.set_density_function(state_equations.density_linear_pressure,**n_den_params)
    n_phase.set_d_density_function(state_equations.d_density_linear_pressure,**n_den_params)
    n_phase.set_volume_fraction_function(state_equations.volume_fraction,**n_en_params)

    return n_phase

def setup_phases(model,s_phase,w_phase,n_phase):
    """
    Set up the model phases.
    """
    model.solid = s_phase
    model.w_phase = w_phase
    model.n_phase = n_phase

def setup_relative_permeabilities(psk_model,w_phase,n_phase):
    """
    Set up the relative permeabilities.
    """
    w_phase.set_rel_perm_function(psk_model.w_relative_permability)
    n_phase.set_rel_perm_function(psk_model.n_relative_permability)


def set_vg_model(ewr,enr,porosity,m,n,alpha ):
    """
    Create the VanGenuchten model.
    """
    vg = state_equations_interface.VanGenuchten(
        m = m,
        n = n,
        alpha = alpha,
        porosity = porosity,
        e_wr = ewr,
        e_nr = enr,
        )

    return vg

def setup_interface(psk_model ):
    """
    Set up the model interface.
    """
    wn_interface = TCAT2Phase.properties.interface.Interface()
    wn_interface.set_pc_function(psk_model.capillary_pressure)
    wn_interface.set_d_pc_function(psk_model.d_capillary_pressure)
    wn_interface.set_volume_fraction_function(psk_model.volume_fraction_w)
    wn_interface.set_d_volume_fraction_function(psk_model.d_volume_fraction_w)

    return wn_interface

def setup_hysteresis_parkerlenhard(model,wn_interface_imb,
                                    wn_interface_dr,
                                    drainage_new,
                                    p_c_reversal,
                                    p_c_endpoint,
                                    e_w_reversal,
                                    e_w_endpoint,
                                    p_c_old,
                                    e_w_old,
                                    hysteresis_tol):
    """
    Set up the model hysteresis.
    """
    model.hysteresis = True
    model.interface = state_equations_hysteresis.ParkerLenhard(
        model,
        wn_interface_imb,
        wn_interface_dr,
        drainage_new,
        p_c_reversal,
        p_c_endpoint,
        e_w_reversal,
        e_w_endpoint,
        p_c_old,
        e_w_old,
        hysteresis_tol
        )


def setup_petsc_solver(bdf_order,
                       ts_max_steps,
                       ts_max_reject,
                       max_snes_failures,
                       rtol,
                       atol,
                       pc_type):
    """
    Set up the ts solver parameters.
    """
    # Numerical Parameters
    OptDB = PETSc.Options()

    # Time-Stepper Options
    OptDB.setValue('snes_fd_color', None)
    OptDB.setValue('ts_type', 'bdf')
    OptDB.setValue('ts_bdf_order', bdf_order)
    OptDB.setValue('ts_max_steps',ts_max_steps)
    OptDB.setValue('ts_max_reject',ts_max_reject)
    OptDB.setValue('ts_monitor', None)

    # Nonlinear Solver Options
    OptDB.setValue('ts_max_snes_failures',max_snes_failures)

    # Linear Solver Options
    OptDB.setValue('ts_rtol', rtol)
    OptDB.setValue('ts_atol', atol)
    # Preconditioner Options
    OptDB.setValue('pc_type', pc_type)
