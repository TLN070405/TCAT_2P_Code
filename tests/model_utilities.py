import TCAT2Phase
from TCAT2Phase.state_equations import state_equations_interface

def initialize_model(domain_size,dims,nodes,model_type,form):
    """
    initialize model
    """
    domain =  TCAT2Phase.properties.domain.initialize_domain(
                                dims = dims,
                                domain_size = domain_size,
                                )

    # Discretization
    nodes = 3

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

def set_solid_phase(porosity,permeability,gravity ):
    """
    Set solid phase properties
    """
    s_phase = TCAT2Phase.properties.phase.Solid('s')
    s_phase.set_porosity(porosity)
    s_phase.set_permeability(permeability)
    s_phase.set_gravity(gravity)

    return s_phase

def set_wet_phase(ewr,viscosity ):
    """
    Set wetting phase properties
    """
    w_phase = TCAT2Phase.properties.phase.Fluid('w')
    w_phase.set_residual_volume_fraction(ewr)
    w_phase.set_viscosity(viscosity)

    return w_phase

def set_nonwet_phase(enr,viscosity ):
    """
    Set non-wetting phase properties
    """
    n_phase = TCAT2Phase.properties.phase.Fluid('n')
    n_phase.set_residual_volume_fraction(enr)
    n_phase.set_viscosity(viscosity)

    return n_phase

def setup_phases(model,s_phase,w_phase,n_phase ):
    """
    Set up the model phases.
    """
    model.solid = s_phase
    model.w_phase = w_phase
    model.n_phase = n_phase

def create_vg_model(s_phase,w_phase,n_phase,m,n,alpha ):
    """
    Create a VanGenuchten model.
    """
    vg = state_equations_interface.VanGenuchten(
        m = m,
        n = n,
        alpha = alpha,
        porosity = s_phase.porosity,
        e_wr = w_phase.residual_volume_fraction,
        e_nr = n_phase.residual_volume_fraction,
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
