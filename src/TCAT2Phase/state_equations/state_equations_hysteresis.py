"""hysteresis.py"""

def pc_hysteretic(model_imb,model_dr, ew, drainage, ew_reversal, ew_endpoint, pc_reversal, pc_endpoint):
    """
    Hysteretic capillary pressure
    mpc  : main primary curve (drainage or imbibition)
    """
    if(drainage==True):
       model=model_dr
    else:
       model=model_imb

    ew_reversal_mpc = model.w_phase.volume_fraction_function(pc_reversal)
    ew_endpoint_mpc = model.w_phase.volume_fraction_function(pc_endpoint)
    ew_new          = ((ew-ew_endpoint)/(ew_reversal-ew_endpoint))*(ew_reversal_mpc-ew_endpoint_mpc) + ew_endpoint_mpc
    pc_hysteretic   = model.interface.pc_function(ew_new)   

    return pc_hysteretic

def d_pc_hysteretic(model_imb,model_dr, ew, drainage, ew_reversal, ew_endpoint, pc_reversal, pc_endpoint):
    """
    Derivative of hysteretic capillary pressure over the w-phase folume fraction
    """
    if(drainage==True):
       model=model_dr
    else:
       model=model_imb

    ew_reversal_mpc = model.w_phase.volume_fraction_function(pc_reversal)
    ew_endpoint_mpc = model.w_phase.volume_fraction_function(pc_endpoint)
    ew_new          = ((ew-ew_endpoint)/(ew_reversal-ew_endpoint))*(ew_reversal_mpc-ew_endpoint_mpc) + ew_endpoint_mpc
    slope           = (ew_reversal_mpc-ew_endpoint_mpc)/(ew_reversal-ew_endpoint)
    d_pc_hystereric = slope * model.interface.d_pc_function(ew_new)

    return d_pc_hystereric

def e_hysteretic(model_imb,model_dr, pc, drainage, ew_reversal, ew_endpoint, pc_reversal, pc_endpoint):
    """
    Hysteretic volume fraction of w phase  
    """
    if(drainage==True):
       model=model_dr
    else:
       model=model_imb

    ew_reversal_mpc = model.w_phase.volume_fraction_function(pc_reversal)
    ew_endpoint_mpc = model.w_phase.volume_fraction_function(pc_endpoint)
    ew              = model.w_phase.volume_fraction_function(pc)
    ew_hystereric   =(((ew-ew_endpoint_mpc)/(ew_reversal_mpc-ew_endpoint_mpc))*(ew_reversal-ew_endpoint)) + ew_endpoint 

    return ew_hystereric
    

def get_data_hysteresis(data_hysteresis):
    """
    Hysteretic volume fraction of w phase  
    """
    drainage          = data_hysteresis[0]
    ew_reversal = data_hysteresis[1]
    ew_endpoint = data_hysteresis[2]
    pc_reversal = data_hysteresis[3]
    pc_endpoint = data_hysteresis[4]

    return drainage,ew_reversal,ew_endpoint,pc_reversal,pc_endpoint