"""hysteresis.py"""

def pc_hysteretic(model_imb,model_dr, ew, data_hysteresis):
    """
    Hysteretic capillary pressure
    mpc  : main primary curve (drainage or imbibition)
    ih   : index hysteresis
    """
    ih, ew_reversal, ew_endpoint, pc_reversal, pc_endpoint  = get_data_hysteresis(data_hysteresis)
    if(ih==-1):
       ew_reversal_mpc = model_imb.w_phase.volume_fraction_function(pc_reversal)
       ew_endpoint_mpc = model_imb.w_phase.volume_fraction_function(pc_endpoint)
       ew_new          = ((ew-ew_endpoint)/(ew_reversal-ew_endpoint))*(ew_reversal_mpc-ew_endpoint_mpc) + ew_endpoint_mpc
       pc_hysteretic   = model_imb.interface.pc_function(ew_new)
    elif(ih==1):
       ew_reversal_mpc = model_dr.w_phase.volume_fraction_function(pc_reversal)
       ew_endpoint_mpc = model_dr.w_phase.volume_fraction_function(pc_endpoint)
       ew_new          = ((ew-ew_endpoint)/(ew_reversal-ew_endpoint))*(ew_reversal_mpc-ew_endpoint_mpc) + ew_endpoint_mpc
       pc_hysteretic   = model_dr.interface.pc_function(ew_new)
    return pc_hysteretic

def d_pc_hysteretic(model_imb,model_dr, ew, data_hysteresis):
    """
    Derivative of hysteretic capillary pressure over the w-phase folume fraction
    """
    ih, ew_reversal, ew_endpoint, pc_reversal, pc_endpoint  = get_data_hysteresis(data_hysteresis)
    if(ih==-1):
       ew_reversal_mpc = model_imb.w_phase.volume_fraction_function(pc_reversal)
       ew_endpoint_mpc = model_imb.w_phase.volume_fraction_function(pc_endpoint)
       ew_new          = ((ew-ew_endpoint)/(ew_reversal-ew_endpoint))*(ew_reversal_mpc-ew_endpoint_mpc) + ew_endpoint_mpc
       slope           = (ew_reversal_mpc-ew_endpoint_mpc)/(ew_reversal-ew_endpoint)
       d_pc_hystereric = slope * model_imb.interface.d_pc_function(ew_new)
    elif(ih==1):
       ew_reversal_mpc = model_dr.w_phase.volume_fraction_function(pc_reversal)
       ew_endpoint_mpc = model_dr.w_phase.volume_fraction_function(pc_endpoint)
       ew_new          = ((ew-ew_endpoint)/(ew_reversal-ew_endpoint))*(ew_reversal_mpc-ew_endpoint_mpc) + ew_endpoint_mpc
       slope           = (ew_reversal_mpc-ew_endpoint_mpc)/(ew_reversal-ew_endpoint)
       d_pc_hystereric = slope * model_dr.interface.d_pc_function(ew_new)
    return d_pc_hystereric

def e_hysteretic(model_imb,model_dr, pc, data_hysteresis):
    """
    Hysteretic volume fraction of w phase  
    """
    ih, ew_reversal, ew_endpoint, pc_reversal, pc_endpoint  = get_data_hysteresis(data_hysteresis)
    if(ih == -1):
       ew_reversal_mpc = model_imb.w_phase.volume_fraction_function(pc_reversal)
       ew_endpoint_mpc = model_imb.w_phase.volume_fraction_function(pc_endpoint)
       ew              = model_imb.w_phase.volume_fraction_function(pc)
       ew_hystereric   =(((ew-ew_endpoint_mpc)/(ew_reversal_mpc-ew_endpoint_mpc))*(ew_reversal-ew_endpoint)) + ew_endpoint 
    elif(ih == 1):
       ew_reversal_mpc = model_dr.w_phase.volume_fraction_function(pc_reversal)
       ew_endpoint_mpc = model_dr.w_phase.volume_fraction_function(pc_endpoint)
       ew              = model_dr.w_phase.volume_fraction_function(pc)
       ew_hystereric   =(((ew-ew_endpoint_mpc)/(ew_reversal_mpc-ew_endpoint_mpc))*(ew_reversal-ew_endpoint)) + ew_endpoint 
    return ew_hystereric
    

def get_data_hysteresis(data_hysteresis):
    """
    Hysteretic volume fraction of w phase  
    """
    ih          = data_hysteresis[0]
    ew_reversal = data_hysteresis[1]
    ew_endpoint = data_hysteresis[2]
    pc_reversal = data_hysteresis[3]
    pc_endpoint = data_hysteresis[4]

    return ih,ew_reversal,ew_endpoint,pc_reversal,pc_endpoint