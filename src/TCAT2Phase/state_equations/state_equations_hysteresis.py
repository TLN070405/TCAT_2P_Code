"""hysteresis.py"""
class Hysteresis:
    """
    Hysteresis Class to hold functions and parameters
    """
    def __init__(self,imb_pc,dr_pc):
        """
        Initialize the imbibition and drainage interfaces
        """
        self.imbibition    = imb_pc
        self.drainage      = dr_pc

    def pc_function(self,ew, drain, ew_reversal, ew_endpoint, pc_reversal, pc_endpoint):
        """
        Hysteretic capillary pressure
        mpc  : main primary curve (drainage or imbibition)
        """
        if(drain==True):
           interface = self.drainage 
        else:
           interface = self.imbibition

        ew_reversal_mpc =  interface.volume_fraction_function(pc_reversal)
        ew_endpoint_mpc =  interface.volume_fraction_function(pc_endpoint)
        ew_new          = ((ew-ew_endpoint)/(ew_reversal-ew_endpoint))*(ew_reversal_mpc-ew_endpoint_mpc) + ew_endpoint_mpc
        pc_hysteretic   = interface.pc_function(ew_new)   

        return pc_hysteretic

    def d_pc_function(self, ew, drain, ew_reversal, ew_endpoint, pc_reversal, pc_endpoint):
        """
        Derivative of hysteretic capillary pressure over the w-phase folume fraction
        """
        if(drain==True):
           interface = self.drainage 
        else:
           interface = self.imbibition

        ew_reversal_mpc = interface.volume_fraction_function(pc_reversal)
        ew_endpoint_mpc = interface.volume_fraction_function(pc_endpoint)
        ew_new          = ((ew-ew_endpoint)/(ew_reversal-ew_endpoint))*(ew_reversal_mpc-ew_endpoint_mpc) + ew_endpoint_mpc
        slope           = (ew_reversal_mpc-ew_endpoint_mpc)/(ew_reversal-ew_endpoint)
        d_pc_hystereric = slope * interface.d_pc_function(ew_new)

        return d_pc_hystereric

    def ew_hysteretic(self, pc, drain, ew_reversal, ew_endpoint, pc_reversal, pc_endpoint):
        """
        Hysteretic volume fraction of w phase  
        """
        if(drain==True):
           interface = self.drainage 
        else:
           interface = self.imbibition

        ew_reversal_mpc = interface.volume_fraction_function(pc_reversal)
        ew_endpoint_mpc = interface.volume_fraction_function(pc_endpoint)
        ew              = interface.volume_fraction_function(pc)

        ew_hystereric   =(((ew-ew_endpoint_mpc)/(ew_reversal_mpc-ew_endpoint_mpc))*(ew_reversal-ew_endpoint)) + ew_endpoint 

        return ew_hystereric
