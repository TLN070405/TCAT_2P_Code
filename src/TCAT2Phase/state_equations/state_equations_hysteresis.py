"""hysteresis.py"""
class Hysteresis:
    """
    Hysteresis Class to hold functions and parameters
    """
    def __init__(self,model,imb_pc,dr_pc):
        """
        Initialize the imbibition and drainage interfaces
        """
        self.imbibition    = imb_pc
        self.drainage      = dr_pc
        self.model         = model 
    def pc_function(self,ew):
        """
        Hysteretic capillary pressure
        mpc  : main primary curve (drainage or imbibition)
        """
        drain,ew_reversal,ew_endpoint,pc_reversal,pc_endpoint = get_data_hysteresis(self.model)

        if(drain==True):
           interface = self.drainage 
        else:
           interface = self.imbibition

        ew_reversal_mpc =  interface.volume_fraction_function(pc_reversal)
        ew_endpoint_mpc =  interface.volume_fraction_function(pc_endpoint)
        ew_new          = ((ew-ew_endpoint)/(ew_reversal-ew_endpoint))*(ew_reversal_mpc-ew_endpoint_mpc) + ew_endpoint_mpc
        pc_hysteretic   = interface.pc_function(ew_new)   

        return pc_hysteretic

    def d_pc_function(self, ew):
        """
        Derivative of hysteretic capillary pressure over the w-phase folume fraction
        """
        drain,ew_reversal,ew_endpoint,pc_reversal,pc_endpoint = get_data_hysteresis(self.model)

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

    def volume_fraction_function(self, pc):
        """
        Hysteretic volume fraction of w phase  
        """
        drain,ew_reversal,ew_endpoint,pc_reversal,pc_endpoint = get_data_hysteresis(self.model)

        if(drain==True):
           interface = self.drainage 
        else:
           interface = self.imbibition

        ew_reversal_mpc = interface.volume_fraction_function(pc_reversal)
        ew_endpoint_mpc = interface.volume_fraction_function(pc_endpoint)
        ew              = interface.volume_fraction_function(pc)
        ew_hystereric   =((ew-ew_endpoint_mpc)/(ew_reversal_mpc-ew_endpoint_mpc))*(ew_reversal-ew_endpoint) + ew_endpoint 

        return ew_hystereric

    def d_volume_fraction_function(self, pc):
        """
        Hysteretic derivative volume fraction of w phase  
        """
        drain,ew_reversal,ew_endpoint,pc_reversal,pc_endpoint = get_data_hysteresis(self.model)

        if(drain==True):
           interface = self.drainage 
        else:
           interface = self.imbibition

        ew_reversal_mpc = interface.volume_fraction_function(pc_reversal)
        ew_endpoint_mpc = interface.volume_fraction_function(pc_endpoint)
        slope           =(ew_reversal-ew_endpoint) /(ew_reversal_mpc-ew_endpoint_mpc)
        dew_hystereric  =slope*interface.d_volume_fraction_function(pc)

        return dew_hystereric
    
def get_data_hysteresis(model):
   """
   function to get the hysetresis data from model.
   """

   current_cell= model.current_cell
   state       = model.data_hysteresis['state']

   if(state=="new"):
      drain       = model.data_hysteresis['drainage_new'][current_cell]
      pc_reversal = model.data_hysteresis['p_c_reversal'][current_cell]
      pc_endpoint = model.data_hysteresis['p_c_endpoint'][current_cell]
      ew_reversal = model.data_hysteresis['e_w_reversal'][current_cell]
      ew_endpoint = model.data_hysteresis['e_w_endpoint'][current_cell]
      
   elif(state=="old"):
      drain       = model.data_hysteresis['drainage_old'][current_cell]
      pc_reversal = model.data_hysteresis['p_c_reversal_old'][current_cell]
      pc_endpoint = model.data_hysteresis['p_c_endpoint_old'][current_cell]
      ew_reversal = model.data_hysteresis['e_w_reversal_old'][current_cell]
      ew_endpoint = model.data_hysteresis['e_w_endpoint_old'][current_cell]
      
   else:
      print("state can only took two values -new- or -old-")

   return drain,ew_reversal,ew_endpoint,pc_reversal,pc_endpoint