"""hysteresis models.py"""
import dataclasses
from dataclasses import field
import numpy as np

@dataclasses.dataclass
class ParkerLenharddata:
    """
    Parker Lenhard hysteresis model necessary arrays and data
    """
    drainage_new: np.ndarray
    pc_reversal: np.ndarray
    pc_endpoint: np.ndarray
    ew_reversal: np.ndarray
    ew_endpoint: np.ndarray
    ew_hysteresis: np.ndarray
    pc_hysteresis: np.ndarray
    hysteresis_tol: float

    drainage_old: np.ndarray = field(default_factory=lambda: np.array([]))
    ew_endpoint_hysteresis: np.ndarray = field(default_factory=lambda: np.array([]))
    pc_endpoint_hysteresis: np.ndarray = field(default_factory=lambda: np.array([]))
    counter_hysteresis: int = 1

class ParkerLenhard:
    """
    Hysteresis Class to hold functions and parameters
    """
    def __init__(self,model,imb_interface,dr_interface,
            drainage_new,
            pc_reversal,
            pc_endpoint,
            ew_reversal,
            ew_endpoint,
            ew_hysteresis,
            pc_hysteresis,
            hysteresis_tol):
        """
        Initialize the imbibition and drainage interfaces
        """
        self.data = ParkerLenharddata(
            drainage_new,
            pc_reversal,
            pc_endpoint,
            ew_reversal,
            ew_endpoint,
            ew_hysteresis,
            pc_hysteresis,
            hysteresis_tol)

        self.data.drainage_old    = np.array(drainage_new)
        self.data.ew_endpoint_hysteresis   = np.tile(ew_endpoint[:, np.newaxis], (1, 100000))
        self.data.pc_endpoint_hysteresis   = np.tile(pc_endpoint[:, np.newaxis], (1, 100000))
        self.e_w           = np.zeros_like(model.nodes)
        self.p_w           = np.zeros_like(model.nodes)
        self.p_n           = np.zeros_like(model.nodes)
        self.p_c           = np.zeros_like(model.nodes)
        self.imbibition    = imb_interface
        self.drainage      = dr_interface
        self.model         = model

    def pc_function(self,ew):
        """
        Hysteretic capillary pressure
        mpc  : main primary curve (drainage or imbibition)
        """
        drainage,ew_reversal,ew_endpoint,pc_reversal,pc_endpoint = self.get_data_hysteresis()

        if drainage:
            interface = self.drainage
        else:
            interface = self.imbibition

        ew_reversal_mpc = interface.volume_fraction_function(pc_reversal)
        ew_endpoint_mpc = interface.volume_fraction_function(pc_endpoint)
        ew_new          = ((ew-ew_endpoint)/(ew_reversal-ew_endpoint))*(ew_reversal_mpc-ew_endpoint_mpc) + ew_endpoint_mpc
        pc_hysteretic   = interface.pc_function(ew_new)

        return pc_hysteretic

    def d_pc_function(self, ew):
        """
        Derivative of hysteretic capillary pressure over the w-phase folume fraction
        """
        drainage,ew_reversal,ew_endpoint,pc_reversal,pc_endpoint = self.get_data_hysteresis()

        if drainage:
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
        drainage,ew_reversal,ew_endpoint,pc_reversal,pc_endpoint = self.get_data_hysteresis()

        if drainage:
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
        drainage,ew_reversal,ew_endpoint,pc_reversal,pc_endpoint = self.get_data_hysteresis()

        if drainage:
            interface = self.drainage
        else:
            interface = self.imbibition

        ew_reversal_mpc = interface.volume_fraction_function(pc_reversal)
        ew_endpoint_mpc = interface.volume_fraction_function(pc_endpoint)
        slope           = (ew_reversal-ew_endpoint) /(ew_reversal_mpc-ew_endpoint_mpc)
        dew_hystereric  = slope*interface.d_volume_fraction_function(pc)

        return dew_hystereric

    def get_data_hysteresis(self):
        """
        function to get the hysetresis reversal and endpoints at current time.
        This function is only used by the hysteretic volume_fraction, d_volume_fraction, pc_function and d_pc_function.

        """
        n= self.model.current_cell

        drainage    = self.data.drainage_new[n]
        pc_reversal = self.data.pc_reversal[n]
        pc_endpoint = self.data.pc_endpoint[n]
        ew_reversal = self.data.ew_reversal[n]
        ew_endpoint = self.data.ew_endpoint[n]

        return drainage,ew_reversal,ew_endpoint,pc_reversal,pc_endpoint

    def determine_hysteresis_state(self,switch_condition):
        """
        function to update the current process at each node (drainage or imbibition ) based on 
        the switch condition (which is different for each node). 
        If the switch condition is greater than than some user defined tolerance (hysteresis_tol), 
        an imbibition process will take place. Else, the current process is a drainage. 
        The theoretical value of  "hysteresis_tol" is zero, but to avoid oscillations in the solution a bigger value is preferred.
        """
        n=self.model.current_cell

        if switch_condition > self.data.hysteresis_tol:
            self.data.drainage_old[n] = self.data.drainage_new[n]
            self.data.drainage_new[n] = False
        elif switch_condition < self.data.hysteresis_tol:
            self.data.drainage_old[n] = self.data.drainage_new[n]
            self.data.drainage_new[n] = True

        if self.data.drainage_new[n] != self.data.drainage_old[n]:
            self.update_scaling_points(self.e_w[n], self.p_c[n])
     
    def update_scaling_points(self,ew,pc):
        """
        function to update hysteresis  reversal and end points. each node has 2 reversal points 
        and 2 endpoints, which means 4 scaling points for each node.

        """
        n = self.model.current_cell

        # Update reversal and endpoints (ew and pc)
        self.data.pc_endpoint[n]     = self.data.pc_reversal[n]
        self.data.pc_reversal[n]     = pc
        self.data.ew_endpoint[n]     = self.data.ew_reversal[n]
        self.data.ew_reversal[n]     = ew

    def store_hysteresis_results(self,ew,pc):
        """
        function to store hysteresis results of ew and pc. These results are used as old values (for switching condition)
        and for visualization purposes.
        """
        n = self.model.current_cell

        # Store hysteretic ew and pc for visualization
        self.data.ew_hysteresis[n,self.data.counter_hysteresis] = ew
        self.data.pc_hysteresis[n,self.data.counter_hysteresis] = pc
        self.data.ew_endpoint_hysteresis[n,self.data.counter_hysteresis] = self.data.ew_endpoint[n]
        self.data.pc_endpoint_hysteresis[n,self.data.counter_hysteresis] = self.data.pc_endpoint[n]

    def model_update_hysteresis(self,ts):
        """
        main function to update process, update scaling points and store results.
        """
        nodes = self.model.nodes
        counter  = self.data.counter_hysteresis
        ew_hysteresis = self.data.ew_hysteresis
        pc_hysteresis = self.data.pc_hysteresis
        for n in range(0,nodes):
            self.model.current_cell = n
            #self.determine_hysteresis_state(self.e_w[n]-ew_hysteresis[n,counter-1]+pc_hysteresis[n,counter-1]-self.p_c[n])
            self.determine_hysteresis_state(self.e_w[n]-ew_hysteresis[n,counter-1])
            self.store_hysteresis_results(self.e_w[n], self.p_n[n]-self.p_w[n])

        self.data.counter_hysteresis += 1
