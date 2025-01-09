"""hysteresis models.py"""
import dataclasses
from dataclasses import field
import numpy as np

@dataclasses.dataclass
class ParkerLenharddata:
    """
    Parker Lenhard hysteresis model necessary arrays and data
    """
    drainage_process_new: np.ndarray
    pc_reversal: np.ndarray
    pc_endpoint: np.ndarray
    ew_reversal: np.ndarray
    ew_endpoint: np.ndarray
    ew_hysteresis: np.ndarray
    pc_hysteresis: np.ndarray
    process_hysteresis: np.ndarray
    switch_condition: str
    hysteresis_tol: float
    max_reversal_points: int

    drainage_process_old: np.ndarray = field(default_factory=lambda: np.array([]))
    switch_condition_value:  np.ndarray = field(default_factory=lambda: np.array([]))

    counter_hysteresis: int = 1

class ParkerLenhard:
    """
    Hysteresis Class to hold functions and parameters
    """
    def __init__(self,model,imb_interface,dr_interface,
            drainage_process_new,
            pc_reversal,
            pc_endpoint,
            ew_reversal,
            ew_endpoint,
            ew_hysteresis,
            pc_hysteresis,
            process_hysteresis,
            switch_condition,
            hysteresis_tol,
            max_reversal_points):
        """
        Initialize the imbibition and drainage interfaces
        """
        self.data = ParkerLenharddata(
            drainage_process_new,
            pc_reversal,
            pc_endpoint,
            ew_reversal,
            ew_endpoint,
            ew_hysteresis,
            pc_hysteresis,
            process_hysteresis,
            switch_condition,
            hysteresis_tol,
            max_reversal_points)

        self.data.drainage_process_old = np.array(drainage_process_new)
        self.data.switch_condition_value  = np.array(process_hysteresis)
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
        drainage_process,ew_reversal,ew_endpoint,pc_reversal,pc_endpoint = self.get_data_hysteresis()

        if drainage_process:
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
        drainage_process,ew_reversal,ew_endpoint,pc_reversal,pc_endpoint = self.get_data_hysteresis()

        if drainage_process:
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
        drainage_process,ew_reversal,ew_endpoint,pc_reversal,pc_endpoint = self.get_data_hysteresis()

        if drainage_process:
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
        drainage_process,ew_reversal,ew_endpoint,pc_reversal,pc_endpoint = self.get_data_hysteresis()

        if drainage_process:
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

        drainage_process = self.data.drainage_process_new[n]
        pc_reversal = self.data.pc_reversal[n][0]
        pc_endpoint = self.data.pc_endpoint[n][0]
        ew_reversal = self.data.ew_reversal[n][0]
        ew_endpoint = self.data.ew_endpoint[n][0]

        return drainage_process,ew_reversal,ew_endpoint,pc_reversal,pc_endpoint

    def determine_hysteresis_state(self,switch_condition):
        """
        function to update the current process at each node (drainage or imbibition ) based on 
        the switch condition (which is different for each node). 
        If the switch condition is greater than than some user defined tolerance (hysteresis_tol), 
        an imbibition process will take place. Else, the current process is a drainage. 
        The theoretical value of  "hysteresis_tol" is zero, but to avoid oscillations 
        in the solution a bigger value is preferred.
        """
        n=self.model.current_cell

        if switch_condition > self.data.hysteresis_tol:
            self.data.drainage_process_old[n] = self.data.drainage_process_new[n]
            self.data.drainage_process_new[n] = False
        elif switch_condition < -self.data.hysteresis_tol:
            self.data.drainage_process_old[n] = self.data.drainage_process_new[n]
            self.data.drainage_process_new[n] = True
        else :
            self.data.drainage_process_old[n] = self.data.drainage_process_new[n]

        if self.data.drainage_process_new[n] != self.data.drainage_process_old[n]:
            self.update_scaling_points(self.e_w[n], self.p_c[n])

        if (self.data.drainage_process_new[n]==True and self.e_w[n] < self.data.ew_endpoint[n][0]) or (self.data.drainage_process_new[n]==False and self.e_w[n] > self.data.ew_endpoint[n][0]):
            for i in range( 0, self.data.max_reversal_points-2):
                self.data.pc_endpoint[n][i]  = self.data.pc_endpoint[n][i+2]
                self.data.pc_reversal[n][i]  = self.data.pc_reversal[n][i+2]
                self.data.ew_endpoint[n][i]  = self.data.ew_endpoint[n][i+2]
                self.data.ew_reversal[n][i]  = self.data.ew_reversal[n][i+2]

    def update_scaling_points(self,ew,pc):
        """
        function to update hysteresis  reversal and end points. each node has 2 reversal points 
        and 2 endpoints, which means 4 scaling points for each node.

        """
        n = self.model.current_cell
        for i in range(self.data.max_reversal_points-1, 0, -1):
            self.data.pc_endpoint[n][i]  = self.data.pc_endpoint[n][i-1]
            self.data.pc_reversal[n][i]  = self.data.pc_reversal[n][i-1]
            self.data.ew_endpoint[n][i]  = self.data.ew_endpoint[n][i-1]
            self.data.ew_reversal[n][i]  = self.data.ew_reversal[n][i-1]
        # Update reversal and endpoints (ew and pc)
        self.data.pc_endpoint[n][0]     = self.data.pc_reversal[n][0]
        self.data.pc_reversal[n][0]     = pc
        self.data.ew_endpoint[n][0]     = self.data.ew_reversal[n][0]
        self.data.ew_reversal[n][0]     = ew

    def store_hysteresis_results(self,ew,pc):
        """
        function to store hysteresis results of ew and pc. These results are used as old values (for switching condition)
        and for visualization purposes.
        """
        n = self.model.current_cell

        # Store hysteretic ew and pc for visualization
        self.data.ew_hysteresis[n,self.data.counter_hysteresis] = ew
        self.data.pc_hysteresis[n,self.data.counter_hysteresis] = pc
        if self.data.drainage_process_new[n] == True:
            self.data.process_hysteresis[n,self.data.counter_hysteresis] = -1
        else:
            self.data.process_hysteresis[n,self.data.counter_hysteresis] = 1

    def model_update_hysteresis(self,ts):
        """
        main function to update process, update scaling points and store results.
        """
        nodes = self.model.nodes
        counter  = self.data.counter_hysteresis

        if self.data.switch_condition == 'p_c':
            switch_condition = self.data.pc_hysteresis[:,counter-1] - self.p_c
        elif self.data.switch_condition == 'e_w':
            switch_condition = self.e_w - self.data.ew_hysteresis[:,counter-1]
        else:
            print('switch_condition should be either e_w or p_c')

        for n in range(0,nodes):
            self.model.current_cell = n
            self.data.switch_condition_value[n,counter] = switch_condition[n]
            self.determine_hysteresis_state(switch_condition[n])
            self.store_hysteresis_results(self.e_w[n], self.p_n[n]-self.p_w[n])

        self.data.counter_hysteresis += 1
