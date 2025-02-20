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
    sa_reversal: np.ndarray
    sa_endpoint: np.ndarray
    snte_reversal: np.ndarray
    snte_endpoint: np.ndarray
    path_is_pdc: np.ndarray
    ew_hysteresis: np.ndarray
    pc_hysteresis: np.ndarray
    process_hysteresis: np.ndarray
    time_hysteresis: np.ndarray
    switch_condition: str
    hysteresis_tol: float
    max_reversals: int

    drainage_process_old: np.ndarray = field(default_factory=lambda: np.array([]))
    switch_condition_value: np.ndarray = field(default_factory=lambda: np.array([]))
    reversals_number : np.ndarray = field(default_factory=lambda: np.array([]))
    exceeds_number   : np.ndarray = field(default_factory=lambda: np.array([]))
    counter_hysteresis: int = 1

class ParkerLenhard:
    """
    Hysteresis Class to hold functions and parameters
    """
    def __init__(self,model,imb_interface,dr_interface,
            drainage_process_new,
            pc_reversal,
            pc_endpoint,
            sa_reversal,
            sa_endpoint,
            snte_reversal,
            snte_endpoint,
            path_is_pdc,
            ew_hysteresis,
            pc_hysteresis,
            process_hysteresis,
            time_hysteresis,
            switch_condition,
            hysteresis_tol,
            max_reversals):
        """
        Initialize the imbibition and drainage interfaces
        """
        self.data = ParkerLenharddata(
            drainage_process_new,
            pc_reversal,
            pc_endpoint,
            sa_reversal,
            sa_endpoint,
            snte_reversal,
            snte_endpoint,
            path_is_pdc,
            ew_hysteresis,
            pc_hysteresis,
            process_hysteresis,
            time_hysteresis,
            switch_condition,
            hysteresis_tol,
            max_reversals)

        self.data.drainage_process_old   = np.array(drainage_process_new)
        self.data.switch_condition_value = np.array(process_hysteresis)
        self.data.reversals_number = np.zeros(model.nodes)
        self.data.exceeds_number   = np.zeros(model.nodes)

        self.e_w           = np.zeros(model.nodes)
        self.p_w           = np.zeros(model.nodes)
        self.p_n           = np.zeros(model.nodes)
        self.p_c           = np.zeros(model.nodes)
        self.s_a           = np.zeros(model.nodes)
        self.s_nt_e        = np.zeros(model.nodes)
        self.imbibition    = imb_interface
        self.drainage      = dr_interface
        self.model         = model


    def volume_fraction_function(self, pc):
        """
        Hysteretic volume fraction of w phase  
        """
        e_wr = self.model.w_phase.residual_volume_fraction
        porosity = self.model.solid.porosity

        sa = self.apparent_saturation_function(pc)
        snte = self.linear_effective_entrapped_saturation_nw(sa)
        ew_hysteretic = (sa - snte) * (porosity-e_wr) + e_wr

        return ew_hysteretic

    def d_volume_fraction_function(self, pc):
        """
        Hysteretic derivative volume fraction of w phase  
        """

        e_wr = self.model.w_phase.residual_volume_fraction
        porosity = self.model.solid.porosity

        dsa   = self.d_apparent_saturation_function( pc)
        dsnte = self.d_linear_effective_entrapped_saturation_nw(dsa)

        dew_hysteretic = (porosity-e_wr)*(dsa-dsnte)

        return dew_hysteretic

    def apparent_saturation_function(self, pc):
        """
        Hysteretic volume fraction of w phase  
        """
        drainage_process,sa_reversal,sa_endpoint,pc_reversal,pc_endpoint,snte_reversal,snte_endpoint = self.get_data_hysteresis()
        e_wr = self.model.w_phase.residual_volume_fraction
        e_nr = self.model.n_phase.residual_volume_fraction
        e_ws = self.model.solid.porosity - e_nr - e_wr

        if drainage_process:
            interface = self.drainage
        else:
            interface = self.imbibition

        sa_reversal_mpc = (interface.volume_fraction_function(pc_reversal)-e_wr)/e_ws
        sa_endpoint_mpc = (interface.volume_fraction_function(pc_endpoint)-e_wr)/e_ws
        sa              = (interface.volume_fraction_function(pc)-e_wr)/e_ws
        sa_hystereric   = ((sa-sa_endpoint_mpc)/(sa_reversal_mpc-sa_endpoint_mpc))*(sa_reversal-sa_endpoint) + sa_endpoint

        return sa_hystereric

    def d_apparent_saturation_function(self, pc):
        """
        Hysteretic apparent saturation with respect to pc dSa/dpc
        """
        drainage_process,sa_reversal,sa_endpoint,pc_reversal,pc_endpoint,snte_reversal,snte_endpoint = self.get_data_hysteresis()
        e_wr = self.model.w_phase.residual_volume_fraction
        e_nr = self.model.n_phase.residual_volume_fraction
        e_ws = self.model.solid.porosity - e_nr - e_wr

        if drainage_process:
            interface = self.drainage
        else:
            interface = self.imbibition

        sa_reversal_mpc = (interface.volume_fraction_function(pc_reversal)-e_wr)/e_ws
        sa_endpoint_mpc = (interface.volume_fraction_function(pc_endpoint)-e_wr)/e_ws
        dsa              = interface.d_volume_fraction_function(pc)/e_ws
        dsa_hystereric   = ((sa_reversal-sa_endpoint)/(sa_reversal_mpc-sa_endpoint_mpc))*dsa

        return dsa_hystereric

    def lands_residual_effective_saturation_nw(self):
        """
        maximum entrapped effective nw saturation from a reversal point on the primary drainage curve 
        (Lands formula)
        """
        drainage_process,sa_reversal,sa_endpoint,pc_reversal,pc_endpoint,snte_reversal,snte_endpoint = self.get_data_hysteresis()
        e_wr = self.model.w_phase.residual_volume_fraction
        e_nr = self.model.n_phase.residual_volume_fraction
        porosity = self.model.solid.porosity

        if e_nr == 0:
            s_nr_e = 0.0
        else:
            r_nw = ((porosity-e_wr)/e_nr)-1
            s_nr_e = (1 - sa_reversal)/(1 + r_nw*(1 - sa_reversal))

        return s_nr_e

    def linear_effective_entrapped_saturation_nw(self,sa):
        """
        Linear evolution of effective entrapped nw saturation from Parker Lenhard formula  snt_eff = a * sa + b
        sa : apparent saturation
        """
        drainage_process,sa_reversal,sa_endpoint,pc_reversal,pc_endpoint,snte_reversal,snte_endpoint = self.get_data_hysteresis()

        a = (snte_endpoint-snte_reversal)/(sa_endpoint-sa_reversal)
        b = snte_endpoint - a * sa_endpoint
        snte = a * sa + b

        return snte

    def d_linear_effective_entrapped_saturation_nw(self,dsa):
        """
        Linear evolution of effective entrapped nw saturation from Parker Lenhard formula  snt_eff = a * sa + b
        sa : apparent saturation
        """
        drainage_process,sa_reversal,sa_endpoint,pc_reversal,pc_endpoint,snte_reversal,snte_endpoint = self.get_data_hysteresis()

        a = (snte_endpoint-snte_reversal)/(sa_endpoint-sa_reversal)
        dsnte = a * dsa

        return dsnte


    def get_data_hysteresis(self):
        """
        function to get the hysetresis reversal and endpoints at current time.
        """
        n= self.model.current_cell

        drainage_process = self.data.drainage_process_new[n]
        pc_reversal = self.data.pc_reversal[n][0]
        pc_endpoint = self.data.pc_endpoint[n][0]
        sa_reversal = self.data.sa_reversal[n][0]
        sa_endpoint = self.data.sa_endpoint[n][0]
        snte_reversal = self.data.snte_reversal[n][0]
        snte_endpoint = self.data.snte_endpoint[n][0]

        return drainage_process,sa_reversal,sa_endpoint,pc_reversal,pc_endpoint,snte_reversal,snte_endpoint

    def determine_hysteresis_state(self,switch_condition):
        """
        function to update the current process at each node (drainage or imbibition ) based on 
        the switch condition (which is different for each node). 
        If the switch condition is greater than than some user defined tolerance (hysteresis_tol), 
        an imbibition process will take place. Else, the current process is a drainage. 
        The theoretical value of  "hysteresis_tol" is zero, but to avoid oscillations 
        in the solution a bigger value is preferred.
        """
        #update the current process
        n=self.model.current_cell
        self.data.drainage_process_old[n] = self.data.drainage_process_new[n]
        if switch_condition > self.data.hysteresis_tol:
            self.data.drainage_process_new[n] = False
        elif switch_condition < -self.data.hysteresis_tol:
            self.data.drainage_process_new[n] = True
        #update the scaling points
        if self.data.drainage_process_new[n] != self.data.drainage_process_old[n]:
            self.update_scaling_points(self.s_a[n], self.s_nt_e[n], self.p_c[n])

        # if there is an exceeding of the current endpoint replace the current scaling points with the correspondent previous ones.
        if (self.data.drainage_process_new[n]== True and self.s_a[n] < self.data.sa_endpoint[n][0]) or (self.data.drainage_process_new[n]== False and self.s_a[n] > self.data.sa_endpoint[n][0]):
            # keep track of the number of exceeds.
            self.data.exceeds_number[n]+= 1
            for i in range( 0, self.data.max_reversals-2):
                self.data.pc_endpoint[n][i]   = self.data.pc_endpoint[n][i+2]
                self.data.pc_reversal[n][i]   = self.data.pc_reversal[n][i+2]
                self.data.sa_endpoint[n][i]   = self.data.sa_endpoint[n][i+2]
                self.data.sa_reversal[n][i]   = self.data.sa_reversal[n][i+2]
                self.data.snte_endpoint[n][i] = self.data.snte_endpoint[n][i+1]
                self.data.snte_reversal[n][i] = self.data.snte_reversal[n][i+1]

    def update_scaling_points(self,sa,snte,pc):
        """
        function to update hysteresis  reversal and end points. each node has 2 reversal points 
        and 2 endpoints, which means 4 scaling points for each node.
        """
        n = self.model.current_cell
        # keep track of the number of reversals
        self.data.reversals_number[n] += 1
        # store the scaling points by sliding the indices. This is needed in case we exceed an endpoint.
        for i in range(self.data.max_reversals-1, 0, -1):
            self.data.pc_endpoint[n][i]  = self.data.pc_endpoint[n][i-1]
            self.data.pc_reversal[n][i]  = self.data.pc_reversal[n][i-1]
            self.data.sa_endpoint[n][i]  = self.data.sa_endpoint[n][i-1]
            self.data.sa_reversal[n][i]  = self.data.sa_reversal[n][i-1]
            self.data.snte_endpoint[n][i]  = self.data.snte_endpoint[n][i-1]
            self.data.snte_reversal[n][i]  = self.data.snte_reversal[n][i-1]

        # verify if we are on the primary drainage curve
        if self.data.reversals_number[n] - 2*self.data.exceeds_number[n] == 1:
            self.data.path_is_pdc[n] = True
        else:
            self.data.path_is_pdc[n] = False
        # Update reversal and end points
        self.data.sa_endpoint[n][0]  = self.data.sa_reversal[n][0]
        self.data.sa_reversal[n][0]  = sa
        # if we are on the primary drainage curve than Lands formula should be used to estimate the max entrapped nw phase (Snr)
        if self.data.path_is_pdc[n] == True:
            self.data.pc_endpoint[n][0]   = 0.0
            self.data.snte_endpoint[n][0] = self.lands_residual_effective_saturation_nw()
            self.data.snte_reversal[n][0] = 0.0
        # if we are on a scanning curve just do "endpoint=reversal" and "reversal= snte calculated from results of ts solver"
        elif self.data.path_is_pdc[n] == False:
            self.data.pc_endpoint[n][0] = self.data.pc_reversal[n][0]
            self.data.snte_endpoint[n][0] = self.data.snte_reversal[n][0]
            self.data.snte_reversal[n][0] = snte
        else:
            print("path_is_pdc tooks only the values False or True")

        self.data.pc_reversal[n][0] = pc


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
        self.data.time_hysteresis[self.data.counter_hysteresis] = ts.getTime()

        if self.data.switch_condition == 'p_c':
            switch_condition = self.data.pc_hysteresis[:,counter-1] - self.p_c
        elif self.data.switch_condition == 'e_w':
            switch_condition = self.e_w - self.data.ew_hysteresis[:,counter-1]
        else:
            print('switch_condition is a string and should be either e_w or p_c')

        for n in range(0,nodes):
            self.model.current_cell = n
            self.data.switch_condition_value[n,counter] = switch_condition[n]
            self.s_a[n]    = self.apparent_saturation_function(self.p_c[n])
            self.s_nt_e[n] = self.linear_effective_entrapped_saturation_nw(self.s_a[n])
            self.determine_hysteresis_state(switch_condition[n])
            self.store_hysteresis_results(self.e_w[n], self.p_n[n]-self.p_w[n])

        self.data.counter_hysteresis += 1
