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
    sa_reversal: np.ndarray
    sa_endpoint: np.ndarray
    snte_reversal: np.ndarray
    snte_endpoint: np.ndarray
    path_is_pdc: np.ndarray
    switch_condition: str
    hysteresis_tol: float
    max_reversals: int
    max_time_steps: int

    drainage_process_old   : np.ndarray = field(default_factory=lambda: np.array([]))
    switch_condition_value : np.ndarray = field(default_factory=lambda: np.array([]))
    reversals_number       : np.ndarray = field(default_factory=lambda: np.array([]))
    exceeds_number         : np.ndarray = field(default_factory=lambda: np.array([]))
    ew_hysteresis          : np.ndarray = field(default_factory=lambda: np.array([]))
    pc_hysteresis          : np.ndarray = field(default_factory=lambda: np.array([]))
    process_hysteresis     : np.ndarray = field(default_factory=lambda: np.array([]))
    time                   : np.ndarray = field(default_factory=lambda: np.array([]))
    counter_hysteresis     : int = 1

class ParkerLenhard:
    """
    Hysteresis Class to hold functions and parameters
    """
    def __init__(self, model, imb_interface, dr_interface, **kwargs):
        """
        Initialize the imbibition and drainage interfaces
        Accept parameters in any order using **kwargs
        """
        # Default required keys for the data class
        required_keys = [
            'drainage_process_new', 'pc_reversal', 'pc_endpoint', 'ew_reversal', 'ew_endpoint',
            'sa_reversal', 'sa_endpoint', 'snte_reversal', 'snte_endpoint', 'path_is_pdc',
            'switch_condition', 'hysteresis_tol', 'max_reversals', 'max_time_steps'
        ]
        
        # Check if all required parameters are provided
        for key in required_keys:
            if key not in kwargs:
                raise ValueError(f"Missing required parameter: {key}")
        
        # Initialize the ParkerLenharddata dataclass with values from kwargs
        self.data = ParkerLenharddata(**kwargs)

        # Initialize other attributes
        self.data.drainage_process_old   = np.array(self.data.drainage_process_new)
        self.data.ew_hysteresis          = np.zeros((model.nodes, self.data.max_time_steps))
        self.data.pc_hysteresis          = np.zeros((model.nodes, self.data.max_time_steps))
        self.data.process_hysteresis     = np.zeros((model.nodes, self.data.max_time_steps))
        self.data.switch_condition_value = np.zeros((model.nodes, self.data.max_time_steps))
        self.data.time                   = np.zeros(self.data.max_time_steps)
        self.data.reversals_number = np.zeros(model.nodes)
        self.data.exceeds_number   = np.zeros(model.nodes)
        
        self.e_w           = np.zeros(model.nodes)
        self.p_w           = np.zeros(model.nodes)
        self.p_n           = np.zeros(model.nodes)
        self.p_c           = np.zeros(model.nodes)
        self.s_a           = np.zeros(model.nodes)
        self.s_nt_e        = np.zeros(model.nodes)
        self.primary_variables = {}
        self.imbibition    = imb_interface
        self.drainage      = dr_interface
        self.model         = model

    def pc_function(self,ew):
        """
        Hysteretic capillary pressure
        mpc  : main primary curve (drainage or imbibition)
        """
        drainage_process,sa_reversal,sa_endpoint,pc_reversal,pc_endpoint,snte_reversal,snte_endpoint,ew_reversal,ew_endpoint = self.get_data_hysteresis()
        e_wr = self.model.w_phase.residual_volume_fraction
        e_nr = self.model.n_phase.residual_volume_fraction
        e_ws = self.model.solid.porosity - e_nr - e_wr

        if drainage_process:
            interface = self.drainage
        else:
            interface = self.imbibition

        sa_reversal_mpc = (interface.volume_fraction_function(pc_reversal)-e_wr)/e_ws
        sa_endpoint_mpc = (interface.volume_fraction_function(pc_endpoint)-e_wr)/e_ws
        snte            = self.linear_effective_entrapped_saturation_nw_ew(ew)
        sa              = (ew-e_wr)/(self.model.solid.porosity-e_wr) +  snte
        sa_hysteretic   = ((sa-sa_endpoint)/(sa_reversal-sa_endpoint))*(sa_reversal_mpc-sa_endpoint_mpc) + sa_endpoint_mpc
        pc_hysteretic   = interface.pc_function(sa_hysteretic * e_ws + e_wr)

        return pc_hysteretic

    def d_pc_function(self, ew):
        """
        Derivative of hysteretic capillary pressure over the w-phase volume fraction
        """
        drainage_process,sa_reversal,sa_endpoint,pc_reversal,pc_endpoint,snte_reversal,snte_endpoint,ew_reversal,ew_endpoint = self.get_data_hysteresis()

        e_wr = self.model.w_phase.residual_volume_fraction
        e_nr = self.model.n_phase.residual_volume_fraction
        e_ws = self.model.solid.porosity - e_nr - e_wr
        if drainage_process:
            interface = self.drainage
        else:
            interface = self.imbibition
        
        sa_reversal_mpc = (interface.volume_fraction_function(pc_reversal)-e_wr)/e_ws
        sa_endpoint_mpc = (interface.volume_fraction_function(pc_endpoint)-e_wr)/e_ws

        snte            = self.linear_effective_entrapped_saturation_nw_ew(ew)
        sa              = (ew-e_wr)/(self.model.solid.porosity-e_wr) +  snte
        sa_hysteretic   = ((sa-sa_endpoint)/(sa_reversal-sa_endpoint))*(sa_reversal_mpc-sa_endpoint_mpc) + sa_endpoint_mpc

        dsnte           = self.dlinear_effective_entrapped_saturation_nw_ew(ew)
        dsa             = 1./(self.model.solid.porosity-e_wr) +  dsnte
        dsa_hysteretic  = ((sa_reversal_mpc-sa_endpoint_mpc)/(sa_reversal-sa_endpoint))*dsa

        d_pc_hysteretic = e_ws*dsa_hysteretic*interface.d_pc_function(sa_hysteretic)

        return d_pc_hysteretic

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

        dsa   = self.d_apparent_saturation_function(pc)
        dsnte = self.d_linear_effective_entrapped_saturation_nw(dsa)

        dew_hysteretic = (porosity-e_wr)*(dsa-dsnte)

        return dew_hysteretic

    def apparent_saturation_function(self, pc):
        """
        Hysteretic volume fraction of w phase  
        """
        drainage_process,sa_reversal,sa_endpoint,pc_reversal,pc_endpoint,snte_reversal,snte_endpoint,ew_reversal,ew_endpoint = self.get_data_hysteresis()
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
        drainage_process,sa_reversal,sa_endpoint,pc_reversal,pc_endpoint,snte_reversal,snte_endpoint,ew_reversal,ew_endpoint = self.get_data_hysteresis()
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
        drainage_process,sa_reversal,sa_endpoint,pc_reversal,pc_endpoint,snte_reversal,snte_endpoint,ew_reversal,ew_endpoint = self.get_data_hysteresis()
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
        drainage_process,sa_reversal,sa_endpoint,pc_reversal,pc_endpoint,snte_reversal,snte_endpoint,ew_reversal,ew_endpoint = self.get_data_hysteresis()

        a = (snte_endpoint-snte_reversal)/(sa_endpoint-sa_reversal)
        b = snte_endpoint - a * sa_endpoint
        snte = a * sa + b

        return snte

    def linear_effective_entrapped_saturation_nw_ew(self,ew):
        """
        Linear evolution of effective entrapped nw saturation from Parker Lenhard formula  snt_eff = a * ew + b
        """
        drainage_process,sa_reversal,sa_endpoint,pc_reversal,pc_endpoint,snte_reversal,snte_endpoint,ew_reversal,ew_endpoint = self.get_data_hysteresis()

        a = (snte_endpoint-snte_reversal)/(ew_endpoint-ew_reversal)
        b = snte_endpoint - a * sa_endpoint
        snte = a * ew + b

        return snte

    def d_linear_effective_entrapped_saturation_nw(self,dsa):
        """
        Linear evolution of effective entrapped nw saturation from Parker Lenhard formula  snt_eff = a * sa + b
        sa : apparent saturation
        """
        drainage_process,sa_reversal,sa_endpoint,pc_reversal,pc_endpoint,snte_reversal,snte_endpoint,ew_reversal,ew_endpoint = self.get_data_hysteresis()

        a = (snte_endpoint-snte_reversal)/(sa_endpoint-sa_reversal)
        dsnte = a * dsa

        return dsnte

    def dlinear_effective_entrapped_saturation_nw_ew(self,ew):
        """
        Linear evolution of effective entrapped nw saturation from Parker Lenhard formula  snt_eff = a * ew + b
        sa : apparent saturation
        """
        drainage_process,sa_reversal,sa_endpoint,pc_reversal,pc_endpoint,snte_reversal,snte_endpoint,ew_reversal,ew_endpoint = self.get_data_hysteresis()

        a = (snte_endpoint-snte_reversal)/(ew_endpoint-ew_reversal)
        b = snte_endpoint - a * sa_endpoint
        dsnte = a

        return dsnte

    def get_data_hysteresis(self):
        """
        function to get the hysetresis reversal and endpoints at current time.
        """
        n= self.model.current_cell

        drainage_process = self.data.drainage_process_new[n]
        pc_reversal = self.data.pc_reversal[n][0]
        pc_endpoint = self.data.pc_endpoint[n][0]
        ew_reversal = self.data.ew_reversal[n][0]
        ew_endpoint = self.data.ew_endpoint[n][0]
        sa_reversal = self.data.sa_reversal[n][0]
        sa_endpoint = self.data.sa_endpoint[n][0]
        snte_reversal = self.data.snte_reversal[n][0]
        snte_endpoint = self.data.snte_endpoint[n][0]

        return drainage_process,sa_reversal,sa_endpoint,pc_reversal,pc_endpoint,snte_reversal,snte_endpoint,ew_reversal,ew_endpoint

    def determine_hysteresis_state(self,switch_condition):
        """
        function to update the current process at each node (drainage or imbibition ) based on 
        the switch condition (which is different for each node).
        If the switch condition is greater than some user defined tolerance (hysteresis_tol), 
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

        # This part of code is to update the scaling points in case of a reversal
        if self.data.drainage_process_new[n] != self.data.drainage_process_old[n]:
            self.update_scaling_points_reversal_case(self.s_a[n], self.s_nt_e[n], self.p_c[n])

        # This part of code is to update the scaling points in case of exceeding the current endpoint (without reversal).
        if (self.data.drainage_process_new[n]== True and self.s_a[n] < self.data.sa_endpoint[n][0]) or (self.data.drainage_process_new[n]== False and self.s_a[n] > self.data.sa_endpoint[n][0]):
            self.update_scaling_points_exceeding_case()


    def update_scaling_points_reversal_case(self,sa,snte,pc):
        """
        function to update hysteresis  reversal and end points. each node has 2 reversal points 
        and 2 endpoints, which means 4 scaling points for each node.
        """
        e_wr = self.model.w_phase.residual_volume_fraction
        n = self.model.current_cell
        # keep track of the number of reversals
        self.data.reversals_number[n] +=  1
        # store the scaling points by sliding the indices. This is needed in case we exceed an endpoint.
        for i in range(self.data.max_reversals-1, 0, -1):
            self.data.pc_endpoint[n][i]  = self.data.pc_endpoint[n][i-1]
            self.data.pc_reversal[n][i]  = self.data.pc_reversal[n][i-1]
            self.data.ew_endpoint[n][i]  = self.data.ew_endpoint[n][i-1]
            self.data.ew_reversal[n][i]  = self.data.ew_reversal[n][i-1]
            self.data.sa_endpoint[n][i]  = self.data.sa_endpoint[n][i-1]
            self.data.sa_reversal[n][i]  = self.data.sa_reversal[n][i-1]
            self.data.snte_endpoint[n][i]  = self.data.snte_endpoint[n][i-1]
            self.data.snte_reversal[n][i]  = self.data.snte_reversal[n][i-1]

        # verify if how far we are from the main process curve (process=dr or imb)
        current_path_index = self.data.reversals_number[n] - 2*self.data.exceeds_number[n]
        # Update reversal and end points

        self.data.sa_endpoint[n][0]  = self.data.sa_reversal[n][0]
        self.data.sa_reversal[n][0]  = sa
        # if we are on the primary drainage curve than Lands formula should be used to estimate the max entrapped nw phase (Snr)
        if current_path_index == 1 and self.data.path_is_pdc[n]== True:
            self.data.snte_endpoint[n][0] = self.lands_residual_effective_saturation_nw()
            self.data.snte_reversal[n][0] = 0.0
        # if we are on a scanning curve just do "endpoint=reversal" and "reversal= snte calculated from results of ts solver"
        elif current_path_index is not 0 or 1:
            self.data.snte_endpoint[n][0] = self.data.snte_reversal[n][0]
            self.data.snte_reversal[n][0] = snte

        self.data.pc_endpoint[n][0]  = self.data.pc_reversal[n][0]
        self.data.pc_reversal[n][0]  = pc
        self.data.ew_endpoint[n][0]  = self.data.ew_reversal[n][0]
        self.data.ew_reversal[n][0]  = (self.data.sa_reversal[n][0] -  self.data.snte_reversal[n][0])*(self.model.solid.porosity-e_wr)+e_wr

    def update_scaling_points_exceeding_case(self):
        """
        function to update hysteresis  reversal and end points. each node has 2 reversal points 
        and 2 endpoints, which means 4 scaling points for each node.
        """
        n=self.model.current_cell
        # keep track of the number of exceeds.
        self.data.exceeds_number[n] +=  1
        # Update scaling points in case we have an exceeding of endpoint using the second oldest scaling points.
        for i in range( 0, self.data.max_reversals-2):
            self.data.pc_endpoint[n][i]   = self.data.pc_endpoint[n][i+2]
            self.data.pc_reversal[n][i]   = self.data.pc_reversal[n][i+2]
            self.data.ew_endpoint[n][i]   = self.data.ew_endpoint[n][i+2]
            self.data.ew_reversal[n][i]   = self.data.ew_reversal[n][i+2]
            self.data.sa_endpoint[n][i]   = self.data.sa_endpoint[n][i+2]
            self.data.sa_reversal[n][i]   = self.data.sa_reversal[n][i+2]
            self.data.snte_endpoint[n][i] = self.data.snte_endpoint[n][i+2]
            self.data.snte_reversal[n][i] = self.data.snte_reversal[n][i+2]
        # If we get back to primary drainage curve then set entrapped scaling points to zero.
        if self.data.reversals_number[n] - 2*self.data.exceeds_number[n] == 0 and self.data.path_is_pdc[n] == True:
            self.data.snte_endpoint[n][0] = 0.0
            self.data.snte_reversal[n][0] = 0.0

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
        self.data.time[counter] = ts.getTime()

        #Get primary variables from solver
        self.e_w = self.primary_variables['e_w']
        self.p_w = self.primary_variables['p_w']
        self.p_n = self.primary_variables['p_n']
        self.p_c = self.primary_variables['p_c']

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
