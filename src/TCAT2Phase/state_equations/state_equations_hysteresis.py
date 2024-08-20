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
    pc_old: np.ndarray
    ew_old: np.ndarray
    hysteresis_tol: float

    drainage_old: np.ndarray = field(default_factory=lambda: np.array([]))
    pc_reversal_old: np.ndarray =field(default_factory=lambda: np.array([]))
    pc_endpoint_old: np.ndarray =field(default_factory=lambda: np.array([]))
    ew_reversal_old: np.ndarray =field(default_factory=lambda: np.array([]))
    ew_endpoint_old: np.ndarray =field(default_factory=lambda: np.array([]))
    ew_hysteresis: np.ndarray = field(default_factory=lambda: np.array([]))
    pc_hysteresis: np.ndarray = field(default_factory=lambda: np.array([]))

    state: int = 0
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
            pc_old,
            ew_old,
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
            pc_old,
            ew_old,
            hysteresis_tol
            )
        self.data.drainage_old    = np.array(drainage_new)
        self.data.pc_reversal_old = np.array(pc_reversal)
        self.data.pc_endpoint_old = np.array(pc_endpoint)
        self.data.ew_reversal_old = np.array(ew_reversal)
        self.data.ew_endpoint_old = np.array(ew_endpoint)
        self.data.ew_hysteresis   = np.tile(ew_old[:, np.newaxis], (1, 100000))
        self.data.pc_hysteresis   = np.tile(pc_old[:, np.newaxis], (1, 100000))

        self.imbibition    = imb_interface
        self.drainage      = dr_interface
        self.model         = model

    def pc_function(self,ew):
        """
        Hysteretic capillary pressure
        mpc  : main primary curve (drainage or imbibition)
        """
        drain,ew_reversal,ew_endpoint,pc_reversal,pc_endpoint = self.get_data_hysteresis()

        if drain:
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
        drain,ew_reversal,ew_endpoint,pc_reversal,pc_endpoint = self.get_data_hysteresis()

        if drain:
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
        drain,ew_reversal,ew_endpoint,pc_reversal,pc_endpoint = self.get_data_hysteresis()

        if drain:
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
        drain,ew_reversal,ew_endpoint,pc_reversal,pc_endpoint = self.get_data_hysteresis()

        if drain:
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
        function to get the hysetresis data.
        """
        n= self.model.current_cell

        if self.data.state == 1:
            drain       = self.data.drainage_new[n]
            pc_reversal = self.data.pc_reversal[n]
            pc_endpoint = self.data.pc_endpoint[n]
            ew_reversal = self.data.ew_reversal[n]
            ew_endpoint = self.data.ew_endpoint[n]

        else:
            drain       = self.data.drainage_old[n]
            pc_reversal = self.data.pc_reversal_old[n]
            pc_endpoint = self.data.pc_endpoint_old[n]
            ew_reversal = self.data.ew_reversal_old[n]
            ew_endpoint = self.data.ew_endpoint_old[n]

        return drain,ew_reversal,ew_endpoint,pc_reversal,pc_endpoint

    def update_process(self,switch_condition):
        """
        function to update the process even drainage or imbibition
        """
        n=self.model.current_cell
        if switch_condition > self.data.hysteresis_tol:
            self.data.drainage_old[n] = self.data.drainage_new[n]
            self.data.drainage_new[n] = False
        else:
            self.data.drainage_old[n] = self.data.drainage_new[n]
            self.data.drainage_new[n] = True

    def update_hysteresis_capillary_pressure(self,ew_new,pc_new):
        """
        function to update hysteresis parameters when solution is in terms of capillary pressure
        """
        model = self.model
        self.data.state = 1
        n = model.current_cell

        # Store hysteretic ew and pc for visualization
        self.data.ew_hysteresis[n,self.data.counter_hysteresis] = ew_new
        self.data.pc_hysteresis[n,self.data.counter_hysteresis] = pc_new

        # Update the process even drainage or imbibition
        self.update_process(self.data.pc_old[n]-pc_new)

        # Update old capillary pressure
        self.data.pc_old[n] = pc_new

        # Update reversal and endpoints (ew and pc)
        if self.data.drainage_new[n] != self.data.drainage_old[n]:

            self.data.pc_reversal_old[n] = self.data.pc_reversal[n]
            self.data.pc_endpoint_old[n] = self.data.pc_endpoint[n]
            self.data.ew_endpoint_old[n] = self.data.ew_endpoint[n]
            self.data.ew_reversal_old[n] = self.data.ew_reversal[n]

            self.data.state = 0
            self.data.ew_endpoint[n]     = self.data.ew_reversal[n]
            self.data.ew_reversal[n]     = model.interface.volume_fraction_function(pc_new)
            self.data.pc_endpoint[n]     = self.data.pc_reversal[n]
            self.data.pc_reversal[n]     = pc_new

    def update_hysteresis_volume_fraction(self,ew_new,pc_new):
        """
        function to update hysteresis parameters when solution is in terms of volume fraction
        """
        model = self.model
        self.data.state = 1
        n = model.current_cell

        # Store hysteretic ew and pc for visualization
        self.data.ew_hysteresis[n,self.data.counter_hysteresis] = ew_new
        self.data.pc_hysteresis[n,self.data.counter_hysteresis] = pc_new

        # Update the process even drainage or imbibition
        self.update_process(ew_new-self.data.ew_old[n])

        # Update old volume fraction
        self.data.ew_old[n] = ew_new

        # Update reversal and endpoints (ew and pc)
        if self.data.drainage_new[n] != self.data.drainage_old[n]:

            self.data.pc_reversal_old[n] = self.data.pc_reversal[n]
            self.data.pc_endpoint_old[n] = self.data.pc_endpoint[n]
            self.data.ew_endpoint_old[n] = self.data.ew_endpoint[n]
            self.data.ew_reversal_old[n] = self.data.ew_reversal[n]

            self.data.state = 0
            self.data.pc_endpoint[n]     = self.data.pc_reversal[n]
            self.data.pc_reversal[n]     = model.interface.pc_function(ew_new)
            self.data.ew_endpoint[n]     = self.data.ew_reversal[n]
            self.data.ew_reversal[n]     = ew_new

    def model_update_hysteresis(self,solution_now):
        """
        function to update hysteresis parameters depending on model and formulation.
        """
        model = self.model
        nodes = model.nodes
        form  = model.model.name
        num_u = model.model.num_u

        if form =='extended-two-phase-spp':
            for n in range(0,nodes):
                model.current_cell = n
                ew_new = solution_now[num_u*n]
                pc_new = solution_now[num_u*n+2]-solution_now[num_u*n+1]
                self.update_hysteresis_volume_fraction(ew_new, pc_new)
        elif form == 'two-phase-sp':
            for n in range(0,nodes):
                model.current_cell = n
                ew_new = solution_now[num_u*n]
                self.data.state = 1
                pc_new = model.interface.pc_function(ew_new)
                self.update_hysteresis_volume_fraction(ew_new, pc_new)
        elif form == 'extended-richards-sp':
            for n in range(0,nodes):
                model.current_cell = n
                ew_new = solution_now[num_u*n]
                pc_new = -solution_now[num_u*n+1]
                self.update_hysteresis_volume_fraction(ew_new, pc_new)
        elif form == 'two-phase-pp':
            for n in range(0,nodes):
                model.current_cell = n
                pc_new = solution_now[num_u*n+1]-solution_now[num_u*n]
                self.data.state = 1
                ew_new = model.interface.volume_fraction_function(pc_new)
                self.update_hysteresis_capillary_pressure(ew_new,pc_new)
        elif form == 'richards-pr':
            for n in range(0,nodes):
                model.current_cell = n
                pc_new = -solution_now[n]
                self.data.state = 1
                ew_new = model.interface.volume_fraction_function(pc_new)
                self.update_hysteresis_capillary_pressure(ew_new,pc_new)
        else:
            raise ValueError(f"Unsupported form: {form}")

        self.data.counter_hysteresis += 1
