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
        drain,ew_reversal,ew_endpoint,pc_reversal,pc_endpoint = get_data_hysteresis(self.model)

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
        drain,ew_reversal,ew_endpoint,pc_reversal,pc_endpoint = get_data_hysteresis(self.model)

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
        drain,ew_reversal,ew_endpoint,pc_reversal,pc_endpoint = get_data_hysteresis(self.model)

        if drain:
            interface = self.drainage
        else:
            interface = self.imbibition

        ew_reversal_mpc = interface.volume_fraction_function(pc_reversal)
        ew_endpoint_mpc = interface.volume_fraction_function(pc_endpoint)
        slope           = (ew_reversal-ew_endpoint) /(ew_reversal_mpc-ew_endpoint_mpc)
        dew_hystereric  = slope*interface.d_volume_fraction_function(pc)

        return dew_hystereric

    def update_hysteresis_parameters(self,ts):
        """
        Hysteresis function
        """
        model = self.model

        drainage_old    = model.data_hysteresis['drainage_old']
        drainage_new    = model.data_hysteresis['drainage_new']
        pc_reversal     = model.data_hysteresis['p_c_reversal']
        pc_endpoint     = model.data_hysteresis['p_c_endpoint']
        ew_reversal     = model.data_hysteresis['e_w_reversal']
        ew_endpoint     = model.data_hysteresis['e_w_endpoint']
        pc_reversal_old = model.data_hysteresis['p_c_reversal_old']
        pc_endpoint_old = model.data_hysteresis['p_c_endpoint_old']
        ew_reversal_old = model.data_hysteresis['e_w_reversal_old']
        ew_endpoint_old = model.data_hysteresis['e_w_endpoint_old']
        pc_old          = model.data_hysteresis['p_c_old']
        ew_old          = model.data_hysteresis['e_w_old']

        ew_hysteresis   = model.data_hysteresis['e_w_hysteresis']
        pc_hysteresis   = model.data_hysteresis['p_c_hysteresis']
        hysteresis_tol  = model.data_hysteresis['hysteresis_tol']

        nodes = model.nodes
        form = model.model.name

        if ts.getStepNumber()==model.counter_hysteresis:
            solution_now = ts.getSolution()

            if form == 'extended-two-phase-spp':
                for n in range(0,nodes):
                    model.data_hysteresis['state'] = "new"
                    model.current_cell = n
                    pc_old[n]          = model.interface.pc_function(solution_now[3*n])
                    if solution_now[3*n]-ew_old[n]>hysteresis_tol:
                        drainage_old[n] = drainage_new[n]
                        drainage_new[n] = False
                    else:
                        drainage_old[n] = drainage_new[n]
                        drainage_new[n] = True
                    #
                    ew_old[n]  = solution_now[3*n]
                    # store hysteresis ew and pc just for visualization
                    ew_hysteresis[n,model.counter_hysteresis] = ew_old[n]
                    pc_hysteresis[n,model.counter_hysteresis] = pc_old[n]
                    # update reversal and endpoints (ew and pc)
                    if drainage_new[n] != drainage_old[n]:
                        model.data_hysteresis['state'] = "old"
                        model.current_cell = n

                        pc_reversal_old[n] = pc_reversal[n]
                        pc_endpoint_old[n] = pc_endpoint[n]
                        ew_endpoint_old[n] = ew_endpoint[n]
                        ew_reversal_old[n] = ew_reversal[n]

                        pc_endpoint[n]     = pc_reversal[n]
                        pc_reversal[n]     = model.interface.pc_function(solution_now[3*n])
                        ew_endpoint[n]     = ew_reversal[n]
                        ew_reversal[n]     = solution_now[3*n]
            elif form == 'two-phase-sp':
                for n in range(0,nodes):
                    model.data_hysteresis['state'] = "new"
                    model.current_cell = n
                    pc_old[n]          = model.interface.pc_function(solution_now[2*n])
                    if solution_now[2*n]-ew_old[n]>hysteresis_tol:
                        drainage_old[n] = drainage_new[n]
                        drainage_new[n] = False
                    else:
                        drainage_old[n] = drainage_new[n]
                        drainage_new[n] = True
                    #
                    ew_old[n]  = solution_now[2*n]
                    #  store hysteresis ew and pc just for visualization
                    ew_hysteresis[n,model.counter_hysteresis] = ew_old[n]
                    pc_hysteresis[n,model.counter_hysteresis] = pc_old[n]
                    # update reversal and endpoints (ew and pc)
                    if drainage_new[n] != drainage_old[n]:
                        model.data_hysteresis['state'] = "old"
                        model.current_cell = n

                        pc_reversal_old[n] = pc_reversal[n]
                        pc_endpoint_old[n] = pc_endpoint[n]
                        ew_endpoint_old[n] = ew_endpoint[n]
                        ew_reversal_old[n] = ew_reversal[n]

                        pc_endpoint[n]     = pc_reversal[n]
                        pc_reversal[n]     = model.interface.pc_function(solution_now[2*n])
                        ew_endpoint[n]     = ew_reversal[n]
                        ew_reversal[n]     = solution_now[2*n]

            elif form == 'two-phase-pp':
                for n in range(0,nodes):
                    model.data_hysteresis['state'] = "new"
                    model.current_cell = n
                    ew_old[n]  = model.interface.volume_fraction_function(solution_now[2*n+1]-solution_now[2*n])
                    if pc_old[n]-(solution_now[2*n+1]-solution_now[2*n])>hysteresis_tol:
                        drainage_old[n] = drainage_new[n]
                        drainage_new[n] = False
                    else:
                        drainage_old[n] = drainage_new[n]
                        drainage_new[n] = True
                    #
                    pc_old[n]         = solution_now[2*n+1]-solution_now[2*n]
                    #  store hysteresis ew and pc just for visualization
                    ew_hysteresis[n,model.counter_hysteresis] = ew_old[n]
                    pc_hysteresis[n,model.counter_hysteresis] = pc_old[n]
                    # update reversal and endpoints (ew and pc)
                    if drainage_new[n] != drainage_old[n]:
                        model.data_hysteresis['state'] = "old"
                        model.current_cell = n

                        pc_reversal_old[n] = pc_reversal[n]
                        pc_endpoint_old[n] = pc_endpoint[n]
                        ew_endpoint_old[n] = ew_endpoint[n]
                        ew_reversal_old[n] = ew_reversal[n]

                        ew_endpoint[n]     = ew_reversal[n]
                        ew_reversal[n]     = model.interface.volume_fraction_function(solution_now[2*n+1]-solution_now[2*n])
                        pc_endpoint[n]     = pc_reversal[n]
                        pc_reversal[n]     = solution_now[2*n+1]-solution_now[2*n]

            elif form == 'richards-pr':
                for n in range(0,nodes):
                    model.data_hysteresis['state'] = "new"
                    model.current_cell = n
                    ew_old[n]  = model.interface.volume_fraction_function(-solution_now[n])
                    if pc_old[n]-(-solution_now[n])>hysteresis_tol:
                        drainage_old[n] = drainage_new[n]
                        drainage_new[n] = False
                    else:
                        drainage_old[n] = drainage_new[n]
                        drainage_new[n] = True
                    #
                    pc_old[n]           = -solution_now[n]
                    #  store hysteresis ew and pc just for visualization
                    ew_hysteresis[n,model.counter_hysteresis] = ew_old[n]
                    pc_hysteresis[n,model.counter_hysteresis] = pc_old[n]
                    # update reversal and endpoints (ew and pc)
                    if drainage_new[n] != drainage_old[n]:
                        model.data_hysteresis['state'] = "old"
                        model.current_cell = n

                        pc_reversal_old[n] = pc_reversal[n]
                        pc_endpoint_old[n] = pc_endpoint[n]
                        ew_endpoint_old[n] = ew_endpoint[n]
                        ew_reversal_old[n] = ew_reversal[n]

                        ew_endpoint[n]     = ew_reversal[n]
                        ew_reversal[n]     = model.interface.volume_fraction_function(-solution_now[n])
                        pc_endpoint[n]     = pc_reversal[n]
                        pc_reversal[n]     = -solution_now[n]
            elif form == 'extended-richards-sp':
                for n in range(0,nodes):
                    model.data_hysteresis['state'] = "new"
                    model.current_cell = n
                    pc_old[n]          = model.interface.pc_function(solution_now[2*n])
                    if solution_now[2*n]-ew_old[n]>hysteresis_tol:
                        drainage_old[n] = drainage_new[n]
                        drainage_new[n] = False
                    else:
                        drainage_old[n] = drainage_new[n]
                        drainage_new[n] = True
                    #
                    ew_old[n]  = solution_now[2*n]
                    #  store hysteresis ew and pc just for visualization
                    ew_hysteresis[n,model.counter_hysteresis] = ew_old[n]
                    pc_hysteresis[n,model.counter_hysteresis] = -solution_now[2*n+1]
                    # update reversal and endpoints (ew and pc)
                    if drainage_new[n] != drainage_old[n]:
                        model.data_hysteresis['state'] = "old"
                        model.current_cell = n

                        pc_reversal_old[n] = pc_reversal[n]
                        pc_endpoint_old[n] = pc_endpoint[n]
                        ew_endpoint_old[n] = ew_endpoint[n]
                        ew_reversal_old[n] = ew_reversal[n]

                        pc_endpoint[n]     = pc_reversal[n]
                        pc_reversal[n]     = model.interface.pc_function(solution_now[2*n])
                        ew_endpoint[n]     = ew_reversal[n]
                        ew_reversal[n]     = solution_now[2*n]

            model.counter_hysteresis += 1



def get_data_hysteresis(model):
    """
    function to get the hysetresis data from model.
    """

    current_cell= model.current_cell
    state       = model.data_hysteresis['state']

    if state=="new":
        drain       = model.data_hysteresis['drainage_new'][current_cell]
        pc_reversal = model.data_hysteresis['p_c_reversal'][current_cell]
        pc_endpoint = model.data_hysteresis['p_c_endpoint'][current_cell]
        ew_reversal = model.data_hysteresis['e_w_reversal'][current_cell]
        ew_endpoint = model.data_hysteresis['e_w_endpoint'][current_cell]

    elif state=="old":
        drain       = model.data_hysteresis['drainage_old'][current_cell]
        pc_reversal = model.data_hysteresis['p_c_reversal_old'][current_cell]
        pc_endpoint = model.data_hysteresis['p_c_endpoint_old'][current_cell]
        ew_reversal = model.data_hysteresis['e_w_reversal_old'][current_cell]
        ew_endpoint = model.data_hysteresis['e_w_endpoint_old'][current_cell]

    else:
        print("state can only took two values -new- or -old-")

    return drain,ew_reversal,ew_endpoint,pc_reversal,pc_endpoint
