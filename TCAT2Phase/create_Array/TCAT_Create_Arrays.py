import numpy as np

def CreateArrays(Nx,time_interval):
    ################# Constants arrays################################
    Ve_wn_eq             = np.zeros([Nx], dtype=np.double) 
    ########################Arrays for Boundary conditions############
    VEw_bc               = np.zeros([Nx], dtype=np.double) 
    VPw_bc               = np.zeros([Nx], dtype=np.double) 
    VEwn_bc              = np.zeros([Nx], dtype=np.double) 
    VJwn_bc              = np.zeros([Nx], dtype=np.double) 
    VWxwn_bc             = np.zeros([Nx], dtype=np.double) 
    VKwn_bc              = np.zeros([Nx], dtype=np.double) 
    ####################(Arrays to store results)#######################
    Ve_w_Time_Results    = np.zeros([len(time_interval),Nx], dtype=np.double) 
    Vp_w_Time_Results    = np.zeros([len(time_interval),Nx], dtype=np.double) 
    Ve_wn_Time_Results   = np.zeros([len(time_interval),Nx], dtype=np.double) 
    Vj_wn_Time_Results   = np.zeros([len(time_interval),Nx], dtype=np.double) 
    Vw_xwn_Time_Results  = np.zeros([len(time_interval),Nx], dtype=np.double) 
    Vkn_wn_Time_Results  = np.zeros([len(time_interval),Nx], dtype=np.double) 

    ##############(Arrays to store time of simulation)#################
    Clock_Array          = np.empty([len(time_interval)], dtype=float) 
    ftime                = np.empty([len(time_interval)], dtype=float) 
    
    return Ve_wn_eq,VEw_bc,VPw_bc,VEwn_bc,VJwn_bc,VWxwn_bc,VKwn_bc,Ve_w_Time_Results,Vp_w_Time_Results,Ve_wn_Time_Results,Vj_wn_Time_Results,Vw_xwn_Time_Results,Vkn_wn_Time_Results,Clock_Array,ftime
